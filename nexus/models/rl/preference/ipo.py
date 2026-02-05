"""
Identity Preference Optimization (IPO)
Paper: "A General Theoretical Paradigm to Understand Learning from Human Feedback" (Azar et al., 2023)

IPO is a bounded variant of DPO that:
- Replaces the unbounded log-sigmoid loss with a squared-error (identity) loss
- Provides explicit regularization toward the reference policy
- Avoids the overfitting failure mode of DPO where the model becomes
  deterministic on preferred responses
- Has provable convergence guarantees under the general RLHF framework

Loss formulation:
    L_IPO = E[ (log_ratio_chosen - log_ratio_rejected - 1/(2*beta))^2 ]

    where:
        log_ratio_chosen  = log pi(y_w|x) - log pi_ref(y_w|x)
        log_ratio_rejected = log pi(y_l|x) - log pi_ref(y_l|x)
        beta = inverse temperature (regularization strength)

    The 1/(2*beta) term sets the target margin: the optimal policy should
    have a log-ratio difference of exactly 1/(2*beta) between chosen and
    rejected responses. This prevents the loss from driving the difference
    to infinity as in DPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from ....core.base import NexusModule
from ....core.mixins import ConfigValidatorMixin


class IPOAgent(NexusModule, ConfigValidatorMixin):
    """
    Identity Preference Optimization Agent for bounded LLM alignment.

    IPO addresses the overfitting problem of DPO by replacing the log-sigmoid
    loss with a squared-error loss that has a finite optimum. The regularization
    target 1/(2*beta) ensures the policy does not deviate too far from the
    reference while still learning to prefer chosen over rejected responses.

    Args:
        config: Configuration dictionary with:
            - policy: The policy network (language model) to align
            - reference_policy: Frozen reference policy for log-ratio computation
            - beta: Regularization strength / inverse temperature (default: 0.1)
            - learning_rate: Optimizer learning rate (default: 1e-6)
            - max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.validate_config(config, required_keys=["policy", "reference_policy"])

        # Policy networks
        self.policy = config["policy"]
        self.reference_policy = config["reference_policy"]

        # Freeze reference policy
        for param in self.reference_policy.parameters():
            param.requires_grad = False

        # Hyperparameters
        self.beta = config.get("beta", 0.1)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.get("learning_rate", 1e-6),
            weight_decay=config.get("weight_decay", 0.01),
        )

    def _get_per_token_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities for a given model.

        Args:
            model: Language model to evaluate.
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).

        Returns:
            Per-token log probabilities of shape (batch_size, seq_len - 1).
        """
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        per_token_log_probs = log_probs.gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        return per_token_log_probs

    def _compute_sequence_log_ratio(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the sequence-level log-probability ratio between policy and reference.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            action_mask: Binary mask for response tokens (batch_size, seq_len).

        Returns:
            Sequence-level log ratios of shape (batch_size,).
        """
        policy_log_probs = self._get_per_token_log_probs(
            self.policy, input_ids, attention_mask
        )

        with torch.no_grad():
            ref_log_probs = self._get_per_token_log_probs(
                self.reference_policy, input_ids, attention_mask
            )

        # Per-token log ratio
        token_log_ratios = policy_log_probs - ref_log_probs

        # Mask to response tokens
        response_mask = action_mask[:, 1:]
        masked_log_ratios = token_log_ratios * response_mask

        # Sum over response tokens for sequence-level log ratio
        sequence_log_ratios = masked_log_ratios.sum(dim=-1)
        return sequence_log_ratios

    def compute_ipo_loss(
        self,
        chosen_log_ratios: torch.Tensor,
        rejected_log_ratios: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the IPO squared-error loss.

        The loss penalizes deviation of (chosen_log_ratio - rejected_log_ratio)
        from the target margin 1/(2*beta).

        Args:
            chosen_log_ratios: Log ratios for chosen responses (batch_size,).
            rejected_log_ratios: Log ratios for rejected responses (batch_size,).

        Returns:
            Dictionary with loss and diagnostic metrics.
        """
        # Difference in log ratios
        log_ratio_diff = chosen_log_ratios - rejected_log_ratios

        # Target margin
        target = 1.0 / (2.0 * self.beta)

        # Squared-error (identity) loss
        loss = ((log_ratio_diff - target) ** 2).mean()

        # Accuracy: fraction where chosen is preferred
        accuracy = (chosen_log_ratios > rejected_log_ratios).float().mean()

        # How close the mean difference is to the target
        mean_diff = log_ratio_diff.mean()

        return {
            "loss": loss,
            "accuracy": accuracy.detach(),
            "mean_log_ratio_diff": mean_diff.detach(),
            "target_margin": torch.tensor(target, device=loss.device),
            "chosen_log_ratio_mean": chosen_log_ratios.mean().detach(),
            "rejected_log_ratio_mean": rejected_log_ratios.mean().detach(),
        }

    def forward(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        chosen_action_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        rejected_action_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing IPO loss on chosen/rejected pairs.

        Args:
            chosen_input_ids: Token IDs for chosen responses (batch_size, seq_len_c).
            chosen_attention_mask: Attention mask for chosen (batch_size, seq_len_c).
            chosen_action_mask: Response token mask for chosen (batch_size, seq_len_c).
            rejected_input_ids: Token IDs for rejected responses (batch_size, seq_len_r).
            rejected_attention_mask: Attention mask for rejected (batch_size, seq_len_r).
            rejected_action_mask: Response token mask for rejected (batch_size, seq_len_r).

        Returns:
            Dictionary with loss and diagnostic metrics.
        """
        # Compute sequence-level log ratios
        chosen_log_ratios = self._compute_sequence_log_ratio(
            chosen_input_ids, chosen_attention_mask, chosen_action_mask
        )
        rejected_log_ratios = self._compute_sequence_log_ratio(
            rejected_input_ids, rejected_attention_mask, rejected_action_mask
        )

        return self.compute_ipo_loss(chosen_log_ratios, rejected_log_ratios)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one optimization step on a batch of preference pairs.

        Args:
            batch: Dictionary containing:
                - chosen_input_ids: Token IDs for chosen responses (batch_size, seq_len)
                - chosen_attention_mask: Attention mask for chosen (batch_size, seq_len)
                - chosen_action_mask: Response token mask for chosen (batch_size, seq_len)
                - rejected_input_ids: Token IDs for rejected responses (batch_size, seq_len)
                - rejected_attention_mask: Attention mask for rejected (batch_size, seq_len)
                - rejected_action_mask: Response token mask for rejected (batch_size, seq_len)

        Returns:
            Dictionary of scalar loss metrics.
        """
        loss_dict = self.forward(
            chosen_input_ids=batch["chosen_input_ids"],
            chosen_attention_mask=batch["chosen_attention_mask"],
            chosen_action_mask=batch.get(
                "chosen_action_mask", batch["chosen_attention_mask"]
            ),
            rejected_input_ids=batch["rejected_input_ids"],
            rejected_attention_mask=batch["rejected_attention_mask"],
            rejected_action_mask=batch.get(
                "rejected_action_mask", batch["rejected_attention_mask"]
            ),
        )

        loss = loss_dict["loss"]

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": loss_dict["accuracy"].item(),
            "mean_log_ratio_diff": loss_dict["mean_log_ratio_diff"].item(),
            "target_margin": loss_dict["target_margin"].item(),
            "chosen_log_ratio_mean": loss_dict["chosen_log_ratio_mean"].item(),
            "rejected_log_ratio_mean": loss_dict["rejected_log_ratio_mean"].item(),
        }
