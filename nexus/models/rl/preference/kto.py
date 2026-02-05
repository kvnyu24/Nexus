"""
Kahneman-Tversky Optimization (KTO)
Paper: "KTO: Model Alignment as Prospect Theoretic Optimization" (Ethayarajh et al., 2024)

KTO is a preference optimization algorithm that:
- Requires only binary feedback (good/bad) rather than pairwise preferences
- Grounded in Kahneman & Tversky's prospect theory from behavioral economics
- Models loss aversion: humans weight losses more heavily than equivalent gains
- Computes a reference point (z_ref) from the KL divergence of the policy from reference
- Applies asymmetric weighting (lambda_good, lambda_bad) to desirable vs undesirable examples
- More data-efficient than DPO since it does not require paired comparisons

Loss formulation:
    For desirable examples (y_good):
        L_good = lambda_good * (1 - sigma(beta * (log_ratio - z_ref)))
    For undesirable examples (y_bad):
        L_bad  = lambda_bad  * (1 - sigma(-beta * (log_ratio - z_ref)))
    where:
        log_ratio = log pi(y|x) - log pi_ref(y|x)
        z_ref     = E[KL(pi || pi_ref)] (estimated from the batch)
        sigma     = sigmoid function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from ....core.base import NexusModule
from ....core.mixins import ConfigValidatorMixin


class KTOAgent(NexusModule, ConfigValidatorMixin):
    """
    Kahneman-Tversky Optimization Agent for LLM alignment with binary feedback.

    Unlike DPO which requires paired (chosen, rejected) examples, KTO operates
    on individual examples labeled as desirable or undesirable. This makes it
    more practical for real-world feedback collection where pairwise comparisons
    are expensive or unavailable.

    Args:
        config: Configuration dictionary with:
            - policy: The policy network (language model) to align
            - reference_policy: Frozen reference policy for KL computation
            - beta: Inverse temperature controlling deviation from reference (default: 0.1)
            - lambda_good: Weight for desirable examples (default: 1.0)
            - lambda_bad: Weight for undesirable examples; set > lambda_good for
              loss aversion (default: 1.0)
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
        self.lambda_good = config.get("lambda_good", 1.0)
        self.lambda_bad = config.get("lambda_bad", 1.0)
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
        # Shift: predict token t+1 from position t
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        # Gather the log probs of the actual next tokens
        per_token_log_probs = log_probs.gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        return per_token_log_probs

    def _compute_sequence_log_ratios(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-example sum of log-probability ratios (policy vs reference).

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            action_mask: Mask for response tokens of shape (batch_size, seq_len).

        Returns:
            Sequence-level log ratios of shape (batch_size,).
        """
        # Policy log probs
        policy_log_probs = self._get_per_token_log_probs(
            self.policy, input_ids, attention_mask
        )

        # Reference log probs (no gradient)
        with torch.no_grad():
            ref_log_probs = self._get_per_token_log_probs(
                self.reference_policy, input_ids, attention_mask
            )

        # Per-token log ratio
        token_log_ratios = policy_log_probs - ref_log_probs

        # Mask to response tokens only (shifted by 1 to align with logit shift)
        response_mask = action_mask[:, 1:]
        masked_log_ratios = token_log_ratios * response_mask

        # Sum over tokens to get sequence-level log ratio
        sequence_log_ratios = masked_log_ratios.sum(dim=-1)
        return sequence_log_ratios

    def _compute_kl_reference_point(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the KL reference point z_ref from the current batch.

        z_ref = E[KL(pi || pi_ref)] estimated as the mean of per-token
        log ratios across all response tokens in the batch.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            action_mask: Mask for response tokens of shape (batch_size, seq_len).

        Returns:
            Scalar reference point z_ref.
        """
        with torch.no_grad():
            policy_log_probs = self._get_per_token_log_probs(
                self.policy, input_ids, attention_mask
            )
            ref_log_probs = self._get_per_token_log_probs(
                self.reference_policy, input_ids, attention_mask
            )
            token_log_ratios = policy_log_probs - ref_log_probs
            response_mask = action_mask[:, 1:]

            # Mean KL across all response tokens in the batch
            total_tokens = response_mask.sum().clamp(min=1.0)
            z_ref = (token_log_ratios * response_mask).sum() / total_tokens

        return z_ref

    def compute_kto_loss(
        self,
        log_ratios: torch.Tensor,
        is_desirable: torch.Tensor,
        z_ref: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the KTO loss based on prospect theory.

        Args:
            log_ratios: Sequence-level log ratios of shape (batch_size,).
            is_desirable: Binary indicator, 1.0 for good examples, 0.0 for bad.
                Shape (batch_size,).
            z_ref: Scalar KL reference point.

        Returns:
            Dictionary containing total_loss and per-category losses.
        """
        good_mask = is_desirable.bool()
        bad_mask = ~good_mask

        # Prospect-theoretic value function
        # Good examples: want sigma(beta * (log_ratio - z_ref)) to be high
        # Bad examples: want sigma(-beta * (log_ratio - z_ref)) to be high (i.e., log_ratio < z_ref)
        loss = torch.zeros(1, device=log_ratios.device)
        metrics = {}

        if good_mask.any():
            good_log_ratios = log_ratios[good_mask]
            good_losses = 1.0 - torch.sigmoid(
                self.beta * (good_log_ratios - z_ref)
            )
            weighted_good_loss = self.lambda_good * good_losses.mean()
            loss = loss + weighted_good_loss
            metrics["good_loss"] = weighted_good_loss.detach()
        else:
            metrics["good_loss"] = torch.tensor(0.0, device=log_ratios.device)

        if bad_mask.any():
            bad_log_ratios = log_ratios[bad_mask]
            bad_losses = 1.0 - torch.sigmoid(
                -self.beta * (bad_log_ratios - z_ref)
            )
            weighted_bad_loss = self.lambda_bad * bad_losses.mean()
            loss = loss + weighted_bad_loss
            metrics["bad_loss"] = weighted_bad_loss.detach()
        else:
            metrics["bad_loss"] = torch.tensor(0.0, device=log_ratios.device)

        metrics["total_loss"] = loss
        return metrics

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
        is_desirable: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing KTO loss.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            action_mask: Binary mask for response/action tokens (batch_size, seq_len).
            is_desirable: Binary labels, 1.0 for desirable, 0.0 for undesirable.
                Shape (batch_size,).

        Returns:
            Dictionary with loss values and diagnostic metrics.
        """
        # Compute reference point from the full batch
        z_ref = self._compute_kl_reference_point(
            input_ids, attention_mask, action_mask
        )

        # Compute sequence-level log ratios
        log_ratios = self._compute_sequence_log_ratios(
            input_ids, attention_mask, action_mask
        )

        # Compute KTO loss
        loss_dict = self.compute_kto_loss(log_ratios, is_desirable, z_ref)

        loss_dict["z_ref"] = z_ref.detach()
        loss_dict["mean_log_ratio"] = log_ratios.mean().detach()
        return loss_dict

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one optimization step on a batch of binary-feedback examples.

        Args:
            batch: Dictionary containing:
                - input_ids: Token IDs (batch_size, seq_len)
                - attention_mask: Attention mask (batch_size, seq_len)
                - action_mask: Mask for response tokens (batch_size, seq_len)
                - is_desirable: Binary labels (batch_size,), 1=good, 0=bad

        Returns:
            Dictionary of scalar loss metrics.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        action_mask = batch.get("action_mask", attention_mask)
        is_desirable = batch["is_desirable"]

        # Forward pass
        loss_dict = self.forward(
            input_ids, attention_mask, action_mask, is_desirable
        )
        total_loss = loss_dict["total_loss"]

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "good_loss": loss_dict["good_loss"].item(),
            "bad_loss": loss_dict["bad_loss"].item(),
            "z_ref": loss_dict["z_ref"].item(),
            "mean_log_ratio": loss_dict["mean_log_ratio"].item(),
        }
