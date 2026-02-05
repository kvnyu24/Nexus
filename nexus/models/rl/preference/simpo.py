"""
Simple Preference Optimization (SimPO)
Paper: "SimPO: Simple Preference Optimization with a Reference-Free Reward" (Meng et al., 2024)

SimPO is a reference-free preference optimization algorithm that:
- Eliminates the need for a reference model, reducing memory and compute by ~50%
- Uses the average log probability of a sequence as an implicit reward signal
- Applies length normalization to prevent the model from favoring longer responses
- Introduces a target reward margin (gamma) in the Bradley-Terry objective to
  ensure a minimum quality gap between chosen and rejected responses

Loss formulation:
    r(y|x) = (1/|y|) * sum_{t=1}^{|y|} log pi(y_t | x, y_{<t})   (length-normalized)
    L_SimPO = -log sigma(beta * (r(y_w|x) - r(y_l|x) - gamma))

    where:
        y_w = chosen (winning) response
        y_l = rejected (losing) response
        gamma = target reward margin (encourages a gap between chosen/rejected)
        beta = inverse temperature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from ....core.base import NexusModule
from ....core.mixins import ConfigValidatorMixin


class SimPOAgent(NexusModule, ConfigValidatorMixin):
    """
    Simple Preference Optimization Agent for reference-free LLM alignment.

    SimPO removes the reference model from the preference optimization pipeline,
    instead relying on length-normalized log probabilities as implicit rewards.
    A target reward margin gamma ensures that the model learns a meaningful
    quality gap between preferred and dispreferred responses.

    Args:
        config: Configuration dictionary with:
            - policy: The policy network (language model) to align
            - beta: Inverse temperature for the Bradley-Terry model (default: 2.0)
            - gamma: Target reward margin between chosen and rejected (default: 0.5)
            - length_normalization: Whether to normalize rewards by sequence length
              (default: True)
            - learning_rate: Optimizer learning rate (default: 1e-6)
            - max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.validate_config(config, required_keys=["policy"])

        # Policy network
        self.policy = config["policy"]

        # Hyperparameters
        self.beta = config.get("beta", 2.0)
        self.gamma = config.get("gamma", 0.5)
        self.length_normalization = config.get("length_normalization", True)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.get("learning_rate", 1e-6),
            weight_decay=config.get("weight_decay", 0.01),
        )

    def _get_per_token_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities from the policy.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).

        Returns:
            Per-token log probabilities of shape (batch_size, seq_len - 1).
        """
        outputs = self.policy(input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        per_token_log_probs = log_probs.gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        return per_token_log_probs

    def compute_implicit_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the implicit reward for each sequence as (optionally length-normalized)
        sum of log probabilities over response tokens.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            action_mask: Mask for response tokens of shape (batch_size, seq_len).

        Returns:
            Implicit rewards of shape (batch_size,).
        """
        per_token_log_probs = self._get_per_token_log_probs(
            input_ids, attention_mask
        )

        # Align mask with the shifted log probs
        response_mask = action_mask[:, 1:]
        masked_log_probs = per_token_log_probs * response_mask

        # Sum log probs over response tokens
        sum_log_probs = masked_log_probs.sum(dim=-1)

        if self.length_normalization:
            # Normalize by the number of response tokens per example
            response_lengths = response_mask.sum(dim=-1).clamp(min=1.0)
            rewards = sum_log_probs / response_lengths
        else:
            rewards = sum_log_probs

        return rewards

    def compute_simpo_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the SimPO loss with target reward margin.

        Args:
            chosen_rewards: Implicit rewards for chosen responses (batch_size,).
            rejected_rewards: Implicit rewards for rejected responses (batch_size,).

        Returns:
            Dictionary with loss and diagnostic metrics.
        """
        # Bradley-Terry objective with margin
        logits = self.beta * (chosen_rewards - rejected_rewards - self.gamma)
        loss = -F.logsigmoid(logits).mean()

        # Accuracy: fraction where chosen is preferred
        accuracy = (chosen_rewards > rejected_rewards).float().mean()

        # Reward margin (how much chosen exceeds rejected on average)
        reward_margin = (chosen_rewards - rejected_rewards).mean()

        return {
            "loss": loss,
            "accuracy": accuracy.detach(),
            "reward_margin": reward_margin.detach(),
            "chosen_reward_mean": chosen_rewards.mean().detach(),
            "rejected_reward_mean": rejected_rewards.mean().detach(),
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
        Forward pass computing SimPO loss on chosen/rejected pairs.

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
        # Compute implicit rewards for both sequences
        chosen_rewards = self.compute_implicit_reward(
            chosen_input_ids, chosen_attention_mask, chosen_action_mask
        )
        rejected_rewards = self.compute_implicit_reward(
            rejected_input_ids, rejected_attention_mask, rejected_action_mask
        )

        return self.compute_simpo_loss(chosen_rewards, rejected_rewards)

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
            "reward_margin": loss_dict["reward_margin"].item(),
            "chosen_reward_mean": loss_dict["chosen_reward_mean"].item(),
            "rejected_reward_mean": loss_dict["rejected_reward_mean"].item(),
        }
