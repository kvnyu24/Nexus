"""
Odds Ratio Preference Optimization (ORPO)
Paper: "ORPO: Monolithic Preference Optimization without Reference Model" (Hong et al., 2024)

ORPO is a monolithic alignment algorithm that:
- Combines supervised fine-tuning (SFT) and preference alignment in a single pass
- Requires no reference model, reducing memory and compute overhead
- Uses an odds-ratio-based penalty to discourage generation of rejected responses
- Jointly optimizes the SFT objective on chosen responses and a log-odds penalty

Loss formulation:
    L_SFT    = -E[ (1/|y_w|) * sum log pi(y_w_t | x, y_w_{<t}) ]
    odds(y)  = P(y|x) / (1 - P(y|x))
    log_OR   = log(odds(y_w|x)) - log(odds(y_l|x))
    L_OR     = -log sigma(log_OR)
    L_ORPO   = L_SFT + lambda * L_OR

    where:
        y_w   = chosen (winning) response
        y_l   = rejected (losing) response
        lambda = weighting coefficient for the odds-ratio penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from ....core.base import NexusModule
from ....core.mixins import ConfigValidatorMixin


class ORPOAgent(NexusModule, ConfigValidatorMixin):
    """
    Odds Ratio Preference Optimization Agent for monolithic LLM alignment.

    ORPO unifies SFT and alignment into one training objective, removing the
    separate SFT pre-training stage and the reference model. The odds-ratio
    penalty naturally discourages rejected responses by contrasting the
    likelihood odds of chosen vs. rejected completions.

    Args:
        config: Configuration dictionary with:
            - policy: The policy network (language model) to align
            - lambda_weight: Weight for the odds-ratio penalty (default: 1.0)
            - learning_rate: Optimizer learning rate (default: 1e-6)
            - max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.validate_config(config, required_keys=["policy"])

        # Policy network
        self.policy = config["policy"]

        # Hyperparameters
        self.lambda_weight = config.get("lambda_weight", 1.0)
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

    def _compute_average_log_prob(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the average log probability over response tokens.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            action_mask: Binary mask for response tokens (batch_size, seq_len).

        Returns:
            Average log probability per example of shape (batch_size,).
        """
        per_token_log_probs = self._get_per_token_log_probs(
            input_ids, attention_mask
        )
        response_mask = action_mask[:, 1:]
        masked_log_probs = per_token_log_probs * response_mask
        sum_log_probs = masked_log_probs.sum(dim=-1)
        response_lengths = response_mask.sum(dim=-1).clamp(min=1.0)
        return sum_log_probs / response_lengths

    def compute_sft_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the SFT (negative log-likelihood) loss on chosen responses.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            action_mask: Binary mask for response tokens (batch_size, seq_len).

        Returns:
            Scalar SFT loss.
        """
        per_token_log_probs = self._get_per_token_log_probs(
            input_ids, attention_mask
        )
        response_mask = action_mask[:, 1:]
        masked_nll = -per_token_log_probs * response_mask
        total_tokens = response_mask.sum().clamp(min=1.0)
        sft_loss = masked_nll.sum() / total_tokens
        return sft_loss

    def compute_log_odds(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the log-odds of the response: log(P(y|x) / (1 - P(y|x))).

        We compute this at the token level and average over response tokens:
            log_odds = (1/|y|) * sum_t [ log p(y_t) - log(1 - p(y_t)) ]

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            action_mask: Binary mask for response tokens (batch_size, seq_len).

        Returns:
            Average log-odds per example of shape (batch_size,).
        """
        per_token_log_probs = self._get_per_token_log_probs(
            input_ids, attention_mask
        )
        response_mask = action_mask[:, 1:]

        # log(1 - p) = log(1 - exp(log_p))
        # Use log1p(-exp(log_p)) for numerical stability
        per_token_log_one_minus_probs = torch.log1p(
            -torch.exp(per_token_log_probs.clamp(max=-1e-7))
        )

        # log_odds_t = log_p_t - log(1 - p_t)
        per_token_log_odds = per_token_log_probs - per_token_log_one_minus_probs

        masked_log_odds = per_token_log_odds * response_mask
        response_lengths = response_mask.sum(dim=-1).clamp(min=1.0)
        avg_log_odds = masked_log_odds.sum(dim=-1) / response_lengths
        return avg_log_odds

    def compute_orpo_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        chosen_action_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        rejected_action_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the full ORPO loss: SFT + lambda * odds-ratio penalty.

        Args:
            chosen_input_ids: Token IDs for chosen responses.
            chosen_attention_mask: Attention mask for chosen responses.
            chosen_action_mask: Response token mask for chosen.
            rejected_input_ids: Token IDs for rejected responses.
            rejected_attention_mask: Attention mask for rejected responses.
            rejected_action_mask: Response token mask for rejected.

        Returns:
            Dictionary with component losses and diagnostic metrics.
        """
        # SFT loss on chosen responses
        sft_loss = self.compute_sft_loss(
            chosen_input_ids, chosen_attention_mask, chosen_action_mask
        )

        # Log-odds for chosen and rejected
        chosen_log_odds = self.compute_log_odds(
            chosen_input_ids, chosen_attention_mask, chosen_action_mask
        )
        rejected_log_odds = self.compute_log_odds(
            rejected_input_ids, rejected_attention_mask, rejected_action_mask
        )

        # Log odds ratio
        log_odds_ratio = chosen_log_odds - rejected_log_odds

        # Odds-ratio loss (Bradley-Terry on log-odds)
        or_loss = -F.logsigmoid(log_odds_ratio).mean()

        # Total ORPO loss
        total_loss = sft_loss + self.lambda_weight * or_loss

        # Accuracy diagnostic
        accuracy = (chosen_log_odds > rejected_log_odds).float().mean()

        return {
            "total_loss": total_loss,
            "sft_loss": sft_loss.detach(),
            "or_loss": or_loss.detach(),
            "accuracy": accuracy.detach(),
            "log_odds_ratio_mean": log_odds_ratio.mean().detach(),
            "chosen_log_odds_mean": chosen_log_odds.mean().detach(),
            "rejected_log_odds_mean": rejected_log_odds.mean().detach(),
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
        Forward pass computing the ORPO loss.

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
        return self.compute_orpo_loss(
            chosen_input_ids,
            chosen_attention_mask,
            chosen_action_mask,
            rejected_input_ids,
            rejected_attention_mask,
            rejected_action_mask,
        )

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
            "sft_loss": loss_dict["sft_loss"].item(),
            "or_loss": loss_dict["or_loss"].item(),
            "accuracy": loss_dict["accuracy"].item(),
            "log_odds_ratio_mean": loss_dict["log_odds_ratio_mean"].item(),
            "chosen_log_odds_mean": loss_dict["chosen_log_odds_mean"].item(),
            "rejected_log_odds_mean": loss_dict["rejected_log_odds_mean"].item(),
        }
