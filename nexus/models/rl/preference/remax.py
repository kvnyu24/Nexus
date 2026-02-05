"""
ReMax - REINFORCE with Maximum-Reward Baseline
Paper: "ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for
Aligning Large Language Models" (Li et al., 2024)

ReMax is a simplified REINFORCE variant for LLM alignment that:
- Uses the reward from greedy (argmax) decoding as a baseline
- Requires only a single sampled completion plus one greedy completion per prompt
- Eliminates the need for a learned critic/value network entirely
- Far more memory-efficient than PPO (no value model, no multiple rollouts)
- Provides a natural, low-overhead baseline that is strongly correlated
  with the expected reward

Loss formulation:
    For prompt x:
        y_sample ~ pi(.|x)                       (sampled completion)
        y_greedy = argmax pi(.|x)                 (greedy completion)
        r_sample = R(x, y_sample)                 (reward for sample)
        r_greedy = R(x, y_greedy)                 (reward for greedy)
        advantage = r_sample - r_greedy
        L = -advantage * sum_t log pi(y_sample_t | x, y_sample_{<t})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from ....core.base import NexusModule
from ....core.mixins import ConfigValidatorMixin


class ReMaxAgent(NexusModule, ConfigValidatorMixin):
    """
    ReMax Agent for memory-efficient LLM alignment via greedy-baseline REINFORCE.

    ReMax uses the reward of a greedily decoded completion as a control variate
    (baseline) for REINFORCE. This gives a simple, low-variance gradient
    estimator without requiring a value/critic network or multiple samples.

    Args:
        config: Configuration dictionary with:
            - policy: The policy network (language model) to optimize
            - learning_rate: Optimizer learning rate (default: 1e-6)
            - max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
            - temperature: Sampling temperature for generation (default: 1.0)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.validate_config(config, required_keys=["policy"])

        # Policy network
        self.policy = config["policy"]

        # Hyperparameters
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.temperature = config.get("temperature", 1.0)

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

    def compute_remax_loss(
        self,
        sample_log_probs: torch.Tensor,
        sample_rewards: torch.Tensor,
        greedy_rewards: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the ReMax REINFORCE loss with greedy baseline.

        Args:
            sample_log_probs: Per-token log probs for sampled completions
                of shape (batch_size, seq_len - 1).
            sample_rewards: Scalar rewards for sampled completions (batch_size,).
            greedy_rewards: Scalar rewards for greedy completions (batch_size,).
            action_mask: Response token mask (batch_size, seq_len).

        Returns:
            Dictionary with loss and diagnostic metrics.
        """
        # Advantage: sample reward minus greedy baseline
        advantages = sample_rewards - greedy_rewards

        # Normalize advantages for training stability
        adv_std = advantages.std().clamp(min=1e-8)
        normalized_advantages = (advantages - advantages.mean()) / adv_std

        # Masked sum of log probs over response tokens
        response_mask = action_mask[:, 1:]
        masked_log_probs = sample_log_probs * response_mask
        sequence_log_probs = masked_log_probs.sum(dim=-1)  # (batch_size,)

        # REINFORCE loss
        policy_loss = -(
            normalized_advantages.detach() * sequence_log_probs
        ).mean()

        # Fraction of samples that beat the greedy baseline
        improvement_rate = (advantages > 0).float().mean()

        return {
            "loss": policy_loss,
            "mean_sample_reward": sample_rewards.mean().detach(),
            "mean_greedy_reward": greedy_rewards.mean().detach(),
            "mean_advantage": advantages.mean().detach(),
            "std_advantage": advantages.std().detach(),
            "improvement_rate": improvement_rate.detach(),
        }

    def forward(
        self,
        sample_input_ids: torch.Tensor,
        sample_attention_mask: torch.Tensor,
        sample_action_mask: torch.Tensor,
        sample_rewards: torch.Tensor,
        greedy_rewards: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing ReMax loss.

        Args:
            sample_input_ids: Token IDs for sampled completions (batch_size, seq_len).
            sample_attention_mask: Attention mask for samples (batch_size, seq_len).
            sample_action_mask: Response token mask for samples (batch_size, seq_len).
            sample_rewards: Scalar rewards for sampled completions (batch_size,).
            greedy_rewards: Scalar rewards for greedy completions (batch_size,).

        Returns:
            Dictionary with loss and diagnostic metrics.
        """
        # Get log probs for sampled completions
        sample_log_probs = self._get_per_token_log_probs(
            sample_input_ids, sample_attention_mask
        )

        return self.compute_remax_loss(
            sample_log_probs, sample_rewards, greedy_rewards, sample_action_mask
        )

    def generate_sample_and_greedy(
        self,
        prompts: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """
        Generate both a sampled and a greedy completion for each prompt.

        Args:
            prompts: Input prompt IDs of shape (batch_size, prompt_len).
            attention_mask: Attention mask for prompts (batch_size, prompt_len).
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            Tuple of:
                - sample_ids: Full sampled sequences (batch_size, seq_len)
                - sample_attention_mask: Attention mask for sampled sequences
                - sample_action_mask: Mask for generated tokens in sampled sequences
                - greedy_ids: Full greedy sequences (batch_size, seq_len)
                - greedy_attention_mask: Attention mask for greedy sequences
                - greedy_action_mask: Mask for generated tokens in greedy sequences
        """
        batch_size = prompts.shape[0]
        prompt_len = prompts.shape[1]

        with torch.no_grad():
            # Generate sampled completions
            sample_ids = prompts.clone()
            sample_mask = attention_mask.clone()

            for _ in range(max_new_tokens):
                outputs = self.policy(sample_ids, attention_mask=sample_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                next_logits = logits[:, -1, :] / self.temperature
                probs = F.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                sample_ids = torch.cat([sample_ids, next_tokens], dim=1)
                sample_mask = torch.cat(
                    [sample_mask, torch.ones_like(next_tokens)], dim=1
                )

            # Generate greedy completions
            greedy_ids = prompts.clone()
            greedy_mask = attention_mask.clone()

            for _ in range(max_new_tokens):
                outputs = self.policy(greedy_ids, attention_mask=greedy_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                next_logits = logits[:, -1, :]
                next_tokens = next_logits.argmax(dim=-1, keepdim=True)
                greedy_ids = torch.cat([greedy_ids, next_tokens], dim=1)
                greedy_mask = torch.cat(
                    [greedy_mask, torch.ones_like(next_tokens)], dim=1
                )

        # Build action masks
        sample_action_mask = torch.zeros_like(sample_ids)
        sample_action_mask[:, prompt_len:] = 1.0

        greedy_action_mask = torch.zeros_like(greedy_ids)
        greedy_action_mask[:, prompt_len:] = 1.0

        return (
            sample_ids, sample_mask, sample_action_mask,
            greedy_ids, greedy_mask, greedy_action_mask,
        )

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one optimization step.

        Args:
            batch: Dictionary containing:
                - sample_input_ids: Token IDs for sampled completions
                  (batch_size, seq_len)
                - sample_attention_mask: Attention mask for samples
                  (batch_size, seq_len)
                - sample_action_mask: Response token mask for samples
                  (batch_size, seq_len)
                - sample_rewards: Scalar rewards for sampled completions
                  (batch_size,)
                - greedy_rewards: Scalar rewards for greedy completions
                  (batch_size,)

        Returns:
            Dictionary of scalar loss metrics.
        """
        sample_input_ids = batch["sample_input_ids"]
        sample_attention_mask = batch["sample_attention_mask"]
        sample_action_mask = batch.get(
            "sample_action_mask", sample_attention_mask
        )
        sample_rewards = batch["sample_rewards"]
        greedy_rewards = batch["greedy_rewards"]

        loss_dict = self.forward(
            sample_input_ids,
            sample_attention_mask,
            sample_action_mask,
            sample_rewards,
            greedy_rewards,
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
            "mean_sample_reward": loss_dict["mean_sample_reward"].item(),
            "mean_greedy_reward": loss_dict["mean_greedy_reward"].item(),
            "mean_advantage": loss_dict["mean_advantage"].item(),
            "std_advantage": loss_dict["std_advantage"].item(),
            "improvement_rate": loss_dict["improvement_rate"].item(),
        }
