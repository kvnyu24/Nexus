"""
REINFORCE Leave-One-Out (RLOO)
Paper: "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from
Human Feedback in LLMs" (Ahmadian et al., 2024)

RLOO is a variance-reduced policy gradient method for LLM alignment that:
- Generates multiple completions (K samples) per prompt
- Uses a leave-one-out baseline: for each sample i, the baseline is the mean
  reward of the other K-1 samples from the same prompt
- Avoids the need for a learned critic/value network (unlike PPO)
- Achieves lower variance than vanilla REINFORCE while remaining unbiased
- More memory-efficient than PPO since no value model is needed

Loss formulation:
    For prompt x with K samples {y_1, ..., y_K} and rewards {r_1, ..., r_K}:
        baseline_i = (1/(K-1)) * sum_{j != i} r_j
        advantage_i = r_i - baseline_i
        L = -(1/K) * sum_i advantage_i * log pi(y_i | x)

    The leave-one-out baseline is an unbiased estimate of E[r] that uses
    all available information without introducing bias from a learned model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from ....core.base import NexusModule
from ....core.mixins import ConfigValidatorMixin


class RLOOAgent(NexusModule, ConfigValidatorMixin):
    """
    REINFORCE Leave-One-Out Agent for LLM alignment with multi-sample baselines.

    RLOO draws K completions per prompt and uses a leave-one-out estimator
    as the baseline for each sample. This yields a low-variance, unbiased
    policy gradient without requiring a separate critic model.

    Args:
        config: Configuration dictionary with:
            - policy: The policy network (language model) to optimize
            - num_samples: Number of completions per prompt (K) (default: 4)
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
        self.num_samples = config.get("num_samples", 4)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.temperature = config.get("temperature", 1.0)

        if self.num_samples < 2:
            raise ValueError(
                f"num_samples must be >= 2 for leave-one-out baseline, "
                f"got {self.num_samples}"
            )

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

    def compute_leave_one_out_baselines(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute leave-one-out baselines for each sample.

        For sample i in a group of K samples, the baseline is:
            baseline_i = (sum of all rewards - reward_i) / (K - 1)

        Args:
            rewards: Reward tensor of shape (num_prompts, num_samples).

        Returns:
            Baselines of shape (num_prompts, num_samples).
        """
        K = rewards.shape[1]
        # Sum of all rewards per prompt
        reward_sum = rewards.sum(dim=1, keepdim=True)  # (num_prompts, 1)
        # Leave-one-out: (total - self) / (K - 1)
        baselines = (reward_sum - rewards) / (K - 1)
        return baselines

    def compute_rloo_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the RLOO policy gradient loss.

        Args:
            log_probs: Per-token log probabilities of shape
                (num_prompts * num_samples, seq_len - 1).
            rewards: Per-sample rewards of shape (num_prompts * num_samples,).
            action_mask: Response token mask of shape
                (num_prompts * num_samples, seq_len).

        Returns:
            Dictionary with loss and diagnostic metrics.
        """
        total_samples = rewards.shape[0]
        num_prompts = total_samples // self.num_samples

        # Reshape rewards to (num_prompts, num_samples) for baseline computation
        rewards_grouped = rewards.view(num_prompts, self.num_samples)

        # Leave-one-out baselines
        baselines = self.compute_leave_one_out_baselines(rewards_grouped)

        # Advantages
        advantages = rewards_grouped - baselines  # (num_prompts, num_samples)
        advantages_flat = advantages.view(-1)  # (total_samples,)

        # Normalize advantages for training stability
        adv_std = advantages_flat.std().clamp(min=1e-8)
        normalized_advantages = (
            advantages_flat - advantages_flat.mean()
        ) / adv_std

        # Compute sequence-level log probs (masked sum over response tokens)
        response_mask = action_mask[:, 1:]
        masked_log_probs = log_probs * response_mask
        sequence_log_probs = masked_log_probs.sum(dim=-1)  # (total_samples,)

        # REINFORCE loss: -advantage * log_prob
        # Detach advantages to avoid backpropagation through reward/baseline
        policy_loss = -(
            normalized_advantages.detach() * sequence_log_probs
        ).mean()

        return {
            "loss": policy_loss,
            "mean_reward": rewards.mean().detach(),
            "std_reward": rewards.std().detach(),
            "mean_advantage": advantages_flat.mean().detach(),
            "std_advantage": advantages_flat.std().detach(),
            "mean_baseline": baselines.mean().detach(),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing RLOO loss.

        Expects a batch where each prompt has been expanded to num_samples
        completions. The batch should be ordered so that samples for the
        same prompt are contiguous: [prompt1_s1, prompt1_s2, ..., prompt1_sK,
        prompt2_s1, ...].

        Args:
            input_ids: Token IDs of shape (num_prompts * num_samples, seq_len).
            attention_mask: Attention mask (num_prompts * num_samples, seq_len).
            action_mask: Response token mask (num_prompts * num_samples, seq_len).
            rewards: Scalar rewards per completion (num_prompts * num_samples,).

        Returns:
            Dictionary with loss and diagnostic metrics.
        """
        # Get per-token log probs
        log_probs = self._get_per_token_log_probs(input_ids, attention_mask)

        return self.compute_rloo_loss(log_probs, rewards, action_mask)

    def generate_samples(
        self,
        prompts: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate multiple completions per prompt for RLOO training.

        Args:
            prompts: Input prompt IDs of shape (num_prompts, prompt_len).
            attention_mask: Attention mask for prompts (num_prompts, prompt_len).
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            Tuple of:
                - generated_ids: Full sequences (num_prompts * num_samples, seq_len)
                - full_attention_mask: Attention mask for full sequences
                - action_mask: Mask indicating generated tokens
        """
        num_prompts = prompts.shape[0]
        prompt_len = prompts.shape[1]

        # Repeat each prompt num_samples times
        expanded_prompts = prompts.repeat_interleave(self.num_samples, dim=0)
        expanded_mask = attention_mask.repeat_interleave(self.num_samples, dim=0)

        with torch.no_grad():
            current_ids = expanded_prompts
            current_mask = expanded_mask

            for step in range(max_new_tokens):
                outputs = self.policy(current_ids, attention_mask=current_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                next_token_logits = logits[:, -1, :] / self.temperature

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

                current_ids = torch.cat([current_ids, next_tokens], dim=1)
                current_mask = torch.cat(
                    [current_mask, torch.ones_like(next_tokens)], dim=1
                )

        # Build action mask: 0 for prompt tokens, 1 for generated tokens
        action_mask = torch.zeros_like(current_ids)
        action_mask[:, prompt_len:] = 1.0

        return current_ids, current_mask, action_mask

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one optimization step.

        Args:
            batch: Dictionary containing:
                - input_ids: Token IDs (num_prompts * num_samples, seq_len).
                  Samples for the same prompt must be contiguous.
                - attention_mask: Attention mask (num_prompts * num_samples, seq_len)
                - action_mask: Response token mask (num_prompts * num_samples, seq_len)
                - rewards: Per-sample rewards (num_prompts * num_samples,)

        Returns:
            Dictionary of scalar loss metrics.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        action_mask = batch.get("action_mask", attention_mask)
        rewards = batch["rewards"]

        loss_dict = self.forward(
            input_ids, attention_mask, action_mask, rewards
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
            "mean_reward": loss_dict["mean_reward"].item(),
            "std_reward": loss_dict["std_reward"].item(),
            "mean_advantage": loss_dict["mean_advantage"].item(),
            "std_advantage": loss_dict["std_advantage"].item(),
            "mean_baseline": loss_dict["mean_baseline"].item(),
        }
