"""
Group Relative Policy Optimization (GRPO)
Paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (DeepSeek, 2024)

GRPO is a variant of PPO designed for LLM training that:
- Eliminates the need for a separate critic/value network
- Uses group-level baselines computed from multiple samples per prompt
- Applies PPO-style clipping with group-normalized advantages
- More memory efficient than PPO for large language models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List, Union
from ...core.base import NexusModule
from ...core.mixins import ConfigValidatorMixin
import numpy as np


class GRPOAgent(NexusModule, ConfigValidatorMixin):
    """
    Group Relative Policy Optimization Agent.

    GRPO computes advantages using group-level baselines instead of a critic:
    advantage_i = (reward_i - mean(rewards_in_group)) / std(rewards_in_group)

    This is particularly useful for LLM training where maintaining a separate
    value network would be memory-intensive.

    Args:
        config: Configuration dictionary with:
            - policy: The policy network (e.g., language model)
            - group_size: Number of samples per prompt (default: 8)
            - clip_range: PPO clipping parameter (default: 0.2)
            - kl_coef: KL divergence coefficient (default: 0.1)
            - entropy_coef: Entropy bonus coefficient (default: 0.01)
            - max_grad_norm: Gradient clipping norm (default: 1.0)
            - learning_rate: Learning rate (default: 1e-5)
            - beta: Reference model interpolation (default: 0.1)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.validate_config(config, required_keys=["policy"])

        # Policy network (could be an LLM or any policy)
        self.policy = config["policy"]

        # Optional reference policy for KL constraint
        self.reference_policy = config.get("reference_policy", None)

        # Hyperparameters
        self.group_size = config.get("group_size", 8)
        self.clip_range = config.get("clip_range", 0.2)
        self.kl_coef = config.get("kl_coef", 0.1)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.beta = config.get("beta", 0.1)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.get("learning_rate", 1e-5),
            weight_decay=config.get("weight_decay", 0.01)
        )

    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        group_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute group-relative advantages.

        Args:
            rewards: Tensor of shape (batch_size,) where batch_size = num_prompts * group_size
            group_size: Number of samples per prompt

        Returns:
            advantages: Normalized advantages of same shape as rewards
        """
        group_size = group_size or self.group_size
        batch_size = rewards.shape[0]
        num_groups = batch_size // group_size

        # Reshape to (num_groups, group_size)
        rewards_grouped = rewards.view(num_groups, group_size)

        # Compute group statistics
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8

        # Normalize within each group
        advantages = (rewards_grouped - group_mean) / group_std

        # Flatten back to (batch_size,)
        return advantages.view(-1)

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute PPO-style clipped policy loss.

        Args:
            log_probs: Current policy log probabilities
            old_log_probs: Old policy log probabilities (from sampling)
            advantages: Group-normalized advantages
            mask: Optional mask for valid tokens

        Returns:
            policy_loss: Scalar loss tensor
        """
        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

        # PPO loss (negative because we maximize)
        loss1 = -advantages * ratio
        loss2 = -advantages * clipped_ratio
        policy_loss = torch.max(loss1, loss2)

        if mask is not None:
            policy_loss = (policy_loss * mask).sum() / mask.sum()
        else:
            policy_loss = policy_loss.mean()

        return policy_loss

    def compute_kl_penalty(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute KL divergence penalty from reference policy.

        Args:
            log_probs: Current policy log probabilities
            ref_log_probs: Reference policy log probabilities
            mask: Optional mask for valid tokens

        Returns:
            kl_penalty: Scalar KL divergence
        """
        kl = log_probs - ref_log_probs

        if mask is not None:
            kl = (kl * mask).sum() / mask.sum()
        else:
            kl = kl.mean()

        return kl

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update policy using GRPO.

        Args:
            batch: Dictionary containing:
                - input_ids: Input token IDs (batch_size, seq_len)
                - attention_mask: Attention mask (batch_size, seq_len)
                - rewards: Reward for each sample (batch_size,)
                - old_log_probs: Log probs from sampling (batch_size, seq_len)
                - action_mask: Mask for action tokens (batch_size, seq_len)

        Returns:
            Dictionary with loss metrics
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        rewards = batch["rewards"]
        old_log_probs = batch["old_log_probs"]
        action_mask = batch.get("action_mask", attention_mask)

        # Compute group advantages
        advantages = self.compute_group_advantages(rewards)

        # Forward pass through current policy
        outputs = self.policy(input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather log probs for selected actions
        log_probs = log_probs[:, :-1].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        # Expand advantages to match sequence length
        advantages_expanded = advantages.unsqueeze(-1).expand_as(log_probs)

        # Compute policy loss
        policy_loss = self.compute_policy_loss(
            log_probs,
            old_log_probs[:, :-1] if old_log_probs.dim() > 1 else old_log_probs.unsqueeze(-1).expand_as(log_probs),
            advantages_expanded,
            action_mask[:, 1:] if action_mask is not None else None
        )

        # Compute KL penalty if reference policy exists
        kl_loss = torch.tensor(0.0, device=input_ids.device)
        if self.reference_policy is not None:
            with torch.no_grad():
                ref_outputs = self.reference_policy(input_ids, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits if hasattr(ref_outputs, 'logits') else ref_outputs
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_log_probs = ref_log_probs[:, :-1].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

            kl_loss = self.compute_kl_penalty(
                log_probs,
                ref_log_probs,
                action_mask[:, 1:] if action_mask is not None else None
            )

        # Compute entropy bonus
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs.unsqueeze(-1)).sum(dim=-1)
        if action_mask is not None:
            entropy = (entropy[:, :-1] * action_mask[:, 1:]).sum() / action_mask[:, 1:].sum()
        else:
            entropy = entropy.mean()

        # Total loss
        total_loss = policy_loss + self.kl_coef * kl_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "entropy": entropy.item(),
            "mean_advantage": advantages.mean().item(),
            "std_advantage": advantages.std().item()
        }

    def generate_samples(
        self,
        prompts: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate multiple samples per prompt for GRPO training.

        Args:
            prompts: Input prompt IDs (num_prompts, prompt_len)
            attention_mask: Attention mask for prompts
            max_length: Maximum generation length
            temperature: Sampling temperature

        Returns:
            generated_ids: Generated token IDs (num_prompts * group_size, seq_len)
            log_probs: Log probabilities of generated tokens
        """
        num_prompts = prompts.shape[0]

        # Repeat each prompt group_size times
        prompts_expanded = prompts.repeat_interleave(self.group_size, dim=0)
        mask_expanded = attention_mask.repeat_interleave(self.group_size, dim=0)

        # Generate using the policy
        with torch.no_grad():
            generated_ids = []
            log_probs_list = []
            current_ids = prompts_expanded
            current_mask = mask_expanded

            for _ in range(max_length - prompts.shape[1]):
                outputs = self.policy(current_ids, attention_mask=current_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                next_token_logits = logits[:, -1, :] / temperature

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                next_log_probs = F.log_softmax(next_token_logits, dim=-1)
                token_log_probs = next_log_probs.gather(1, next_tokens)

                generated_ids.append(next_tokens)
                log_probs_list.append(token_log_probs)

                # Update for next iteration
                current_ids = torch.cat([current_ids, next_tokens], dim=1)
                current_mask = torch.cat([current_mask, torch.ones_like(next_tokens)], dim=1)

            generated_ids = torch.cat(generated_ids, dim=1)
            log_probs = torch.cat(log_probs_list, dim=1)

        return torch.cat([prompts_expanded, generated_ids], dim=1), log_probs


class GRPOTrainer:
    """
    Trainer class for GRPO that handles the full training loop.

    This includes:
    - Sampling multiple completions per prompt
    - Computing rewards (via reward model or rule-based)
    - Computing group advantages
    - Updating the policy
    """

    def __init__(
        self,
        agent: GRPOAgent,
        reward_fn: callable,
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent = agent
        self.reward_fn = reward_fn
        self.config = config or {}

        self.num_epochs = self.config.get("num_epochs", 1)
        self.batch_size = self.config.get("batch_size", 8)

    def train_step(
        self,
        prompts: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            prompts: Batch of prompt token IDs
            attention_mask: Attention mask for prompts

        Returns:
            Dictionary of training metrics
        """
        # Generate samples
        generated_ids, log_probs = self.agent.generate_samples(
            prompts,
            attention_mask,
            max_length=self.config.get("max_length", 512),
            temperature=self.config.get("temperature", 1.0)
        )

        # Compute rewards
        rewards = self.reward_fn(generated_ids)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, device=prompts.device)

        # Prepare batch
        batch = {
            "input_ids": generated_ids,
            "attention_mask": torch.ones_like(generated_ids),
            "rewards": rewards,
            "old_log_probs": log_probs
        }

        # Update policy
        metrics = self.agent.update(batch)

        # Add reward statistics
        metrics["mean_reward"] = rewards.mean().item()
        metrics["std_reward"] = rewards.std().item()
        metrics["max_reward"] = rewards.max().item()
        metrics["min_reward"] = rewards.min().item()

        return metrics
