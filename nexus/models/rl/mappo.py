"""
Multi-Agent Proximal Policy Optimization (MAPPO)
Paper: "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (Yu et al., 2022)

MAPPO extends PPO to multi-agent settings using Centralized Training with
Decentralized Execution (CTDE):
- Each agent has its own actor (policy) network conditioned on local observations
- A shared centralized critic takes the global state (or joint observations) for
  value estimation during training
- At execution time, each agent acts independently using only its local observation
- PPO clipped surrogate objective is applied per-agent for stable policy updates
- Generalized Advantage Estimation (GAE) is computed using the shared critic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional, Union
from ...core.base import NexusModule
import numpy as np


class SharedCritic(NexusModule):
    """
    Centralized value function for MAPPO.

    Takes the global state (or concatenated observations from all agents) and
    outputs a single scalar value estimate. Used only during training for
    computing advantages and value targets.

    Args:
        state_dim: Dimension of the global state input
        hidden_dim: Hidden layer size
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimate from global state.

        Args:
            state: Global state tensor [batch_size, state_dim]

        Returns:
            Value estimate [batch_size, 1]
        """
        return self.network(state)


class MAPPOActor(NexusModule):
    """
    Decentralized actor (policy) network for a single agent.

    Takes the agent's local observation and outputs a Gaussian policy
    (mean and log_std) over continuous actions.

    Args:
        obs_dim: Dimension of the agent's local observation
        action_dim: Dimension of the action space
        hidden_dim: Hidden layer size
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute policy distribution parameters.

        Args:
            obs: Local observation [batch_size, obs_dim]

        Returns:
            Tuple of (action_mean, action_std)
        """
        features = self.network(obs)
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Get the policy distribution for the given observation."""
        mean, std = self.forward(obs)
        return torch.distributions.Normal(mean, std)


class MAPPOAgent(NexusModule):
    """
    Multi-Agent PPO (MAPPO) Agent with CTDE paradigm.

    Manages multiple agents, each with their own actor network but sharing a
    single centralized critic. Training uses PPO's clipped surrogate objective
    per agent with GAE advantages computed from the shared critic.

    Args:
        config: Configuration dictionary with:
            - num_agents: Number of agents
            - state_dim: Dimension of global state (for centralized critic)
            - obs_dim: Dimension of per-agent local observation
            - action_dim: Dimension of action space (shared across agents)
            - hidden_dim: Hidden layer size (default: 256)
            - clip_range: PPO clipping parameter (default: 0.2)
            - gamma: Discount factor (default: 0.99)
            - gae_lambda: GAE lambda parameter (default: 0.95)
            - value_coef: Value loss coefficient (default: 0.5)
            - entropy_coef: Entropy bonus coefficient (default: 0.01)
            - max_grad_norm: Maximum gradient norm for clipping (default: 0.5)
            - learning_rate: Learning rate (default: 3e-4)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Core dimensions
        self.num_agents = config["num_agents"]
        self.state_dim = config["state_dim"]
        self.obs_dim = config["obs_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)

        # Hyperparameters
        self.clip_range = config.get("clip_range", 0.2)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)

        # Shared centralized critic
        self.critic = SharedCritic(self.state_dim, self.hidden_dim)

        # Per-agent actor networks
        self.actors = nn.ModuleList([
            MAPPOActor(self.obs_dim, self.action_dim, self.hidden_dim)
            for _ in range(self.num_agents)
        ])

        # Optimizers
        self.actor_optimizers = [
            torch.optim.Adam(
                actor.parameters(),
                lr=config.get("learning_rate", 3e-4),
                eps=1e-5
            )
            for actor in self.actors
        ]
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.get("learning_rate", 3e-4),
            eps=1e-5
        )

    def select_action(
        self,
        observations: List[Union[np.ndarray, torch.Tensor]],
        deterministic: bool = False
    ) -> Tuple[List[np.ndarray], List[Dict[str, torch.Tensor]]]:
        """
        Select actions for all agents given their local observations.

        Args:
            observations: List of per-agent observations [num_agents]
            deterministic: If True, use mean action (no sampling)

        Returns:
            Tuple of (actions, infos) where actions is a list of numpy arrays
            and infos contains log_probs and action means per agent
        """
        actions = []
        infos = []

        with torch.no_grad():
            for i, (actor, obs) in enumerate(zip(self.actors, observations)):
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                else:
                    obs_tensor = obs.unsqueeze(0) if obs.dim() == 1 else obs

                dist = actor.get_distribution(obs_tensor)

                if deterministic:
                    action = dist.mean
                else:
                    action = dist.sample()

                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

                actions.append(action.cpu().numpy()[0])
                infos.append({
                    "log_prob": log_prob,
                    "action_mean": dist.mean,
                    "action_std": dist.stddev,
                })

        return actions, infos

    def compute_gae(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation using the shared critic.

        Args:
            values: Value estimates [T]
            rewards: Rewards [T]
            dones: Done flags [T]
            next_values: Next step value estimates [T]

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values

        return advantages, returns

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update all actors and the shared critic using PPO objective.

        Args:
            batch: Dictionary containing per-agent data:
                - observations: List of observation tensors per agent [num_agents x [batch, obs_dim]]
                - actions: List of action tensors per agent [num_agents x [batch, action_dim]]
                - old_log_probs: List of old log probs per agent [num_agents x [batch, 1]]
                - states: Global state tensor [batch, state_dim]
                - rewards: Reward tensor [batch]
                - dones: Done flag tensor [batch]
                - advantages: Pre-computed advantages [batch]
                - returns: Pre-computed returns [batch]

        Returns:
            Dictionary with loss metrics
        """
        states = batch["states"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        # Update shared critic
        values = self.critic(states).squeeze(-1)
        value_loss = F.mse_loss(values, returns)

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Update each agent's actor
        total_policy_loss = 0.0
        total_entropy = 0.0

        for i in range(self.num_agents):
            obs_i = batch["observations"][i]
            actions_i = batch["actions"][i]
            old_log_probs_i = batch["old_log_probs"][i]

            # Get current policy distribution
            dist = self.actors[i].get_distribution(obs_i)
            new_log_probs = dist.log_prob(actions_i).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1).mean()

            # PPO clipped surrogate objective
            ratio = (new_log_probs - old_log_probs_i).exp()
            adv = advantages.unsqueeze(-1) if advantages.dim() == 1 else advantages

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv

            policy_loss = -torch.min(surr1, surr2).mean()
            actor_loss = policy_loss - self.entropy_coef * entropy

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
            self.actor_optimizers[i].step()

            total_policy_loss += policy_loss.item()
            total_entropy += entropy.item()

        return {
            "value_loss": value_loss.item(),
            "policy_loss": total_policy_loss / self.num_agents,
            "entropy": total_entropy / self.num_agents,
        }

    def forward(self, observations: List[torch.Tensor], state: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass for all agents.

        Args:
            observations: List of per-agent observation tensors
            state: Global state tensor for the critic

        Returns:
            Dictionary with actions, log_probs, and value estimate
        """
        actions = []
        log_probs = []

        for i, (actor, obs) in enumerate(zip(self.actors, observations)):
            dist = actor.get_distribution(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            actions.append(action)
            log_probs.append(log_prob)

        value = self.critic(state)

        return {
            "actions": actions,
            "log_probs": log_probs,
            "value": value,
        }
