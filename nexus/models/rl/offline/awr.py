"""
Advantage Weighted Regression (AWR)
Paper: "Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning"
       (Peng et al., 2019)

AWR is a simple offline RL algorithm that:
- Combines policy gradient with weighted behavior cloning
- Weights behavioral actions by their exponential advantage
- Only imitates good actions (positive advantage) from the dataset
- Simple, stable, and easy to implement
- Works well for offline RL and imitation learning

Key idea: π* = argmax E[exp(A(s,a)/β) * log π(a|s)]
where β is a temperature parameter controlling the sharpness of the weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
from ....core.base import NexusModule
import numpy as np


class AWRActor(nn.Module):
    """
    Actor network for AWR with continuous actions.

    Outputs a Gaussian policy: mean and log_std for each action dimension.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer size
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass outputting policy distribution parameters.

        Args:
            state: Input state [batch, state_dim]

        Returns:
            Tuple of (mean, log_std) for Gaussian policy
        """
        features = self.network(state)
        mean = self.mean_head(features)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def get_distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        """Get the policy distribution for the given state."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        return torch.distributions.Normal(mean, std)

    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions under the current policy."""
        dist = self.get_distribution(state)
        return dist.log_prob(action).sum(dim=-1)


class AWRCritic(nn.Module):
    """
    Critic network (value function) for AWR.

    Args:
        state_dim: Dimension of state space
        hidden_dim: Hidden layer size
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing state value.

        Args:
            state: Input state [batch, state_dim]

        Returns:
            Value estimate [batch, 1]
        """
        return self.network(state)


class AWRAgent(NexusModule):
    """
    Advantage Weighted Regression Agent for offline RL.

    AWR trains the policy to imitate actions from the dataset, weighted
    by their exponential advantage. This naturally focuses learning on
    high-quality actions while ignoring poor ones.

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state space
            - action_dim: Dimension of action space
            - hidden_dim: Hidden layer size (default: 256)
            - gamma: Discount factor (default: 0.99)
            - beta: Temperature for advantage weighting (default: 0.3)
            - actor_lr: Actor learning rate (default: 3e-4)
            - critic_lr: Critic learning rate (default: 3e-4)
            - value_iters: Value function update iterations per batch (default: 5)
            - actor_iters: Actor update iterations per batch (default: 1)
            - max_weight: Maximum advantage weight clipping (default: 20.0)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.gamma = config.get("gamma", 0.99)
        self.beta = config.get("beta", 0.3)
        self.value_iters = config.get("value_iters", 5)
        self.actor_iters = config.get("actor_iters", 1)
        self.max_weight = config.get("max_weight", 20.0)

        # Actor and critic networks
        self.actor = AWRActor(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic = AWRCritic(self.state_dim, self.hidden_dim)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.get("actor_lr", 3e-4)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.get("critic_lr", 3e-4)
        )

    def select_action(
        self, state: Union[np.ndarray, torch.Tensor], deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action using the current policy.

        Args:
            state: Current state
            deterministic: If True, use mean action

        Returns:
            Action as numpy array
        """
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            dist = self.actor.get_distribution(state)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()

            return action.cpu().numpy()[0]

    def compute_advantages(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute advantages A(s,a) = r + γV(s') - V(s).

        Args:
            states: States [batch, state_dim]
            actions: Actions [batch, action_dim]
            rewards: Rewards [batch]
            next_states: Next states [batch, state_dim]
            dones: Done flags [batch]

        Returns:
            Advantages [batch]
        """
        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)
            target_values = rewards + self.gamma * (1 - dones) * next_values
            advantages = target_values - values

        return advantages

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update AWR agent: first update critic, then update actor with advantage weights.

        Args:
            batch: Dictionary containing:
                - states: States [batch, state_dim]
                - actions: Actions [batch, action_dim]
                - rewards: Rewards [batch]
                - next_states: Next states [batch, state_dim]
                - dones: Done flags [batch]

        Returns:
            Dictionary with loss metrics
        """
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Ensure rewards and dones are 1D
        if rewards.dim() == 2:
            rewards = rewards.squeeze(-1)
        if dones.dim() == 2:
            dones = dones.squeeze(-1)

        # --- Update Value Function ---
        value_losses = []
        for _ in range(self.value_iters):
            values = self.critic(states).squeeze(-1)
            with torch.no_grad():
                next_values = self.critic(next_states).squeeze(-1)
                target_values = rewards + self.gamma * (1 - dones) * next_values

            value_loss = F.mse_loss(values, target_values)

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

            value_losses.append(value_loss.item())

        # --- Compute Advantages ---
        advantages = self.compute_advantages(states, actions, rewards, next_states, dones)

        # Compute advantage weights: exp(A(s,a) / β)
        weights = torch.exp(advantages / self.beta)
        weights = torch.clamp(weights, max=self.max_weight)

        # --- Update Policy with Advantage Weighting ---
        actor_losses = []
        for _ in range(self.actor_iters):
            log_probs = self.actor.get_log_prob(states, actions)

            # Weighted behavior cloning loss
            # AWR loss: -E[w(s,a) * log π(a|s)]
            actor_loss = -(weights * log_probs).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())

        return {
            "value_loss": np.mean(value_losses),
            "actor_loss": np.mean(actor_losses),
            "mean_advantage": advantages.mean().item(),
            "mean_weight": weights.mean().item(),
            "max_weight": weights.max().item(),
            "mean_value": values.mean().item(),
        }

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing policy and value.

        Args:
            state: Input state

        Returns:
            Dictionary with action, log_prob, and value
        """
        dist = self.actor.get_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        value = self.critic(state)

        return {
            "action": action,
            "log_prob": log_prob,
            "value": value,
        }
