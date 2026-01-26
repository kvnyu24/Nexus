"""
Deep Deterministic Policy Gradient (DDPG)
Paper: "Continuous control with deep reinforcement learning" (Lillicrap et al., 2015)

DDPG is an actor-critic algorithm for continuous action spaces that uses:
- Deterministic policy gradient for the actor
- Q-learning for the critic
- Target networks for stability
- Ornstein-Uhlenbeck noise for exploration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
from ...core.base import NexusModule
import numpy as np


class Actor(NexusModule):
    """Deterministic policy network that outputs continuous actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        max_action: float = 1.0
    ):
        super().__init__()
        self.max_action = max_action

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        # Initialize final layer with small weights
        self.network[-2].weight.data.uniform_(-3e-3, 3e-3)
        self.network[-2].bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.network(state)


class Critic(NexusModule):
    """Q-value network that estimates Q(s, a)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize final layer with small weights
        self.network[-1].weight.data.uniform_(-3e-3, 3e-3)
        self.network[-1].bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise."""

    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2
    ):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class DDPGAgent(NexusModule):
    """
    DDPG Agent for continuous control tasks.

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state space
            - action_dim: Dimension of action space
            - hidden_dim: Hidden layer size (default: 256)
            - max_action: Maximum action value (default: 1.0)
            - actor_lr: Actor learning rate (default: 1e-4)
            - critic_lr: Critic learning rate (default: 1e-3)
            - gamma: Discount factor (default: 0.99)
            - tau: Soft update coefficient (default: 0.005)
            - noise_sigma: Exploration noise std (default: 0.1)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Dimensions
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.max_action = config.get("max_action", 1.0)

        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.actor_lr = config.get("actor_lr", 1e-4)
        self.critic_lr = config.get("critic_lr", 1e-3)

        # Networks
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim, self.max_action)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.hidden_dim, self.max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # Exploration noise
        self.noise = OUNoise(
            self.action_dim,
            sigma=config.get("noise_sigma", 0.1)
        )

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        add_noise: bool = True
    ) -> np.ndarray:
        """Select action using the actor network with optional exploration noise."""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            action = self.actor(state).cpu().numpy()[0]

        if add_noise:
            noise = self.noise.sample()
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update actor and critic networks using a batch of transitions.

        Args:
            batch: Dictionary containing states, actions, rewards, next_states, dones

        Returns:
            Dictionary with actor_loss and critic_loss
        """
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"].unsqueeze(-1) if batch["rewards"].dim() == 1 else batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"].unsqueeze(-1) if batch["dones"].dim() == 1 else batch["dones"]

        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item()
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network parameters."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def reset_noise(self):
        """Reset exploration noise."""
        self.noise.reset()
