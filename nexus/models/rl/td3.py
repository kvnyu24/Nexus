"""
Twin Delayed Deep Deterministic Policy Gradient (TD3)
Paper: "Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al., 2018)

TD3 improves upon DDPG with three key techniques:
1. Twin Critics: Use two Q-networks and take the minimum to address overestimation
2. Delayed Policy Updates: Update policy less frequently than critics
3. Target Policy Smoothing: Add noise to target actions for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
from ...core.base import NexusModule
import numpy as np


class TD3Actor(NexusModule):
    """Deterministic policy network for TD3."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        max_action: float = 1.0
    ):
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)


class TD3Critic(NexusModule):
    """Twin Q-networks for TD3."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


class TD3Agent(NexusModule):
    """
    TD3 Agent for continuous control tasks.

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state space
            - action_dim: Dimension of action space
            - hidden_dim: Hidden layer size (default: 256)
            - max_action: Maximum action value (default: 1.0)
            - actor_lr: Actor learning rate (default: 3e-4)
            - critic_lr: Critic learning rate (default: 3e-4)
            - gamma: Discount factor (default: 0.99)
            - tau: Soft update coefficient (default: 0.005)
            - policy_delay: Delay between policy updates (default: 2)
            - policy_noise: Noise added to target policy (default: 0.2)
            - noise_clip: Range to clip target policy noise (default: 0.5)
            - exploration_noise: Noise for action exploration (default: 0.1)
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
        self.policy_delay = config.get("policy_delay", 2)
        self.policy_noise = config.get("policy_noise", 0.2)
        self.noise_clip = config.get("noise_clip", 0.5)
        self.exploration_noise = config.get("exploration_noise", 0.1)

        # Actor networks
        self.actor = TD3Actor(self.state_dim, self.action_dim, self.hidden_dim, self.max_action)
        self.actor_target = TD3Actor(self.state_dim, self.action_dim, self.hidden_dim, self.max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic networks (twin)
        self.critic = TD3Critic(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_target = TD3Critic(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.get("actor_lr", 3e-4)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.get("critic_lr", 3e-4)
        )

        # Update counter for delayed policy updates
        self.total_updates = 0

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
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update actor and critic networks using a batch of transitions.

        Args:
            batch: Dictionary containing states, actions, rewards, next_states, dones

        Returns:
            Dictionary with losses
        """
        self.total_updates += 1

        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"].unsqueeze(-1) if batch["rewards"].dim() == 1 else batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"].unsqueeze(-1) if batch["dones"].dim() == 1 else batch["dones"]

        # Compute target Q-value with clipped double Q-learning
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target actions
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Take minimum of twin Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        # Update critics
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        result = {"critic_loss": critic_loss.item()}

        # Delayed policy updates
        if self.total_updates % self.policy_delay == 0:
            # Update actor
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            result["actor_loss"] = actor_loss.item()

        return result

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network parameters."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
