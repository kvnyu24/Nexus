"""
TD3+BC: A Simple Baseline for Offline RL
Paper: "A Minimalist Approach to Offline Reinforcement Learning" (Fujimoto & Gu, NeurIPS 2021)

TD3+BC is a simple but effective offline RL algorithm that:
- Extends TD3 (Twin Delayed DDPG) with a behavior cloning (BC) term
- Adds a weighted behavior cloning loss to the actor objective
- The BC weight is normalized by the average Q-value to balance exploration vs exploitation
- Prevents extrapolation error by keeping the policy close to the offline dataset
- Remarkably simple yet competitive with complex offline RL methods

Key insight: a single hyperparameter (α) controls the trade-off between
policy improvement (via TD3) and conservatism (via BC).

Formula: π* = argmax E[Q(s,a) - α * ||a - a_data||²]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
from ....core.base import NexusModule
import numpy as np


class TD3BCActor(nn.Module):
    """
    Actor network for TD3+BC.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer size
        max_action: Maximum action value for output scaling
    """

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting actions."""
        return self.max_action * self.network(state)


class TD3BCCritic(nn.Module):
    """
    Twin Q-network (critic) for TD3+BC.

    Uses two Q-networks to mitigate overestimation bias.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer size
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass outputting both Q-values."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q1 only."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class TD3BCAgent(NexusModule):
    """
    TD3+BC Agent for offline reinforcement learning.

    Combines TD3 (Twin Delayed DDPG) with behavior cloning regularization.
    The actor is trained to maximize Q-values while staying close to the
    offline dataset actions.

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state space
            - action_dim: Dimension of action space
            - hidden_dim: Hidden layer size (default: 256)
            - max_action: Maximum action value (default: 1.0)
            - gamma: Discount factor (default: 0.99)
            - tau: Soft update coefficient (default: 0.005)
            - policy_noise: Noise added to target actions (default: 0.2)
            - noise_clip: Clipping range for target action noise (default: 0.5)
            - policy_freq: Frequency of delayed policy updates (default: 2)
            - alpha: BC regularization weight (default: 2.5)
            - actor_lr: Actor learning rate (default: 3e-4)
            - critic_lr: Critic learning rate (default: 3e-4)
            - normalize_q: Whether to normalize BC weight by Q-value (default: True)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.max_action = config.get("max_action", 1.0)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.policy_noise = config.get("policy_noise", 0.2)
        self.noise_clip = config.get("noise_clip", 0.5)
        self.policy_freq = config.get("policy_freq", 2)
        self.alpha = config.get("alpha", 2.5)
        self.normalize_q = config.get("normalize_q", True)

        # Actor network and target
        self.actor = TD3BCActor(
            self.state_dim, self.action_dim, self.hidden_dim, self.max_action
        )
        self.actor_target = TD3BCActor(
            self.state_dim, self.action_dim, self.hidden_dim, self.max_action
        )
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Twin critic networks and targets
        self.critic = TD3BCCritic(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_target = TD3BCCritic(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.get("actor_lr", 3e-4)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.get("critic_lr", 3e-4)
        )

        # Update counter
        self.total_updates = 0

    def select_action(
        self, state: Union[np.ndarray, torch.Tensor], deterministic: bool = True
    ) -> np.ndarray:
        """
        Select action using the current policy.

        Args:
            state: Current state
            deterministic: If True, use deterministic action (default for offline RL)

        Returns:
            Action as numpy array
        """
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            action = self.actor(state)

            if not deterministic:
                noise = torch.randn_like(action) * 0.1 * self.max_action
                action = (action + noise).clamp(-self.max_action, self.max_action)

            return action.cpu().numpy()[0]

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update TD3+BC agent.

        Args:
            batch: Dictionary containing:
                - states: Current states [batch, state_dim]
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

        # --- Update Critic ---
        with torch.no_grad():
            # Target action with added noise (for smoothing)
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Target Q-values (minimum of twin Q-networks)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2).squeeze(-1)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        current_q1 = current_q1.squeeze(-1)
        current_q2 = current_q2.squeeze(-1)

        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Delayed Policy Update ---
        actor_loss = None
        bc_loss_val = 0.0
        q_loss_val = 0.0

        if self.total_updates % self.policy_freq == 0:
            # Compute policy actions
            policy_actions = self.actor(states)

            # Q-value from critic (use Q1)
            q_values = self.critic.q1_forward(states, policy_actions).squeeze(-1)

            # Behavior cloning loss: ||a_policy - a_data||²
            bc_loss = F.mse_loss(policy_actions, actions)

            # Normalize BC weight by average Q-value (optional)
            if self.normalize_q:
                lmbda = self.alpha / q_values.abs().mean().detach()
            else:
                lmbda = self.alpha

            # TD3+BC actor loss: maximize Q while minimizing BC loss
            # Original paper: min -Q + λ * BC
            actor_loss = -q_values.mean() + lmbda * bc_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update()

            bc_loss_val = bc_loss.item()
            q_loss_val = -q_values.mean().item()

        self.total_updates += 1

        metrics = {
            "critic_loss": critic_loss.item(),
            "mean_q1": current_q1.mean().item(),
            "mean_q2": current_q2.mean().item(),
            "mean_target_q": target_q.mean().item(),
        }

        if actor_loss is not None:
            metrics.update({
                "actor_loss": actor_loss.item(),
                "bc_loss": bc_loss_val,
                "q_maximization": q_loss_val,
            })

        return metrics

    def _soft_update(self):
        """Soft update target networks."""
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing action and Q-values.

        Args:
            state: Input state

        Returns:
            Dictionary with action and Q-values
        """
        action = self.actor(state)
        q1, q2 = self.critic(state, action)

        return {
            "action": action,
            "q1": q1,
            "q2": q2,
            "q": torch.min(q1, q2),
        }
