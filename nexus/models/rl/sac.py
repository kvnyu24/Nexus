"""
Soft Actor-Critic (SAC)
Paper: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning" (Haarnoja et al., 2018)
Paper: "Soft Actor-Critic Algorithms and Applications" (Haarnoja et al., 2019)

SAC is a state-of-the-art algorithm for continuous control that:
- Maximizes both expected return and policy entropy (maximum entropy RL)
- Uses twin Q-networks to address overestimation
- Learns a stochastic policy with automatic temperature adjustment
- Achieves excellent sample efficiency and stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Any, Tuple, Optional, Union
from ...core.base import NexusModule
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class SACGaussianActor(NexusModule):
    """Stochastic Gaussian policy for SAC."""

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
            nn.ReLU()
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability.

        Returns:
            action: Sampled action (squashed with tanh)
            log_prob: Log probability of the action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.max_action

        # Compute log probability with squashing correction
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound (Appendix C of SAC paper)
        log_prob -= torch.log(self.max_action * (1 - torch.tanh(x_t).pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get deterministic action (mean of the policy)."""
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.max_action


class SACCritic(NexusModule):
    """Twin Q-networks for SAC."""

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


class SACAgent(NexusModule):
    """
    Soft Actor-Critic Agent for continuous control tasks.

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state space
            - action_dim: Dimension of action space
            - hidden_dim: Hidden layer size (default: 256)
            - max_action: Maximum action value (default: 1.0)
            - actor_lr: Actor learning rate (default: 3e-4)
            - critic_lr: Critic learning rate (default: 3e-4)
            - alpha_lr: Temperature learning rate (default: 3e-4)
            - gamma: Discount factor (default: 0.99)
            - tau: Soft update coefficient (default: 0.005)
            - init_alpha: Initial temperature value (default: 0.2)
            - auto_alpha: Whether to auto-tune temperature (default: True)
            - target_entropy: Target entropy (default: -action_dim)
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
        self.auto_alpha = config.get("auto_alpha", True)

        # Actor network
        self.actor = SACGaussianActor(
            self.state_dim, self.action_dim, self.hidden_dim, self.max_action
        )

        # Critic networks
        self.critic = SACCritic(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_target = SACCritic(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Temperature (entropy coefficient)
        init_alpha = config.get("init_alpha", 0.2)
        self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(init_alpha)))
        self.target_entropy = config.get("target_entropy", -self.action_dim)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.get("actor_lr", 3e-4)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.get("critic_lr", 3e-4)
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=config.get("alpha_lr", 3e-4)
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Current temperature value."""
        return self.log_alpha.exp()

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False
    ) -> np.ndarray:
        """Select action using the actor network."""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            if deterministic:
                action = self.actor.deterministic_action(state)
            else:
                action, _ = self.actor.sample(state)

            return action.cpu().numpy()[0]

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update actor, critic, and temperature using a batch of transitions.

        Args:
            batch: Dictionary containing states, actions, rewards, next_states, dones

        Returns:
            Dictionary with losses and alpha value
        """
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"].unsqueeze(-1) if batch["rewards"].dim() == 1 else batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"].unsqueeze(-1) if batch["dones"].dim() == 1 else batch["dones"]

        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        q_value = torch.min(q1, q2)

        actor_loss = (self.alpha.detach() * log_probs - q_value).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Soft update target networks
        self._soft_update(self.critic, self.critic_target)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "log_prob": log_probs.mean().item()
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network parameters."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
