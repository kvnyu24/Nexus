"""
Conservative Q-Learning (CQL)
Paper: "Conservative Q-Learning for Offline Reinforcement Learning" (Kumar et al., 2020)

CQL is an offline RL algorithm that:
- Learns a conservative Q-function that lower-bounds the true Q-value
- Penalizes Q-values of out-of-distribution (OOD) actions
- Adds a regularization term to push down Q-values for actions not in the dataset
- Works with both discrete and continuous action spaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Any, Tuple, Optional, Union
from ...core.base import NexusModule
import numpy as np


class CQLCritic(NexusModule):
    """Twin Q-networks for CQL."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()

        def build_q():
            layers = [nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dim, 1))
            return nn.Sequential(*layers)

        self.q1 = build_q()
        self.q2 = build_q()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class CQLGaussianPolicy(NexusModule):
    """Gaussian policy for CQL (SAC-style)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        max_action: float = 1.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super().__init__()
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

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
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.max_action

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - torch.tanh(x_t).pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(state)

        if deterministic:
            return torch.tanh(mean) * self.max_action
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            return torch.tanh(x_t) * self.max_action


class CQLAgent(NexusModule):
    """
    Conservative Q-Learning Agent for offline RL.

    CQL adds a regularizer to SAC that penalizes high Q-values for OOD actions:
    CQL_loss = alpha * (E_{a~policy}[Q(s,a)] - E_{a~data}[Q(s,a)])

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state space
            - action_dim: Dimension of action space
            - hidden_dim: Hidden layer size (default: 256)
            - max_action: Maximum action value (default: 1.0)
            - discount: Discount factor (default: 0.99)
            - tau: Soft update coefficient (default: 0.005)
            - alpha: SAC temperature (default: 0.2)
            - cql_alpha: CQL regularization coefficient (default: 1.0)
            - cql_n_actions: Number of actions to sample for CQL (default: 10)
            - cql_lagrange: Whether to use lagrange version (default: False)
            - cql_target_action_gap: Target gap for lagrange (default: 5.0)
            - auto_alpha: Whether to auto-tune SAC temperature (default: True)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Dimensions
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.max_action = config.get("max_action", 1.0)

        # Hyperparameters
        self.discount = config.get("discount", 0.99)
        self.tau = config.get("tau", 0.005)
        self.cql_n_actions = config.get("cql_n_actions", 10)
        self.cql_lagrange = config.get("cql_lagrange", False)
        self.cql_target_action_gap = config.get("cql_target_action_gap", 5.0)
        self.auto_alpha = config.get("auto_alpha", True)

        # Networks
        self.policy = CQLGaussianPolicy(
            self.state_dim, self.action_dim, self.hidden_dim, self.max_action
        )
        self.critic = CQLCritic(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_target = CQLCritic(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # SAC temperature
        init_alpha = config.get("alpha", 0.2)
        self.log_alpha = nn.Parameter(torch.tensor(np.log(init_alpha)))
        self.target_entropy = -self.action_dim

        # CQL alpha (regularization coefficient)
        init_cql_alpha = config.get("cql_alpha", 1.0)
        if self.cql_lagrange:
            self.log_cql_alpha = nn.Parameter(torch.tensor(np.log(init_cql_alpha)))
        else:
            self.cql_alpha = init_cql_alpha

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.get("policy_lr", 3e-4)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.get("critic_lr", 3e-4)
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=config.get("alpha_lr", 3e-4)
        )
        if self.cql_lagrange:
            self.cql_alpha_optimizer = torch.optim.Adam(
                [self.log_cql_alpha],
                lr=config.get("cql_alpha_lr", 3e-4)
            )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def cql_alpha_value(self) -> torch.Tensor:
        if self.cql_lagrange:
            return self.log_cql_alpha.exp()
        return torch.tensor(self.cql_alpha)

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False
    ) -> np.ndarray:
        """Select action using the policy."""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            action = self.policy.get_action(state, deterministic)
            return action.cpu().numpy()[0]

    def compute_cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CQL regularization loss.

        CQL penalizes Q-values for actions sampled from the policy while
        pushing up Q-values for actions in the dataset.
        """
        batch_size = states.shape[0]
        device = states.device

        # Sample random actions
        random_actions = torch.FloatTensor(
            batch_size * self.cql_n_actions, self.action_dim
        ).uniform_(-self.max_action, self.max_action).to(device)

        # Sample actions from current policy
        states_repeated = states.unsqueeze(1).repeat(1, self.cql_n_actions, 1)
        states_repeated = states_repeated.view(batch_size * self.cql_n_actions, self.state_dim)

        policy_actions, policy_log_probs = self.policy.sample(states_repeated)

        # Compute Q-values for random actions
        q1_rand, q2_rand = self.critic(states_repeated, random_actions)

        # Compute Q-values for policy actions
        q1_policy, q2_policy = self.critic(states_repeated, policy_actions)

        # Compute Q-values for dataset actions
        q1_data, q2_data = self.critic(states, actions)

        # Reshape for logsumexp
        q1_rand = q1_rand.view(batch_size, self.cql_n_actions)
        q2_rand = q2_rand.view(batch_size, self.cql_n_actions)
        q1_policy = q1_policy.view(batch_size, self.cql_n_actions)
        q2_policy = q2_policy.view(batch_size, self.cql_n_actions)
        policy_log_probs = policy_log_probs.view(batch_size, self.cql_n_actions)

        # Importance sampling correction for policy actions
        q1_policy = q1_policy - policy_log_probs.detach()
        q2_policy = q2_policy - policy_log_probs.detach()

        # Concatenate and compute logsumexp
        q1_cat = torch.cat([q1_rand, q1_policy], dim=1)
        q2_cat = torch.cat([q2_rand, q2_policy], dim=1)

        # CQL loss: logsumexp(Q) - E_data[Q]
        cql_loss_q1 = torch.logsumexp(q1_cat, dim=1).mean() - q1_data.mean()
        cql_loss_q2 = torch.logsumexp(q2_cat, dim=1).mean() - q2_data.mean()

        return cql_loss_q1, cql_loss_q2

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update all networks using CQL.

        Args:
            batch: Dictionary containing states, actions, rewards, next_states, dones

        Returns:
            Dictionary with losses
        """
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"].unsqueeze(-1) if batch["rewards"].dim() == 1 else batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"].unsqueeze(-1) if batch["dones"].dim() == 1 else batch["dones"]

        # ==================== Update Critic ====================
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + self.discount * (1 - dones) * target_q

        # Standard TD loss
        current_q1, current_q2 = self.critic(states, actions)
        td_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # CQL regularization
        cql_loss_q1, cql_loss_q2 = self.compute_cql_loss(states, actions)
        cql_loss = cql_loss_q1 + cql_loss_q2

        # Total critic loss
        critic_loss = td_loss + self.cql_alpha_value * cql_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update CQL alpha (lagrange version)
        cql_alpha_loss = torch.tensor(0.0)
        if self.cql_lagrange:
            cql_alpha_loss = -self.log_cql_alpha * (cql_loss - self.cql_target_action_gap).detach()
            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss.backward()
            self.cql_alpha_optimizer.step()

        # ==================== Update Policy ====================
        new_actions, log_probs = self.policy.sample(states)
        q1, q2 = self.critic(states, new_actions)
        q_value = torch.min(q1, q2)

        policy_loss = (self.alpha.detach() * log_probs - q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ==================== Update SAC Alpha ====================
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # ==================== Update Target Networks ====================
        self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": critic_loss.item(),
            "td_loss": td_loss.item(),
            "cql_loss": cql_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha.item(),
            "cql_alpha": self.cql_alpha_value.item(),
            "alpha_loss": alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss,
            "q_value": q_value.mean().item()
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network parameters."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
