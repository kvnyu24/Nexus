"""
Implicit Q-Learning (IQL)
Paper: "Offline Reinforcement Learning with Implicit Q-Learning" (Kostrikov et al., 2022)

IQL is an offline RL algorithm that:
- Avoids querying out-of-distribution actions using expectile regression
- Learns a value function V(s) that approximates max_a Q(s, a) implicitly
- Uses advantage-weighted regression for policy extraction
- Achieves strong performance without explicit policy constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
from ...core.base import NexusModule
import numpy as np


class MLP(nn.Module):
    """Simple MLP with optional layer normalization."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        use_layer_norm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IQLValueNetwork(NexusModule):
    """Value network V(s) for IQL."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()
        self.net = MLP(state_dim, 1, hidden_dim, num_layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class IQLQNetwork(NexusModule):
    """Twin Q-networks for IQL."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden_dim, num_layers)
        self.q2 = MLP(state_dim + action_dim, 1, hidden_dim, num_layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class IQLGaussianPolicy(NexusModule):
    """Gaussian policy for IQL with advantage-weighted regression."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        max_action: float = 1.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0
    ):
        super().__init__()
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = MLP(state_dim, hidden_dim, hidden_dim, num_layers - 1)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(state)

        if deterministic:
            action = torch.tanh(mean) * self.max_action
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            action = torch.tanh(normal.rsample()) * self.max_action

        return action

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Inverse tanh (atanh) to get pre-squashed action
        action_clipped = torch.clamp(action / self.max_action, -0.999, 0.999)
        pre_tanh_action = torch.atanh(action_clipped)

        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(pre_tanh_action)

        # Correction for tanh squashing
        log_prob -= torch.log(self.max_action * (1 - torch.tanh(pre_tanh_action).pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return log_prob


def expectile_loss(pred: torch.Tensor, target: torch.Tensor, expectile: float) -> torch.Tensor:
    """
    Asymmetric L2 loss for expectile regression.

    Args:
        pred: Predicted values
        target: Target values
        expectile: Expectile parameter (tau in paper)

    Returns:
        Expectile loss
    """
    diff = target - pred
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * diff.pow(2)).mean()


class IQLAgent(NexusModule):
    """
    Implicit Q-Learning Agent for offline RL.

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state space
            - action_dim: Dimension of action space
            - hidden_dim: Hidden layer size (default: 256)
            - max_action: Maximum action value (default: 1.0)
            - discount: Discount factor (default: 0.99)
            - tau: Soft update coefficient (default: 0.005)
            - expectile: Expectile for value learning (default: 0.7)
            - temperature: Temperature for AWR (default: 3.0)
            - q_lr: Q-network learning rate (default: 3e-4)
            - v_lr: Value network learning rate (default: 3e-4)
            - policy_lr: Policy learning rate (default: 3e-4)
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
        self.expectile = config.get("expectile", 0.7)
        self.temperature = config.get("temperature", 3.0)

        # Networks
        self.q_network = IQLQNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.q_target = IQLQNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.q_target.load_state_dict(self.q_network.state_dict())

        self.v_network = IQLValueNetwork(self.state_dim, self.hidden_dim)

        self.policy = IQLGaussianPolicy(
            self.state_dim, self.action_dim, self.hidden_dim,
            max_action=self.max_action
        )

        # Optimizers
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.get("q_lr", 3e-4)
        )
        self.v_optimizer = torch.optim.Adam(
            self.v_network.parameters(),
            lr=config.get("v_lr", 3e-4)
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.get("policy_lr", 3e-4)
        )

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

    def update_value(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update value network using expectile regression."""
        states = batch["states"]
        actions = batch["actions"]

        with torch.no_grad():
            q_value = self.q_target.q_min(states, actions)

        v_value = self.v_network(states)
        v_loss = expectile_loss(v_value, q_value, self.expectile)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        return v_loss.item()

    def update_q(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update Q-networks using TD learning."""
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"].unsqueeze(-1) if batch["rewards"].dim() == 1 else batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"].unsqueeze(-1) if batch["dones"].dim() == 1 else batch["dones"]

        with torch.no_grad():
            next_v = self.v_network(next_states)
            target_q = rewards + self.discount * (1 - dones) * next_v

        q1, q2 = self.q_network(states, actions)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        return q_loss.item()

    def update_policy(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update policy using advantage-weighted regression."""
        states = batch["states"]
        actions = batch["actions"]

        with torch.no_grad():
            v_value = self.v_network(states)
            q_value = self.q_target.q_min(states, actions)
            advantage = q_value - v_value

            # Compute weights using exponential advantage
            weights = torch.exp(self.temperature * advantage)
            weights = torch.clamp(weights, max=100.0)  # Clip for stability

        # Policy loss: negative log likelihood weighted by advantages
        log_prob = self.policy.log_prob(states, actions)
        policy_loss = -(weights * log_prob).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update all networks.

        Args:
            batch: Dictionary containing states, actions, rewards, next_states, dones

        Returns:
            Dictionary with losses
        """
        # Update value network
        v_loss = self.update_value(batch)

        # Update Q networks
        q_loss = self.update_q(batch)

        # Update policy
        policy_loss = self.update_policy(batch)

        # Soft update target Q network
        self._soft_update(self.q_network, self.q_target)

        return {
            "v_loss": v_loss,
            "q_loss": q_loss,
            "policy_loss": policy_loss
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network parameters."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
