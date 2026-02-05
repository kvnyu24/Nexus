"""
QR-DQN: Quantile Regression Deep Q-Network
Paper: "Distributional Reinforcement Learning with Quantile Regression"
       (Dabney et al., AAAI 2018)

QR-DQN is a distributional RL algorithm that:
- Models the value distribution using quantile regression instead of C51's categorical approach
- Learns the locations of N quantiles of the return distribution
- Uses quantile Huber loss for training, which is more robust to outliers
- Avoids the need to manually set value range (v_min, v_max) unlike C51
- Provides better flexibility in representing value distributions

Key advantages over C51:
- No need to specify value range ahead of time
- Better representation of tail distributions
- More stable training with quantile regression
- Can handle unbounded rewards
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
from ....core.base import NexusModule
import numpy as np


class QRDQNNetwork(NexusModule):
    """
    QR-DQN Network that outputs quantiles of the action-value distribution.

    Instead of outputting expected Q-values, the network outputs N quantile
    estimates for each action, representing the cumulative distribution.

    Args:
        state_dim: Dimension of state/observation space
        action_dim: Number of discrete actions
        hidden_dim: Hidden layer size (default: 512)
        num_quantiles: Number of quantiles to estimate (default: 200)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_quantiles: int = 200,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.num_quantiles = num_quantiles

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Quantile output layer
        self.quantiles_head = nn.Linear(hidden_dim, action_dim * num_quantiles)

        # Fixed quantile midpoints: τ_i = (i + 0.5) / N for i = 0, ..., N-1
        self.register_buffer(
            "tau",
            torch.arange(0, num_quantiles, dtype=torch.float32) + 0.5
        )
        self.tau = self.tau / num_quantiles

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile estimates for all actions.

        Args:
            state: Input state [batch_size, state_dim]

        Returns:
            Quantile values [batch_size, action_dim, num_quantiles]
        """
        batch_size = state.size(0)
        features = self.features(state)
        quantiles = self.quantiles_head(features)
        quantiles = quantiles.view(batch_size, self.action_dim, self.num_quantiles)
        return quantiles

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute expected Q-values by averaging quantiles.

        Args:
            state: Input state [batch_size, state_dim]

        Returns:
            Expected Q-values [batch_size, action_dim]
        """
        quantiles = self.forward(state)
        q_values = quantiles.mean(dim=-1)
        return q_values


class QRDQNAgent(NexusModule):
    """
    QR-DQN Agent using quantile regression for distributional RL.

    Implements the full QR-DQN algorithm with:
    - Quantile regression loss (quantile Huber loss)
    - Double DQN for target computation
    - Epsilon-greedy exploration
    - Support for prioritized experience replay (external)

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state space
            - action_dim: Number of discrete actions
            - hidden_dim: Hidden layer size (default: 512)
            - num_quantiles: Number of quantiles to estimate (default: 200)
            - gamma: Discount factor (default: 0.99)
            - tau: Soft update coefficient (default: 0.005)
            - learning_rate: Learning rate (default: 5e-5)
            - epsilon_start: Initial epsilon for exploration (default: 1.0)
            - epsilon_end: Final epsilon (default: 0.01)
            - epsilon_decay: Epsilon decay rate (default: 0.995)
            - kappa: Quantile Huber loss threshold (default: 1.0)
            - max_grad_norm: Maximum gradient norm (default: 10.0)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 512)
        self.num_quantiles = config.get("num_quantiles", 200)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.kappa = config.get("kappa", 1.0)
        self.max_grad_norm = config.get("max_grad_norm", 10.0)

        # Epsilon-greedy exploration
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)

        # Online and target networks
        self.online_network = QRDQNNetwork(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.num_quantiles,
        )
        self.target_network = QRDQNNetwork(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.num_quantiles,
        )
        self.target_network.load_state_dict(self.online_network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=config.get("learning_rate", 5e-5),
        )

    def select_action(
        self, state: Union[np.ndarray, torch.Tensor], training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state/observation
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            q_values = self.online_network.get_q_values(state)
            return q_values.argmax(dim=-1).item()

    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _quantile_huber_loss(
        self,
        quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute quantile Huber loss.

        The quantile Huber loss combines:
        - Quantile regression loss: asymmetric weights based on quantile level
        - Huber loss: smooth L1 loss for robustness to outliers

        Args:
            quantiles: Current quantile estimates [batch, num_quantiles]
            target_quantiles: Target quantile values [batch, num_quantiles]
            tau: Quantile levels [num_quantiles]

        Returns:
            Quantile Huber loss (scalar)
        """
        # Compute TD errors (element-wise)
        # quantiles: [batch, N], target_quantiles: [batch, N']
        # We want all pairwise differences: [batch, N, N']
        td_errors = target_quantiles.unsqueeze(-1) - quantiles.unsqueeze(1)
        # td_errors: [batch, N', N]

        # Huber loss
        abs_errors = td_errors.abs()
        huber_loss = torch.where(
            abs_errors <= self.kappa,
            0.5 * td_errors ** 2,
            self.kappa * (abs_errors - 0.5 * self.kappa),
        )

        # Quantile regression weighting
        # tau: [N], expand to [1, 1, N]
        tau_expanded = tau.unsqueeze(0).unsqueeze(0)
        quantile_weight = torch.abs(tau_expanded - (td_errors < 0).float())

        # Final loss
        loss = (quantile_weight * huber_loss).mean()
        return loss

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update QR-DQN using quantile regression.

        Args:
            batch: Dictionary containing:
                - states: Current states [batch, state_dim]
                - actions: Actions taken [batch] (integer)
                - rewards: Rewards [batch]
                - next_states: Next states [batch, state_dim]
                - dones: Terminal flags [batch]
                - weights: Importance sampling weights [batch] (optional, for PER)

        Returns:
            Dictionary with loss metrics
        """
        states = batch["states"]
        actions = batch["actions"].long()
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        weights = batch.get("weights", torch.ones_like(rewards))

        batch_size = states.size(0)

        # Current quantile estimates for taken actions
        current_quantiles = self.online_network(states)  # [batch, actions, N]
        current_quantiles = current_quantiles.gather(
            1, actions.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, self.num_quantiles)
        ).squeeze(1)  # [batch, N]

        # Target quantile computation with Double DQN
        with torch.no_grad():
            # Double DQN: select actions with online network
            next_q_values = self.online_network.get_q_values(next_states)
            next_actions = next_q_values.argmax(dim=-1)  # [batch]

            # Get target quantiles for selected actions
            target_quantiles = self.target_network(next_states)  # [batch, actions, N]
            target_quantiles = target_quantiles.gather(
                1, next_actions.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, self.num_quantiles)
            ).squeeze(1)  # [batch, N]

            # Bellman backup: r + γ * Z(s', a*)
            target_quantiles = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * target_quantiles

        # Quantile Huber loss
        loss = self._quantile_huber_loss(
            current_quantiles, target_quantiles, self.online_network.tau
        )

        # Apply importance sampling weights (for PER)
        if weights is not None and not torch.all(weights == 1.0):
            loss = (loss * weights.mean()).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_network.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        # Soft update target network
        self._soft_update()

        # Decay epsilon
        self.decay_epsilon()

        # Compute Q-values for logging
        with torch.no_grad():
            q_values = self.online_network.get_q_values(states)
            mean_q = q_values.mean().item()

        return {
            "loss": loss.item(),
            "mean_q_value": mean_q,
            "epsilon": self.epsilon,
        }

    def _soft_update(self):
        """Soft update target network parameters."""
        for param, target_param in zip(
            self.online_network.parameters(),
            self.target_network.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing quantiles and Q-values.

        Args:
            state: Input state [batch, state_dim]

        Returns:
            Dictionary with quantiles, q_values, and selected actions
        """
        quantiles = self.online_network(state)
        q_values = quantiles.mean(dim=-1)
        actions = q_values.argmax(dim=-1)

        return {
            "quantiles": quantiles,
            "q_values": q_values,
            "actions": actions,
        }
