"""
Rainbow DQN: Combining Improvements in Deep Reinforcement Learning
Paper: "Rainbow: Combining Improvements in Deep Reinforcement Learning"
       (Hessel et al., AAAI 2018)

Rainbow DQN integrates six key extensions to DQN into a single agent:

1. Double DQN (van Hasselt et al., 2016): Reduces overestimation by decoupling
   action selection and evaluation in the target computation.

2. Dueling Architecture (Wang et al., 2016): Separates state value and advantage
   streams for better generalization across actions.

3. Prioritized Experience Replay (Schaul et al., 2016): Samples important
   transitions more frequently based on TD error priority.

4. Multi-step Learning (Sutton, 1988): Uses n-step returns for faster reward
   propagation and reduced bias.

5. Distributional RL / C51 (Bellemare et al., 2017): Models the full value
   distribution over a fixed set of atoms instead of just the mean.

6. Noisy Networks (Fortunato et al., 2018): Replaces epsilon-greedy with
   learned parametric noise in network weights for state-dependent exploration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
from ....core.base import NexusModule
import numpy as np
import math


class NoisyLinear(nn.Module):
    """
    Noisy Linear layer for exploration via learned parametric noise.

    Replaces standard nn.Linear with factorized Gaussian noise on weights
    and biases. The noise parameters are learned, enabling state-dependent
    exploration that adapts over training.

    Reference: "Noisy Networks for Exploration" (Fortunato et al., 2018)

    Args:
        in_features: Number of input features
        out_features: Number of output features
        std_init: Initial standard deviation for noise (default: 0.5)
    """

    def __init__(
        self, in_features: int, out_features: int, std_init: float = 0.5
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Factorized noise buffers
        self.register_buffer(
            "weight_epsilon", torch.empty(out_features, in_features)
        )
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self._init_parameters()
        self.reset_noise()

    def _init_parameters(self):
        """Initialize mu and sigma parameters."""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    @staticmethod
    def _factorized_noise(size: int) -> torch.Tensor:
        """Generate factorized Gaussian noise: f(x) = sign(x) * sqrt(|x|)."""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        """Sample new factorized noise for weights and biases."""
        epsilon_in = self._factorized_noise(self.in_features)
        epsilon_out = self._factorized_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with noisy weights.

        During training, weights are: mu + sigma * epsilon
        During eval (no noise reset), uses last sampled noise.

        Args:
            x: Input tensor [batch, in_features]

        Returns:
            Output tensor [batch, out_features]
        """
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)


class RainbowNetwork(NexusModule):
    """
    Rainbow DQN Network combining Dueling + Noisy + Distributional (C51).

    Architecture:
    - Shared feature extractor with noisy linear layers
    - Dueling streams: separate value and advantage branches
    - Each branch outputs a distribution over atoms (C51)

    Args:
        state_dim: Dimension of state/observation space
        action_dim: Number of discrete actions
        hidden_dim: Hidden layer size (default: 512)
        num_atoms: Number of atoms for distributional RL (default: 51)
        v_min: Minimum value support (default: -10)
        v_max: Maximum value support (default: 10)
        noisy_std: Initial noise standard deviation (default: 0.5)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        noisy_std: float = 0.5,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Atom support
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, num_atoms)
        )
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # Dueling: Value stream (noisy)
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim, num_atoms, noisy_std),
        )

        # Dueling: Advantage stream (noisy)
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim * num_atoms, noisy_std),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute action-value distributions.

        Args:
            state: Input state [batch_size, state_dim]

        Returns:
            Log probability distributions over atoms per action
            [batch_size, action_dim, num_atoms]
        """
        batch_size = state.size(0)
        features = self.features(state)

        # Value stream: V(s) distribution
        value = self.value_stream(features)
        value = value.view(batch_size, 1, self.num_atoms)

        # Advantage stream: A(s, a) distribution
        advantage = self.advantage_stream(features)
        advantage = advantage.view(batch_size, self.action_dim, self.num_atoms)

        # Dueling combination: Q = V + A - mean(A)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Log-softmax over atoms for each action
        log_probs = F.log_softmax(q_atoms, dim=-1)

        return log_probs

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute expected Q-values from the distributions.

        Args:
            state: Input state [batch_size, state_dim]

        Returns:
            Expected Q-values [batch_size, action_dim]
        """
        log_probs = self.forward(state)
        probs = log_probs.exp()
        q_values = (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return q_values

    def reset_noise(self):
        """Reset noise for all noisy layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class RainbowAgent(NexusModule):
    """
    Rainbow DQN Agent combining six DQN improvements.

    Implements the full Rainbow algorithm with:
    - Double DQN target computation
    - Dueling network architecture
    - Noisy nets for exploration (no epsilon-greedy)
    - Distributional RL (C51) for value distributions
    - N-step return bootstrapping
    - Support for prioritized experience replay (external)

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state space
            - action_dim: Number of discrete actions
            - hidden_dim: Hidden layer size (default: 512)
            - num_atoms: Number of distributional atoms (default: 51)
            - v_min: Minimum value support (default: -10)
            - v_max: Maximum value support (default: 10)
            - n_step: Number of steps for multi-step returns (default: 3)
            - gamma: Discount factor (default: 0.99)
            - tau: Soft update coefficient (default: 0.005)
            - learning_rate: Learning rate (default: 6.25e-5)
            - noisy_std: Initial noise standard deviation (default: 0.5)
            - max_grad_norm: Maximum gradient norm (default: 10.0)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 512)
        self.num_atoms = config.get("num_atoms", 51)
        self.v_min = config.get("v_min", -10.0)
        self.v_max = config.get("v_max", 10.0)
        self.n_step = config.get("n_step", 3)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.max_grad_norm = config.get("max_grad_norm", 10.0)
        self.noisy_std = config.get("noisy_std", 0.5)

        # Derived
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        # Online and target networks
        self.online_network = RainbowNetwork(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.num_atoms,
            self.v_min,
            self.v_max,
            self.noisy_std,
        )
        self.target_network = RainbowNetwork(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.num_atoms,
            self.v_min,
            self.v_max,
            self.noisy_std,
        )
        self.target_network.load_state_dict(self.online_network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=config.get("learning_rate", 6.25e-5),
            eps=1.5e-4,
        )

    def select_action(
        self, state: Union[np.ndarray, torch.Tensor], training: bool = True
    ) -> int:
        """
        Select action using the noisy network (no epsilon-greedy needed).

        Args:
            state: Current state/observation
            training: If True, use noisy exploration; if False, still uses
                      current noise (call reset_noise for fresh exploration)

        Returns:
            Selected action index
        """
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            q_values = self.online_network.get_q_values(state)
            return q_values.argmax(dim=-1).item()

    def _compute_projected_distribution(
        self,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the projected target distribution for C51.

        Uses Double DQN: action selection from online network, evaluation
        from target network. Projects the target distribution onto the
        fixed support using the Bellman update.

        Args:
            next_states: Next states [batch, state_dim]
            rewards: N-step rewards [batch]
            dones: Done flags [batch]

        Returns:
            Projected target distribution [batch, num_atoms]
        """
        batch_size = next_states.size(0)
        support = self.online_network.support

        # Double DQN: select actions with online network
        with torch.no_grad():
            # Reset noise for target to get different exploration
            self.online_network.reset_noise()
            self.target_network.reset_noise()

            # Action selection from online network
            next_q_values = self.online_network.get_q_values(next_states)
            next_actions = next_q_values.argmax(dim=-1)  # [batch]

            # Get target distribution for selected actions
            target_log_probs = self.target_network(next_states)
            # target_log_probs: [batch, action_dim, num_atoms]
            next_actions_expanded = next_actions.unsqueeze(-1).unsqueeze(-1).expand(
                batch_size, 1, self.num_atoms
            )
            target_probs = target_log_probs.exp().gather(1, next_actions_expanded)
            target_probs = target_probs.squeeze(1)  # [batch, num_atoms]

            # Compute projected support: Tz = r + gamma^n * z
            gamma_n = self.gamma ** self.n_step
            tz = rewards.unsqueeze(-1) + gamma_n * (1 - dones.unsqueeze(-1)) * support.unsqueeze(0)
            tz = tz.clamp(self.v_min, self.v_max)

            # Compute projection indices
            b = (tz - self.v_min) / self.delta_z
            lower = b.floor().long()
            upper = b.ceil().long()

            # Clamp indices
            lower = lower.clamp(0, self.num_atoms - 1)
            upper = upper.clamp(0, self.num_atoms - 1)

            # Distribute probability mass
            projected = torch.zeros(batch_size, self.num_atoms, device=next_states.device)

            # Handle case where lower == upper
            eq_mask = (lower == upper)

            # Proportional distribution
            upper_frac = b - lower.float()
            lower_frac = 1.0 - upper_frac

            projected.scatter_add_(1, lower, target_probs * lower_frac)
            projected.scatter_add_(1, upper, target_probs * upper_frac)

        return projected

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update Rainbow DQN using distributional Bellman backup.

        Args:
            batch: Dictionary containing:
                - states: Current states [batch, state_dim]
                - actions: Actions taken [batch] (integer)
                - rewards: N-step discounted rewards [batch]
                - next_states: States after n steps [batch, state_dim]
                - dones: Terminal flags [batch]
                - weights: Importance sampling weights [batch] (optional, for PER)

        Returns:
            Dictionary with loss metrics and TD errors (for PER updates)
        """
        states = batch["states"]
        actions = batch["actions"].long()
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        weights = batch.get("weights", torch.ones_like(rewards))

        batch_size = states.size(0)

        # Compute projected target distribution
        target_probs = self._compute_projected_distribution(
            next_states, rewards, dones
        )

        # Get current distribution for taken actions
        log_probs = self.online_network(states)  # [batch, actions, atoms]
        actions_expanded = actions.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, 1, self.num_atoms
        )
        current_log_probs = log_probs.gather(1, actions_expanded).squeeze(1)
        # current_log_probs: [batch, num_atoms]

        # Cross-entropy loss (KL divergence)
        elementwise_loss = -(target_probs * current_log_probs).sum(dim=-1)

        # Weighted loss (for prioritized replay)
        loss = (weights * elementwise_loss).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_network.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        # Reset noise for next forward pass
        self.online_network.reset_noise()
        self.target_network.reset_noise()

        # Soft update target network
        self._soft_update()

        # Compute Q-values for logging
        with torch.no_grad():
            q_values = self.online_network.get_q_values(states)
            mean_q = q_values.mean().item()

        return {
            "loss": loss.item(),
            "mean_q_value": mean_q,
            "td_errors": elementwise_loss.detach().cpu().numpy(),
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
        Forward pass computing Q-value distributions and expected Q-values.

        Args:
            state: Input state [batch, state_dim]

        Returns:
            Dictionary with log_probs, q_values, and selected actions
        """
        log_probs = self.online_network(state)
        q_values = self.online_network.get_q_values(state)
        actions = q_values.argmax(dim=-1)

        return {
            "log_probs": log_probs,
            "q_values": q_values,
            "actions": actions,
        }
