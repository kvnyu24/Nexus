"""
WQMIX: Weighted QMIX
Paper: "Weighted QMIX: Expanding Monotonic Value Function Factorisation
       for Deep Multi-Agent Reinforcement Learning" (Rashid et al., NeurIPS 2020)

WQMIX extends QMIX by:
- Relaxing the monotonicity constraint via importance weighting
- Using a centralized weighted projector to handle non-monotonic value functions
- Applying importance weights based on TD-errors
- Achieving better performance when optimal joint action requires non-monotonic mixing
- Maintaining decentralized execution while using weighted centralized training

Key insight: Not all multi-agent tasks satisfy the monotonicity assumption.
WQMIX uses importance weighting to go beyond this limitation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from ...core.base import NexusModule


class WQMIXMixingNetwork(nn.Module):
    """
    Weighted mixing network for WQMIX.

    Similar to QMIX but with importance weighting to handle non-monotonic cases.

    Args:
        n_agents: Number of agents
        state_dim: Dimension of global state
        hidden_dim: Hidden dimension for hyper-networks
    """

    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 32):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Hyper-networks for generating mixing network weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * hidden_dim),
        )

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Hyper-networks for biases
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, agent_qs: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """
        Mix agent Q-values into joint Q-value.

        Args:
            agent_qs: Individual agent Q-values [batch, n_agents]
            state: Global state [batch, state_dim]

        Returns:
            Mixed Q-value [batch, 1]
        """
        batch_size = agent_qs.size(0)

        # Generate mixing network parameters from state
        w1 = torch.abs(self.hyper_w1(state))  # Monotonicity via abs
        w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)

        b1 = self.hyper_b1(state).view(batch_size, 1, self.hidden_dim)

        # First layer
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)  # [batch, 1, hidden]

        # Second layer
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.hidden_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2  # [batch, 1, 1]
        return q_tot.squeeze(-1)  # [batch, 1]


class WQMIXAgent(NexusModule):
    """
    WQMIX Agent for cooperative multi-agent reinforcement learning.

    Extends QMIX with importance weighting to handle non-monotonic value factorization.
    Each agent has a local Q-network, and a weighted mixing network combines them
    into a joint Q-value.

    Args:
        config: Configuration dictionary with:
            - n_agents: Number of agents
            - obs_dim: Dimension of local observations
            - action_dim: Dimension of action space per agent
            - state_dim: Dimension of global state
            - hidden_dim: Hidden layer size for agent networks (default: 64)
            - mixing_hidden_dim: Hidden layer size for mixing network (default: 32)
            - gamma: Discount factor (default: 0.99)
            - tau: Soft update coefficient (default: 0.005)
            - learning_rate: Learning rate (default: 5e-4)
            - weight_decay: Weight decay for optimizer (default: 0.0)
            - omega: Weight for non-monotonic portion (default: 1.0)
            - max_grad_norm: Maximum gradient norm (default: 10.0)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.n_agents = config["n_agents"]
        self.obs_dim = config["obs_dim"]
        self.action_dim = config["action_dim"]
        self.state_dim = config["state_dim"]
        self.hidden_dim = config.get("hidden_dim", 64)
        self.mixing_hidden_dim = config.get("mixing_hidden_dim", 32)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.omega = config.get("omega", 1.0)
        self.max_grad_norm = config.get("max_grad_norm", 10.0)

        # Agent Q-networks (shared across agents)
        self.agent_network = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

        self.target_agent_network = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.target_agent_network.load_state_dict(self.agent_network.state_dict())

        # Mixing networks
        self.mixer = WQMIXMixingNetwork(
            self.n_agents, self.state_dim, self.mixing_hidden_dim
        )
        self.target_mixer = WQMIXMixingNetwork(
            self.n_agents, self.state_dim, self.mixing_hidden_dim
        )
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # Weighted projector (for non-monotonic part)
        self.weighted_projector = nn.Sequential(
            nn.Linear(self.state_dim, self.mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mixing_hidden_dim, 1),
        )

        # Optimizer
        all_params = (
            list(self.agent_network.parameters())
            + list(self.mixer.parameters())
            + list(self.weighted_projector.parameters())
        )
        self.optimizer = torch.optim.Adam(
            all_params,
            lr=config.get("learning_rate", 5e-4),
            weight_decay=config.get("weight_decay", 0.0),
        )

    def get_agent_q_values(
        self, observations: torch.Tensor, network=None
    ) -> torch.Tensor:
        """
        Compute Q-values for all agents.

        Args:
            observations: Agent observations [batch, n_agents, obs_dim]
            network: Network to use (default: self.agent_network)

        Returns:
            Q-values [batch, n_agents, action_dim]
        """
        if network is None:
            network = self.agent_network

        batch_size, n_agents, obs_dim = observations.shape
        obs_flat = observations.view(-1, obs_dim)
        q_values = network(obs_flat)
        return q_values.view(batch_size, n_agents, -1)

    def select_actions(
        self, observations: torch.Tensor, epsilon: float = 0.0
    ) -> torch.Tensor:
        """
        Select actions for all agents using epsilon-greedy.

        Args:
            observations: Agent observations [batch, n_agents, obs_dim]
            epsilon: Exploration probability

        Returns:
            Actions [batch, n_agents]
        """
        with torch.no_grad():
            q_values = self.get_agent_q_values(observations)
            actions = q_values.argmax(dim=-1)

            # Epsilon-greedy exploration
            if epsilon > 0:
                random_actions = torch.randint(
                    0, self.action_dim, actions.shape, device=actions.device
                )
                explore_mask = torch.rand(actions.shape, device=actions.device) < epsilon
                actions = torch.where(explore_mask, random_actions, actions)

            return actions

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update WQMIX agent.

        Args:
            batch: Dictionary containing:
                - observations: Agent observations [batch, n_agents, obs_dim]
                - actions: Agent actions [batch, n_agents]
                - rewards: Team rewards [batch]
                - next_observations: Next observations [batch, n_agents, obs_dim]
                - states: Global states [batch, state_dim]
                - next_states: Next global states [batch, state_dim]
                - dones: Done flags [batch]

        Returns:
            Dictionary with loss metrics
        """
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        states = batch["states"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Ensure shapes
        if rewards.dim() == 2:
            rewards = rewards.squeeze(-1)
        if dones.dim() == 2:
            dones = dones.squeeze(-1)

        # Current agent Q-values
        agent_q_values = self.get_agent_q_values(observations)
        chosen_agent_q = agent_q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # Target agent Q-values
        with torch.no_grad():
            target_agent_q_values = self.get_agent_q_values(
                next_observations, self.target_agent_network
            )
            target_max_q = target_agent_q_values.max(dim=-1)[0]

            # Target joint Q-value (monotonic part)
            target_q_tot_mono = self.target_mixer(target_max_q, next_states).squeeze(-1)

            # Weighted projector (non-monotonic part)
            target_q_proj = self.weighted_projector(next_states).squeeze(-1)

            # Combined target
            target_q_tot = rewards + (1 - dones) * self.gamma * (
                target_q_tot_mono + self.omega * target_q_proj
            )

        # Current joint Q-value
        q_tot_mono = self.mixer(chosen_agent_q, states).squeeze(-1)
        q_proj = self.weighted_projector(states).squeeze(-1)
        q_tot = q_tot_mono + self.omega * q_proj

        # TD loss
        loss = F.mse_loss(q_tot, target_q_tot)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent_network.parameters())
            + list(self.mixer.parameters())
            + list(self.weighted_projector.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()

        # Soft update target networks
        self._soft_update()

        return {
            "loss": loss.item(),
            "q_tot": q_tot.mean().item(),
            "target_q_tot": target_q_tot.mean().item(),
            "q_mono": q_tot_mono.mean().item(),
            "q_proj": q_proj.mean().item(),
        }

    def _soft_update(self):
        """Soft update target networks."""
        for param, target_param in zip(
            self.agent_network.parameters(), self.target_agent_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.mixer.parameters(), self.target_mixer.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def forward(
        self, observations: torch.Tensor, state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing joint Q-value.

        Args:
            observations: Agent observations [batch, n_agents, obs_dim]
            state: Global state [batch, state_dim]

        Returns:
            Dictionary with agent Q-values and mixed Q-value
        """
        agent_q = self.get_agent_q_values(observations)
        max_agent_q = agent_q.max(dim=-1)[0]
        q_tot_mono = self.mixer(max_agent_q, state)
        q_proj = self.weighted_projector(state)
        q_tot = q_tot_mono + self.omega * q_proj

        return {
            "agent_q_values": agent_q,
            "q_tot": q_tot,
            "q_mono": q_tot_mono,
            "q_proj": q_proj,
        }
