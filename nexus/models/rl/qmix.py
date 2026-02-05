"""
QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning
Paper: "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning"
       (Rashid et al., ICML 2018)

QMIX addresses cooperative multi-agent reinforcement learning by:
- Learning individual Q-value functions per agent conditioned on local observations
- Mixing individual Q-values into a joint Q-value via a monotonic mixing network
- The mixing network uses hypernetworks conditioned on the global state to produce
  mixing weights, constrained to be non-negative (ensuring monotonicity)
- Monotonicity guarantees that argmax over the joint Q equals the collection of
  per-agent argmax actions, enabling decentralized execution
- Target networks with soft updates provide stable training targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional, Union
from ...core.base import NexusModule
import numpy as np
import copy


class QMIXNetwork(NexusModule):
    """
    Individual agent Q-network for QMIX.

    Each agent has its own Q-network that takes the agent's local observation
    and outputs Q-values for each discrete action.

    Args:
        obs_dim: Dimension of the agent's local observation
        action_dim: Number of discrete actions
        hidden_dim: Hidden layer size
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for all actions.

        Args:
            obs: Agent observation [batch_size, obs_dim]

        Returns:
            Q-values [batch_size, action_dim]
        """
        return self.network(obs)


class MixingNetwork(NexusModule):
    """
    Hypernetwork-based mixing network for QMIX.

    Mixes individual agent Q-values into a joint Q-value using state-dependent
    weights. The weights are produced by hypernetworks and passed through an
    absolute value activation to ensure monotonicity (non-negative weights),
    guaranteeing that the joint Q-value is monotonically increasing in each
    agent's individual Q-value.

    Args:
        num_agents: Number of agents
        state_dim: Dimension of the global state
        mixing_dim: Dimension of the mixing embedding
    """

    def __init__(self, num_agents: int, state_dim: int, mixing_dim: int = 32):
        super().__init__()

        self.num_agents = num_agents
        self.mixing_dim = mixing_dim

        # Hypernetwork for first layer weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, num_agents * mixing_dim)
        )

        # Hypernetwork for first layer bias
        self.hyper_b1 = nn.Linear(state_dim, mixing_dim)

        # Hypernetwork for second layer weights
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, mixing_dim)
        )

        # Hypernetwork for second layer bias (scalar output)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, 1)
        )

    def forward(
        self, agent_qs: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """
        Mix individual agent Q-values into a joint Q-value.

        Args:
            agent_qs: Individual agent Q-values [batch_size, num_agents]
            state: Global state [batch_size, state_dim]

        Returns:
            Joint Q-value [batch_size, 1]
        """
        batch_size = agent_qs.size(0)

        # Reshape agent Q-values
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)

        # First layer: state-dependent weights (enforced non-negative via abs)
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(batch_size, self.num_agents, self.mixing_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.mixing_dim)

        # First layer forward pass
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # Second layer: state-dependent weights (enforced non-negative via abs)
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch_size, self.mixing_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        # Second layer forward pass
        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.squeeze(-1).squeeze(-1)

        return q_total


class QMIXAgent(NexusModule):
    """
    QMIX Agent for cooperative multi-agent reinforcement learning.

    Manages individual agent Q-networks and the mixing network. Each agent
    selects actions based on its local Q-values (decentralized execution),
    while the mixing network is used during training to compute joint
    Q-values for the TD loss (centralized training).

    Args:
        config: Configuration dictionary with:
            - num_agents: Number of cooperative agents
            - state_dim: Dimension of the global state
            - obs_dim: Dimension of per-agent local observation
            - action_dim: Number of discrete actions per agent
            - hidden_dim: Hidden layer size for agent networks (default: 64)
            - mixing_dim: Mixing network embedding dimension (default: 32)
            - gamma: Discount factor (default: 0.99)
            - tau: Soft update coefficient for target networks (default: 0.005)
            - learning_rate: Learning rate (default: 3e-4)
            - epsilon: Initial exploration rate (default: 1.0)
            - epsilon_min: Minimum exploration rate (default: 0.05)
            - epsilon_decay: Exploration decay rate (default: 0.995)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Core dimensions
        self.num_agents = config["num_agents"]
        self.state_dim = config["state_dim"]
        self.obs_dim = config["obs_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 64)
        self.mixing_dim = config.get("mixing_dim", 32)

        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.05)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)

        # Individual agent Q-networks
        self.agent_networks = nn.ModuleList([
            QMIXNetwork(self.obs_dim, self.action_dim, self.hidden_dim)
            for _ in range(self.num_agents)
        ])

        # Mixing network
        self.mixing_network = MixingNetwork(
            self.num_agents, self.state_dim, self.mixing_dim
        )

        # Target networks (deep copy)
        self.target_agent_networks = nn.ModuleList([
            copy.deepcopy(net) for net in self.agent_networks
        ])
        self.target_mixing_network = copy.deepcopy(self.mixing_network)

        # Freeze target networks
        for net in self.target_agent_networks:
            for param in net.parameters():
                param.requires_grad = False
        for param in self.target_mixing_network.parameters():
            param.requires_grad = False

        # Optimizer for all online parameters
        params = (
            list(self.agent_networks.parameters()) +
            list(self.mixing_network.parameters())
        )
        self.optimizer = torch.optim.Adam(
            params, lr=config.get("learning_rate", 3e-4)
        )

    def select_action(
        self,
        observations: List[Union[np.ndarray, torch.Tensor]],
        training: bool = True
    ) -> List[int]:
        """
        Select actions for all agents using epsilon-greedy.

        Args:
            observations: List of per-agent observations
            training: If True, use epsilon-greedy exploration

        Returns:
            List of integer actions, one per agent
        """
        actions = []

        with torch.no_grad():
            for i, (net, obs) in enumerate(zip(self.agent_networks, observations)):
                if training and np.random.random() < self.epsilon:
                    action = np.random.randint(self.action_dim)
                else:
                    if isinstance(obs, np.ndarray):
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    else:
                        obs_tensor = obs.unsqueeze(0) if obs.dim() == 1 else obs

                    q_values = net(obs_tensor)
                    action = q_values.argmax(dim=-1).item()

                actions.append(action)

        return actions

    def _get_agent_q_values(
        self,
        observations: List[torch.Tensor],
        actions: List[torch.Tensor],
        target: bool = False
    ) -> torch.Tensor:
        """
        Get the Q-values for chosen actions from all agents.

        Args:
            observations: List of observation tensors per agent
            actions: List of action tensors per agent
            target: If True, use target networks

        Returns:
            Agent Q-values [batch_size, num_agents]
        """
        networks = self.target_agent_networks if target else self.agent_networks
        agent_qs = []

        for i, (net, obs, act) in enumerate(zip(networks, observations, actions)):
            q_values = net(obs)
            # Gather Q-values for the chosen actions
            q = q_values.gather(1, act.long().unsqueeze(-1)).squeeze(-1)
            agent_qs.append(q)

        return torch.stack(agent_qs, dim=-1)

    def _get_target_max_q_values(
        self, next_observations: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Get maximum Q-values from target networks for next observations.

        Args:
            next_observations: List of next observation tensors per agent

        Returns:
            Max target Q-values [batch_size, num_agents]
        """
        agent_qs = []

        for i, (net, obs) in enumerate(zip(self.target_agent_networks, next_observations)):
            q_values = net(obs)
            max_q = q_values.max(dim=-1)[0]
            agent_qs.append(max_q)

        return torch.stack(agent_qs, dim=-1)

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update agent Q-networks and mixing network.

        Args:
            batch: Dictionary containing:
                - observations: List of observation tensors per agent
                - actions: List of action tensors per agent
                - next_observations: List of next observation tensors per agent
                - state: Global state tensor [batch, state_dim]
                - next_state: Next global state tensor [batch, state_dim]
                - rewards: Shared team reward tensor [batch]
                - dones: Done flag tensor [batch]

        Returns:
            Dictionary with loss metrics
        """
        state = batch["state"]
        next_state = batch["next_state"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        # Get current joint Q-value
        agent_qs = self._get_agent_q_values(
            batch["observations"], batch["actions"], target=False
        )
        q_total = self.mixing_network(agent_qs, state)

        # Get target joint Q-value
        with torch.no_grad():
            target_agent_qs = self._get_target_max_q_values(batch["next_observations"])
            target_q_total = self.target_mixing_network(target_agent_qs, next_state)
            target = rewards + self.gamma * (1 - dones) * target_q_total

        # TD loss
        loss = F.mse_loss(q_total, target.detach())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent_networks.parameters()) +
            list(self.mixing_network.parameters()),
            10.0
        )
        self.optimizer.step()

        # Soft update target networks
        self._soft_update()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {
            "loss": loss.item(),
            "q_total": q_total.mean().item(),
            "target_q_total": target.mean().item(),
            "epsilon": self.epsilon,
        }

    def _soft_update(self):
        """Soft update all target network parameters."""
        for source, target in zip(
            self.agent_networks.parameters(),
            self.target_agent_networks.parameters()
        ):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)

        for source, target in zip(
            self.mixing_network.parameters(),
            self.target_mixing_network.parameters()
        ):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)

    def forward(
        self,
        observations: List[torch.Tensor],
        state: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Forward pass computing individual Q-values and joint Q-value.

        Args:
            observations: List of per-agent observation tensors
            state: Global state tensor

        Returns:
            Dictionary with agent Q-values and joint Q-value
        """
        all_q_values = []
        for net, obs in zip(self.agent_networks, observations):
            all_q_values.append(net(obs))

        # Get greedy actions and their Q-values
        greedy_actions = [q.argmax(dim=-1) for q in all_q_values]
        agent_qs = torch.stack([
            q.gather(1, a.unsqueeze(-1)).squeeze(-1)
            for q, a in zip(all_q_values, greedy_actions)
        ], dim=-1)

        q_total = self.mixing_network(agent_qs, state)

        return {
            "agent_q_values": all_q_values,
            "greedy_actions": greedy_actions,
            "agent_qs": agent_qs,
            "q_total": q_total,
        }
