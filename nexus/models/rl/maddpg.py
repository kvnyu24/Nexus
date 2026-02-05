"""
MADDPG: Multi-Agent Deep Deterministic Policy Gradient
Paper: "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
       (Lowe et al., NIPS 2017)

MADDPG extends DDPG to multi-agent settings with:
- Centralized training, decentralized execution (CTDE)
- Each agent has its own actor (policy) that only uses local observations
- Centralized critic that takes all agents' observations and actions
- Handles mixed cooperative-competitive scenarios
- Agents can learn from each other through shared experience

Key features:
- Actor: π_i(o_i) - decentralized, uses only local obs
- Critic: Q_i(o_1,...,o_n, a_1,...,a_n) - centralized, uses global info
- At execution, only actors are used (decentralized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from ...core.base import NexusModule
import numpy as np


class MADDPGActor(nn.Module):
    """
    Actor network for MADDPG (decentralized).

    Each agent has its own actor that only uses local observations.

    Args:
        obs_dim: Dimension of local observation
        action_dim: Dimension of action space
        hidden_dim: Hidden layer size
        max_action: Maximum action value
    """

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int = 64, max_action: float = 1.0
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting actions."""
        return self.max_action * self.network(obs)


class MADDPGCritic(nn.Module):
    """
    Centralized critic for MADDPG.

    Takes all agents' observations and actions as input to compute Q-value.

    Args:
        total_obs_dim: Sum of all agents' observation dimensions
        total_action_dim: Sum of all agents' action dimensions
        hidden_dim: Hidden layer size
    """

    def __init__(
        self, total_obs_dim: int, total_action_dim: int, hidden_dim: int = 64
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, all_obs: torch.Tensor, all_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass computing centralized Q-value.

        Args:
            all_obs: Concatenated observations from all agents [batch, total_obs_dim]
            all_actions: Concatenated actions from all agents [batch, total_action_dim]

        Returns:
            Q-value [batch, 1]
        """
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.network(x)


class MADDPGAgent(NexusModule):
    """
    MADDPG Agent for multi-agent reinforcement learning.

    Implements centralized training with decentralized execution for multiple agents.
    Each agent learns its own policy while critics have access to global information.

    Args:
        config: Configuration dictionary with:
            - n_agents: Number of agents
            - obs_dims: List of observation dimensions per agent
            - action_dims: List of action dimensions per agent
            - hidden_dim: Hidden layer size (default: 64)
            - max_action: Maximum action value (default: 1.0)
            - gamma: Discount factor (default: 0.99)
            - tau: Soft update coefficient (default: 0.01)
            - actor_lr: Actor learning rate (default: 1e-3)
            - critic_lr: Critic learning rate (default: 1e-3)
            - noise_std: Exploration noise standard deviation (default: 0.1)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.n_agents = config["n_agents"]
        self.obs_dims = config["obs_dims"]
        self.action_dims = config["action_dims"]
        self.hidden_dim = config.get("hidden_dim", 64)
        self.max_action = config.get("max_action", 1.0)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.01)
        self.noise_std = config.get("noise_std", 0.1)

        self.total_obs_dim = sum(self.obs_dims)
        self.total_action_dim = sum(self.action_dims)

        # Create actors and critics for each agent
        self.actors = nn.ModuleList()
        self.actor_targets = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.critic_targets = nn.ModuleList()

        for i in range(self.n_agents):
            # Actor (decentralized, uses only local obs)
            actor = MADDPGActor(
                self.obs_dims[i], self.action_dims[i], self.hidden_dim, self.max_action
            )
            actor_target = MADDPGActor(
                self.obs_dims[i], self.action_dims[i], self.hidden_dim, self.max_action
            )
            actor_target.load_state_dict(actor.state_dict())

            # Critic (centralized, uses all obs and actions)
            critic = MADDPGCritic(
                self.total_obs_dim, self.total_action_dim, self.hidden_dim
            )
            critic_target = MADDPGCritic(
                self.total_obs_dim, self.total_action_dim, self.hidden_dim
            )
            critic_target.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.critics.append(critic)
            self.critic_targets.append(critic_target)

        # Optimizers for each agent
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=config.get("actor_lr", 1e-3))
            for actor in self.actors
        ]
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=config.get("critic_lr", 1e-3))
            for critic in self.critics
        ]

    def select_actions(
        self, observations: List[torch.Tensor], add_noise: bool = True
    ) -> List[np.ndarray]:
        """
        Select actions for all agents (decentralized execution).

        Args:
            observations: List of observations, one per agent
            add_noise: Whether to add exploration noise

        Returns:
            List of actions (numpy arrays), one per agent
        """
        actions = []
        with torch.no_grad():
            for i, obs in enumerate(observations):
                if isinstance(obs, np.ndarray):
                    obs = torch.FloatTensor(obs)
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)

                action = self.actors[i](obs)

                if add_noise:
                    noise = torch.randn_like(action) * self.noise_std * self.max_action
                    action = (action + noise).clamp(-self.max_action, self.max_action)

                actions.append(action.cpu().numpy()[0])

        return actions

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update all MADDPG agents.

        Args:
            batch: Dictionary containing:
                - observations: List of observation tensors [batch, obs_dim_i] per agent
                - actions: List of action tensors [batch, action_dim_i] per agent
                - rewards: List of reward tensors [batch] per agent
                - next_observations: List of next observation tensors per agent
                - dones: Tensor of done flags [batch]

        Returns:
            Dictionary with loss metrics
        """
        observations = batch["observations"]  # List of [batch, obs_dim_i]
        actions = batch["actions"]  # List of [batch, action_dim_i]
        rewards = batch["rewards"]  # List of [batch]
        next_observations = batch["next_observations"]
        dones = batch["dones"]  # [batch]

        if dones.dim() == 2:
            dones = dones.squeeze(-1)

        # Concatenate all observations and actions for centralized critic
        all_obs = torch.cat(observations, dim=-1)
        all_actions = torch.cat(actions, dim=-1)
        all_next_obs = torch.cat(next_observations, dim=-1)

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        # Update each agent
        for agent_i in range(self.n_agents):
            # --- Update Critic ---
            with torch.no_grad():
                # Get target actions from all agents' target actors
                next_actions = [
                    self.actor_targets[i](next_observations[i])
                    for i in range(self.n_agents)
                ]
                all_next_actions = torch.cat(next_actions, dim=-1)

                # Target Q-value
                target_q = self.critic_targets[agent_i](all_next_obs, all_next_actions).squeeze(-1)
                target_q = rewards[agent_i] + (1 - dones) * self.gamma * target_q

            # Current Q-value
            current_q = self.critics[agent_i](all_obs, all_actions).squeeze(-1)

            # Critic loss
            critic_loss = F.mse_loss(current_q, target_q)

            self.critic_optimizers[agent_i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[agent_i].step()

            total_critic_loss += critic_loss.item()

            # --- Update Actor ---
            # Compute actions from all agents (current agent uses trainable actor, others use current actors)
            current_actions = []
            for i in range(self.n_agents):
                if i == agent_i:
                    # Use trainable actor for gradient flow
                    current_actions.append(self.actors[i](observations[i]))
                else:
                    # Use detached actions from other agents
                    current_actions.append(self.actors[i](observations[i]).detach())

            all_current_actions = torch.cat(current_actions, dim=-1)

            # Actor loss: maximize Q-value
            actor_loss = -self.critics[agent_i](all_obs, all_current_actions).mean()

            self.actor_optimizers[agent_i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_i].step()

            total_actor_loss += actor_loss.item()

        # Soft update target networks for all agents
        self._soft_update()

        return {
            "total_actor_loss": total_actor_loss / self.n_agents,
            "total_critic_loss": total_critic_loss / self.n_agents,
            "mean_q": current_q.mean().item(),
        }

    def _soft_update(self):
        """Soft update all target networks."""
        for i in range(self.n_agents):
            for param, target_param in zip(
                self.actors[i].parameters(), self.actor_targets[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.critics[i].parameters(), self.critic_targets[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def forward(
        self, observations: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing actions for all agents.

        Args:
            observations: List of observations, one per agent

        Returns:
            Dictionary with actions per agent
        """
        actions = []
        for i, obs in enumerate(observations):
            action = self.actors[i](obs)
            actions.append(action)

        all_obs = torch.cat(observations, dim=-1)
        all_actions = torch.cat(actions, dim=-1)

        # Compute Q-values from all critics
        q_values = [
            self.critics[i](all_obs, all_actions)
            for i in range(self.n_agents)
        ]

        return {
            "actions": actions,
            "q_values": q_values,
        }
