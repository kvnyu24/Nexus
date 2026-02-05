"""
Random Network Distillation (RND)
Paper: "Exploration by Random Network Distillation" (Burda et al., ICLR 2019)

RND is an exploration bonus method that:
- Uses a fixed, randomly initialized target network as a source of prediction targets
- Trains a predictor network to match the output of the fixed network
- The prediction error serves as an intrinsic reward: novel states yield higher error
  because the predictor has not been trained on them
- Unlike ICM, RND does not require an inverse model and avoids the "noisy TV" problem
  because the target is deterministic and state-only (no dependence on actions)
- Computationally simple and highly effective for hard exploration problems

Key properties:
- Intrinsic reward = MSE(predictor(state), fixed_network(state))
- The fixed network acts as a random but consistent feature extractor
- As the predictor improves on frequently visited states, intrinsic reward decreases
- Novel (unvisited) states maintain high prediction error, encouraging exploration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
from ...core.base import NexusModule
import numpy as np


class FixedRandomNetwork(NexusModule):
    """
    Fixed, randomly initialized target network.

    This network is never trained. It provides deterministic, consistent
    feature mappings for any input state. The predictor network is trained
    to match these outputs.

    Args:
        state_dim: Dimension of state/observation space
        feature_dim: Dimension of output feature space
        hidden_dim: Hidden layer size
    """

    def __init__(
        self, state_dim: int, feature_dim: int = 256, hidden_dim: int = 256
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Freeze all parameters (never trained)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute fixed random features.

        Args:
            state: Input state [batch_size, state_dim]

        Returns:
            Feature vector [batch_size, feature_dim]
        """
        return self.network(state)


class PredictorNetwork(NexusModule):
    """
    Trainable predictor network for RND.

    Trained to match the output of the fixed random network. The prediction
    error on novel states serves as the intrinsic exploration reward.

    Args:
        state_dim: Dimension of state/observation space
        feature_dim: Dimension of output feature space (must match target)
        hidden_dim: Hidden layer size
    """

    def __init__(
        self, state_dim: int, feature_dim: int = 256, hidden_dim: int = 256
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute predicted features.

        Args:
            state: Input state [batch_size, state_dim]

        Returns:
            Predicted feature vector [batch_size, feature_dim]
        """
        return self.network(state)


class RNDModule(NexusModule):
    """
    Random Network Distillation module.

    Combines the fixed target network and trainable predictor network.
    Provides methods to compute intrinsic rewards and update the predictor.

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state/observation space
            - feature_dim: Dimension of feature space (default: 256)
            - hidden_dim: Hidden layer size (default: 256)
            - learning_rate: Predictor learning rate (default: 1e-3)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.state_dim = config["state_dim"]
        self.feature_dim = config.get("feature_dim", 256)
        self.hidden_dim = config.get("hidden_dim", 256)

        # Fixed target network (random, frozen)
        self.target = FixedRandomNetwork(
            self.state_dim, self.feature_dim, self.hidden_dim
        )

        # Trainable predictor network
        self.predictor = PredictorNetwork(
            self.state_dim, self.feature_dim, self.hidden_dim
        )

        # Running statistics for observation normalization
        self.register_buffer("obs_mean", torch.zeros(self.state_dim))
        self.register_buffer("obs_var", torch.ones(self.state_dim))
        self.register_buffer("obs_count", torch.tensor(1e-4))

        # Running statistics for reward normalization
        self.register_buffer("reward_mean", torch.tensor(0.0))
        self.register_buffer("reward_var", torch.tensor(1.0))
        self.register_buffer("reward_count", torch.tensor(1e-4))

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=config.get("learning_rate", 1e-3),
        )

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations using running statistics."""
        return (obs - self.obs_mean) / (self.obs_var.sqrt() + 1e-8)

    def _update_obs_stats(self, obs: torch.Tensor):
        """Update running observation statistics."""
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0)
        batch_count = obs.shape[0]

        total_count = self.obs_count + batch_count
        delta = batch_mean - self.obs_mean
        new_mean = self.obs_mean + delta * batch_count / total_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.obs_count * batch_count / total_count

        self.obs_mean = new_mean
        self.obs_var = m2 / total_count
        self.obs_count = total_count

    def _normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Normalize intrinsic reward using running statistics."""
        return reward / (self.reward_var.sqrt() + 1e-8)

    def _update_reward_stats(self, reward: torch.Tensor):
        """Update running reward statistics."""
        batch_mean = reward.mean()
        batch_var = reward.var()
        batch_count = reward.numel()

        total_count = self.reward_count + batch_count
        delta = batch_mean - self.reward_mean
        new_mean = self.reward_mean + delta * batch_count / total_count
        m_a = self.reward_var * self.reward_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.reward_count * batch_count / total_count

        self.reward_mean = new_mean
        self.reward_var = m2 / total_count
        self.reward_count = total_count

    def compute_intrinsic_reward(
        self, state: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute intrinsic reward as MSE between predictor and target outputs.

        Args:
            state: Current state [batch_size, state_dim]
            normalize: Whether to normalize the intrinsic reward

        Returns:
            Intrinsic reward [batch_size]
        """
        with torch.no_grad():
            # Normalize observation
            normalized_state = self._normalize_obs(state)

            # Compute features
            target_features = self.target(normalized_state)
            predicted_features = self.predictor(normalized_state)

            # MSE as intrinsic reward
            intrinsic_reward = (target_features - predicted_features).pow(2).mean(dim=-1)

            if normalize:
                self._update_reward_stats(intrinsic_reward)
                intrinsic_reward = self._normalize_reward(intrinsic_reward)

            return intrinsic_reward

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the predictor network to better match the target.

        Args:
            batch: Dictionary containing:
                - states: State observations [batch, state_dim]

        Returns:
            Dictionary with loss metrics
        """
        states = batch["states"]

        # Update observation statistics
        self._update_obs_stats(states)
        normalized_states = self._normalize_obs(states)

        # Compute features
        target_features = self.target(normalized_states).detach()
        predicted_features = self.predictor(normalized_states)

        # MSE loss
        loss = F.mse_loss(predicted_features, target_features)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute intrinsic reward for logging
        with torch.no_grad():
            intrinsic_reward = (target_features - predicted_features).pow(2).mean(dim=-1)

        return {
            "rnd_loss": loss.item(),
            "mean_intrinsic_reward": intrinsic_reward.mean().item(),
            "max_intrinsic_reward": intrinsic_reward.max().item(),
            "min_intrinsic_reward": intrinsic_reward.min().item(),
        }

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing both target and predicted features.

        Args:
            state: Input state [batch, state_dim]

        Returns:
            Dictionary with target_features, predicted_features, intrinsic_reward
        """
        normalized_state = self._normalize_obs(state)
        target_features = self.target(normalized_state)
        predicted_features = self.predictor(normalized_state)
        intrinsic_reward = (target_features - predicted_features).pow(2).mean(dim=-1)

        return {
            "target_features": target_features,
            "predicted_features": predicted_features,
            "intrinsic_reward": intrinsic_reward,
        }


class RNDWrapper(NexusModule):
    """
    Wrapper that combines any RL agent with RND for exploration.

    Similar to ICMWrapper, this wraps a base agent and augments rewards
    with RND-based intrinsic rewards for exploration bonuses.

    Args:
        config: Configuration dictionary with:
            - agent: The base RL agent instance
            - rnd_config: Configuration for RNDModule with:
                - state_dim: State dimension
                - feature_dim: RND feature dimension (default: 256)
                - hidden_dim: Hidden layer size (default: 256)
                - learning_rate: Predictor learning rate (default: 1e-3)
            - intrinsic_reward_scale: Scale for intrinsic rewards (default: 1.0)
            - extrinsic_reward_scale: Scale for extrinsic rewards (default: 1.0)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.agent = config["agent"]
        self.rnd = RNDModule(config["rnd_config"])

        self.intrinsic_scale = config.get("intrinsic_reward_scale", 1.0)
        self.extrinsic_scale = config.get("extrinsic_reward_scale", 1.0)

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ):
        """
        Select action using the base agent.

        Args:
            state: Current state/observation
            **kwargs: Additional arguments passed to the base agent

        Returns:
            Action selected by the base agent
        """
        return self.agent.select_action(state, **kwargs)

    def compute_combined_reward(
        self,
        state: torch.Tensor,
        extrinsic_reward: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined intrinsic + extrinsic reward.

        Args:
            state: Current state (used for RND intrinsic reward)
            extrinsic_reward: Environment reward

        Returns:
            Combined reward
        """
        intrinsic_reward = self.rnd.compute_intrinsic_reward(state)
        combined = (
            self.extrinsic_scale * extrinsic_reward
            + self.intrinsic_scale * intrinsic_reward
        )
        return combined

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update both RND and the base agent.

        Args:
            batch: Dictionary containing states, actions, rewards, next_states, dones

        Returns:
            Dictionary with all loss metrics
        """
        # Update RND predictor
        rnd_metrics = self.rnd.update(batch)

        # Compute combined rewards
        states = batch["states"]
        extrinsic_rewards = batch["rewards"]

        combined_rewards = self.compute_combined_reward(states, extrinsic_rewards)

        # Update batch with combined rewards
        batch_with_intrinsic = batch.copy()
        batch_with_intrinsic["rewards"] = combined_rewards

        # Update agent
        agent_metrics = self.agent.update(batch_with_intrinsic)

        # Combine metrics
        metrics = {**rnd_metrics, **agent_metrics}
        metrics["combined_reward"] = combined_rewards.mean().item()
        metrics["extrinsic_reward"] = extrinsic_rewards.mean().item()

        return metrics

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing RND features and intrinsic reward.

        Args:
            state: Input state

        Returns:
            Dictionary with RND outputs
        """
        return self.rnd(state)
