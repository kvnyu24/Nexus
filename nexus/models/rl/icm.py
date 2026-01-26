"""
Intrinsic Curiosity Module (ICM)
Paper: "Curiosity-driven Exploration by Self-Supervised Prediction" (Pathak et al., 2017)

ICM is an exploration method that:
- Generates intrinsic rewards based on prediction error
- Uses a forward model to predict next state features from current state and action
- Uses an inverse model to predict action from state transition
- Encourages exploration of novel states where predictions are uncertain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
from ...core.base import NexusModule
import numpy as np


class FeatureEncoder(NexusModule):
    """
    Feature encoder that maps observations to a learned feature space.

    Can handle both vector observations and image observations.
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        is_image: bool = False,
        image_channels: int = 3
    ):
        super().__init__()
        self.is_image = is_image
        self.feature_dim = feature_dim

        if is_image:
            # CNN encoder for images
            self.encoder = nn.Sequential(
                nn.Conv2d(image_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, feature_dim)  # Assumes 84x84 input
            )
        else:
            # MLP encoder for vectors
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ForwardModel(NexusModule):
    """
    Forward dynamics model that predicts next state features.

    Given current state features and action, predicts the features of the next state.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        discrete_actions: bool = True
    ):
        super().__init__()
        self.discrete_actions = discrete_actions
        self.action_dim = action_dim

        input_dim = feature_dim + (action_dim if not discrete_actions else action_dim)

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(
        self,
        state_features: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        if self.discrete_actions:
            # One-hot encode discrete actions
            if action.dim() == 1:
                action = F.one_hot(action.long(), num_classes=self.action_dim).float()
        else:
            if action.dim() == 1:
                action = action.unsqueeze(-1)

        x = torch.cat([state_features, action], dim=-1)
        return self.model(x)


class InverseModel(NexusModule):
    """
    Inverse dynamics model that predicts action from state transition.

    Given current and next state features, predicts the action taken.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        discrete_actions: bool = True
    ):
        super().__init__()
        self.discrete_actions = discrete_actions

        self.model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(
        self,
        state_features: torch.Tensor,
        next_state_features: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([state_features, next_state_features], dim=-1)
        return self.model(x)


class ICM(NexusModule):
    """
    Intrinsic Curiosity Module.

    Provides intrinsic rewards based on prediction error of a forward dynamics model.
    The inverse model helps learn a good feature representation.

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state/observation space
            - action_dim: Dimension of action space
            - feature_dim: Dimension of learned features (default: 256)
            - hidden_dim: Hidden layer size (default: 256)
            - discrete_actions: Whether actions are discrete (default: True)
            - is_image: Whether observations are images (default: False)
            - beta: Weight for forward vs inverse loss (default: 0.2)
            - eta: Scaling factor for intrinsic reward (default: 0.01)
            - learning_rate: Learning rate (default: 1e-3)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.feature_dim = config.get("feature_dim", 256)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.discrete_actions = config.get("discrete_actions", True)
        self.is_image = config.get("is_image", False)

        # Hyperparameters
        self.beta = config.get("beta", 0.2)  # Weight for forward loss
        self.eta = config.get("eta", 0.01)   # Intrinsic reward scaling

        # Networks
        self.encoder = FeatureEncoder(
            self.state_dim, self.feature_dim, self.hidden_dim,
            self.is_image, config.get("image_channels", 3)
        )

        self.forward_model = ForwardModel(
            self.feature_dim, self.action_dim, self.hidden_dim, self.discrete_actions
        )

        self.inverse_model = InverseModel(
            self.feature_dim, self.action_dim, self.hidden_dim, self.discrete_actions
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.forward_model.parameters()) +
            list(self.inverse_model.parameters()),
            lr=config.get("learning_rate", 1e-3)
        )

    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute intrinsic reward based on forward model prediction error.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            Intrinsic reward (scalar per sample)
        """
        with torch.no_grad():
            # Encode states
            state_features = self.encoder(state)
            next_state_features = self.encoder(next_state)

            # Predict next state features
            predicted_features = self.forward_model(state_features, action)

            # Intrinsic reward = prediction error (L2 norm)
            intrinsic_reward = 0.5 * (predicted_features - next_state_features).pow(2).sum(dim=-1)

            return self.eta * intrinsic_reward

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing features and predictions.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            predicted_action: Action predicted by inverse model
            predicted_next_features: Next state features predicted by forward model
            actual_next_features: Actual next state features
        """
        # Encode states
        state_features = self.encoder(state)
        next_state_features = self.encoder(next_state)

        # Inverse model: predict action from transition
        predicted_action = self.inverse_model(state_features, next_state_features)

        # Forward model: predict next state features
        predicted_next_features = self.forward_model(state_features, action)

        return predicted_action, predicted_next_features, next_state_features

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update ICM models.

        Args:
            batch: Dictionary containing states, actions, next_states

        Returns:
            Dictionary with losses
        """
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]

        # Forward pass
        predicted_actions, predicted_features, actual_features = self.forward(
            states, actions, next_states
        )

        # Inverse model loss
        if self.discrete_actions:
            inverse_loss = F.cross_entropy(predicted_actions, actions.long())
        else:
            inverse_loss = F.mse_loss(predicted_actions, actions)

        # Forward model loss (don't backprop through actual features)
        forward_loss = 0.5 * (predicted_features - actual_features.detach()).pow(2).mean()

        # Total loss
        total_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Compute intrinsic reward for logging
        with torch.no_grad():
            intrinsic_reward = self.compute_intrinsic_reward(states, actions, next_states)

        return {
            "icm_loss": total_loss.item(),
            "inverse_loss": inverse_loss.item(),
            "forward_loss": forward_loss.item(),
            "mean_intrinsic_reward": intrinsic_reward.mean().item()
        }


class ICMWrapper(NexusModule):
    """
    Wrapper that combines an RL agent with ICM for curiosity-driven exploration.

    Args:
        config: Configuration dictionary with:
            - agent: The base RL agent
            - icm_config: Configuration for ICM
            - intrinsic_reward_scale: Scale for intrinsic rewards (default: 1.0)
            - extrinsic_reward_scale: Scale for extrinsic rewards (default: 1.0)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.agent = config["agent"]
        self.icm = ICM(config["icm_config"])

        self.intrinsic_scale = config.get("intrinsic_reward_scale", 1.0)
        self.extrinsic_scale = config.get("extrinsic_reward_scale", 1.0)

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        **kwargs
    ):
        """Select action using the base agent."""
        return self.agent.select_action(state, **kwargs)

    def compute_combined_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        extrinsic_reward: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined intrinsic + extrinsic reward.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            extrinsic_reward: Environment reward

        Returns:
            Combined reward
        """
        intrinsic_reward = self.icm.compute_intrinsic_reward(state, action, next_state)
        combined = (
            self.extrinsic_scale * extrinsic_reward +
            self.intrinsic_scale * intrinsic_reward
        )
        return combined

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update both ICM and the base agent.

        Args:
            batch: Dictionary containing states, actions, rewards, next_states, dones

        Returns:
            Dictionary with all losses
        """
        # Update ICM
        icm_metrics = self.icm.update(batch)

        # Compute combined rewards
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        extrinsic_rewards = batch["rewards"]

        combined_rewards = self.compute_combined_reward(
            states, actions, next_states, extrinsic_rewards
        )

        # Update batch with combined rewards
        batch_with_intrinsic = batch.copy()
        batch_with_intrinsic["rewards"] = combined_rewards

        # Update agent
        agent_metrics = self.agent.update(batch_with_intrinsic)

        # Combine metrics
        metrics = {**icm_metrics, **agent_metrics}
        metrics["combined_reward"] = combined_rewards.mean().item()

        return metrics
