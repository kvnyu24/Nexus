"""
Model-Based Policy Optimization (MBPO)
Paper: "When to Trust Your Model: Model-Based Policy Optimization" (Janner et al., NeurIPS 2019)

MBPO bridges model-based and model-free RL by:
- Learning an ensemble of dynamics models from real environment data
- Using the learned models to generate short branched rollouts starting from
  real states, augmenting the replay buffer with synthetic data
- Training a model-free algorithm (SAC) on the combined real + synthetic data
- The ensemble provides uncertainty estimates, and elite model selection
  improves prediction quality
- Short rollout lengths mitigate compounding model errors

Key insight: monotonic policy improvement is guaranteed under bounded model error,
and short branched rollouts keep the effective model error small.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional, Union
from ...core.base import NexusModule
import numpy as np


class ProbabilisticDynamicsModel(NexusModule):
    """
    Single probabilistic dynamics model that predicts a Gaussian distribution
    over (next_state_delta, reward) given (state, action).

    Predicts the mean and log-variance of the next state delta and reward,
    enabling uncertainty-aware predictions.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer size
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.output_dim = state_dim + 1  # next_state_delta + reward

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, self.output_dim)
        self.logvar_head = nn.Linear(hidden_dim, self.output_dim)

        # Learnable bounds for log-variance
        self.max_logvar = nn.Parameter(torch.ones(1, self.output_dim) * 0.5)
        self.min_logvar = nn.Parameter(torch.ones(1, self.output_dim) * -10.0)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and log-variance of (next_state_delta, reward).

        Args:
            state: Current state [batch, state_dim]
            action: Action taken [batch, action_dim]

        Returns:
            Tuple of (mean, logvar) each [batch, state_dim + 1]
        """
        x = torch.cat([state, action], dim=-1)
        features = self.network(x)

        mean = self.mean_head(features)
        logvar = self.logvar_head(features)

        # Soft-clamp log-variance
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar

    def predict(
        self, state: torch.Tensor, action: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state and reward.

        Args:
            state: Current state [batch, state_dim]
            action: Action [batch, action_dim]
            deterministic: If True, return mean prediction

        Returns:
            Tuple of (next_state, reward)
        """
        mean, logvar = self.forward(state, action)

        if deterministic:
            prediction = mean
        else:
            std = (0.5 * logvar).exp()
            prediction = mean + std * torch.randn_like(std)

        # Split into next_state_delta and reward
        next_state_delta = prediction[:, :-1]
        reward = prediction[:, -1:]

        next_state = state + next_state_delta

        return next_state, reward


class EnsembleDynamicsModel(NexusModule):
    """
    Ensemble of probabilistic dynamics models.

    Maintains an ensemble of independent dynamics models for uncertainty
    estimation. Elite models (those with lowest validation loss) are selected
    for generating rollouts, improving prediction quality.

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state space
            - action_dim: Dimension of action space
            - hidden_dim: Hidden layer size for each model (default: 256)
            - ensemble_size: Number of models in ensemble (default: 7)
            - elite_size: Number of elite models for prediction (default: 5)
            - model_lr: Learning rate for dynamics models (default: 3e-4)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.ensemble_size = config.get("ensemble_size", 7)
        self.elite_size = config.get("elite_size", 5)

        # Create ensemble of models
        self.models = nn.ModuleList([
            ProbabilisticDynamicsModel(
                self.state_dim, self.action_dim, self.hidden_dim
            )
            for _ in range(self.ensemble_size)
        ])

        # Track elite model indices (initially all models are elite)
        self.register_buffer(
            "elite_indices",
            torch.arange(self.elite_size)
        )

        # Per-model optimizers
        self.optimizers = [
            torch.optim.Adam(
                model.parameters(),
                lr=config.get("model_lr", 3e-4),
                weight_decay=1e-5,
            )
            for model in self.models
        ]

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through all ensemble members.

        Args:
            state: Current state [batch, state_dim]
            action: Action [batch, action_dim]

        Returns:
            Tuple of (means, logvars) lists, one per model
        """
        means, logvars = [], []
        for model in self.models:
            mean, logvar = model(state, action)
            means.append(mean)
            logvars.append(logvar)
        return means, logvars

    def predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state and reward using a randomly selected elite model.

        Args:
            state: Current state [batch, state_dim]
            action: Action [batch, action_dim]
            deterministic: If True, use mean prediction

        Returns:
            Tuple of (next_state, reward)
        """
        # Randomly select an elite model for each sample
        elite_idx = self.elite_indices[
            torch.randint(0, self.elite_size, (1,)).item()
        ]
        model = self.models[elite_idx]
        return model.predict(state, action, deterministic)

    def predict_with_uncertainty(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with ensemble disagreement as uncertainty estimate.

        Args:
            state: Current state [batch, state_dim]
            action: Action [batch, action_dim]

        Returns:
            Tuple of (mean_next_state, mean_reward, uncertainty)
        """
        predictions = []
        for idx in self.elite_indices:
            model = self.models[idx]
            next_state, reward = model.predict(state, action, deterministic=True)
            predictions.append(torch.cat([next_state, reward], dim=-1))

        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0).mean(dim=-1, keepdim=True)

        return mean_pred[:, :-1], mean_pred[:, -1:], uncertainty

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update all ensemble members on real transition data.

        Args:
            batch: Dictionary containing:
                - states: Current states [batch, state_dim]
                - actions: Actions [batch, action_dim]
                - next_states: Next states [batch, state_dim]
                - rewards: Rewards [batch, 1]

        Returns:
            Dictionary with loss metrics
        """
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        rewards = batch["rewards"]
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)

        # Target: (delta_state, reward)
        targets = torch.cat([next_states - states, rewards], dim=-1)

        total_loss = 0.0
        individual_losses = []

        for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            mean, logvar = model(states, actions)

            # Gaussian negative log-likelihood
            inv_var = (-logvar).exp()
            mse_loss = ((mean - targets) ** 2 * inv_var).mean()
            var_loss = logvar.mean()

            # Regularize logvar bounds
            bound_loss = 0.01 * (
                model.max_logvar.sum() - model.min_logvar.sum()
            )

            loss = mse_loss + var_loss + bound_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            individual_losses.append(loss.item())

        # Update elite indices based on losses
        sorted_indices = torch.tensor(individual_losses).argsort()
        self.elite_indices = sorted_indices[: self.elite_size]

        return {
            "model_loss": total_loss / self.ensemble_size,
            "best_model_loss": min(individual_losses),
            "worst_model_loss": max(individual_losses),
        }


class MBPOAgent(NexusModule):
    """
    Model-Based Policy Optimization Agent.

    Wraps a model-free SAC agent with a learned ensemble dynamics model.
    Generates short branched rollouts from real data using the learned model
    to augment training data, improving sample efficiency.

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state space
            - action_dim: Dimension of action space
            - ensemble_size: Number of ensemble models (default: 7)
            - elite_size: Number of elite models (default: 5)
            - rollout_length: Length of model rollouts (default: 1)
            - rollout_batch_size: Batch size for model rollouts (default: 256)
            - real_ratio: Ratio of real data in training batch (default: 0.05)
            - model_lr: Dynamics model learning rate (default: 3e-4)
            - hidden_dim: Hidden dim for dynamics model (default: 256)
            - agent_config: Config dict for the underlying SAC agent
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.rollout_length = config.get("rollout_length", 1)
        self.rollout_batch_size = config.get("rollout_batch_size", 256)
        self.real_ratio = config.get("real_ratio", 0.05)

        # Ensemble dynamics model
        self.dynamics = EnsembleDynamicsModel(config)

        # Model-free agent (SAC)
        # Import here to avoid circular imports
        from .sac import SACAgent

        agent_config = config.get("agent_config", {})
        agent_config.setdefault("state_dim", self.state_dim)
        agent_config.setdefault("action_dim", self.action_dim)
        self.agent = SACAgent(agent_config)

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select action using the underlying SAC agent.

        Args:
            state: Current state
            deterministic: If True, use deterministic action

        Returns:
            Action as numpy array
        """
        return self.agent.select_action(state, deterministic)

    def generate_rollouts(
        self,
        start_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate short branched rollouts from starting states using the
        learned dynamics model.

        Args:
            start_states: Real states to branch from [batch, state_dim]

        Returns:
            Dictionary with synthetic transitions:
                - states, actions, rewards, next_states, dones
        """
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []

        states = start_states

        with torch.no_grad():
            for t in range(self.rollout_length):
                # Select actions using the current policy
                actions_np = np.array([
                    self.agent.select_action(s.cpu().numpy())
                    for s in states
                ])
                actions = torch.FloatTensor(actions_np).to(states.device)

                # Predict next state and reward using the dynamics model
                next_states, rewards = self.dynamics.predict(states, actions)

                # Simple done prediction: terminate if state is too far from origin
                # (can be replaced with a learned termination function)
                dones = torch.zeros(states.size(0), 1, device=states.device)

                all_states.append(states)
                all_actions.append(actions)
                all_rewards.append(rewards)
                all_next_states.append(next_states)
                all_dones.append(dones)

                states = next_states

        return {
            "states": torch.cat(all_states, dim=0),
            "actions": torch.cat(all_actions, dim=0),
            "rewards": torch.cat(all_rewards, dim=0),
            "next_states": torch.cat(all_next_states, dim=0),
            "dones": torch.cat(all_dones, dim=0),
        }

    def update_model(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the dynamics model on real environment data.

        Args:
            batch: Real transition data

        Returns:
            Dictionary with model loss metrics
        """
        return self.dynamics.update(batch)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Full MBPO update: update dynamics model, generate rollouts,
        and update the SAC agent.

        Args:
            batch: Dictionary containing real transitions:
                - states, actions, rewards, next_states, dones

        Returns:
            Dictionary with all loss metrics
        """
        # Step 1: Update dynamics model on real data
        model_metrics = self.dynamics.update(batch)

        # Step 2: Generate synthetic rollouts from real states
        start_indices = torch.randperm(batch["states"].size(0))[:self.rollout_batch_size]
        start_states = batch["states"][start_indices]
        synthetic_batch = self.generate_rollouts(start_states)

        # Step 3: Combine real and synthetic data for agent update
        real_batch_size = max(
            1, int(self.real_ratio * self.rollout_batch_size)
        )
        synthetic_batch_size = self.rollout_batch_size - real_batch_size

        # Sample from real data
        real_indices = torch.randperm(batch["states"].size(0))[:real_batch_size]
        real_subset = {
            k: v[real_indices] for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }

        # Sample from synthetic data
        syn_size = synthetic_batch["states"].size(0)
        syn_indices = torch.randperm(syn_size)[:synthetic_batch_size]
        synthetic_subset = {
            k: v[syn_indices] for k, v in synthetic_batch.items()
        }

        # Normalize dimensions: ensure all tensors are 1D for scalars, 2D for vectors
        for key in synthetic_subset:
            syn_t = synthetic_subset[key]
            if key in real_subset:
                real_t = real_subset[key]
                # Match dimensions: squeeze synthetic 2D to 1D if real is 1D
                if real_t.dim() == 1 and syn_t.dim() == 2 and syn_t.size(-1) == 1:
                    synthetic_subset[key] = syn_t.squeeze(-1)
                elif real_t.dim() == 2 and syn_t.dim() == 1:
                    synthetic_subset[key] = syn_t.unsqueeze(-1)

        # Combine
        combined_batch = {}
        for key in real_subset:
            if key in synthetic_subset:
                combined_batch[key] = torch.cat(
                    [real_subset[key], synthetic_subset[key]], dim=0
                )
            else:
                combined_batch[key] = real_subset[key]

        # Ensure correct shapes for SAC (expects 1D rewards and dones)
        if combined_batch["rewards"].dim() == 2:
            combined_batch["rewards"] = combined_batch["rewards"].squeeze(-1)
        if combined_batch["dones"].dim() == 2:
            combined_batch["dones"] = combined_batch["dones"].squeeze(-1)

        # Step 4: Update SAC agent on combined data
        agent_metrics = self.agent.update(combined_batch)

        # Combine all metrics
        metrics = {}
        for k, v in model_metrics.items():
            metrics[f"dynamics/{k}"] = v
        for k, v in agent_metrics.items():
            metrics[f"agent/{k}"] = v
        metrics["synthetic_data_size"] = float(syn_size)

        return metrics

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Forward pass through the dynamics model.

        Args:
            state: Current state
            action: Action

        Returns:
            Dictionary with predicted next state, reward, and uncertainty
        """
        next_state, reward, uncertainty = self.dynamics.predict_with_uncertainty(
            state, action
        )
        return {
            "next_state": next_state,
            "reward": reward,
            "uncertainty": uncertainty,
        }
