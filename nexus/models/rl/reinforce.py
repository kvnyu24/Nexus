"""
REINFORCE (Monte Carlo Policy Gradient)
Paper: "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (Williams, 1992)

REINFORCE is the foundational policy gradient algorithm that:
- Uses Monte Carlo returns to estimate policy gradients
- Directly optimizes the policy without learning a value function
- Can use a baseline to reduce variance
- Serves as the basis for more advanced policy gradient methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Dict, Any, Tuple, Optional, Union, List
from ...core.base import NexusModule
import numpy as np


class DiscretePolicy(NexusModule):
    """Policy network for discrete action spaces."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(state), dim=-1)

    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        probs = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = self.forward(state)
        dist = Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class ContinuousPolicy(NexusModule):
    """Policy network for continuous action spaces."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        max_action: float = 1.0,
        log_std_init: float = 0.0
    ):
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(state)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_action(self, state: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Apply tanh squashing
        action_squashed = torch.tanh(action) * self.max_action
        return action_squashed.detach().cpu().numpy(), log_prob

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        # Inverse tanh to get pre-squashed action
        action_clipped = torch.clamp(action / self.max_action, -0.999, 0.999)
        pre_tanh_action = torch.atanh(action_clipped)

        log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Baseline(NexusModule):
    """State-value baseline for variance reduction."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class REINFORCEAgent(NexusModule):
    """
    REINFORCE Agent with optional baseline.

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state space
            - action_dim: Dimension of action space
            - hidden_dim: Hidden layer size (default: 128)
            - discrete: Whether action space is discrete (default: True)
            - max_action: Maximum action value for continuous (default: 1.0)
            - gamma: Discount factor (default: 0.99)
            - learning_rate: Learning rate (default: 1e-3)
            - use_baseline: Whether to use a learned baseline (default: True)
            - baseline_lr: Baseline learning rate (default: 1e-3)
            - entropy_coef: Entropy bonus coefficient (default: 0.01)
            - normalize_returns: Whether to normalize returns (default: True)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 128)
        self.discrete = config.get("discrete", True)
        self.gamma = config.get("gamma", 0.99)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.normalize_returns = config.get("normalize_returns", True)
        self.use_baseline = config.get("use_baseline", True)

        # Policy network
        if self.discrete:
            self.policy = DiscretePolicy(self.state_dim, self.action_dim, self.hidden_dim)
        else:
            self.policy = ContinuousPolicy(
                self.state_dim, self.action_dim, self.hidden_dim,
                max_action=config.get("max_action", 1.0)
            )

        # Baseline (optional)
        self.baseline = None
        if self.use_baseline:
            self.baseline = Baseline(self.state_dim, self.hidden_dim)
            self.baseline_optimizer = torch.optim.Adam(
                self.baseline.parameters(),
                lr=config.get("baseline_lr", 1e-3)
            )

        # Policy optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.get("learning_rate", 1e-3)
        )

        # Episode storage
        self.saved_log_probs: List[torch.Tensor] = []
        self.saved_rewards: List[float] = []
        self.saved_states: List[torch.Tensor] = []
        self.saved_actions: List[torch.Tensor] = []

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor]
    ) -> Union[int, np.ndarray]:
        """Select action and store log probability."""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        self.saved_states.append(state)

        if self.discrete:
            action, log_prob = self.policy.get_action(state)
            self.saved_actions.append(torch.tensor([action]))
        else:
            action, log_prob = self.policy.get_action(state)
            self.saved_actions.append(torch.FloatTensor(action))

        self.saved_log_probs.append(log_prob)
        return action

    def store_reward(self, reward: float):
        """Store reward for the current step."""
        self.saved_rewards.append(reward)

    def compute_returns(self) -> torch.Tensor:
        """Compute discounted returns for the episode."""
        returns = []
        R = 0

        for r in reversed(self.saved_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)

        if self.normalize_returns and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update(self) -> Dict[str, float]:
        """
        Update policy using REINFORCE.

        Should be called at the end of each episode.

        Returns:
            Dictionary with loss and other metrics
        """
        if len(self.saved_rewards) == 0:
            return {"policy_loss": 0.0}

        # Compute returns
        returns = self.compute_returns()

        # Stack saved tensors
        states = torch.cat(self.saved_states, dim=0)
        log_probs = torch.stack(self.saved_log_probs)

        if self.discrete:
            actions = torch.cat(self.saved_actions)
        else:
            actions = torch.stack(self.saved_actions)

        # Compute baseline values and update baseline
        baseline_loss = torch.tensor(0.0)
        advantages = returns

        if self.use_baseline and self.baseline is not None:
            values = self.baseline(states).squeeze()
            advantages = returns - values.detach()

            # Update baseline
            baseline_loss = F.mse_loss(values, returns)
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

        # Compute policy loss
        # Re-evaluate log probs for proper gradient flow
        log_probs_new, entropies = self.policy.evaluate(states, actions)

        policy_loss = -(log_probs_new * advantages).mean()
        entropy_loss = -entropies.mean()

        total_loss = policy_loss + self.entropy_coef * entropy_loss

        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Clear episode storage
        self.clear_episode()

        return {
            "policy_loss": policy_loss.item(),
            "entropy": -entropy_loss.item(),
            "baseline_loss": baseline_loss.item() if isinstance(baseline_loss, torch.Tensor) else baseline_loss,
            "mean_return": returns.mean().item(),
            "mean_advantage": advantages.mean().item()
        }

    def clear_episode(self):
        """Clear episode storage."""
        self.saved_log_probs = []
        self.saved_rewards = []
        self.saved_states = []
        self.saved_actions = []


class REINFORCEWithBaseline(REINFORCEAgent):
    """Alias for REINFORCE with baseline enabled."""

    def __init__(self, config: Dict[str, Any]):
        config["use_baseline"] = True
        super().__init__(config)


class VanillaREINFORCE(REINFORCEAgent):
    """Alias for vanilla REINFORCE without baseline."""

    def __init__(self, config: Dict[str, Any]):
        config["use_baseline"] = False
        super().__init__(config)
