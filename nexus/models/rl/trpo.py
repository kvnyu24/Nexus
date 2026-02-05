"""
Trust Region Policy Optimization (TRPO)
Paper: "Trust Region Policy Optimization" (Schulman et al., ICML 2015)

TRPO is a policy gradient method that:
- Guarantees monotonic policy improvement through constrained optimization
- Uses a KL divergence constraint to limit policy updates
- Employs the conjugate gradient algorithm to efficiently solve the constrained optimization
- Computes natural gradients using the Fisher information matrix
- Provides theoretical guarantees on policy improvement

Key features:
- KL-constrained optimization: max E[advantage] subject to KL(π_old || π_new) ≤ δ
- Conjugate gradient for efficient natural gradient computation
- Line search with backtracking to ensure constraint satisfaction
- GAE (Generalized Advantage Estimation) for variance reduction
- More stable than vanilla policy gradient but more complex than PPO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
from ...core.base import NexusModule
import numpy as np


class TRPOActor(nn.Module):
    """
    Actor network for TRPO with continuous actions.

    Outputs a Gaussian policy: mean and log_std for each action dimension.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer size
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass outputting policy distribution parameters.

        Args:
            state: Input state [batch, state_dim]

        Returns:
            Tuple of (mean, log_std) for Gaussian policy
        """
        features = self.network(state)
        mean = self.mean_head(features)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def get_distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        """Get the policy distribution for the given state."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        return torch.distributions.Normal(mean, std)


class TRPOCritic(nn.Module):
    """
    Critic network (value function) for TRPO.

    Args:
        state_dim: Dimension of state space
        hidden_dim: Hidden layer size
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing state value.

        Args:
            state: Input state [batch, state_dim]

        Returns:
            Value estimate [batch, 1]
        """
        return self.network(state)


class TRPOAgent(NexusModule):
    """
    Trust Region Policy Optimization Agent.

    Implements TRPO with:
    - KL-constrained policy optimization
    - Conjugate gradient for natural gradient computation
    - Line search with backtracking
    - GAE for advantage estimation

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state space
            - action_dim: Dimension of action space
            - hidden_dim: Hidden layer size (default: 256)
            - gamma: Discount factor (default: 0.99)
            - lambda_: GAE lambda (default: 0.95)
            - max_kl: Maximum KL divergence (default: 0.01)
            - damping: Damping coefficient for CG (default: 0.1)
            - cg_iters: Conjugate gradient iterations (default: 10)
            - backtrack_iters: Line search backtrack iterations (default: 10)
            - backtrack_coeff: Line search backtrack coefficient (default: 0.8)
            - value_lr: Value function learning rate (default: 1e-3)
            - value_iters: Value function update iterations (default: 5)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.gamma = config.get("gamma", 0.99)
        self.lambda_ = config.get("lambda_", 0.95)
        self.max_kl = config.get("max_kl", 0.01)
        self.damping = config.get("damping", 0.1)
        self.cg_iters = config.get("cg_iters", 10)
        self.backtrack_iters = config.get("backtrack_iters", 10)
        self.backtrack_coeff = config.get("backtrack_coeff", 0.8)
        self.value_iters = config.get("value_iters", 5)

        # Actor and critic networks
        self.actor = TRPOActor(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic = TRPOCritic(self.state_dim, self.hidden_dim)

        # Value function optimizer
        self.value_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.get("value_lr", 1e-3),
        )

    def select_action(
        self, state: Union[np.ndarray, torch.Tensor], deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action using the current policy.

        Args:
            state: Current state
            deterministic: If True, use mean action

        Returns:
            Action as numpy array
        """
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            dist = self.actor.get_distribution(state)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()

            return action.cpu().numpy()[0]

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Rewards [batch]
            values: State values [batch]
            dones: Done flags [batch]
            next_values: Next state values [batch]

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * (1 - dones[t]) * next_value - values[t]
            advantages[t] = delta + self.gamma * self.lambda_ * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        returns = advantages + values
        return advantages, returns

    def compute_kl(
        self,
        states: torch.Tensor,
        old_mean: torch.Tensor,
        old_log_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between old and new policies.

        Args:
            states: States [batch, state_dim]
            old_mean: Old policy means [batch, action_dim]
            old_log_std: Old policy log stds [batch, action_dim]

        Returns:
            Mean KL divergence (scalar)
        """
        new_mean, new_log_std = self.actor(states)

        # KL(N(μ_old, σ_old) || N(μ_new, σ_new))
        old_std = old_log_std.exp()
        new_std = new_log_std.exp()

        kl = (
            new_log_std - old_log_std
            + (old_std ** 2 + (old_mean - new_mean) ** 2) / (2.0 * new_std ** 2)
            - 0.5
        )
        return kl.sum(dim=-1).mean()

    def flat_grad(self, loss: torch.Tensor, parameters) -> torch.Tensor:
        """Compute flattened gradient."""
        grads = torch.autograd.grad(loss, parameters, create_graph=True)
        return torch.cat([grad.view(-1) for grad in grads])

    def conjugate_gradient(
        self,
        fisher_vector_product,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve Ax = b using conjugate gradient, where A is the Fisher information matrix.

        Args:
            fisher_vector_product: Function computing Fisher-vector product
            b: Target vector (policy gradient)

        Returns:
            Solution x (natural gradient direction)
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)

        for _ in range(self.cg_iters):
            Ap = fisher_vector_product(p)
            alpha = rdotr / torch.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            p = r + (new_rdotr / rdotr) * p
            rdotr = new_rdotr

            if rdotr < 1e-10:
                break

        return x

    def fisher_vector_product(
        self, states: torch.Tensor, old_mean: torch.Tensor, old_log_std: torch.Tensor
    ):
        """
        Create a function that computes Fisher-vector products.

        Returns a function that takes a vector and returns F * vector,
        where F is the Fisher information matrix.
        """
        def fvp(v: torch.Tensor) -> torch.Tensor:
            # Compute KL divergence
            kl = self.compute_kl(states, old_mean, old_log_std)

            # Compute gradient of KL w.r.t. parameters
            kl_grad = self.flat_grad(kl, self.actor.parameters())

            # Compute gradient-vector product
            gvp = torch.dot(kl_grad, v)

            # Compute Hessian-vector product (second derivative)
            hvp = self.flat_grad(gvp, self.actor.parameters())

            # Add damping
            return hvp + self.damping * v

        return fvp

    def set_flat_params(self, flat_params: torch.Tensor):
        """Set model parameters from a flat vector."""
        offset = 0
        for param in self.actor.parameters():
            param_length = param.numel()
            param.data.copy_(
                flat_params[offset:offset + param_length].view_as(param)
            )
            offset += param_length

    def get_flat_params(self) -> torch.Tensor:
        """Get model parameters as a flat vector."""
        return torch.cat([param.view(-1) for param in self.actor.parameters()])

    def line_search(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_mean: torch.Tensor,
        old_log_std: torch.Tensor,
        old_loss: torch.Tensor,
        full_step: torch.Tensor,
    ) -> bool:
        """
        Backtracking line search to ensure improvement and KL constraint.

        Args:
            states: States
            actions: Actions
            advantages: Advantages
            old_mean: Old policy means
            old_log_std: Old policy log stds
            old_loss: Old policy loss
            full_step: Full natural gradient step

        Returns:
            True if a valid step was found, False otherwise
        """
        old_params = self.get_flat_params()

        for i in range(self.backtrack_iters):
            step_size = self.backtrack_coeff ** i
            new_params = old_params + step_size * full_step
            self.set_flat_params(new_params)

            # Check KL constraint
            with torch.no_grad():
                kl = self.compute_kl(states, old_mean, old_log_std)
                if kl > self.max_kl:
                    continue

            # Check improvement
            new_loss = self._compute_policy_loss(states, actions, advantages)
            if new_loss < old_loss:
                return True

        # If no valid step found, restore old parameters
        self.set_flat_params(old_params)
        return False

    def _compute_policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute policy loss (negative expected advantage)."""
        dist = self.actor.get_distribution(states)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return -(log_probs * advantages).mean()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        TRPO update: natural policy gradient with KL constraint.

        Args:
            batch: Dictionary containing:
                - states: States [batch, state_dim]
                - actions: Actions [batch, action_dim]
                - rewards: Rewards [batch]
                - next_states: Next states [batch, state_dim]
                - dones: Done flags [batch]

        Returns:
            Dictionary with loss metrics
        """
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Compute values and advantages
        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)
            advantages, returns = self.compute_gae(rewards, values, dones, next_values)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Store old policy for KL computation
            old_mean, old_log_std = self.actor(states)

        # --- Update value function ---
        for _ in range(self.value_iters):
            value_pred = self.critic(states).squeeze(-1)
            value_loss = F.mse_loss(value_pred, returns)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        # --- Update policy with TRPO ---
        # Compute policy gradient
        policy_loss = self._compute_policy_loss(states, actions, advantages)
        policy_grad = self.flat_grad(policy_loss, self.actor.parameters())

        # Compute natural gradient using conjugate gradient
        fvp_fn = self.fisher_vector_product(states, old_mean, old_log_std)
        natural_grad = self.conjugate_gradient(fvp_fn, policy_grad)

        # Compute step size: √(2δ / g^T F^{-1} g)
        gHg = torch.dot(natural_grad, fvp_fn(natural_grad))
        step_size = torch.sqrt(2 * self.max_kl / (gHg + 1e-8))
        full_step = step_size * natural_grad

        # Line search with backtracking
        old_loss = policy_loss.detach()
        success = self.line_search(
            states, actions, advantages, old_mean, old_log_std, old_loss, full_step
        )

        # Compute final KL for logging
        with torch.no_grad():
            final_kl = self.compute_kl(states, old_mean, old_log_std)

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "kl_divergence": final_kl.item(),
            "line_search_success": float(success),
            "mean_advantage": advantages.mean().item(),
            "mean_value": values.mean().item(),
        }

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing policy and value.

        Args:
            state: Input state

        Returns:
            Dictionary with action, log_prob, and value
        """
        dist = self.actor.get_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        value = self.critic(state)

        return {
            "action": action,
            "log_prob": log_prob,
            "value": value,
        }
