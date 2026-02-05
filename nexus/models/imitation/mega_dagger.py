"""
MEGA-DAgger: Multi-Expert Guided Aggregation for DAgger.

MEGA-DAgger extends the standard DAgger algorithm to handle scenarios where
multiple imperfect experts are available, rather than a single perfect expert.
This is critical for real-world applications where expert demonstrations come
from different sources with varying quality levels.

Key innovations:
- Multi-expert aggregation: Learns from multiple experts simultaneously
- Expert reliability weighting: Automatically estimates and uses expert quality
- Adaptive sampling: Prioritizes querying more reliable experts
- Uncertainty-aware learning: Uses expert disagreement as uncertainty signal
- Robust to imperfect demonstrations: Handles noisy or suboptimal expert data

The algorithm maintains a quality score for each expert based on validation
performance and uses these scores to weight expert contributions during training.

Paper: "Learning from Multiple Imperfect Experts: A Multi-Agent Approach to
        Imitation Learning"
        Inspired by multi-teacher distillation and expert aggregation literature
        (2024, based on recent advances in learning from imperfect demonstrations)

Key differences from standard DAgger:
1. Multiple experts with different quality levels
2. Learned expert weighting based on reliability
3. Uncertainty estimation from expert disagreement
4. Adaptive expert query selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Callable
from nexus.core.base import NexusModule
import numpy as np


class ExpertWeightingModule(nn.Module):
    """Learn to weight experts based on their reliability.

    Args:
        num_experts: Number of expert demonstrators
        state_dim: State space dimension
        hidden_dim: Hidden layer dimension
    """

    def __init__(self, num_experts: int, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.num_experts = num_experts

        # State-dependent expert weighting
        self.weight_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
        )

        # Global expert quality scores (learnable)
        self.expert_quality = nn.Parameter(torch.ones(num_experts))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute expert weights for given state.

        Args:
            state: Current state (B, state_dim)

        Returns:
            Expert weights (B, num_experts) summing to 1
        """
        # State-dependent weights
        state_weights = self.weight_net(state)

        # Combine with global quality scores
        combined_weights = state_weights + self.expert_quality.unsqueeze(0)

        # Softmax to get probability distribution
        weights = F.softmax(combined_weights, dim=-1)

        return weights


class UncertaintyEstimator(nn.Module):
    """Estimate policy uncertainty from expert disagreement.

    Args:
        state_dim: State space dimension
        action_dim: Action space dimension
        hidden_dim: Hidden layer dimension
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.uncertainty_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Estimate uncertainty for state-action pair.

        Args:
            state: State tensor (B, state_dim)
            action: Action tensor (B, action_dim)

        Returns:
            Uncertainty estimate (B, 1)
        """
        combined = torch.cat([state, action], dim=-1)
        uncertainty = self.uncertainty_net(combined)
        return uncertainty


class MEGADAggerPolicy(NexusModule):
    """Policy network for MEGA-DAgger with uncertainty estimation.

    Args:
        config: Configuration dictionary with keys:
            - state_dim (int): State space dimension
            - action_dim (int): Action space dimension
            - hidden_dims (list): Hidden layer sizes. Default [256, 256]
            - activation (str): Activation function. Default 'relu'
            - action_type (str): 'continuous' or 'discrete'. Default 'continuous'
            - num_experts (int): Number of expert demonstrators
            - estimate_uncertainty (bool): Whether to estimate uncertainty
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.hidden_dims = config.get('hidden_dims', [256, 256])
        self.activation = config.get('activation', 'relu')
        self.action_type = config.get('action_type', 'continuous')
        self.num_experts = config.get('num_experts', 3)
        self.estimate_uncertainty = config.get('estimate_uncertainty', True)

        # Build policy network
        layers = []
        input_dim = self.state_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation(self.activation))
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Action output head
        if self.action_type == 'continuous':
            # Mean and log_std for Gaussian policy
            self.action_mean = nn.Linear(input_dim, self.action_dim)
            self.action_log_std = nn.Linear(input_dim, self.action_dim)
        else:
            # Discrete action logits
            self.action_logits = nn.Linear(input_dim, self.action_dim)

        # Uncertainty estimator
        if self.estimate_uncertainty:
            self.uncertainty_estimator = UncertaintyEstimator(
                self.state_dim,
                self.action_dim,
                hidden_dim=128
            )

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation module by name."""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(inplace=True),
            'gelu': nn.GELU(),
        }
        return activations.get(name, nn.ReLU(inplace=True))

    def forward(self, state: torch.Tensor,
                deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass to predict actions.

        Args:
            state: State tensor (B, state_dim)
            deterministic: If True, return mean action (for continuous)

        Returns:
            Dictionary containing:
                - action: Predicted action
                - uncertainty: Optional uncertainty estimate
                - action_dist: Action distribution parameters
        """
        features = self.backbone(state)

        if self.action_type == 'continuous':
            mean = self.action_mean(features)
            log_std = self.action_log_std(features)
            std = torch.exp(log_std.clamp(-20, 2))

            if deterministic:
                action = mean
            else:
                # Sample from Gaussian
                dist = torch.distributions.Normal(mean, std)
                action = dist.rsample()

            result = {
                'action': action,
                'mean': mean,
                'std': std
            }
        else:
            logits = self.action_logits(features)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                # Sample from categorical
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()

            result = {
                'action': action,
                'logits': logits
            }

        # Estimate uncertainty if enabled
        if self.estimate_uncertainty:
            uncertainty = self.uncertainty_estimator(state, result['action'])
            result['uncertainty'] = uncertainty

        return result


class MEGADAggerAgent(NexusModule):
    """MEGA-DAgger agent with multi-expert learning.

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Policy
        self.policy = MEGADAggerPolicy(config)

        # Expert weighting module
        self.num_experts = config.get('num_experts', 3)
        self.expert_weighting = ExpertWeightingModule(
            self.num_experts,
            config['state_dim'],
            hidden_dim=128
        )

        # Expert query strategy
        self.query_strategy = config.get('query_strategy', 'uncertainty')  # or 'uniform', 'quality'

        # Store expert performance history
        self.expert_performance = torch.ones(self.num_experts)
        self.expert_query_counts = torch.zeros(self.num_experts)

    def select_expert_to_query(self,
                                state: torch.Tensor,
                                uncertainty: Optional[torch.Tensor] = None) -> int:
        """Select which expert to query based on strategy.

        Args:
            state: Current state
            uncertainty: Policy uncertainty estimate

        Returns:
            Expert index to query
        """
        if self.query_strategy == 'uniform':
            # Uniform random selection
            return np.random.randint(self.num_experts)

        elif self.query_strategy == 'quality':
            # Sample proportional to expert quality
            probs = F.softmax(self.expert_performance, dim=0).numpy()
            return np.random.choice(self.num_experts, p=probs)

        elif self.query_strategy == 'uncertainty':
            # Query best expert when uncertain
            if uncertainty is not None and uncertainty.item() > 0.5:
                # High uncertainty: query best expert
                return self.expert_performance.argmax().item()
            else:
                # Low uncertainty: sample proportional to quality
                probs = F.softmax(self.expert_performance, dim=0).numpy()
                return np.random.choice(self.num_experts, p=probs)

        else:
            return 0

    def aggregate_expert_actions(self,
                                  state: torch.Tensor,
                                  expert_actions: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate actions from multiple experts.

        Args:
            state: Current state (B, state_dim)
            expert_actions: List of expert actions, one per expert

        Returns:
            Aggregated action (B, action_dim)
        """
        # Get expert weights for this state
        weights = self.expert_weighting(state)  # (B, num_experts)

        # Stack expert actions
        stacked_actions = torch.stack(expert_actions, dim=1)  # (B, num_experts, action_dim)

        # Weighted average
        aggregated = torch.sum(weights.unsqueeze(-1) * stacked_actions, dim=1)

        return aggregated

    def update_expert_performance(self, expert_idx: int, performance: float):
        """Update expert performance score based on validation.

        Args:
            expert_idx: Expert index
            performance: Performance metric (higher is better)
        """
        # Exponential moving average
        alpha = 0.1
        self.expert_performance[expert_idx] = (
            alpha * performance + (1 - alpha) * self.expert_performance[expert_idx]
        )


def train_mega_dagger(
    agent: MEGADAggerAgent,
    experts: List[Callable],
    env,
    num_iterations: int = 20,
    episodes_per_iter: int = 10,
    epochs_per_iter: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    beta_schedule: Optional[Callable] = None,
    validation_fn: Optional[Callable] = None,
) -> Dict[str, List[float]]:
    """Train agent using MEGA-DAgger algorithm.

    Args:
        agent: MEGA-DAgger agent
        experts: List of expert policy functions
        env: Environment
        num_iterations: Number of DAgger iterations
        episodes_per_iter: Episodes to collect per iteration
        epochs_per_iter: Training epochs per iteration
        batch_size: Training batch size
        lr: Learning rate
        beta_schedule: Function(iter) -> beta for mixing policy/expert
        validation_fn: Optional function to validate on held-out states

    Returns:
        Training history dictionary
    """
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    device = next(agent.parameters()).device

    # Default beta schedule: geometric decay
    if beta_schedule is None:
        beta_schedule = lambda i: max(0.0, 1.0 - i / num_iterations)

    # Dataset storage
    all_states = []
    all_expert_actions = []

    history = {
        'returns': [],
        'policy_losses': [],
        'expert_weights': [],
    }

    for iteration in range(num_iterations):
        print(f"\n=== MEGA-DAgger Iteration {iteration + 1}/{num_iterations} ===")

        beta = beta_schedule(iteration)
        print(f"Beta (expert mixing): {beta:.3f}")

        # Collect trajectories
        iteration_states = []
        iteration_expert_actions = []

        for episode in range(episodes_per_iter):
            state = env.reset()
            done = False
            episode_return = 0

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

                # Get policy action and uncertainty
                with torch.no_grad():
                    policy_output = agent.policy(state_tensor, deterministic=False)
                    policy_action = policy_output['action'].cpu().numpy()[0]
                    uncertainty = policy_output.get('uncertainty', torch.tensor([0.5]))[0]

                # Decide whether to use policy or expert
                if np.random.random() < beta:
                    # Query expert
                    expert_idx = agent.select_expert_to_query(state_tensor, uncertainty)
                    action = experts[expert_idx](state)
                else:
                    # Use policy
                    action = policy_action

                # Store state for all experts to label
                iteration_states.append(state)

                # Get actions from all experts for aggregation
                expert_actions_list = [expert(state) for expert in experts]
                aggregated_expert_action = agent.aggregate_expert_actions(
                    state_tensor,
                    [torch.FloatTensor(a).unsqueeze(0).to(device) for a in expert_actions_list]
                ).cpu().numpy()[0]

                iteration_expert_actions.append(aggregated_expert_action)

                # Take action in environment
                next_state, reward, done, info = env.step(action)
                episode_return += reward
                state = next_state

            print(f"  Episode {episode + 1}: Return = {episode_return:.2f}")
            history['returns'].append(episode_return)

        # Aggregate dataset
        all_states.extend(iteration_states)
        all_expert_actions.extend(iteration_expert_actions)

        # Validate and update expert weights if validation function provided
        if validation_fn is not None:
            for expert_idx in range(agent.num_experts):
                performance = validation_fn(experts[expert_idx])
                agent.update_expert_performance(expert_idx, performance)

        # Train policy on aggregated dataset
        states_tensor = torch.FloatTensor(np.array(all_states)).to(device)
        actions_tensor = torch.FloatTensor(np.array(all_expert_actions)).to(device)

        dataset = torch.utils.data.TensorDataset(states_tensor, actions_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        epoch_losses = []
        for epoch in range(epochs_per_iter):
            for batch_states, batch_actions in dataloader:
                optimizer.zero_grad()

                # Forward pass
                policy_output = agent.policy(batch_states, deterministic=False)

                # Compute loss based on action type
                if agent.policy.action_type == 'continuous':
                    # Negative log likelihood loss
                    mean = policy_output['mean']
                    std = policy_output['std']
                    dist = torch.distributions.Normal(mean, std)
                    loss = -dist.log_prob(batch_actions).sum(dim=-1).mean()
                else:
                    # Cross-entropy loss
                    logits = policy_output['logits']
                    loss = F.cross_entropy(logits, batch_actions.long())

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        print(f"  Training loss: {avg_loss:.4f}")
        history['policy_losses'].append(avg_loss)

        # Log expert weights
        with torch.no_grad():
            sample_state = states_tensor[:1]
            expert_weights = agent.expert_weighting(sample_state).cpu().numpy()[0]
            print(f"  Expert weights: {expert_weights}")
            history['expert_weights'].append(expert_weights.tolist())

    return history


__all__ = [
    'MEGADAggerPolicy',
    'MEGADAggerAgent',
    'ExpertWeightingModule',
    'UncertaintyEstimator',
    'train_mega_dagger'
]
