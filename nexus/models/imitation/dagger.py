"""
DAgger: Dataset Aggregation for Imitation Learning.

Paper: "A Reduction of Imitation Learning and Structured Prediction to No-Regret Learning"
       Ross, Gordon, & Bagnell, AISTATS 2011
       https://arxiv.org/abs/1011.0686

DAgger addresses the distribution mismatch problem in behavior cloning by iteratively
collecting data under the learned policy and querying the expert for labels. This
creates a dataset that covers states the policy actually visits, leading to more
robust imitation learning.

Key innovations:
- Interactive expert querying during policy training
- Aggregates datasets from all policy iterations
- Beta schedule for mixing expert and learned policy during rollouts
- Provably reduces compounding errors in sequential prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Callable
from nexus.core.base import NexusModule


class DAggerPolicy(NexusModule):
    """Learnable policy network for DAgger.

    Args:
        config: Configuration dictionary with keys:
            - state_dim (int): State space dimension
            - action_dim (int): Action space dimension
            - hidden_dims (list): Hidden layer sizes. Default [256, 256]
            - activation (str): Activation function. Default 'relu'
            - action_type (str): 'continuous' or 'discrete'. Default 'continuous'
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.hidden_dims = config.get('hidden_dims', [256, 256])
        self.activation = config.get('activation', 'relu')
        self.action_type = config.get('action_type', 'continuous')

        # Build network
        layers = []
        input_dim = self.state_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))

            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())

            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Output head
        if self.action_type == 'continuous':
            # Mean and log_std for Gaussian policy
            self.mean_head = nn.Linear(input_dim, self.action_dim)
            self.log_std_head = nn.Linear(input_dim, self.action_dim)
        else:
            # Logits for categorical policy
            self.logits_head = nn.Linear(input_dim, self.action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict action.

        Args:
            state: State tensor [batch, state_dim]

        Returns:
            Action tensor [batch, action_dim]
        """
        features = self.backbone(state)

        if self.action_type == 'continuous':
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, -20, 2)  # Stability
            std = log_std.exp()

            # Sample action (for training with noise, return mean for inference)
            if self.training:
                action = mean + std * torch.randn_like(mean)
            else:
                action = mean

            return action
        else:
            logits = self.logits_head(features)
            # Sample action for training, argmax for inference
            if self.training:
                action_probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(action_probs, 1).squeeze(-1)
            else:
                action = logits.argmax(dim=-1)

            return action

    def compute_loss(
        self,
        states: torch.Tensor,
        expert_actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute supervised learning loss.

        Args:
            states: State tensor
            expert_actions: Expert action labels

        Returns:
            Loss scalar
        """
        features = self.backbone(states)

        if self.action_type == 'continuous':
            mean = self.mean_head(features)
            # MSE loss for continuous actions
            loss = F.mse_loss(mean, expert_actions)
        else:
            logits = self.logits_head(features)
            # Cross-entropy loss for discrete actions
            loss = F.cross_entropy(logits, expert_actions.long())

        return loss


class DAggerAgent(NexusModule):
    """DAgger agent with iterative data aggregation.

    Args:
        config: Configuration dictionary with keys:
            - state_dim (int): State space dimension
            - action_dim (int): Action space dimension
            - policy_config (dict): Config for DAggerPolicy
            - expert_policy (callable): Expert policy function
            - beta_schedule (str): 'linear', 'exponential', or 'constant'. Default 'linear'
            - beta_start (float): Initial beta value. Default 1.0
            - beta_end (float): Final beta value. Default 0.0
            - num_iterations (int): Number of DAgger iterations. Default 10
            - learning_rate (float): Learning rate. Default 1e-3
            - batch_size (int): Batch size for training. Default 64
            - epochs_per_iter (int): Training epochs per iteration. Default 10
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Create policy
        policy_config = config.get('policy_config', {})
        policy_config.update({
            'state_dim': config['state_dim'],
            'action_dim': config['action_dim']
        })
        self.policy = DAggerPolicy(policy_config)

        # Expert policy (function that takes state and returns action)
        self.expert_policy = config['expert_policy']

        # Beta schedule parameters
        self.beta_schedule = config.get('beta_schedule', 'linear')
        self.beta_start = config.get('beta_start', 1.0)
        self.beta_end = config.get('beta_end', 0.0)
        self.num_iterations = config.get('num_iterations', 10)

        # Training parameters
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.batch_size = config.get('batch_size', 64)
        self.epochs_per_iter = config.get('epochs_per_iter', 10)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate
        )

        # Aggregated dataset
        self.dataset_states = []
        self.dataset_actions = []

    def _compute_beta(self, iteration: int) -> float:
        """Compute beta value for mixing expert and learned policy.

        Args:
            iteration: Current iteration number

        Returns:
            Beta value in [0, 1]
        """
        if self.beta_schedule == 'linear':
            beta = self.beta_start + (self.beta_end - self.beta_start) * (
                iteration / max(self.num_iterations - 1, 1)
            )
        elif self.beta_schedule == 'exponential':
            decay_rate = -torch.log(torch.tensor(self.beta_end / self.beta_start)) / self.num_iterations
            beta = self.beta_start * torch.exp(-decay_rate * iteration).item()
        else:  # constant
            beta = self.beta_start

        return max(min(beta, 1.0), 0.0)

    def select_action(
        self,
        state: torch.Tensor,
        beta: float
    ) -> torch.Tensor:
        """Select action using beta-weighted mix of expert and learned policy.

        Args:
            state: Current state
            beta: Mixing parameter (1.0 = pure expert, 0.0 = pure policy)

        Returns:
            Selected action
        """
        # With probability beta, use expert; otherwise use learned policy
        if torch.rand(1).item() < beta:
            action = self.expert_policy(state)
        else:
            with torch.no_grad():
                self.policy.training = False
                action = self.policy(state)
                self.policy.training = True

        return action

    def collect_data(
        self,
        env,
        iteration: int,
        num_episodes: int = 10
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Collect data using mixed policy with expert labels.

        Args:
            env: Environment to interact with
            iteration: Current DAgger iteration
            num_episodes: Number of episodes to collect

        Returns:
            Tuple of (states, expert_actions)
        """
        beta = self._compute_beta(iteration)
        states = []
        expert_actions = []

        for episode in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                # Execute action from mixed policy
                action = self.select_action(state_tensor, beta)

                # Query expert for label
                expert_action = self.expert_policy(state_tensor)

                # Store state and expert action
                states.append(state_tensor)
                expert_actions.append(expert_action)

                # Take step in environment
                next_state, reward, done, info = env.step(action.cpu().numpy())
                state = next_state

        return states, expert_actions

    def update_dataset(
        self,
        new_states: List[torch.Tensor],
        new_actions: List[torch.Tensor]
    ) -> None:
        """Aggregate new data into dataset.

        Args:
            new_states: New state observations
            new_actions: New expert action labels
        """
        self.dataset_states.extend(new_states)
        self.dataset_actions.extend(new_actions)

    def train_policy(self) -> float:
        """Train policy on aggregated dataset.

        Returns:
            Average training loss
        """
        if len(self.dataset_states) == 0:
            return 0.0

        # Create dataset
        states = torch.cat(self.dataset_states, dim=0)
        actions = torch.cat(self.dataset_actions, dim=0)

        dataset_size = states.size(0)
        total_loss = 0.0
        num_batches = 0

        self.policy.training = True

        for epoch in range(self.epochs_per_iter):
            # Shuffle data
            indices = torch.randperm(dataset_size)

            for i in range(0, dataset_size, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]

                # Compute loss
                loss = self.policy.compute_loss(batch_states, batch_actions)

                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def train(
        self,
        env,
        num_iterations: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """Full DAgger training loop.

        Args:
            env: Environment to train in
            num_iterations: Number of iterations (overrides config if provided)

        Returns:
            Dictionary with training statistics
        """
        if num_iterations is None:
            num_iterations = self.num_iterations

        losses = []
        betas = []

        print(f"Starting DAgger training for {num_iterations} iterations...")

        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")

            # Compute beta for this iteration
            beta = self._compute_beta(iteration)
            betas.append(beta)
            print(f"  Beta: {beta:.3f}")

            # Collect data
            print("  Collecting data...")
            new_states, new_actions = self.collect_data(env, iteration)
            print(f"  Collected {len(new_states)} transitions")

            # Aggregate data
            self.update_dataset(new_states, new_actions)
            print(f"  Total dataset size: {len(self.dataset_states)}")

            # Train policy
            print("  Training policy...")
            avg_loss = self.train_policy()
            losses.append(avg_loss)
            print(f"  Average loss: {avg_loss:.4f}")

        print("\nDAgger training complete!")

        return {
            'losses': losses,
            'betas': betas
        }


# Convenience function
def train_with_dagger(
    env,
    expert_policy: Callable,
    state_dim: int,
    action_dim: int,
    num_iterations: int = 10,
    action_type: str = 'continuous'
):
    """Convenience function to train a policy with DAgger.

    Args:
        env: Environment to train in
        expert_policy: Expert policy function
        state_dim: State space dimension
        action_dim: Action space dimension
        num_iterations: Number of DAgger iterations
        action_type: 'continuous' or 'discrete'

    Returns:
        Trained policy
    """
    config = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'expert_policy': expert_policy,
        'num_iterations': num_iterations,
        'policy_config': {'action_type': action_type}
    }

    agent = DAggerAgent(config)
    agent.train(env, num_iterations)

    return agent.policy
