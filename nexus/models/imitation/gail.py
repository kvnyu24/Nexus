"""
GAIL: Generative Adversarial Imitation Learning.

Paper: "Generative Adversarial Imitation Learning"
       Ho & Ermon, NeurIPS 2016
       https://arxiv.org/abs/1606.03476

GAIL frames imitation learning as a generative adversarial problem. A discriminator
learns to distinguish expert behavior from policy behavior, while the policy is
trained to fool the discriminator. This avoids explicit reward function learning
and achieves strong performance with few expert demonstrations.

Key innovations:
- GAN-style adversarial training for imitation
- Discriminator provides implicit reward signal
- No need for explicit inverse RL reward learning
- Works well with off-the-shelf policy gradient algorithms (TRPO, PPO)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from nexus.core.base import NexusModule


class GAILDiscriminator(NexusModule):
    """Discriminator network for GAIL.

    Classifies state-action pairs as coming from expert (1) or policy (0).

    Args:
        config: Configuration dictionary with keys:
            - state_dim (int): State space dimension
            - action_dim (int): Action space dimension
            - hidden_dims (list): Hidden layer sizes. Default [256, 256]
            - activation (str): Activation function. Default 'tanh'
            - use_spectral_norm (bool): Use spectral normalization. Default False
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.hidden_dims = config.get('hidden_dims', [256, 256])
        self.activation = config.get('activation', 'tanh')
        self.use_spectral_norm = config.get('use_spectral_norm', False)

        # Build network
        layers = []
        input_dim = self.state_dim + self.action_dim

        for hidden_dim in self.hidden_dims:
            linear = nn.Linear(input_dim, hidden_dim)
            if self.use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)

            if self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))

            input_dim = hidden_dim

        # Output layer
        linear = nn.Linear(input_dim, 1)
        if self.use_spectral_norm:
            linear = nn.utils.spectral_norm(linear)
        layers.append(linear)

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass of discriminator.

        Args:
            state: State tensor [batch, state_dim]
            action: Action tensor [batch, action_dim]

        Returns:
            Logits for being expert data [batch, 1]
        """
        x = torch.cat([state, action], dim=-1)
        logits = self.network(x)
        return logits

    def predict_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict probability of being expert data.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Probability [batch, 1]
        """
        logits = self.forward(state, action)
        return torch.sigmoid(logits)

    def compute_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute GAIL reward signal for the policy.

        The reward is based on how well the policy fools the discriminator.
        Common formulations:
        - r = log(D(s,a)) (original GAIL)
        - r = -log(1 - D(s,a)) (non-saturating)
        - r = log(D(s,a)) - log(1 - D(s,a)) (least squares)

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Reward tensor [batch, 1]
        """
        with torch.no_grad():
            d = self.predict_prob(state, action)
            # Non-saturating reward (better gradients)
            reward = -torch.log(1 - d + 1e-8)
        return reward


class GAILAgent(NexusModule):
    """GAIL agent combining discriminator with policy learning.

    Args:
        config: Configuration dictionary with keys:
            - state_dim (int): State space dimension
            - action_dim (int): Action space dimension
            - discriminator_config (dict): Config for GAILDiscriminator
            - policy: Policy network (any RL agent)
            - disc_lr (float): Discriminator learning rate. Default 3e-4
            - policy_lr (float): Policy learning rate. Default 3e-4
            - disc_steps (int): Discriminator updates per policy update. Default 5
            - grad_penalty_coef (float): Gradient penalty coefficient. Default 10.0
            - use_grad_penalty (bool): Use gradient penalty for discriminator. Default True
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Create discriminator
        disc_config = config.get('discriminator_config', {})
        disc_config.update({
            'state_dim': config['state_dim'],
            'action_dim': config['action_dim']
        })
        self.discriminator = GAILDiscriminator(disc_config)

        # Policy (provided externally, e.g., PPO agent)
        self.policy = config['policy']

        # Optimizers
        disc_lr = config.get('disc_lr', 3e-4)
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=disc_lr
        )

        # Training parameters
        self.disc_steps = config.get('disc_steps', 5)
        self.grad_penalty_coef = config.get('grad_penalty_coef', 10.0)
        self.use_grad_penalty = config.get('use_grad_penalty', True)

    def _compute_gradient_penalty(
        self,
        expert_states: torch.Tensor,
        expert_actions: torch.Tensor,
        policy_states: torch.Tensor,
        policy_actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-style training.

        Args:
            expert_states: Expert states
            expert_actions: Expert actions
            policy_states: Policy states
            policy_actions: Policy actions

        Returns:
            Gradient penalty scalar
        """
        batch_size = expert_states.size(0)
        alpha = torch.rand(batch_size, 1, device=expert_states.device)

        # Interpolate states and actions
        interp_states = alpha * expert_states + (1 - alpha) * policy_states
        interp_actions = alpha * expert_actions + (1 - alpha) * policy_actions

        interp_states.requires_grad_(True)
        interp_actions.requires_grad_(True)

        # Discriminator output on interpolated inputs
        interp_logits = self.discriminator(interp_states, interp_actions)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=interp_logits,
            inputs=[interp_states, interp_actions],
            grad_outputs=torch.ones_like(interp_logits),
            create_graph=True,
            retain_graph=True
        )

        # Concatenate and compute penalty
        grad_concat = torch.cat([g.view(batch_size, -1) for g in gradients], dim=1)
        grad_norm = grad_concat.norm(2, dim=1)
        penalty = ((grad_norm - 1) ** 2).mean()

        return penalty

    def update_discriminator(
        self,
        expert_states: torch.Tensor,
        expert_actions: torch.Tensor,
        policy_states: torch.Tensor,
        policy_actions: torch.Tensor
    ) -> Dict[str, float]:
        """Update discriminator to distinguish expert from policy.

        Args:
            expert_states: States from expert demonstrations
            expert_actions: Actions from expert demonstrations
            policy_states: States from policy rollouts
            policy_actions: Actions from policy rollouts

        Returns:
            Dictionary with discriminator loss statistics
        """
        # Discriminator predictions
        expert_logits = self.discriminator(expert_states, expert_actions)
        policy_logits = self.discriminator(policy_states, policy_actions)

        # Binary cross-entropy loss
        # Expert = 1, Policy = 0
        expert_loss = F.binary_cross_entropy_with_logits(
            expert_logits, torch.ones_like(expert_logits)
        )
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_logits, torch.zeros_like(policy_logits)
        )

        disc_loss = expert_loss + policy_loss

        # Gradient penalty
        if self.use_grad_penalty:
            grad_penalty = self._compute_gradient_penalty(
                expert_states, expert_actions,
                policy_states, policy_actions
            )
            disc_loss += self.grad_penalty_coef * grad_penalty
        else:
            grad_penalty = torch.tensor(0.0)

        # Update discriminator
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            expert_acc = (expert_logits > 0).float().mean()
            policy_acc = (policy_logits < 0).float().mean()

        return {
            'disc_loss': disc_loss.item(),
            'expert_loss': expert_loss.item(),
            'policy_loss': policy_loss.item(),
            'grad_penalty': grad_penalty.item(),
            'expert_acc': expert_acc.item(),
            'policy_acc': policy_acc.item()
        }

    def compute_rewards(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute GAIL rewards for policy training.

        Args:
            states: State tensor
            actions: Action tensor

        Returns:
            Reward tensor
        """
        return self.discriminator.compute_reward(states, actions)

    def update(
        self,
        expert_batch: Tuple[torch.Tensor, torch.Tensor],
        policy_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """Full GAIL update step.

        Args:
            expert_batch: Tuple of (expert_states, expert_actions)
            policy_batch: Tuple of (policy_states, policy_actions, policy_rewards)

        Returns:
            Dictionary with training statistics
        """
        expert_states, expert_actions = expert_batch
        policy_states, policy_actions, _ = policy_batch

        # Update discriminator multiple times
        disc_stats = {}
        for i in range(self.disc_steps):
            stats = self.update_discriminator(
                expert_states, expert_actions,
                policy_states, policy_actions
            )
            if i == self.disc_steps - 1:
                disc_stats = stats

        # Compute GAIL rewards for policy
        gail_rewards = self.compute_rewards(policy_states, policy_actions)

        # Update policy with GAIL rewards
        # (This is handled by the external policy algorithm, e.g., PPO)

        disc_stats['mean_reward'] = gail_rewards.mean().item()
        return disc_stats


# Convenience function
def train_gail(
    policy,
    expert_dataset,
    env,
    num_iterations: int = 1000,
    state_dim: int = None,
    action_dim: int = None
):
    """Convenience function to train a policy with GAIL.

    Args:
        policy: Policy network to train
        expert_dataset: Expert demonstration dataset
        env: Environment for policy rollouts
        num_iterations: Number of training iterations
        state_dim: State space dimension
        action_dim: Action space dimension

    Returns:
        Trained policy
    """
    config = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'policy': policy
    }

    agent = GAILAgent(config)

    for iteration in range(num_iterations):
        # Sample expert batch
        expert_batch = expert_dataset.sample()

        # Collect policy rollouts
        policy_batch = policy.collect_rollouts(env)

        # Update GAIL
        stats = agent.update(expert_batch, policy_batch)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: {stats}")

    return policy
