"""
AIRL: Adversarial Inverse Reinforcement Learning.

AIRL learns reward functions from expert demonstrations that are transferable
across different environments and dynamics. Unlike behavioral cloning or GAIL,
AIRL explicitly disentangles the reward from the dynamics, enabling zero-shot
transfer to new environments with different transition dynamics.

Key innovations:
- Reward function that is invariant to environment dynamics
- Disentangled reward learning via adversarial training
- Transferable rewards: learned rewards work in different environments
- Optimal policy recovery guarantees under certain assumptions
- More stable training than vanilla IRL methods

The discriminator in AIRL is structured to recover rewards rather than just
distinguishing expert vs agent behavior, making the learned rewards interpretable
and reusable.

Paper: "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning"
       Fu et al., ICLR 2018
       https://arxiv.org/abs/1710.11248

Key difference from GAIL:
- GAIL learns a discriminator that only distinguishes trajectories
- AIRL's discriminator structure enables reward function recovery
- AIRL rewards transfer to new dynamics, GAIL's discriminator does not
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Callable
from nexus.core.base import NexusModule
import numpy as np


class RewardNetwork(NexusModule):
    """Reward function network for AIRL.

    Learns state-only or state-action reward function.

    Args:
        config: Configuration dictionary with keys:
            - state_dim (int): State space dimension
            - action_dim (int): Action space dimension
            - hidden_dims (list): Hidden layer sizes. Default [256, 256]
            - state_only (bool): If True, reward depends only on state. Default False
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.hidden_dims = config.get('hidden_dims', [256, 256])
        self.state_only = config.get('state_only', False)

        # Build network
        if self.state_only:
            input_dim = self.state_dim
        else:
            input_dim = self.state_dim + self.action_dim

        layers = []
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor,
                action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute reward for state(-action) pair.

        Args:
            state: State tensor (B, state_dim)
            action: Optional action tensor (B, action_dim)

        Returns:
            Reward tensor (B, 1)
        """
        if self.state_only:
            inp = state
        else:
            if action is None:
                raise ValueError("Action required for state-action reward")
            inp = torch.cat([state, action], dim=-1)

        reward = self.network(inp)
        return reward


class ValueNetwork(NexusModule):
    """Value function network for AIRL discriminator.

    Args:
        config: Configuration dictionary with keys:
            - state_dim (int): State space dimension
            - hidden_dims (list): Hidden layer sizes. Default [256, 256]
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.state_dim = config['state_dim']
        self.hidden_dims = config.get('hidden_dims', [256, 256])

        # Build network
        layers = []
        input_dim = self.state_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute value for state.

        Args:
            state: State tensor (B, state_dim)

        Returns:
            Value tensor (B, 1)
        """
        return self.network(state)


class AIRLDiscriminator(NexusModule):
    """AIRL discriminator with disentangled reward structure.

    The discriminator computes:
        D(s, a, s') = exp(f(s, a, s')) / [exp(f(s, a, s')) + π(a|s)]

    where f(s, a, s') = r(s, a) + γV(s') - V(s) is the advantage function
    approximation.

    This structure enables reward recovery and transfer.

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.gamma = config.get('gamma', 0.99)

        # Reward and value networks
        self.reward_net = RewardNetwork(config)
        self.value_net = ValueNetwork(config)

    def forward(self,
                state: torch.Tensor,
                action: torch.Tensor,
                next_state: torch.Tensor,
                log_pi: torch.Tensor) -> torch.Tensor:
        """Compute discriminator output.

        Args:
            state: Current state (B, state_dim)
            action: Action taken (B, action_dim)
            next_state: Next state (B, state_dim)
            log_pi: Log probability of action under current policy (B, 1)

        Returns:
            Discriminator output (B, 1) - probability of being expert
        """
        # Compute reward
        reward = self.reward_net(state, action)

        # Compute values
        value_s = self.value_net(state)
        value_s_next = self.value_net(next_state)

        # Compute advantage: f(s, a, s') = r(s, a) + γV(s') - V(s)
        advantage = reward + self.gamma * value_s_next - value_s

        # Compute discriminator: D = exp(f) / [exp(f) + π(a|s)]
        # In log space: log D = f - log[exp(f) + exp(log π)]
        # Use log-sum-exp trick for numerical stability
        log_p_expert = advantage
        log_p_policy = log_pi

        # Compute log[exp(f) + exp(log π)] using logsumexp
        max_val = torch.max(log_p_expert, log_p_policy)
        log_sum = max_val + torch.log(
            torch.exp(log_p_expert - max_val) + torch.exp(log_p_policy - max_val)
        )

        log_discriminator = log_p_expert - log_sum
        discriminator = torch.exp(log_discriminator)

        return discriminator

    def compute_reward(self, state: torch.Tensor,
                      action: torch.Tensor) -> torch.Tensor:
        """Compute learned reward for state-action pair.

        Args:
            state: State tensor (B, state_dim)
            action: Action tensor (B, action_dim)

        Returns:
            Reward tensor (B, 1)
        """
        return self.reward_net(state, action)


class AIRLAgent(NexusModule):
    """AIRL agent with policy and discriminator.

    Args:
        config: Configuration dictionary with keys:
            - state_dim (int): State space dimension
            - action_dim (int): Action space dimension
            - hidden_dims (list): Hidden layer sizes. Default [256, 256]
            - action_type (str): 'continuous' or 'discrete'. Default 'continuous'
            - gamma (float): Discount factor. Default 0.99
            - state_only_reward (bool): State-only reward. Default False
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.action_type = config.get('action_type', 'continuous')
        self.gamma = config.get('gamma', 0.99)

        # Policy network
        hidden_dims = config.get('hidden_dims', [256, 256])
        layers = []
        input_dim = self.state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim

        self.policy_backbone = nn.Sequential(*layers)

        if self.action_type == 'continuous':
            self.action_mean = nn.Linear(input_dim, self.action_dim)
            self.action_log_std = nn.Parameter(torch.zeros(self.action_dim))
        else:
            self.action_logits = nn.Linear(input_dim, self.action_dim)

        # Discriminator
        self.discriminator = AIRLDiscriminator(config)

    def get_action_and_log_prob(self, state: torch.Tensor,
                                 deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and its log probability.

        Args:
            state: State tensor (B, state_dim)
            deterministic: If True, return mean action

        Returns:
            Tuple of (action, log_prob)
        """
        features = self.policy_backbone(state)

        if self.action_type == 'continuous':
            mean = self.action_mean(features)
            std = torch.exp(self.action_log_std).expand_as(mean)

            if deterministic:
                action = mean
                # For deterministic, we can't compute proper log prob, use dummy
                log_prob = torch.zeros(state.shape[0], 1, device=state.device)
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.rsample()
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        else:
            logits = self.action_logits(features)
            dist = torch.distributions.Categorical(logits=logits)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action).unsqueeze(-1)

        return action, log_prob

    def forward(self, state: torch.Tensor,
                deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass to get action.

        Args:
            state: State tensor (B, state_dim)
            deterministic: If True, return deterministic action

        Returns:
            Dictionary with action and log_prob
        """
        action, log_prob = self.get_action_and_log_prob(state, deterministic)
        return {'action': action, 'log_prob': log_prob}


def train_airl(
    agent: AIRLAgent,
    expert_trajectories: List[Dict[str, np.ndarray]],
    env,
    num_iterations: int = 100,
    episodes_per_iter: int = 10,
    discriminator_steps: int = 5,
    policy_steps: int = 5,
    batch_size: int = 256,
    lr_policy: float = 3e-4,
    lr_discriminator: float = 3e-4,
    clip_grad: float = 1.0,
) -> Dict[str, List[float]]:
    """Train agent using AIRL algorithm.

    Args:
        agent: AIRL agent
        expert_trajectories: List of expert trajectory dicts with keys:
            'states', 'actions', 'next_states', 'dones'
        env: Environment for policy rollouts
        num_iterations: Number of training iterations
        episodes_per_iter: Episodes to collect per iteration
        discriminator_steps: Discriminator update steps per iteration
        policy_steps: Policy update steps per iteration
        batch_size: Batch size for training
        lr_policy: Policy learning rate
        lr_discriminator: Discriminator learning rate
        clip_grad: Gradient clipping value

    Returns:
        Training history dictionary
    """
    device = next(agent.parameters()).device

    # Optimizers
    policy_optimizer = torch.optim.Adam(
        list(agent.policy_backbone.parameters()) +
        list(agent.action_mean.parameters() if agent.action_type == 'continuous'
             else agent.action_logits.parameters()) +
        ([agent.action_log_std] if agent.action_type == 'continuous' else []),
        lr=lr_policy
    )

    discriminator_optimizer = torch.optim.Adam(
        agent.discriminator.parameters(),
        lr=lr_discriminator
    )

    # Prepare expert data
    expert_states = []
    expert_actions = []
    expert_next_states = []

    for traj in expert_trajectories:
        expert_states.append(traj['states'])
        expert_actions.append(traj['actions'])
        expert_next_states.append(traj['next_states'])

    expert_states = torch.FloatTensor(np.concatenate(expert_states, axis=0)).to(device)
    expert_actions = torch.FloatTensor(np.concatenate(expert_actions, axis=0)).to(device)
    expert_next_states = torch.FloatTensor(np.concatenate(expert_next_states, axis=0)).to(device)

    history = {
        'returns': [],
        'discriminator_losses': [],
        'policy_losses': [],
        'expert_accuracy': [],
    }

    for iteration in range(num_iterations):
        print(f"\n=== AIRL Iteration {iteration + 1}/{num_iterations} ===")

        # Collect agent trajectories
        agent_states_list = []
        agent_actions_list = []
        agent_next_states_list = []
        agent_log_probs_list = []

        for episode in range(episodes_per_iter):
            state = env.reset()
            done = False
            episode_return = 0

            ep_states, ep_actions, ep_next_states, ep_log_probs = [], [], [], []

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = agent(state_tensor, deterministic=False)
                    action = output['action'].cpu().numpy()[0]
                    log_prob = output['log_prob'].cpu().numpy()[0]

                next_state, reward, done, info = env.step(action)

                ep_states.append(state)
                ep_actions.append(action)
                ep_next_states.append(next_state)
                ep_log_probs.append(log_prob)

                episode_return += reward
                state = next_state

            agent_states_list.extend(ep_states)
            agent_actions_list.extend(ep_actions)
            agent_next_states_list.extend(ep_next_states)
            agent_log_probs_list.extend(ep_log_probs)

            print(f"  Episode {episode + 1}: Return = {episode_return:.2f}")
            history['returns'].append(episode_return)

        # Convert to tensors
        agent_states = torch.FloatTensor(np.array(agent_states_list)).to(device)
        agent_actions = torch.FloatTensor(np.array(agent_actions_list)).to(device)
        agent_next_states = torch.FloatTensor(np.array(agent_next_states_list)).to(device)
        agent_log_probs = torch.FloatTensor(np.array(agent_log_probs_list)).to(device)

        # Update discriminator
        disc_losses = []
        for _ in range(discriminator_steps):
            # Sample batches
            expert_idx = np.random.choice(len(expert_states), batch_size, replace=True)
            agent_idx = np.random.choice(len(agent_states), batch_size, replace=True)

            expert_s = expert_states[expert_idx]
            expert_a = expert_actions[expert_idx]
            expert_s_next = expert_next_states[expert_idx]

            agent_s = agent_states[agent_idx]
            agent_a = agent_actions[agent_idx]
            agent_s_next = agent_next_states[agent_idx]
            agent_log_pi = agent_log_probs[agent_idx]

            # Get expert log probs
            with torch.no_grad():
                _, expert_log_pi = agent.get_action_and_log_prob(expert_s)

            # Discriminator outputs
            expert_d = agent.discriminator(expert_s, expert_a, expert_s_next, expert_log_pi)
            agent_d = agent.discriminator(agent_s, agent_a, agent_s_next, agent_log_pi)

            # Binary cross-entropy loss
            expert_loss = -torch.log(expert_d + 1e-8).mean()
            agent_loss = -torch.log(1 - agent_d + 1e-8).mean()
            disc_loss = expert_loss + agent_loss

            discriminator_optimizer.zero_grad()
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.discriminator.parameters(), clip_grad)
            discriminator_optimizer.step()

            disc_losses.append(disc_loss.item())

        avg_disc_loss = np.mean(disc_losses)
        print(f"  Discriminator loss: {avg_disc_loss:.4f}")
        history['discriminator_losses'].append(avg_disc_loss)

        # Compute expert accuracy
        with torch.no_grad():
            expert_idx = np.random.choice(len(expert_states), min(1000, len(expert_states)), replace=False)
            agent_idx = np.random.choice(len(agent_states), min(1000, len(agent_states)), replace=False)

            _, expert_log_pi_eval = agent.get_action_and_log_prob(expert_states[expert_idx])
            _, agent_log_pi_eval = agent.get_action_and_log_prob(agent_states[agent_idx])

            expert_d_eval = agent.discriminator(
                expert_states[expert_idx],
                expert_actions[expert_idx],
                expert_next_states[expert_idx],
                expert_log_pi_eval
            )
            agent_d_eval = agent.discriminator(
                agent_states[agent_idx],
                agent_actions[agent_idx],
                agent_next_states[agent_idx],
                agent_log_pi_eval
            )

            expert_acc = (expert_d_eval > 0.5).float().mean().item()
            agent_acc = (agent_d_eval < 0.5).float().mean().item()
            total_acc = (expert_acc + agent_acc) / 2

        print(f"  Discriminator accuracy: {total_acc:.3f} (Expert: {expert_acc:.3f}, Agent: {agent_acc:.3f})")
        history['expert_accuracy'].append(total_acc)

        # Update policy (maximize reward from discriminator)
        policy_losses = []
        for _ in range(policy_steps):
            agent_idx = np.random.choice(len(agent_states), batch_size, replace=True)

            states = agent_states[agent_idx]
            actions = agent_actions[agent_idx]
            next_states = agent_next_states[agent_idx]

            # Recompute actions and log probs with gradients
            new_actions, log_probs = agent.get_action_and_log_prob(states)

            # Use new actions for reward computation
            with torch.no_grad():
                rewards = agent.discriminator.compute_reward(states, new_actions)

            # Policy gradient loss (maximize rewards)
            # Simple policy gradient: -log π(a|s) * r
            policy_loss = -(log_probs * rewards.detach()).mean()

            policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(agent.policy_backbone.parameters()) +
                list(agent.action_mean.parameters() if agent.action_type == 'continuous'
                     else agent.action_logits.parameters()),
                clip_grad
            )
            policy_optimizer.step()

            policy_losses.append(policy_loss.item())

        avg_policy_loss = np.mean(policy_losses)
        print(f"  Policy loss: {avg_policy_loss:.4f}")
        history['policy_losses'].append(avg_policy_loss)

    return history


__all__ = [
    'RewardNetwork',
    'ValueNetwork',
    'AIRLDiscriminator',
    'AIRLAgent',
    'train_airl'
]
