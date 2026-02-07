# DreamerV3: Mastering Diverse Domains through World Models

## Overview & Motivation

DreamerV3 is a reinforcement learning algorithm that learns a world model of the environment and trains policies purely by imagining trajectories in this learned model. It achieves state-of-the-art performance across diverse domains (Atari, DMC, Minecraft) using a single set of fixed hyperparameters, demonstrating unprecedented generality in model-based RL.

### Key Innovation

**Universal world model with fixed hyperparameters**:
- Works across 150+ different tasks without tuning
- Recurrent State-Space Model (RSSM) for dynamics
- Actor-critic learning entirely in imagination
- Symlog predictions for handling diverse reward scales
- Percentile normalization for stable training

### Problem Statement

Traditional model-free RL algorithms require extensive hyperparameter tuning for each domain. Model-based methods can be sample-efficient but often struggle with:
- Complex high-dimensional observations (images)
- Long-term credit assignment
- Compounding model errors
- Domain-specific tuning requirements

DreamerV3 solves these challenges through:
1. **Robust world modeling**: RSSM captures both deterministic and stochastic dynamics
2. **Imagination-based learning**: Train policy in perfect model rollouts
3. **Universal design**: Fixed hyperparameters work across all tested domains
4. **Scale-invariant predictions**: Symlog transformations handle any reward scale

## Theoretical Background

### World Model Architecture: RSSM

DreamerV3 uses a Recurrent State-Space Model (RSSM) that separates the latent state into deterministic and stochastic components. This architecture enables the model to capture both predictable transitions and inherent stochasticity in the environment.

**Deterministic state** h_t: Captures the recurrent history and temporal dependencies
**Stochastic state** z_t: Represents the unpredictable variations at each timestep

```
# Dynamics model
h_t = f_det(h_t-1, z_t-1, a_t-1)  # Deterministic recurrence (GRU)
z_t ~ p(z_t | h_t)                 # Stochastic prediction (categorical)

# Observation model
o_t ~ p(o_t | h_t, z_t)            # Decode to observations

# Reward model
r_t ~ p(r_t | h_t, z_t)            # Predict rewards

# Continue model (termination)
c_t ~ p(c_t | h_t, z_t)            # Predict episode continuation
```

The RSSM factorizes the joint distribution:

```
p(s_1:T, o_1:T, a_1:T, r_1:T) = ∏_t p(o_t | h_t, z_t) p(r_t | h_t, z_t) p(c_t | h_t, z_t) p(z_t | h_t) p(h_t | h_t-1, z_t-1, a_t-1)
```

### Learning in Imagination

Once the world model is trained, the policy is trained entirely by imagining trajectories in latent space. This decouples environment interaction from policy learning.

```
1. Sample initial state from replay buffer: (h_0, z_0)
2. Imagine trajectory:
   for t in range(imagination_horizon):
       a_t = π(h_t, z_t)              # Actor samples action
       h_t+1 = f_det(h_t, z_t, a_t)   # Deterministic transition
       z_t+1 ~ p(z_t+1 | h_t+1)       # Stochastic sampling
       r_t ~ p(r_t | h_t, z_t)        # Predicted reward
       c_t ~ p(c_t | h_t, z_t)        # Continuation flag

3. Compute λ-returns from imagined rewards
4. Update actor to maximize returns
5. Update critic to predict returns
```

**Advantages of imagination-based learning**:
- No environment interaction during policy training (fast)
- Perfect "replays" without approximation error
- Can train multiple policy updates per environment step
- Compounding errors limited by imagination horizon

### Posterior vs Prior in RSSM

The RSSM uses a variational approach with two distributions:

**Posterior** (uses observation):
```
q(z_t | h_t, o_t) = encoder(h_t, embed(o_t))
```
This is used during world model training to infer the true latent state given the observation.

**Prior** (prediction only):
```
p(z_t | h_t) = predictor(h_t)
```
This is used during imagination to generate trajectories without observations.

The KL divergence between posterior and prior trains the model to make accurate predictions:
```
KL(q(z_t | h_t, o_t) || p(z_t | h_t))
```

## Mathematical Formulation

### World Model Loss

The complete world model objective combines multiple prediction tasks:

```
L_world = L_dynamics + L_observation + L_reward + L_continue
```

**Dynamics Loss** (KL between predicted and actual posterior):
```
L_dynamics = E_t [ KL(q(z_t | h_t, o_t) || p(z_t | h_t)) ]
```

This encourages the prior to match the posterior, enabling accurate predictions without observations.

**Observation Loss** (reconstruction):
```
L_observation = E_t [ -log p(o_t | h_t, z_t) ]
```

For images, this is typically a Gaussian likelihood:
```
p(o_t | h_t, z_t) = N(o_t; μ_decoder(h_t, z_t), σ²)
```

**Reward Loss** (symlog space):
```
L_reward = E_t [ -log p(r_t | h_t, z_t) ]
```

Rewards are predicted in symlog space to handle diverse scales:
```
r_pred = symlog^(-1)(reward_head(h_t, z_t))
```

**Continue Loss** (predicts episode termination):
```
L_continue = E_t [ -log p(c_t | h_t, z_t) ]
```

This is a Bernoulli distribution:
```
p(c_t = 1 | h_t, z_t) = σ(continue_head(h_t, z_t))
```

### Actor-Critic Loss

The policy is trained using actor-critic methods in imagination.

**Critic Loss** (predict λ-return):
```
L_critic = E_τ [ ∑_t (V_θ(h_t, z_t) - λ_return_t)² ]
```

Where λ-return is computed recursively:
```
λ_return_t = r_t + γ·c_t·((1-λ)·V(h_t+1, z_t+1) + λ·λ_return_t+1)
```

This interpolates between TD(0) and Monte Carlo returns based on λ ∈ [0, 1].

**Actor Loss** (maximize value):
```
L_actor = E_τ [ -∑_t (λ_return_t - V(h_t, z_t)).detach() · log π(a_t | h_t, z_t) ]
```

With entropy regularization:
```
L_actor = -E_τ [ ∑_t λ_return_t.detach() · log π(a_t | h_t, z_t) + β·H(π(·|h_t, z_t)) ]
```

The entropy term encourages exploration:
```
H(π) = -E_{a~π} [log π(a)]
```

### Symlog Transformation

DreamerV3 uses symlog to handle diverse reward scales without normalization:

```
symlog(x) = sign(x) · log(|x| + 1)
symexp(x) = sign(x) · (exp(|x|) - 1)
```

**Properties**:
- Linear near zero: symlog(x) ≈ x for |x| < 1
- Logarithmic for large values: compresses large magnitudes
- Symmetric: handles both positive and negative rewards
- Invertible: symexp(symlog(x)) = x

This allows the same network architecture and learning rates to work on rewards ranging from -1000 to +1000.

### Free Bits for KL Regularization

To prevent posterior collapse (where q = p trivially), DreamerV3 uses free bits:

```
L_kl = max(KL(q || p), free_bits)
```

This allows the KL to be below `free_bits` without penalty, giving the model capacity to encode information in the stochastic state.

## High-Level Intuition

Think of DreamerV3 as a human learning to play a video game:

1. **World Model Learning** (Understanding the Game):
   - Watch gameplay (collect data)
   - Build mental model of game physics and rules
   - Predict what happens when you press buttons
   - Understand cause and effect

2. **Policy Learning in Imagination** (Mental Practice):
   - Imagine playing the game in your head
   - Try different strategies mentally
   - Learn which actions lead to high scores
   - Never touch the real game during this phase
   - Practice thousands of episodes in your mind

3. **Execution** (Playing):
   - Use learned policy in real game
   - Collect more data for improving world model
   - Repeat the cycle

**Key Insight**: Most learning happens in imagination (fast, safe, scalable), with minimal real interaction. This is analogous to how humans and animals learn through mental simulation and planning.

## Implementation Details

### Network Architecture

**Encoder** (for images):
```python
Conv2d(3, 32, kernel=4, stride=2)  # 64x64 -> 32x32
ReLU()
Conv2d(32, 64, kernel=4, stride=2)  # 32x32 -> 16x16
ReLU()
Conv2d(64, 128, kernel=4, stride=2)  # 16x16 -> 8x8
ReLU()
Conv2d(128, 256, kernel=4, stride=2)  # 8x8 -> 4x4
ReLU()
Flatten()  # 256 * 4 * 4 = 4096
Linear(4096, 1024)  # Embedding dimension
LayerNorm(1024)
```

**RSSM Core**:
- **Deterministic state**: GRU with 4096 hidden units
  ```python
  h_t = GRU(h_t-1, [z_t-1, a_t-1])  # Concatenate inputs
  ```

- **Stochastic state**: 32 categorical variables × 32 classes each
  ```python
  # Posterior (with observation)
  logits_post = MLP([h_t, embed_t])  # Output: [32, 32]
  z_t_post = OneHot(Categorical(logits_post))  # [1024] one-hot

  # Prior (prediction only)
  logits_prior = MLP(h_t)  # Output: [32, 32]
  z_t_prior = OneHot(Categorical(logits_prior))
  ```

**Decoder** (for images):
```python
Linear(4096 + 1024, 4096)  # h_t + z_t -> flattened spatial
Reshape(4096, [256, 4, 4])
ConvTranspose2d(256, 128, kernel=4, stride=2)  # 4x4 -> 8x8
ReLU()
ConvTranspose2d(128, 64, kernel=4, stride=2)  # 8x8 -> 16x16
ReLU()
ConvTranspose2d(64, 32, kernel=4, stride=2)  # 16x16 -> 32x32
ReLU()
ConvTranspose2d(32, 3, kernel=4, stride=2)  # 32x32 -> 64x64
```

**Reward Predictor**:
```python
MLP(
  input_dim=4096 + 1024,  # h_t + z_t
  hidden_dims=[512, 512, 512],
  output_dim=1,  # Scalar reward in symlog space
  activation=ELU,
  output_activation=None  # Linear output for symlog
)
```

**Continue Predictor**:
```python
MLP(
  input_dim=4096 + 1024,
  hidden_dims=[512, 512, 512],
  output_dim=1,  # Bernoulli logit
  activation=ELU,
  output_activation=None
)
```

**Value Network** (Critic):
```python
MLP(
  input_dim=4096 + 1024,
  hidden_dims=[512, 512, 512],
  output_dim=1,  # Value in symlog space
  activation=ELU,
  output_activation=None
)
```

**Policy Network** (Actor):
```python
# For continuous actions
MLP(
  input_dim=4096 + 1024,
  hidden_dims=[512, 512, 512],
  output_dim=2 * action_dim,  # Mean and log_std
  activation=ELU,
  output_activation=None
)

# For discrete actions
MLP(
  input_dim=4096 + 1024,
  hidden_dims=[512, 512, 512],
  output_dim=action_dim,  # Logits
  activation=ELU,
  output_activation=None
)
```

### Training Procedure

```python
# Pseudo-code for DreamerV3 training loop

import torch
import torch.nn.functional as F
from collections import deque

class DreamerV3:
    def __init__(self, config, env):
        self.env = env
        self.config = config

        # Networks
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.rssm = RSSM(config.rssm_hidden, config.rssm_stochastic)
        self.reward_model = RewardModel()
        self.continue_model = ContinueModel()
        self.actor = Actor()
        self.critic = Critic()

        # Optimizers
        self.world_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.reward_model.parameters()) +
            list(self.continue_model.parameters()),
            lr=config.world_lr
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)

    def train_step(self):
        """Single training iteration."""

        # Phase 1: Collect experience
        self.collect_data(self.config.collect_steps)

        # Phase 2: Train world model
        world_metrics = self.train_world_model(
            self.config.world_model_updates
        )

        # Phase 3: Train policy in imagination
        policy_metrics = self.train_policy(
            self.config.policy_updates
        )

        return {**world_metrics, **policy_metrics}

    def collect_data(self, num_steps):
        """Collect environment experience."""
        obs = self.env.reset()

        # Initialize RSSM state
        h, z = self.rssm.initial_state(batch_size=1)

        for _ in range(num_steps):
            # Encode observation
            embed = self.encoder(obs)

            # Update posterior
            z = self.rssm.posterior(h, embed)

            # Sample action from policy
            with torch.no_grad():
                action = self.actor(h, z).sample()

            # Environment step
            next_obs, reward, done, info = self.env.step(action)

            # Store transition
            self.replay_buffer.add(
                obs, action, reward, next_obs, done
            )

            # Update recurrent state
            h = self.rssm.recurrent(h, z, action)

            if done:
                obs = self.env.reset()
                h, z = self.rssm.initial_state(batch_size=1)
            else:
                obs = next_obs

    def train_world_model(self, num_updates):
        """Train world model components."""
        metrics = {}

        for _ in range(num_updates):
            # Sample sequence batch
            batch = self.replay_buffer.sample_sequences(
                batch_size=self.config.batch_size,
                sequence_length=self.config.sequence_length
            )

            # Encode observations
            embeddings = self.encoder(batch.observations)

            # Unroll RSSM through sequence
            h, z = self.rssm.initial_state(self.config.batch_size)

            # Storage for losses
            kl_losses = []
            recon_losses = []
            reward_losses = []
            continue_losses = []

            for t in range(self.config.sequence_length):
                # Posterior (uses observation)
                z_post, post_dist = self.rssm.posterior(
                    h, embeddings[:, t], return_dist=True
                )

                # Prior (prediction only)
                prior_dist = self.rssm.prior(h, return_dist=True)

                # KL divergence (dynamics loss)
                kl_loss = kl_divergence(post_dist, prior_dist)
                kl_loss = torch.maximum(
                    kl_loss,
                    torch.tensor(self.config.free_bits)
                )
                kl_losses.append(kl_loss.mean())

                # Reconstruction loss
                obs_dist = self.decoder(h, z_post)
                recon_loss = -obs_dist.log_prob(
                    batch.observations[:, t]
                )
                recon_losses.append(recon_loss.mean())

                # Reward prediction loss
                reward_pred = self.reward_model(h, z_post)
                reward_target = symlog(batch.rewards[:, t])
                reward_loss = F.mse_loss(reward_pred, reward_target)
                reward_losses.append(reward_loss)

                # Continue prediction loss
                continue_pred = self.continue_model(h, z_post)
                continue_target = 1.0 - batch.dones[:, t].float()
                continue_loss = F.binary_cross_entropy_with_logits(
                    continue_pred, continue_target
                )
                continue_losses.append(continue_loss)

                # Update recurrent state for next timestep
                h = self.rssm.recurrent(h, z_post, batch.actions[:, t])

            # Aggregate losses
            kl_loss = torch.stack(kl_losses).mean()
            recon_loss = torch.stack(recon_losses).mean()
            reward_loss = torch.stack(reward_losses).mean()
            continue_loss = torch.stack(continue_losses).mean()

            # Total world model loss
            world_loss = (
                kl_loss +
                recon_loss +
                reward_loss +
                continue_loss
            )

            # Optimize
            self.world_optimizer.zero_grad()
            world_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.world_optimizer.param_groups[0]['params'],
                max_norm=self.config.grad_clip
            )
            self.world_optimizer.step()

            # Track metrics
            metrics['kl_loss'] = kl_loss.item()
            metrics['recon_loss'] = recon_loss.item()
            metrics['reward_loss'] = reward_loss.item()
            metrics['continue_loss'] = continue_loss.item()

        return metrics

    def train_policy(self, num_updates):
        """Train actor and critic in imagination."""
        metrics = {}

        for _ in range(num_updates):
            # Sample initial states from replay buffer
            initial_states = self.replay_buffer.sample_states(
                self.config.batch_size
            )
            h, z = initial_states

            # Imagine trajectories
            imagined_states = []
            imagined_actions = []
            imagined_rewards = []
            imagined_continues = []

            for t in range(self.config.imagination_horizon):
                # Store current state
                imagined_states.append((h.detach(), z.detach()))

                # Sample action from policy
                action_dist = self.actor(h, z)
                action = action_dist.rsample()  # Reparameterized sample
                imagined_actions.append(action)

                # Predict reward
                reward = self.reward_model(h, z)
                imagined_rewards.append(symexp(reward))

                # Predict continue
                continue_logit = self.continue_model(h, z)
                continue_prob = torch.sigmoid(continue_logit)
                imagined_continues.append(continue_prob)

                # Imagine next state
                h = self.rssm.recurrent(h, z, action)
                z = self.rssm.prior(h)

            # Stack imagined trajectory
            rewards = torch.stack(imagined_rewards)  # [H, B]
            continues = torch.stack(imagined_continues)  # [H, B]

            # Compute values for all imagined states
            values = []
            for h, z in imagined_states:
                value = self.critic(h, z)
                values.append(symexp(value))
            values = torch.stack(values)  # [H, B]

            # Compute λ-returns
            lambda_returns = self.compute_lambda_returns(
                rewards, values, continues
            )

            # Update critic
            critic_loss = 0.5 * F.mse_loss(
                values, lambda_returns.detach()
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                max_norm=self.config.grad_clip
            )
            self.critic_optimizer.step()

            # Update actor
            # Recompute imagined trajectory (with gradients)
            h, z = initial_states
            actor_loss = 0
            entropy_loss = 0

            for t in range(self.config.imagination_horizon):
                # Policy distribution
                action_dist = self.actor(h, z)
                action = action_dist.rsample()

                # Actor loss: maximize advantage
                value = symexp(self.critic(h, z))
                advantage = lambda_returns[t] - value
                actor_loss -= advantage.detach() * action_dist.log_prob(
                    action
                ).sum(-1)

                # Entropy regularization
                entropy_loss -= action_dist.entropy().sum(-1)

                # Next state
                h = self.rssm.recurrent(h, z, action)
                z = self.rssm.prior(h)

            actor_loss = actor_loss.mean()
            entropy_loss = entropy_loss.mean()

            total_actor_loss = (
                actor_loss +
                self.config.entropy_coef * entropy_loss
            )

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                max_norm=self.config.grad_clip
            )
            self.actor_optimizer.step()

            # Track metrics
            metrics['critic_loss'] = critic_loss.item()
            metrics['actor_loss'] = actor_loss.item()
            metrics['entropy'] = -entropy_loss.item()
            metrics['mean_return'] = lambda_returns.mean().item()

        return metrics

    def compute_lambda_returns(self, rewards, values, continues):
        """Compute λ-returns for advantage estimation."""
        # rewards: [H, B]
        # values: [H, B]
        # continues: [H, B]

        lambda_coef = self.config.lambda_coef
        discount = self.config.discount

        # Bootstrap from last value
        returns = values[-1]

        # Backward pass through time
        lambda_returns = []
        for t in reversed(range(len(rewards))):
            returns = (
                rewards[t] +
                discount * continues[t] * (
                    (1 - lambda_coef) * values[t] +
                    lambda_coef * returns
                )
            )
            lambda_returns.insert(0, returns)

        return torch.stack(lambda_returns)
```

### Complete Nexus Implementation

```python
# nexus/models/world_models/dreamerv3.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
from typing import Tuple, Dict, List

class DreamerV3Config:
    """Configuration for DreamerV3."""

    # RSSM architecture
    rssm_hidden: int = 4096
    rssm_stochastic: int = 32  # Number of categoricals
    rssm_classes: int = 32  # Classes per categorical

    # Network dimensions
    embed_dim: int = 1024
    hidden_dim: int = 512
    num_layers: int = 3

    # Training
    batch_size: int = 16
    sequence_length: int = 64
    imagination_horizon: int = 15

    # Learning rates
    world_lr: float = 1e-4
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5

    # Loss coefficients
    kl_coef: float = 1.0
    recon_coef: float = 1.0
    reward_coef: float = 1.0
    continue_coef: float = 1.0
    entropy_coef: float = 1e-3

    # Regularization
    free_bits: float = 1.0
    grad_clip: float = 100.0

    # RL
    discount: float = 0.99
    lambda_coef: float = 0.95

    # Replay buffer
    buffer_size: int = 1_000_000

    # Data collection
    collect_steps: int = 100
    world_model_updates: int = 100
    policy_updates: int = 100


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithmic transformation."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class Encoder(nn.Module):
    """Encodes observations to embeddings."""

    def __init__(self, obs_shape, embed_dim=1024):
        super().__init__()

        if len(obs_shape) == 3:  # Image
            c, h, w = obs_shape
            self.conv = nn.Sequential(
                nn.Conv2d(c, 32, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.ReLU(),
                nn.Flatten(),
            )

            # Compute flattened size
            with torch.no_grad():
                dummy = torch.zeros(1, c, h, w)
                conv_out = self.conv(dummy)
                conv_dim = conv_out.shape[1]

            self.fc = nn.Sequential(
                nn.Linear(conv_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            )
        else:
            # Vector observations
            self.fc = nn.Sequential(
                nn.Linear(obs_shape[0], embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Tanh()
            )
            self.conv = None

    def forward(self, obs):
        if self.conv is not None:
            x = self.conv(obs)
            return self.fc(x)
        else:
            return self.fc(obs)


class Decoder(nn.Module):
    """Decodes latent states to observations."""

    def __init__(self, state_dim, obs_shape):
        super().__init__()

        if len(obs_shape) == 3:  # Image
            c, h, w = obs_shape

            # Compute initial spatial size
            self.init_h = h // 16
            self.init_w = w // 16
            init_dim = 256 * self.init_h * self.init_w

            self.fc = nn.Linear(state_dim, init_dim)

            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, c, 4, 2, 1),
            )
        else:
            # Vector observations
            self.fc = nn.Linear(state_dim, obs_shape[0])
            self.deconv = None

    def forward(self, state):
        if self.deconv is not None:
            x = self.fc(state)
            x = x.view(-1, 256, self.init_h, self.init_w)
            return self.deconv(x)
        else:
            return self.fc(state)


class RSSM(nn.Module):
    """Recurrent State-Space Model."""

    def __init__(
        self,
        action_dim,
        hidden_dim=4096,
        num_categoricals=32,
        num_classes=32,
        embed_dim=1024
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.stochastic_dim = num_categoricals * num_classes

        # Recurrent model (deterministic state)
        self.rnn = nn.GRUCell(
            self.stochastic_dim + action_dim,
            hidden_dim
        )

        # Prior (predict stochastic state from deterministic)
        self.prior_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ELU(),
            nn.Linear(512, num_categoricals * num_classes)
        )

        # Posterior (infer stochastic state from obs + deterministic)
        self.posterior_mlp = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, 512),
            nn.ELU(),
            nn.Linear(512, num_categoricals * num_classes)
        )

    def initial_state(self, batch_size, device='cpu'):
        """Initialize RSSM state."""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(
            batch_size,
            self.stochastic_dim,
            device=device
        )
        return h, z

    def recurrent(self, h, z, action):
        """Update deterministic state."""
        x = torch.cat([z, action], dim=-1)
        h_next = self.rnn(x, h)
        return h_next

    def prior(self, h, return_dist=False):
        """Predict stochastic state (prior)."""
        logits = self.prior_mlp(h)
        logits = logits.view(-1, self.num_categoricals, self.num_classes)

        dist = td.Independent(
            td.OneHotCategorical(logits=logits),
            1
        )

        if return_dist:
            return dist

        # Sample and flatten
        z = dist.sample()
        z = z.view(-1, self.stochastic_dim)
        return z

    def posterior(self, h, embed, return_dist=False):
        """Infer stochastic state (posterior)."""
        x = torch.cat([h, embed], dim=-1)
        logits = self.posterior_mlp(x)
        logits = logits.view(-1, self.num_categoricals, self.num_classes)

        dist = td.Independent(
            td.OneHotCategorical(logits=logits),
            1
        )

        if return_dist:
            z = dist.sample()
            z = z.view(-1, self.stochastic_dim)
            return z, dist

        z = dist.sample()
        z = z.view(-1, self.stochastic_dim)
        return z

    def get_state_dim(self):
        """Get total state dimension."""
        return self.hidden_dim + self.stochastic_dim


class RewardModel(nn.Module):
    """Predicts rewards from latent states."""

    def __init__(self, state_dim, hidden_dim=512):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h, z):
        state = torch.cat([h, z], dim=-1)
        return self.mlp(state).squeeze(-1)


class ContinueModel(nn.Module):
    """Predicts episode continuation."""

    def __init__(self, state_dim, hidden_dim=512):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h, z):
        state = torch.cat([h, z], dim=-1)
        return self.mlp(state).squeeze(-1)


class Actor(nn.Module):
    """Policy network."""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=512,
        discrete=False
    ):
        super().__init__()

        self.discrete = discrete
        self.action_dim = action_dim

        if discrete:
            # Discrete action space
            self.mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, action_dim)
            )
        else:
            # Continuous action space
            self.mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 2 * action_dim)
            )

    def forward(self, h, z):
        state = torch.cat([h, z], dim=-1)

        if self.discrete:
            logits = self.mlp(state)
            return td.Categorical(logits=logits)
        else:
            out = self.mlp(state)
            mean, log_std = out.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -10, 2)
            std = log_std.exp()
            return td.Normal(mean, std)


class Critic(nn.Module):
    """Value network."""

    def __init__(self, state_dim, hidden_dim=512):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h, z):
        state = torch.cat([h, z], dim=-1)
        return self.mlp(state).squeeze(-1)
```

## Optimization Tricks

### 1. Symlog Predictions

Predict in symlog space for diverse reward scales:

```python
def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

# Predict in symlog space
reward_pred_symlog = reward_model(h, z)
reward_pred = symexp(reward_pred_symlog)

# Loss also in symlog space
reward_target_symlog = symlog(batch.rewards)
loss = F.mse_loss(reward_pred_symlog, reward_target_symlog)
```

**Why it works**:
- Compresses large values logarithmically
- Linear near zero (preserves small rewards)
- Symmetric (handles negative rewards)
- Same network capacity for all scales

### 2. Percentile Normalization

Normalize values by their percentiles across the batch:

```python
def percentile_normalize(x, percentile_low=5, percentile_high=95):
    """Normalize to [0, 1] using percentiles."""
    low = torch.quantile(x, percentile_low / 100)
    high = torch.quantile(x, percentile_high / 100)
    x_norm = (x - low) / (high - low + 1e-8)
    return torch.clamp(x_norm, 0, 1)

# Use for value normalization
values = critic(states)
values_norm = percentile_normalize(values)
```

**Benefits**:
- Robust to outliers (uses percentiles not min/max)
- Adaptive to distribution shifts
- Stable across different domains

### 3. Free Bits for KL Loss

Prevent KL collapse with free bits:

```python
kl_loss = kl_divergence(posterior, prior)
kl_loss = torch.maximum(kl_loss, torch.tensor(free_bits))  # free_bits = 1.0

# Alternatively, per-dimension free bits
kl_per_dim = kl_divergence(posterior, prior, reduce=False)
kl_loss = torch.maximum(kl_per_dim, free_bits).sum()
```

**Why it helps**:
- Allows KL below threshold without penalty
- Prevents information collapse in stochastic state
- Maintains model capacity

### 4. Return Normalization

Normalize returns using exponential moving statistics:

```python
class RunningStats:
    def __init__(self, momentum=0.99):
        self.mean = 0
        self.std = 1
        self.momentum = momentum

    def update(self, x):
        self.mean = self.momentum * self.mean + (1 - self.momentum) * x.mean()
        self.std = self.momentum * self.std + (1 - self.momentum) * x.std()

    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-8)

# Usage
return_stats = RunningStats()
return_stats.update(lambda_returns)
normalized_returns = return_stats.normalize(lambda_returns)
```

### 5. Gradient Clipping

Clip gradients by global norm:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=100.0
)
```

**Prevents**:
- Gradient explosions
- Training instabilities
- NaN/Inf values

### 6. Categorical Representation

Use categorical distribution for stochastic states:

```python
# Instead of Gaussian: z ~ N(μ, σ²)
# Use multiple categoricals: z = [cat_1, cat_2, ..., cat_N]

num_categoricals = 32
num_classes = 32

# Each categorical is one-hot encoded
# Total stochastic dim = 32 * 32 = 1024

# Benefits:
# - Discrete = easier to learn
# - Multiple categoricals = expressive
# - One-hot = differentiable (straight-through estimator not needed)
```

### 7. Separate Learning Rates

Use different learning rates for different components:

```python
world_lr = 1e-4  # World model
actor_lr = 3e-5  # Policy (slower)
critic_lr = 3e-5  # Value function (slower)

# World model learns faster because:
# - Direct supervision from data
# - Reconstruction objectives

# Policy/value learn slower because:
# - Indirect RL signal
# - More sensitive to instability
```

### 8. Imagination Horizon

Use moderate imagination horizon (15 steps):

```python
imagination_horizon = 15  # Not too short, not too long

# Too short (5): Limited credit assignment
# Too long (50): Compounding model errors
# Sweet spot (15): Balance between both
```

## Hyperparameter Guidelines

### Core Hyperparameters

| Parameter | Value | Range | Notes |
|-----------|-------|-------|-------|
| `rssm_hidden` | 4096 | [2048, 8192] | Deterministic state size |
| `rssm_stochastic` | 32 | [16, 64] | Number of categoricals |
| `rssm_classes` | 32 | [16, 64] | Classes per categorical |
| `imagination_horizon` | 15 | [10, 20] | Longer = more model error |
| `batch_size` | 16 | [8, 32] | Sequence batch size |
| `sequence_length` | 64 | [32, 128] | Temporal context |
| `free_bits` | 1.0 | [0.5, 2.0] | KL regularization |
| `discount` | 0.99 | [0.95, 0.999] | Return discount factor |
| `lambda` | 0.95 | [0.9, 0.99] | λ-return parameter |

### Learning Rates

```python
# Default (works across domains)
world_lr = 1e-4
actor_lr = 3e-5
critic_lr = 3e-5

# For small-scale problems (toy envs)
world_lr = 3e-4
actor_lr = 1e-4
critic_lr = 1e-4

# For very large-scale (complex sims)
world_lr = 3e-5
actor_lr = 1e-5
critic_lr = 1e-5
```

### Data Collection

```python
# Typical settings
collect_interval = 100  # Steps between training
prefill_steps = 5000  # Initial random data
replay_buffer_size = 1_000_000

# Fast experimentation
collect_interval = 10
prefill_steps = 1000
replay_buffer_size = 100_000

# Large-scale training
collect_interval = 1000
prefill_steps = 50000
replay_buffer_size = 10_000_000
```

## Experiments & Results

### Atari 100k Benchmark

Performance on 26 Atari games with only 100k environment steps (400k frames):

| Method | Median Human-Normalized Score |
|--------|-------------------------------|
| Data-Efficient Rainbow | 0.42 |
| OTRainbow | 0.49 |
| CURL | 0.52 |
| DrQ | 0.58 |
| SPR | 0.67 |
| MuZero | 1.51 |
| **DreamerV3** | **1.83** |

**Analysis**:
- DreamerV3 achieves superhuman performance (>1.0) on average
- 21% improvement over MuZero
- 4.4x improvement over data-efficient Rainbow
- Uses same hyperparameters for all 26 games

**Per-game highlights**:
- Alien: 227% human performance
- Boxing: 198% human performance
- Breakout: 156% human performance
- Pong: 121% human performance

### DeepMind Control Suite

Continuous control on 20 DMC tasks:

| Method | Median Score | Mean Score |
|--------|--------------|------------|
| SAC | 823 | 801 |
| TD3 | 857 | 834 |
| DrQ-v2 | 918 | 897 |
| Dreamer | 905 | 882 |
| DreamerV2 | 943 | 921 |
| **DreamerV3** | **971** | **953** |

**Task breakdown**:
- Walker-walk: 998 / 1000
- Cartpole-swingup: 887 / 1000
- Reacher-easy: 976 / 1000
- Finger-spin: 993 / 1000
- Cheetah-run: 945 / 1000

**Key observations**:
- Near-optimal performance on most tasks
- Consistent across all 20 environments
- No task-specific tuning required

### Minecraft (Diamond Collection)

Long-horizon sparse reward task in 3D environment:

| Method | Success Rate | Steps to Diamond |
|--------|--------------|------------------|
| MineRL Baseline | 1% | Never |
| Behavioral Cloning | 3% | Never |
| VPT (700M params) | 15% | 24M |
| **DreamerV3** | **31%** | **12M** |

**Achievements**:
- First model-based method to solve diamond collection
- 2x success rate of VPT
- 50% fewer steps to first diamond
- Learns from pixels only (no privileged information)

**Task difficulty**:
- Requires ~20 minutes of gameplay
- Sparse reward (only at diamond collection)
- Complex action space (keyboard + mouse)
- 3D visual navigation

### Generality: Fixed Hyperparameters

DreamerV3 uses **identical hyperparameters** across:

**7 Atari games**:
- Different visual styles
- Different reward scales (-1 to +1)
- Different action spaces (4-18 actions)

**20 DMC tasks**:
- Continuous control
- Different embodiments
- Reward scales (0-1000)

**5 Minecraft tasks**:
- 3D environment
- Long horizon (20 min)
- Sparse rewards
- Complex action space

**Reward scale diversity**:
- Breakout: 0-400
- Pong: -21 to +21
- Alien: 0-7000
- DMC: 0-1000
- Minecraft: 0-1

**No tuning needed** - Same config works everywhere!

### Ablation Studies

**Impact of symlog predictions**:
| Configuration | Atari | DMC | Minecraft |
|---------------|-------|-----|-----------|
| No symlog | 1.2 | 720 | 8% |
| With symlog | 1.83 | 971 | 31% |

**Impact of RSSM size**:
| Hidden Dim | Atari 100k | DMC |
|------------|------------|-----|
| 1024 | 1.4 | 890 |
| 2048 | 1.6 | 920 |
| 4096 | 1.83 | 971 |
| 8192 | 1.85 | 969 |

**Impact of imagination horizon**:
| Horizon | Atari | DMC | Training Speed |
|---------|-------|-----|----------------|
| 5 | 1.5 | 920 | Fast |
| 10 | 1.7 | 950 | Medium |
| 15 | 1.83 | 971 | Medium |
| 20 | 1.81 | 965 | Slow |
| 50 | 1.6 | 910 | Very Slow |

**Optimal**: 15 steps balances credit assignment and model accuracy.

### Sample Efficiency

DreamerV3 vs model-free methods (steps to threshold):

| Environment | DreamerV3 | PPO | SAC | Speedup |
|-------------|-----------|-----|-----|---------|
| Atari (1.0 score) | 100k | 10M | N/A | 100x |
| DMC Walker | 100k | 1M | 500k | 5-10x |
| DMC Cheetah | 200k | 2M | 1M | 5-10x |

**Analysis**:
- DreamerV3 is 5-100x more sample efficient
- Largest gains on complex visual tasks
- Imagination enables offline policy improvement

## Common Pitfalls

### 1. KL Balancing

**Problem**: Posterior and prior diverge or collapse

**Symptoms**:
- KL loss → 0 (collapse: model ignores stochastic state)
- KL loss → ∞ (divergence: posterior ignores prior)
- Poor reconstruction despite low loss

**Solutions**:

```python
# Solution 1: Free bits
kl_loss = torch.maximum(kl, torch.tensor(1.0))

# Solution 2: KL balancing (weighted average)
kl_forward = kl_divergence(posterior, prior.detach())  # Train posterior
kl_reverse = kl_divergence(posterior.detach(), prior)  # Train prior
kl_loss = 0.8 * kl_forward + 0.2 * kl_reverse

# Solution 3: Annealing
kl_weight = min(1.0, step / 10000)  # Gradually increase
kl_loss = kl_weight * kl_divergence(posterior, prior)
```

**Diagnostics**:
```python
# Monitor these metrics
print(f"KL: {kl_loss.item():.3f}")
print(f"Posterior entropy: {posterior.entropy().mean().item():.3f}")
print(f"Prior entropy: {prior.entropy().mean().item():.3f}")
print(f"Posterior-Prior distance: {(posterior.mean - prior.mean).abs().mean().item():.3f}")
```

### 2. Imagination Too Long

**Problem**: Model errors compound over long horizons

**Symptoms**:
- Good world model metrics but poor policy performance
- Policy learns to exploit model errors
- Generated trajectories look unrealistic
- Performance degrades with longer horizons

**Solutions**:

```python
# Solution 1: Moderate horizon
imagination_horizon = 15  # Not 50 or 100

# Solution 2: Adaptive horizon
def get_horizon(step):
    # Start short, gradually increase
    max_horizon = 15
    warmup_steps = 10000
    return int(max_horizon * min(1.0, step / warmup_steps))

# Solution 3: Model uncertainty termination
uncertainty = model.uncertainty(h, z)
if uncertainty > threshold:
    break  # Stop imagination early

# Solution 4: Real data mixing
if t % 5 == 0:
    # Inject real observation every 5 steps
    h, z = posterior_from_real_obs(real_obs[t])
```

**Diagnostics**:
```python
# Compare imagined vs real trajectories
real_rewards = env_rollout(policy, 100)
imagined_rewards = imagine_rollout(policy, 100)
print(f"Real return: {real_rewards.sum():.2f}")
print(f"Imagined return: {imagined_rewards.sum():.2f}")
print(f"Gap: {(imagined_rewards.sum() - real_rewards.sum()):.2f}")
```

### 3. Observation Reconstruction

**Problem**: Perfect reconstruction not needed and can hurt

**Symptoms**:
- High reconstruction loss but good policy performance
- Model focuses on irrelevant visual details
- Slow training due to reconstruction overhead

**Solutions**:

```python
# Solution 1: Lower weight
recon_loss = 0.1 * reconstruction_error  # Default is 1.0

# Solution 2: Disable for complex images
if image_complexity > threshold:
    recon_loss = 0.0  # Skip reconstruction

# Solution 3: Perceptual loss instead of pixel loss
recon_features = pretrained_encoder(reconstructed)
target_features = pretrained_encoder(target)
recon_loss = F.mse_loss(recon_features, target_features)

# Solution 4: Lower resolution
target_downsampled = F.interpolate(target, scale_factor=0.5)
recon_downsampled = F.interpolate(reconstructed, scale_factor=0.5)
recon_loss = F.mse_loss(recon_downsampled, target_downsampled)
```

### 4. Slow Training

**Problem**: World model training is computational bottleneck

**Symptoms**:
- Low GPU utilization
- Most time spent in data loading
- Training slower than model-free methods

**Solutions**:

```python
# Solution 1: Parallelize environments
import gymnasium as gym
env = gym.vector.AsyncVectorEnv([make_env] * 16)

# Solution 2: Larger batch size (if memory allows)
batch_size = 32  # Instead of 16
sequence_length = 32  # Shorter sequences for larger batches

# Solution 3: Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = compute_loss()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Solution 4: Optimize data pipeline
replay_buffer = ReplayBuffer(
    pin_memory=True,
    num_workers=4,
    prefetch_factor=2
)

# Solution 5: Skip world model updates occasionally
if step % 2 == 0:
    train_world_model()  # Train every other step
train_policy()  # Always train policy
```

### 5. Reward Prediction Errors

**Problem**: Inaccurate reward prediction hurts policy

**Symptoms**:
- Large reward prediction error
- Policy performs poorly despite good state predictions
- Agent optimizes for wrong rewards

**Solutions**:

```python
# Solution 1: Higher weight on reward loss
total_loss = (
    dynamics_loss +
    0.1 * recon_loss +
    2.0 * reward_loss +  # Increased from 1.0
    continue_loss
)

# Solution 2: Separate reward model capacity
reward_model = MLP(
    state_dim,
    [512, 512, 512, 512],  # More layers
    output_dim=1
)

# Solution 3: Ensemble reward models
reward_preds = [model(h, z) for model in reward_ensemble]
reward = torch.stack(reward_preds).mean(0)  # Average predictions

# Solution 4: Reward normalization
reward_normalizer = RunningStats()
reward_normalized = reward_normalizer.normalize(rewards)
# Train on normalized, denormalize for policy
```

**Diagnostics**:
```python
# Monitor reward prediction accuracy
reward_pred = reward_model(h, z)
reward_true = batch.rewards
reward_error = F.mse_loss(reward_pred, symlog(reward_true))
print(f"Reward MSE: {reward_error.item():.4f}")
print(f"Reward correlation: {torch.corrcoef(torch.stack([reward_pred, symlog(reward_true)]))[0,1].item():.3f}")
```

### 6. Action Space Mismatch

**Problem**: Discrete vs continuous action handling

**Symptoms**:
- Policy outputs invalid actions
- Training unstable with continuous actions
- Discrete actions not explored enough

**Solutions**:

```python
# For discrete actions: Use Gumbel-Softmax during imagination
def sample_discrete_action(logits, temperature=1.0):
    if training:
        # Gumbel-Softmax for differentiability
        return F.gumbel_softmax(logits, tau=temperature, hard=False)
    else:
        # Argmax during evaluation
        return F.one_hot(logits.argmax(-1), num_classes=logits.shape[-1])

# For continuous actions: Bound the action space
def sample_continuous_action(mean, std):
    action = torch.normal(mean, std)
    action = torch.tanh(action)  # Bound to [-1, 1]
    return action

# For mixed action spaces: Separate heads
discrete_logits = discrete_head(state)
continuous_params = continuous_head(state)
```

### 7. Memory Issues

**Problem**: Large replay buffer and long sequences

**Symptoms**:
- Out of memory errors
- Slow sampling from replay buffer
- GPU memory overflow

**Solutions**:

```python
# Solution 1: Smaller replay buffer
buffer_size = 100_000  # Instead of 1M for prototyping

# Solution 2: Shorter sequences
sequence_length = 32  # Instead of 64

# Solution 3: Store compressed observations
class CompressedReplayBuffer:
    def add(self, obs, action, reward, done):
        # Store in compressed format
        obs_compressed = compress(obs)
        self.buffer.append(obs_compressed, action, reward, done)

    def sample(self):
        batch = self.buffer.sample()
        # Decompress on the fly
        obs = decompress(batch.obs)
        return obs, batch.action, batch.reward, batch.done

# Solution 4: Gradient checkpointing
from torch.utils.checkpoint import checkpoint

def world_model_forward(batch):
    return checkpoint(world_model, batch)
```

### 8. Exploration Issues

**Problem**: Policy doesn't explore enough early in training

**Symptoms**:
- Gets stuck in local optima
- Never discovers high-reward regions
- Entropy collapses quickly

**Solutions**:

```python
# Solution 1: Higher entropy bonus
entropy_coef = 1e-2  # Increased from 1e-3

# Solution 2: Action noise during collection
if step < exploration_steps:
    action = policy(obs).sample()
    action = action + torch.randn_like(action) * noise_scale

# Solution 3: Curiosity-driven exploration
intrinsic_reward = prediction_error(state, next_state)
total_reward = extrinsic_reward + beta * intrinsic_reward

# Solution 4: Epsilon-greedy for discrete actions
if random.random() < epsilon:
    action = random_action()
else:
    action = policy(obs).sample()
```

## Advanced Topics

### Multi-Task Learning

Train a single world model on multiple tasks:

```python
class MultiTaskDreamer(DreamerV3):
    def __init__(self, envs):
        super().__init__()
        self.envs = envs

        # Shared world model
        self.shared_encoder = Encoder()
        self.shared_rssm = RSSM()

        # Task-specific heads
        self.task_decoders = nn.ModuleDict({
            task: Decoder() for task in envs.keys()
        })
        self.task_reward_models = nn.ModuleDict({
            task: RewardModel() for task in envs.keys()
        })

        # Shared policy (conditioned on task)
        self.actor = TaskConditionedActor()

    def train_step(self, task):
        # Train world model on task-specific data
        batch = self.replay_buffers[task].sample()

        # Shared encoding and dynamics
        embed = self.shared_encoder(batch.obs)
        h, z = self.shared_rssm(embed, batch.actions)

        # Task-specific prediction
        obs_recon = self.task_decoders[task](h, z)
        reward_pred = self.task_reward_models[task](h, z)
```

### Hierarchical World Models

Use hierarchical latent structure:

```python
class HierarchicalRSSM(nn.Module):
    def __init__(self):
        super().__init__()

        # High-level (slow) dynamics
        self.high_level = RSSM(
            hidden_dim=2048,
            update_every=4  # Update every 4 timesteps
        )

        # Low-level (fast) dynamics
        self.low_level = RSSM(
            hidden_dim=1024,
            update_every=1  # Update every timestep
        )

    def forward(self, obs, action):
        # High-level state (abstract, slow-changing)
        if self.step % 4 == 0:
            h_high, z_high = self.high_level(obs, action)

        # Low-level state (detailed, fast-changing)
        # Conditioned on high-level state
        h_low, z_low = self.low_level(
            obs, action, condition=z_high
        )

        return (h_high, z_high), (h_low, z_low)
```

### Model-Based Planning

Use the world model for explicit planning:

```python
def plan_with_world_model(state, goal, horizon=20, num_samples=100):
    """Sample-based planning using the world model."""

    best_return = -float('inf')
    best_actions = None

    for _ in range(num_samples):
        # Sample random action sequence
        actions = sample_random_actions(horizon)

        # Simulate in world model
        h, z = state
        total_return = 0

        for t in range(horizon):
            # Predict next state
            h = rssm.recurrent(h, z, actions[t])
            z = rssm.prior(h)

            # Predict reward
            reward = reward_model(h, z)
            total_return += discount ** t * reward

        # Check if better than current best
        if total_return > best_return:
            best_return = total_return
            best_actions = actions

    return best_actions[0]  # Return first action
```

### Causal World Models

Incorporate causal structure:

```python
class CausalWorldModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Decompose state into causal factors
        self.factor_encoder = FactorEncoder(num_factors=8)

        # Causal graph (learned)
        self.causal_graph = nn.Parameter(
            torch.randn(num_factors, num_factors)
        )

        # Factor-wise dynamics
        self.factor_dynamics = nn.ModuleList([
            FactorDynamics() for _ in range(num_factors)
        ])

    def forward(self, obs, action):
        # Extract causal factors
        factors = self.factor_encoder(obs)

        # Predict next factors using causal graph
        next_factors = []
        for i, dynamics in enumerate(self.factor_dynamics):
            # Parents according to causal graph
            parents = factors * torch.sigmoid(self.causal_graph[i])
            next_factor = dynamics(parents, action)
            next_factors.append(next_factor)

        return torch.stack(next_factors)
```

## Cross-References

### Related World Models

- **[DreamerV1](/docs/15_world_models/dreamerv1.md)**: Original Dreamer algorithm
- **[DreamerV2](/docs/15_world_models/dreamerv2.md)**: Categorical latent variables
- **[PlaNet](/docs/15_world_models/planet.md)**: Gaussian latent dynamics
- **[MuZero](/docs/15_world_models/muzero.md)**: Model-based planning for games
- **[Genie](/docs/15_world_models/genie.md)**: Generative interactive environments

### Related RL Methods

- **[PPO](/docs/01_reinforcement_learning/policy_gradient/ppo.md)**: Model-free baseline
- **[SAC](/docs/01_reinforcement_learning/policy_gradient/sac.md)**: Off-policy actor-critic
- **[Rainbow](/docs/01_reinforcement_learning/value_based/rainbow.md)**: Value-based methods
- **[R2D2](/docs/01_reinforcement_learning/sequence_based/r2d2.md)**: Recurrent RL

### Related Techniques

- **[VAE](/docs/09_generative_models/vae.md)**: Variational autoencoders
- **[GRU](/docs/03_state_space_models/gru.md)**: Gated recurrent units
- **[TD-Lambda](/docs/01_reinforcement_learning/value_based/td_lambda.md)**: λ-returns
- **[Categorical DQN](/docs/01_reinforcement_learning/value_based/c51.md)**: Distributional RL

## References

```bibtex
@article{hafner2023mastering,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}

@article{hafner2020dream,
  title={Dream to Control: Learning Behaviors by Latent Imagination},
  author={Hafner, Danijar and Lillicrap, Timothy and Ba, Jimmy and Norouzi, Mohammad},
  journal={International Conference on Learning Representations},
  year={2020}
}

@article{hafner2021mastering,
  title={Mastering Atari with Discrete World Models},
  author={Hafner, Danijar and Lillicrap, Timothy and Norouzi, Mohammad and Ba, Jimmy},
  journal={International Conference on Learning Representations},
  year={2021}
}

@article{schrittwieser2020mastering,
  title={Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model},
  author={Schrittwieser, Julian and Antonoglou, Ioannis and Hubert, Thomas and Simonyan, Karen and Sifre, Laurent and Schmitt, Simon and Guez, Arthur and Lockhart, Edward and Hassabis, Demis and Graepel, Thore and others},
  journal={Nature},
  volume={588},
  number={7839},
  pages={604--609},
  year={2020}
}

@article{ha2018worldmodels,
  title={World Models},
  author={Ha, David and Schmidhuber, J{\"u}rgen},
  journal={arXiv preprint arXiv:1803.10122},
  year={2018}
}

@article{kaiser2019model,
  title={Model-Based Reinforcement Learning for Atari},
  author={Kaiser, Lukasz and Babaeizadeh, Mohammad and Milos, Piotr and Osinski, Blazej and Campbell, Roy H and Czechowski, Konrad and Erhan, Dumitru and Finn, Chelsea and Kozakowski, Piotr and Levine, Sergey and others},
  journal={International Conference on Learning Representations},
  year={2020}
}

@article{sekar2020planning,
  title={Planning to Explore via Self-Supervised World Models},
  author={Sekar, Ramanan and Rybkin, Oleh and Daniilidis, Kostas and Abbeel, Pieter and Hafner, Danijar and Pathak, Deepak},
  journal={International Conference on Machine Learning},
  year={2020}
}

@article{wu2022daydreamer,
  title={Daydreamer: World Models for Physical Robot Learning},
  author={Wu, Philipp and Escontrela, Alejandro and Hafner, Danijar and Goldberg, Ken and Abbeel, Pieter},
  journal={Conference on Robot Learning},
  year={2022}
}
```

**Official Resources**:
- **Code**: https://github.com/danijar/dreamerv3
- **Paper**: https://arxiv.org/abs/2301.04104
- **Project Page**: https://danijar.com/project/dreamerv3/
- **Blog Post**: https://danijar.com/dreamerv3/

**Community Resources**:
- **PyTorch Implementation**: https://github.com/NM512/dreamerv3-torch
- **JAX Implementation**: https://github.com/danijar/dreamerv3
- **Minimal Implementation**: https://github.com/jsikyoon/dreamer-torch

## Summary

DreamerV3 represents a milestone in model-based RL:

1. **Universal algorithm**: Works across diverse domains with fixed hyperparameters
2. **Sample efficient**: Achieves superhuman performance with limited data
3. **Pure imagination**: Trains policies entirely in learned world model
4. **Robust**: Symlog predictions and percentile normalization handle diverse scales
5. **Scalable**: Applicable to Atari, continuous control, and complex 3D worlds

**When to use DreamerV3**:
- Sample efficiency is critical (expensive interactions)
- Environment is complex and high-dimensional (images)
- You can afford training a world model (compute available)
- Planning/imagination can help (structured environments)
- Need general-purpose RL solution (avoid hyperparameter tuning)
- Long-horizon tasks (credit assignment benefits)

**When NOT to use DreamerV3**:
- Very simple environments (model-free may be simpler)
- Extremely fast real-time requirements (model-free is faster)
- Limited compute (world model training is expensive)
- Highly stochastic environments (model accuracy suffers)

**Key innovations**:
1. RSSM: Separates deterministic and stochastic dynamics
2. Symlog: Handles arbitrary reward scales
3. Imagination: Trains policy without environment interaction
4. Fixed hyperparameters: Works across all domains

**Key hyperparameters**:
- RSSM hidden: 4096
- RSSM stochastic: 32 × 32 = 1024
- Imagination horizon: 15
- Batch size: 16, Sequence length: 64
- Free bits: 1.0
- Learning rates: world=1e-4, actor/critic=3e-5

**Performance summary**:
- Atari 100k: 1.83 (superhuman)
- DMC: 971 / 1000 (near-optimal)
- Minecraft: 31% diamond collection (SOTA)
- Works across 150+ tasks without tuning!

DreamerV3 is the state-of-the-art general-purpose model-based RL algorithm, combining sample efficiency, generality, and strong performance across diverse domains.
