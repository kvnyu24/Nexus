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

## Theoretical Background

### World Model Architecture: RSSM

DreamerV3 uses a Recurrent State-Space Model that separates:
- **Deterministic state** h_t: Recurrent hidden state
- **Stochastic state** z_t: Sampled latent variables

```
# Dynamics model
h_t = f_det(h_t-1, z_t-1, a_t-1)  # Deterministic recurrence
z_t ~ p(z_t | h_t)                 # Stochastic prediction

# Observation model
o_t ~ p(o_t | h_t, z_t)            # Decode to observations

# Reward model
r_t ~ p(r_t | h_t, z_t)            # Predict rewards

# Continue model (termination)
c_t ~ p(c_t | h_t, z_t)            # Predict episode continuation
```

### Learning in Imagination

Once the world model is trained, the policy is trained entirely by imagining trajectories:

```
1. Sample initial state from replay buffer: (h_0, z_0)
2. Imagine trajectory:
   for t in range(imagination_horizon):
       a_t = π(h_t, z_t)              # Actor
       h_t+1 = f_det(h_t, z_t, a_t)
       z_t+1 ~ p(z_t+1 | h_t+1)
       r_t ~ p(r_t | h_t, z_t)
       
3. Compute returns (λ-returns)
4. Update actor to maximize returns
5. Update critic to predict returns
```

## Mathematical Formulation

### World Model Loss

```
L_world = L_dynamics + L_observation + L_reward + L_continue
```

**Dynamics Loss** (KL between predicted and actual posterior):
```
L_dynamics = KL(q(z_t | h_t, o_t) || p(z_t | h_t))
```

**Observation Loss** (reconstruction):
```
L_observation = -log p(o_t | h_t, z_t)
```

**Reward Loss**:
```
L_reward = -log p(r_t | h_t, z_t)
```

**Continue Loss** (predicts episode termination):
```
L_continue = -log p(c_t | h_t, z_t)
```

### Actor-Critic Loss

**Critic Loss** (predict value):
```
L_critic = 0.5 * (V(h_t, z_t) - λ_return_t)²
```

Where λ-return is:
```
λ_return_t = r_t + γ·c_t·(λ·λ_return_t+1 + (1-λ)·V(h_t+1, z_t+1))
```

**Actor Loss** (maximize value):
```
L_actor = -V(h_t, z_t)  # Reinforce
```

Or with entropy regularization:
```
L_actor = -V(h_t, z_t) - β·H(π(·|h_t, z_t))
```

### Symlog Transformation

DreamerV3 uses symlog to handle diverse reward scales:

```
symlog(x) = sign(x) · log(|x| + 1)
symexp(x) = sign(x) · (exp(|x|) - 1)
```

This allows the same hyperparameters to work on rewards from -1000 to +1000.

## High-Level Intuition

Think of DreamerV3 as a human learning to play a video game:

1. **World Model Learning** (Understanding the Game):
   - Watch gameplay (collect data)
   - Build mental model of game physics and rules
   - Predict what happens when you press buttons

2. **Policy Learning in Imagination** (Mental Practice):
   - Imagine playing the game in your head
   - Try different strategies mentally
   - Learn which actions lead to high scores
   - Never touch the real game during this phase

3. **Execution** (Playing):
   - Use learned policy in real game
   - Collect more data for improving world model
   - Repeat the cycle

**Key Insight**: Most learning happens in imagination (fast, safe, scalable), with minimal real interaction.

## Implementation Details

### Network Architecture

**Encoder**:
- CNN for images: Conv(32, 4, 2) → Conv(64, 4, 2) → Conv(128, 4, 2) → Conv(256, 4, 2)
- Output: 1024-dim vector

**RSSM Core**:
- Deterministic state: GRU with 4096 hidden units
- Stochastic state: 32 categorical variables × 32 classes = 1024-dim one-hot

**Decoder**:
- Transposed CNN: matching encoder architecture
- Outputs: mean and variance for Gaussian observation distribution

**Reward Predictor**:
- MLP: 512 → 512 → 512 → 1 (symlog space)

**Value Network** (Critic):
- MLP: 512 → 512 → 512 → 1 (symlog space)

**Policy Network** (Actor):
- MLP: 512 → 512 → 512 → action_dim
- Outputs: mean and std for continuous actions, or logits for discrete

### Training Procedure

```python
# Pseudo-code for DreamerV3

# Phase 1: Collect experience
for step in range(env_steps):
    action = policy(observation)
    next_obs, reward, done = env.step(action)
    replay_buffer.add(obs, action, reward, next_obs, done)
    
    if done:
        observation = env.reset()

# Phase 2: Train world model
for _ in range(world_model_updates):
    batch = replay_buffer.sample(batch_size, sequence_length)
    
    # Encode observations
    embeddings = encoder(batch.observations)
    
    # Dynamics: predict next stochastic state
    h, z = rssm.initial_state(batch_size)
    dynamics_loss = 0
    for t in range(sequence_length):
        # Posterior (using observation)
        z_posterior = rssm.posterior(h, embeddings[t])
        # Prior (prediction from previous state)
        z_prior = rssm.prior(h)
        
        # KL loss
        dynamics_loss += kl_divergence(z_posterior, z_prior)
        
        # Update recurrent state
        h = rssm.recurrent(h, z_posterior, batch.actions[t])
    
    # Reconstruction loss
    obs_recon = decoder(h, z_posterior)
    recon_loss = -log_prob(obs_recon, batch.observations)
    
    # Reward prediction loss
    reward_pred = reward_model(h, z_posterior)
    reward_loss = -log_prob(reward_pred, symlog(batch.rewards))
    
    # Total world model loss
    world_loss = dynamics_loss + recon_loss + reward_loss
    world_loss.backward()

# Phase 3: Train policy in imagination
for _ in range(policy_updates):
    # Sample starting states from replay buffer
    initial_states = replay_buffer.sample_states(batch_size)
    
    # Imagine trajectories
    states, actions, rewards = [], [], []
    h, z = initial_states
    for t in range(imagination_horizon):
        # Sample action from policy
        action = policy(h, z)
        
        # Imagine next state
        h_next = rssm.recurrent(h, z, action)
        z_next = rssm.prior(h_next)
        
        # Predict reward
        reward = reward_model(h, z)
        
        states.append((h, z))
        actions.append(action)
        rewards.append(reward)
        
        h, z = h_next, z_next
    
    # Compute λ-returns
    values = critic(states)
    returns = compute_lambda_returns(rewards, values)
    
    # Update critic
    critic_loss = (values - returns.detach()).pow(2).mean()
    critic_loss.backward()
    
    # Update actor
    actor_loss = -(returns.detach() * log_prob(actions)).mean()
    actor_loss.backward()
```

### Code Reference

Note: DreamerV3 is not yet implemented in Nexus, but would follow this structure:

```python
# Conceptual API
from nexus.models.world_models import DreamerV3

config = {
    "rssm_hidden": 4096,
    "rssm_categorical": 32,
    "rssm_classes": 32,
    "imagination_horizon": 15,
    "batch_size": 16,
    "sequence_length": 64,
}

agent = DreamerV3(config, env)

# Training loop
for step in range(total_steps):
    # Collect experience
    agent.collect_data(env)
    
    # Train world model and policy
    metrics = agent.train_step()
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
reward_pred_symlog = reward_model(state)
reward_pred = symexp(reward_pred_symlog)
```

### 2. Percentile Normalization

Normalize values by their percentiles across the batch:

```python
def percentile_normalize(x, percentile_low=5, percentile_high=95):
    low = torch.quantile(x, percentile_low / 100)
    high = torch.quantile(x, percentile_high / 100)
    x_norm = (x - low) / (high - low + 1e-8)
    return torch.clamp(x_norm, 0, 1)
```

### 3. Free Bits for KL Loss

Prevent KL collapse with free bits:

```python
kl_loss = kl_divergence(posterior, prior)
kl_loss = torch.maximum(kl_loss, free_bits)  # free_bits = 1.0
```

### 4. Return Normalization

Normalize returns using exponential moving statistics:

```python
# Track moving statistics
return_mean = 0.99 * return_mean + 0.01 * returns.mean()
return_std = 0.99 * return_std + 0.01 * returns.std()

# Normalize
returns_norm = (returns - return_mean) / (return_std + 1e-8)
```

### 5. Gradient Clipping

Clip gradients by global norm:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
```

## Experiments & Results

### Atari 100k Benchmark

| Method | Median Human-Normalized Score |
|--------|-------------------------------|
| PPO | 0.3 |
| Rainbow | 0.5 |
| MuZero | 1.5 |
| **DreamerV3** | **1.8** |

Superhuman performance with 100k steps!

### DeepMind Control Suite

| Method | Median Score |
|--------|--------------|
| SAC | 823 |
| TD3 | 857 |
| Dreamer | 905 |
| **DreamerV3** | **971** |

State-of-the-art on continuous control.

### Minecraft (Diamond Collection)

| Method | Success Rate |
|--------|--------------|
| MineRL | 1% |
| VPT | 15% |
| **DreamerV3** | **31%** |

First to solve diamond collection from pixels!

### Generality: Fixed Hyperparameters

DreamerV3 uses the same hyperparameters across:
- 7 Atari games
- 20 DMC tasks
- 5 Minecraft tasks
- Reward scales from -1000 to +1000

No tuning needed!

## Common Pitfalls

### 1. KL Balancing

**Problem**: Posterior and prior diverge or collapse
**Solution**: Use free bits + gradient scaling

```python
kl_loss = torch.maximum(kl, 1.0)  # Free bits
kl_loss = 0.5 * kl_loss  # Scale down
```

### 2. Imagination Too Long

**Problem**: Model errors compound over long horizons
**Solution**: Use moderate horizon (15 steps)

```python
imagination_horizon = 15  # Not 50 or 100
```

### 3. Observation Reconstruction

**Problem**: Perfect reconstruction not needed
**Solution**: Lower weight or even disable

```python
recon_loss = 0.1 * reconstruction_error  # Lower weight
```

### 4. Slow Training

**Problem**: World model training is slow
**Solution**: Parallelize environment interaction

```python
# Use vectorized environments
env = gym.vector.AsyncVectorEnv([make_env] * 16)
```

### 5. Reward Prediction Errors

**Problem**: Inaccurate reward prediction hurts policy
**Solution**: Higher weight on reward loss

```python
total_loss = dynamics_loss + recon_loss + 2.0 * reward_loss
```

## References

```bibtex
@article{hafner2023mastering,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}
```

**Official Code**: https://github.com/danijar/dreamerv3
**Paper**: https://arxiv.org/abs/2301.04104

## Summary

DreamerV3 represents a milestone in model-based RL:

1. **Universal algorithm**: Works across diverse domains with fixed hyperparameters
2. **Sample efficient**: Achieves superhuman performance with limited data
3. **Pure imagination**: Trains policies entirely in learned world model
4. **Robust**: Symlog predictions and percentile normalization handle diverse scales

**When to use DreamerV3**:
- Sample efficiency is critical
- Environment is complex and high-dimensional
- You can afford training a world model
- Planning/imagination can help
- Need general-purpose RL solution

**Key hyperparameters**:
- RSSM hidden: 4096
- Imagination horizon: 15
- Batch size: 16, Sequence length: 64
- Free bits: 1.0
- Works across all tested domains!
