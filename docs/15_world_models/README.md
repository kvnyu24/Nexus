# World Models

This directory contains comprehensive documentation for world models implemented in Nexus. World models learn compact representations of environments and their dynamics, enabling agents to plan, predict, and act in complex environments through learned imagination.

## Table of Contents

1. [DreamerV3](#dreamerv3)
2. [I-JEPA (Image World Model)](#i-jepa)
3. [V-JEPA 2 (Video World Model)](#v-jepa-2)
4. [Genie (Generative Interactive Environments)](#genie)

## Overview

World models are learned representations of how environments evolve over time. They enable:

- **Planning**: Simulate future trajectories before acting
- **Exploration**: Imagine novel states and strategies
- **Transfer**: Reuse learned dynamics across tasks
- **Sample Efficiency**: Train policies in imagination
- **Understanding**: Learn interpretable environment structure

### What is a World Model?

A world model learns:
```
s_t+1 = f(s_t, a_t)  # Dynamics: next state given current state and action
```

In practice, world models learn in latent space:
```
z_t+1 ~ p(z_t+1 | z_t, a_t)  # Latent dynamics
x̂_t = decode(z_t)            # Observation reconstruction
r̂_t = reward(z_t, a_t)       # Reward prediction
```

## Why World Models?

### Advantages

1. **Sample Efficiency**: Learn from imagined rollouts, not just real experience
2. **Long-term Planning**: Simulate far into the future
3. **Zero-shot Transfer**: Adapt to new tasks without new data
4. **Interpretability**: Visualize learned dynamics
5. **Safety**: Test policies in simulation before deployment

### Applications

- **Robotics**: Simulate robot behavior before execution
- **Game AI**: Plan strategies in complex games
- **Autonomous Vehicles**: Predict traffic dynamics
- **Video Understanding**: Model temporal dynamics
- **Interactive Environments**: Generate playable worlds

## Algorithm Landscape

### Model-Based RL (with Actions)

#### DreamerV3
- **File**: [dreamerv3.md](dreamerv3.md)
- **Difficulty**: Advanced
- **Key Concepts**: Recurrent world model, actor-critic in imagination
- **Training**: End-to-end with policy learning
- **Use Case**: Sample-efficient RL, continuous control

### Self-Supervised World Models (Action-Free)

#### I-JEPA (Image)
- **File**: [ijepa.md](ijepa.md)
- **Difficulty**: Intermediate
- **Key Concepts**: Masked prediction in representation space
- **Training**: Self-supervised on images
- **Use Case**: Learning visual representations, scene understanding

#### V-JEPA 2 (Video)
- **File**: [vjepa2.md](vjepa2.md)
- **Difficulty**: Intermediate-Advanced
- **Key Concepts**: Spatiotemporal dynamics, future prediction
- **Training**: Self-supervised on videos
- **Use Case**: Video understanding, robotics, dynamics learning

### Generative World Models

#### Genie
- **File**: [genie.md](genie.md)
- **Difficulty**: Advanced
- **Key Concepts**: Action-free training, latent actions, video generation
- **Training**: Self-supervised on internet videos
- **Use Case**: Interactive world generation, game environments

## Comparison Table

| Method | Actions Required | Output Space | Training Data | Primary Use Case |
|--------|-----------------|--------------|---------------|------------------|
| DreamerV3 | ✅ Explicit | Latent + Pixels | RL Episodes | RL with planning |
| I-JEPA | ❌ | Representations | Images | Visual understanding |
| V-JEPA 2 | ❌ | Representations | Videos | Temporal dynamics |
| Genie | ✅ Latent | Pixels/Video | Videos | World generation |

## Core Concepts

### Latent Dynamics Models

Instead of modeling pixel-level dynamics (computationally expensive), world models operate in latent space:

```python
# Encode observations to latent
z_t = encoder(o_t)

# Predict latent dynamics
z_t+1 = dynamics_model(z_t, a_t)

# Decode back to observations
o_t+1 = decoder(z_t+1)
```

**Benefits**:
- Lower dimensional
- Faster to compute
- More semantic
- Better generalization

### Recurrent vs Non-Recurrent

**Recurrent Models (DreamerV3)**:
```python
h_t = recurrent_model(h_t-1, z_t, a_t)
z_t+1 ~ predictor(h_t)
```
- Maintain hidden state
- Better for partially observable environments
- Model long-term dependencies

**Non-Recurrent Models (I-JEPA, V-JEPA)**:
```python
z_target = predict(z_context)
```
- Simpler architecture
- Parallel processing
- Better for fully observable states

### Stochastic vs Deterministic

**Stochastic Models (DreamerV3, Genie)**:
```python
z_t+1 ~ N(μ(h_t), σ(h_t))  # Sample from distribution
```
- Model uncertainty
- More robust to randomness
- Enable diverse predictions

**Deterministic Models (I-JEPA, V-JEPA)**:
```python
z_t+1 = f(z_t)  # Deterministic prediction
```
- Simpler training
- Faster inference
- Good for deterministic environments

### Action Representation

**Explicit Actions (DreamerV3)**:
- Actions are provided by the environment
- Model learns: p(s_{t+1} | s_t, a_t)

**Latent Actions (Genie)**:
- Actions inferred from video transitions
- Model learns: p(s_{t+1} | s_t, a_latent)
- Enables training on action-free video data

**Action-Free (I-JEPA, V-JEPA)**:
- No actions at all
- Model learns: p(s_{t+1} | s_t)
- Focuses on natural dynamics

## Training Paradigms

### 1. Model-Based RL

Train world model jointly with policy:

```python
# Phase 1: Collect real experience
real_transitions = env.step(policy)

# Phase 2: Train world model
world_model.fit(real_transitions)

# Phase 3: Train policy in imagination
for _ in range(imagination_steps):
    imagined_transitions = world_model.imagine(policy)
    policy.update(imagined_transitions)
```

**Example**: DreamerV3

### 2. Self-Supervised Pre-training

Pre-train world model, then use for downstream tasks:

```python
# Phase 1: Pre-train on passive data (videos)
world_model.pretrain(video_dataset)

# Phase 2: Fine-tune for downstream task
features = world_model.encode(task_data)
task_model.train(features, labels)
```

**Example**: I-JEPA, V-JEPA 2

### 3. Generative Pre-training

Learn world model from internet-scale data:

```python
# Train on diverse videos
world_model.train(internet_videos)

# Generate interactive environments
env = world_model.generate(initial_frame, actions)
```

**Example**: Genie

## Implementation Patterns

### Basic World Model Structure

```python
class WorldModel(nn.Module):
    def __init__(self):
        self.encoder = Encoder()
        self.dynamics = DynamicsModel()
        self.decoder = Decoder()
        self.reward = RewardPredictor()
        
    def encode(self, observations):
        return self.encoder(observations)
    
    def imagine(self, states, actions):
        next_states = self.dynamics(states, actions)
        observations = self.decoder(next_states)
        rewards = self.reward(next_states, actions)
        return next_states, observations, rewards
    
    def train_step(self, obs, actions, next_obs, rewards):
        # Encode
        z = self.encode(obs)
        z_next = self.encode(next_obs)
        
        # Predict dynamics
        z_next_pred = self.dynamics(z, actions)
        
        # Reconstruction
        obs_recon = self.decoder(z)
        
        # Reward prediction
        reward_pred = self.reward(z, actions)
        
        # Losses
        dynamics_loss = F.mse_loss(z_next_pred, z_next)
        recon_loss = F.mse_loss(obs_recon, obs)
        reward_loss = F.mse_loss(reward_pred, rewards)
        
        return dynamics_loss + recon_loss + reward_loss
```

### Using World Models for Planning

```python
def plan_with_world_model(world_model, current_state, horizon=10):
    """
    Plan future actions using world model.
    """
    best_actions = None
    best_reward = -float('inf')
    
    # Random shooting
    for _ in range(num_candidates):
        actions = sample_random_actions(horizon)
        
        # Simulate trajectory in world model
        state = current_state
        total_reward = 0
        for a in actions:
            state, _, reward = world_model.imagine(state, a)
            total_reward += reward
        
        if total_reward > best_reward:
            best_reward = total_reward
            best_actions = actions
    
    return best_actions[0]  # Execute first action
```

## When to Use Each Method

### Use DreamerV3 when:
- You have a RL problem with explicit actions
- Sample efficiency is critical
- Environment is complex (high-dimensional)
- You can afford recurrent models
- Planning improves performance

### Use I-JEPA when:
- You have unlabeled images
- No actions available
- You need visual representations
- Computational efficiency matters
- Downstream tasks are image-based

### Use V-JEPA 2 when:
- You have videos (action-free)
- You need spatiotemporal understanding
- Robotics or video understanding is the goal
- Training on large-scale video data
- Zero-shot transfer to control

### Use Genie when:
- You want to generate interactive environments
- Training data has no action labels
- Game generation is the application
- You need playable simulations
- Internet-scale video data available

## Performance Comparison

### Sample Efficiency (RL Tasks)

| Method | Atari (100k steps) | DMC (500k steps) |
|--------|-------------------|------------------|
| Model-Free (SAC) | 0.5x human | 0.7x human |
| Model-Free (PPO) | 0.3x human | 0.5x human |
| **DreamerV3** | **1.2x human** | **1.5x human** |

DreamerV3 achieves superhuman performance with less data!

### Representation Quality

| Method | ImageNet Linear Probe | Video Classification |
|--------|----------------------|---------------------|
| Supervised | 78.3% | 82.1% |
| I-JEPA | 80.3% | - |
| V-JEPA 2 | - | 92.7% |

Self-supervised world models learn better representations than supervised learning!

## Common Pitfalls

### 1. Model Bias / Compounding Errors

**Problem**: Errors accumulate when imagining far into the future

**Symptoms**:
- Good 1-step predictions, terrible long-term
- Policy overfits to model errors
- Imagined trajectories diverge from reality

**Solutions**:
```python
# Use shorter imagination horizons
imagination_horizon = 10  # Not 100

# Mix real and imagined data
real_data = sample_real_data(batch_size // 2)
imagined_data = model.imagine(batch_size // 2)
train_data = concat(real_data, imagined_data)

# Regularize model (uncertainty estimation)
z_next ~ N(μ, σ)  # Stochastic model
```

### 2. Observation vs State

**Problem**: Confusing observations (images) with states (full info)

**Symptoms**:
- Model fails in partially observable environments
- Temporal dependencies ignored

**Solutions**:
```python
# Use recurrent models for partial observability
h_t = RNN(h_{t-1}, z_t, a_t)  # Maintain belief state

# Or use frame stacking
obs = concat([frame_t, frame_{t-1}, frame_{t-2}, frame_{t-3}])
```

### 3. Reward Prediction Errors

**Problem**: Inaccurate reward model leads to bad policies

**Symptoms**:
- Policy finds fake rewards in imagination
- Real-world performance poor despite good imagined rewards

**Solutions**:
```python
# Use conservative reward estimates
reward = min(reward_pred, reward_pred_conservative)

# Separate reward model training
reward_model.train(real_transitions_only)

# Prioritize reward accuracy in loss
loss = dynamics_loss + 10.0 * reward_loss  # Higher weight
```

### 4. Catastrophic Forgetting

**Problem**: Model forgets old data when learning new data

**Symptoms**:
- Performance degrades over time
- Model fails on previously mastered tasks

**Solutions**:
```python
# Replay buffer for world model
model_buffer.add(new_transitions)
train_batch = model_buffer.sample(batch_size)

# Regularization
loss += kl_divergence(new_params, old_params)
```

### 5. Scalability Issues

**Problem**: World models are computationally expensive

**Symptoms**:
- Training too slow
- Imagination slower than real interaction
- Memory issues

**Solutions**:
```python
# Latent imagination (not pixel-level)
z_next = dynamics(z, a)  # Fast

# Parallel imagination
imagined_rollouts = model.imagine_parallel(batch_of_states)

# Mixed precision
with torch.cuda.amp.autocast():
    prediction = model.forward(state, action)
```

## Key Papers

### Foundational

1. **World Models**: [World Models](https://arxiv.org/abs/1803.10122) (Ha & Schmidhuber, 2018)
   - Introduced world models for RL

2. **PlaNet**: [Learning Latent Dynamics for Planning](https://arxiv.org/abs/1811.04551) (Hafner et al., 2019)
   - Recurrent latent dynamics models

3. **Dreamer**: [Dream to Control](https://arxiv.org/abs/1912.01603) (Hafner et al., 2020)
   - Policy learning in latent imagination

### Covered Methods

4. **DreamerV3**: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) (Hafner et al., 2023)
5. **I-JEPA**: [Self-Supervised Learning from Images](https://arxiv.org/abs/2301.08243) (Assran et al., 2023)
6. **V-JEPA**: [Revisiting Feature Prediction for Video](https://arxiv.org/abs/2404.08471) (Bardes et al., 2024)
7. **Genie**: [Generative Interactive Environments](https://arxiv.org/abs/2402.15391) (Bruce et al., 2024)

## Additional Resources

### Tutorials
- [World Models Tutorial (Yannic Kilcher)](https://www.youtube.com/watch?v=...)
- [Model-Based RL Course (Sergey Levine)](https://rail.eecs.berkeley.edu/deeprlcourse/)

### Implementations
- [DreamerV3 Official](https://github.com/danijar/dreamerv3)
- [I-JEPA Official](https://github.com/facebookresearch/ijepa)
- [V-JEPA Official](https://github.com/facebookresearch/jepa)

### Benchmarks
- [DMC (DeepMind Control)](https://github.com/deepmind/dm_control)
- [Atari 100k](https://github.com/google-research/batch_rl_icml2020)

## File Structure

```
15_world_models/
├── README.md              # This file
├── dreamerv3.md          # DreamerV3
├── ijepa.md              # I-JEPA (see also ssl/ijepa.md)
├── vjepa2.md             # V-JEPA 2 (see also ssl/vjepa2.md)
└── genie.md              # Genie
```

Note: I-JEPA and V-JEPA 2 have detailed documentation in `docs/12_self_supervised_learning/` as they are primarily SSL methods that also serve as world models.

## Getting Started

### Recommended Learning Path

1. **Start with I-JEPA** (simplest world model)
   - Understand representation prediction
   - Learn about EMA target encoders
   - See world modeling without actions

2. **Explore V-JEPA 2** (temporal dynamics)
   - Add time dimension
   - Understand spatiotemporal prediction
   - See video-based world models

3. **Study DreamerV3** (full RL world model)
   - Integrate actions and rewards
   - Learn recurrent latent dynamics
   - See policy learning in imagination

4. **Advanced: Genie** (generative world model)
   - Understand latent actions
   - Learn world generation
   - See internet-scale pre-training

### Quick Start: Simple World Model

```python
import torch
from nexus.models.world_models import SimpleWorldModel

# Define world model
config = {
    "obs_dim": 64*64*3,
    "action_dim": 6,
    "latent_dim": 256,
    "hidden_dim": 512
}

world_model = SimpleWorldModel(config)

# Collect data
obs, actions, next_obs, rewards = collect_data(env)

# Train world model
for epoch in range(num_epochs):
    loss = world_model.train_step(obs, actions, next_obs, rewards)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Use for planning
current_obs = env.reset()
current_state = world_model.encode(current_obs)
planned_action = plan_with_world_model(world_model, current_state)
next_obs, reward, done = env.step(planned_action)
```

## Summary

World models are a powerful paradigm for learning environment dynamics. They enable:

1. **Sample-efficient RL**: Learn policies in imagination (DreamerV3)
2. **Self-supervised learning**: Learn representations from videos (I-JEPA, V-JEPA)
3. **World generation**: Create interactive environments (Genie)

**Key Takeaways**:
- Operate in latent space for efficiency
- Use recurrence for partial observability
- Balance model accuracy with policy performance
- Mix real and imagined data to avoid overfitting

**Next Steps**:
- Read method-specific documentation
- Implement a simple world model
- Try DreamerV3 on a control task
- Explore V-JEPA for robotics applications

Happy world modeling!
