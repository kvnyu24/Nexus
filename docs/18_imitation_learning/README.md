# Imitation Learning Methods

This directory contains comprehensive documentation for imitation learning algorithms implemented in Nexus. Imitation learning enables agents to learn behaviors by observing expert demonstrations, bypassing the need for explicit reward engineering.

## Table of Contents

1. [GAIL (Generative Adversarial Imitation Learning)](#gail)
2. [DAgger (Dataset Aggregation)](#dagger)
3. [MEGA-DAgger](#mega-dagger)
4. [AIRL (Adversarial Inverse Reinforcement Learning)](#airl)

## Overview

Imitation learning addresses a fundamental question: **How can an agent learn to perform a task by watching an expert, without knowing the underlying reward function?**

This is crucial because:
- Reward engineering is often difficult and time-consuming
- Expert demonstrations are frequently available (human demonstrations, recorded trajectories)
- Many tasks are easier to demonstrate than to specify formally

### Core Paradigms

**Behavioral Cloning (BC)**: Directly supervised learning from expert state-action pairs
- Simple but suffers from distributional shift
- No exploration of states not visited by expert

**Inverse Reinforcement Learning (IRL)**: Learn the reward function that explains expert behavior
- Recovers underlying objectives
- Computationally expensive (requires solving RL in inner loop)

**Adversarial Imitation**: Use discriminators to distinguish expert from policy behavior
- Avoids explicit reward learning
- More sample efficient than IRL
- Combines benefits of BC and IRL

## Algorithm Progression

We recommend studying these algorithms in the following order:

### 1. DAgger (Start Here)
- **File**: [dagger.md](dagger.md)
- **Difficulty**: Beginner-Intermediate
- **Key Concepts**: Interactive learning, covariate shift, expert queries
- **Use Case**: Learning from imperfect demonstrations with expert feedback

### 2. GAIL
- **File**: [gail.md](gail.md)
- **Difficulty**: Intermediate
- **Key Concepts**: Adversarial training, discriminator rewards, GAN stability
- **Use Case**: Learning complex behaviors from expert demonstrations

### 3. AIRL
- **File**: [airl.md](airl.md)
- **Difficulty**: Advanced
- **Key Concepts**: Reward function recovery, disentangling dynamics, transfer learning
- **Use Case**: When you need interpretable rewards or transfer to new environments

### 4. MEGA-DAgger (Advanced)
- **File**: [mega_dagger.md](mega_dagger.md)
- **Difficulty**: Advanced
- **Key Concepts**: Model-based learning, world models, safety-aware exploration
- **Use Case**: Safety-critical domains with limited expert interaction

## Comparison Table

| Algorithm | Paradigm | Expert Queries | Reward Recovery | Sample Efficiency | Complexity |
|-----------|----------|----------------|-----------------|-------------------|------------|
| DAgger | Interactive BC | Required | ❌ | High (with expert) | Low |
| GAIL | Adversarial | Not Required | ❌ | Medium | Medium |
| AIRL | Adversarial IRL | Not Required | ✅ | Medium | High |
| MEGA-DAgger | Model-Based | Minimal | ❌ | Very High | Very High |

## Detailed Comparisons

### Sample Efficiency

**Most to Least Efficient:**
1. **MEGA-DAgger**: Uses learned world model for planning, minimizes expert queries
2. **DAgger**: Direct expert queries reduce compound errors
3. **GAIL/AIRL**: Require many environment interactions to train policy and discriminator

### Computational Cost

**Least to Most Expensive:**
1. **DAgger**: Simple supervised learning with occasional expert queries
2. **GAIL**: Policy optimization + discriminator training
3. **AIRL**: Additional reward function learning and disentanglement
4. **MEGA-DAgger**: World model learning + planning + expert queries

### When Expert Access is Limited

**Best to Worst:**
1. **GAIL**: Works with fixed dataset of demonstrations
2. **AIRL**: Also works with fixed dataset
3. **MEGA-DAgger**: Uses world model to minimize queries
4. **DAgger**: Requires frequent expert access during training

## When to Use Each Algorithm

### Use DAgger when:
- You have access to an expert that can provide labels during training
- Covariate shift is a major concern
- You want a simple, interpretable approach
- Computational resources are limited
- Real-time expert feedback is available

**Typical Applications:**
- Autonomous driving with human supervisor
- Robot manipulation with human corrections
- Game playing with expert annotations

### Use GAIL when:
- You have a fixed dataset of expert demonstrations
- Expert access during training is not available
- You want to learn complex, multi-modal behaviors
- Sample efficiency during training is not critical
- You don't need an interpretable reward function

**Typical Applications:**
- Learning from human gameplay recordings
- Robotics with demonstration datasets
- Character animation from motion capture

### Use AIRL when:
- You need to recover interpretable reward functions
- Transfer to new environment dynamics is required
- You want to understand the expert's objectives
- Computational cost is acceptable
- Domain knowledge can inform reward structure

**Typical Applications:**
- Learning human preferences for alignment
- Transfer learning across robot morphologies
- Understanding expert decision-making
- Multi-task learning with shared rewards

### Use MEGA-DAgger when:
- Expert access is expensive or dangerous
- Safety is critical (avoid bad states)
- Sample efficiency is paramount
- You can learn accurate world models
- Planning in model space is feasible

**Typical Applications:**
- High-stakes medical procedures
- Autonomous vehicles (minimize dangerous situations)
- Expensive robotic systems
- Space exploration

## Core Concepts

### Distributional Shift

The fundamental challenge in imitation learning: the learner's state distribution differs from the expert's.

```
Expert: s₀ → s₁ → s₂ → s₃ (expert states)
Learner: s₀ → s₁' → s₂'' → s₃''' (different states due to errors)
```

**Solutions:**
- **DAgger**: Query expert on learner's states
- **GAIL/AIRL**: Train policy to match expert distribution
- **MEGA-DAgger**: Use world model to avoid bad states

### Adversarial Training

GAIL and AIRL use a discriminator D(s,a) that classifies state-action pairs:
- D(s,a) = 1 for expert demonstrations
- D(s,a) = 0 for policy rollouts

The discriminator's output provides a reward signal:
```
r(s,a) = -log(1 - D(s,a))  # GAIL
r(s,a) = log(D(s,a)) - log(1 - D(s,a))  # AIRL
```

### Model-Based Acceleration

MEGA-DAgger learns a world model M(s,a) → s' to:
- Simulate trajectories without environment interaction
- Plan using learned dynamics
- Identify states where expert input is needed
- Train policy in imagination

## Best Practices

### Data Collection

1. **Expert Quality**: Ensure demonstrations are truly expert-level
   - Suboptimal demonstrations hurt all methods
   - GAIL/AIRL particularly sensitive to noisy data

2. **Diversity**: Collect demonstrations from diverse scenarios
   - Cover edge cases and rare events
   - Multiple experts can improve robustness

3. **Labeling**: For DAgger, expert must label policy-visited states
   - Make labeling interface efficient
   - Consider active learning for query selection

### Training Stability

1. **GAIL/AIRL**: Use techniques from GAN training
   - Gradient penalty for discriminator
   - Spectral normalization
   - Batch normalization
   - Careful learning rate tuning

2. **DAgger**: Balance dataset mixing
   - β-decay schedule for expert data weighting
   - Don't discard early expert demonstrations

3. **MEGA-DAgger**: World model accuracy is critical
   - Use uncertainty estimates
   - Fall back to expert when model is uncertain
   - Iteratively improve model with real data

### Evaluation

1. **Performance Metrics**:
   - Task success rate
   - Distance to expert trajectory
   - Environment-specific rewards

2. **Distributional Metrics**:
   - State visitation frequency
   - Action distribution similarity
   - Trajectory diversity

3. **Sample Efficiency**:
   - Number of expert demonstrations needed
   - Number of environment interactions
   - Number of expert queries (DAgger, MEGA-DAgger)

## Implementation Overview

All implementations in Nexus follow a consistent API:

```python
from nexus.models.imitation import GAILAgent, DAggerAgent, AIRLAgent, MEGADAggerAgent

# GAIL Example
config = {
    "state_dim": 17,
    "action_dim": 6,
    "hidden_dims": [256, 256],
    "policy_lr": 3e-4,
    "discriminator_lr": 3e-4,
    "use_spectral_norm": True
}

agent = GAILAgent(config)

# Training loop
for epoch in range(num_epochs):
    # Collect policy rollouts
    policy_batch = collect_rollouts(agent, env)

    # Sample expert demonstrations
    expert_batch = expert_buffer.sample(batch_size)

    # Update discriminator and policy
    metrics = agent.update(policy_batch, expert_batch)

# DAgger Example
config = {
    "state_dim": 17,
    "action_dim": 6,
    "hidden_dims": [256, 256],
    "learning_rate": 3e-4,
    "beta_decay": 0.95
}

agent = DAggerAgent(config)

# Training loop
for epoch in range(num_epochs):
    # Collect policy rollouts
    states, _ = collect_rollouts(agent, env)

    # Query expert for labels on policy-visited states
    expert_actions = expert.label(states)

    # Update policy
    metrics = agent.update(states, expert_actions)
```

## Common Challenges and Solutions

### Challenge 1: Compounding Errors (Distributional Shift)

**Problem**: Small errors accumulate over time, leading to state distributions unseen during training.

**Solutions**:
- **DAgger**: Query expert on learner-visited states
- **GAIL/AIRL**: Match state-action distributions via adversarial training
- **MEGA-DAgger**: Use world model to plan and avoid error-prone states
- **All**: Add noise to expert demonstrations during training

### Challenge 2: Insufficient Expert Data

**Problem**: Limited demonstrations lead to overfitting and poor generalization.

**Solutions**:
- Data augmentation (trajectory perturbations)
- Regularization (dropout, weight decay)
- Ensemble methods
- Active learning to request more data where needed

### Challenge 3: GAN Training Instability (GAIL/AIRL)

**Problem**: Discriminator and policy training can be unstable, leading to mode collapse or divergence.

**Solutions**:
- Gradient penalty (WGAN-GP style)
- Spectral normalization on discriminator
- Lower discriminator learning rate
- Multiple discriminator updates per policy update
- Batch normalization

### Challenge 4: Reward Ambiguity (GAIL)

**Problem**: Many reward functions can explain the same behavior.

**Solutions**:
- Use AIRL if reward recovery is important
- Add reward shaping or prior knowledge
- Constrain reward function space
- Multi-task learning with shared rewards

### Challenge 5: World Model Errors (MEGA-DAgger)

**Problem**: Inaccurate world model leads to poor planning and policy learning.

**Solutions**:
- Ensemble world models for uncertainty
- Use model only where confident
- Mix model-based and model-free updates
- Continuously update model with real data
- Conservative planning with pessimistic models

## Key Papers

### Foundational

1. **Behavioral Cloning Basics**:
   - [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686) (Ross et al., AISTATS 2011)

2. **DAgger**:
   - [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686) (Ross et al., AISTATS 2011)

3. **GAIL**:
   - [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476) (Ho & Ermon, NeurIPS 2016)

4. **AIRL**:
   - [Learning Robust Rewards with Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248) (Fu et al., ICLR 2018)

5. **MEGA-DAgger**:
   - [Model-based Generative Adversarial Imitation Learning](https://arxiv.org/abs/2004.03763) (Vuong et al., CoRL 2020)

### Surveys and Tutorials

1. **Imitation Learning Survey**: [An Algorithmic Perspective on Imitation Learning](https://arxiv.org/abs/1811.06711) (Osa et al., 2018)
2. **Inverse RL Survey**: [A Survey of Inverse Reinforcement Learning: Techniques, Applications, and Open Problems](https://arxiv.org/abs/1806.06877) (Arora & Doshi, 2021)

## Additional Resources

### Tutorials
- [Stanford CS237A: Imitation Learning](https://web.stanford.edu/class/cs237a/)
- [Berkeley CS287: Advanced Robotics - Imitation Learning Module](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/)
- [Imitation Learning Tutorial by Sergey Levine](https://sites.google.com/view/icml2018-imitation-learning/)

### Implementations
- [imitation](https://github.com/HumanCompatibleAI/imitation): Clean implementations of IL algorithms
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3): Includes GAIL
- [rlkit](https://github.com/rail-berkeley/rlkit): Research codebase with AIRL

### Datasets
- [D4RL](https://github.com/rail-berkeley/d4rl): Offline RL and IL benchmark datasets
- [RoboMimic](https://robomimic.github.io/): Robot manipulation demonstrations
- [Atari demonstrations](https://github.com/hmandell/atari_demonstrations)

## File Structure

```
18_imitation_learning/
├── README.md              # This file
├── gail.md               # Generative Adversarial Imitation Learning
├── dagger.md             # Dataset Aggregation
├── mega_dagger.md        # Model-based DAgger
└── airl.md               # Adversarial Inverse RL
```

## Getting Started

1. **New to Imitation Learning?** Start with [DAgger](dagger.md) to understand the distributional shift problem and interactive learning
2. **Have expert demonstrations?** Jump to [GAIL](gail.md) for adversarial imitation learning
3. **Need interpretable rewards?** Study [AIRL](airl.md) for reward recovery
4. **Limited expert access?** Explore [MEGA-DAgger](mega_dagger.md) for model-based efficiency

Each algorithm documentation includes:
- Theoretical foundations
- Mathematical formulations
- Implementation details
- Code walkthroughs
- Optimization tricks
- Common pitfalls
- Experimental results
