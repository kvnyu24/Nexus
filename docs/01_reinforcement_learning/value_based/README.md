# Value-Based Reinforcement Learning Methods

This directory contains comprehensive documentation for value-based reinforcement learning algorithms implemented in Nexus. Value-based methods learn to estimate the value (expected return) of states or state-action pairs and use these estimates to derive policies.

## Table of Contents

1. [DQN (Deep Q-Network)](#dqn)
2. [Double DQN](#double-dqn)
3. [Dueling DQN](#dueling-dqn)
4. [Rainbow DQN](#rainbow-dqn)
5. [C51 (Categorical DQN)](#c51)
6. [QR-DQN (Quantile Regression DQN)](#qr-dqn)

## Overview

Value-based methods are a fundamental class of reinforcement learning algorithms that learn to estimate value functions:

- **State Value Function V(s)**: Expected return from state s
- **Action Value Function Q(s,a)**: Expected return from taking action a in state s

The key insight is that if we know the optimal Q-values Q*(s,a), we can derive the optimal policy by simply selecting the action with the highest Q-value at each state.

## Core Concepts

### Q-Learning Foundation

All algorithms in this section build upon the Q-learning update rule:

```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

Where:
- α is the learning rate
- γ is the discount factor
- r is the immediate reward
- s' is the next state

### Deep Q-Networks

Deep Q-Networks (DQN) use neural networks to approximate Q-values, enabling RL to scale to high-dimensional state spaces like images. The key innovations that made DQN work include:

1. **Experience Replay**: Store transitions in a buffer and sample mini-batches randomly
2. **Target Network**: Use a separate, slowly-updated network for computing targets
3. **Frame Stacking**: Stack consecutive frames to capture temporal information

## Algorithm Progression

We recommend studying these algorithms in the following order:

### 1. DQN (Start Here)
- **File**: [dqn.md](dqn.md)
- **Difficulty**: Beginner
- **Key Concepts**: Experience replay, target networks, epsilon-greedy exploration
- **Use Case**: Learning the fundamentals of deep RL

### 2. Double DQN
- **File**: [double_dqn.md](double_dqn.md)
- **Difficulty**: Beginner-Intermediate
- **Key Concepts**: Overestimation bias, decoupling action selection and evaluation
- **Use Case**: More stable Q-value estimates

### 3. Dueling DQN
- **File**: [dueling_dqn.md](dueling_dqn.md)
- **Difficulty**: Intermediate
- **Key Concepts**: Network architecture, value/advantage decomposition
- **Use Case**: Better generalization across actions

### 4. C51 (Categorical DQN)
- **File**: [c51.md](c51.md)
- **Difficulty**: Advanced
- **Key Concepts**: Distributional RL, categorical representations
- **Use Case**: Modeling uncertainty and risk

### 5. QR-DQN
- **File**: [qrdqn.md](qrdqn.md)
- **Difficulty**: Advanced
- **Key Concepts**: Quantile regression, flexible distributions
- **Use Case**: Better tail modeling, unbounded rewards

### 6. Rainbow DQN (Capstone)
- **File**: [rainbow.md](rainbow.md)
- **Difficulty**: Advanced
- **Key Concepts**: Integration of multiple improvements
- **Use Case**: State-of-the-art performance on Atari

## Comparison Table

| Algorithm | Overestimation Fix | Architecture | Distribution | Exploration | Complexity |
|-----------|-------------------|--------------|--------------|-------------|------------|
| DQN | ❌ | Standard | Point Estimate | ε-greedy | Low |
| Double DQN | ✅ | Standard | Point Estimate | ε-greedy | Low |
| Dueling DQN | ❌ | Value/Advantage | Point Estimate | ε-greedy | Medium |
| C51 | ❌ | Standard | Categorical | ε-greedy | High |
| QR-DQN | ✅ | Standard | Quantile | ε-greedy | High |
| Rainbow | ✅ | Value/Advantage | Categorical | Noisy Nets | Very High |

## When to Use Each Algorithm

### Use DQN when:
- You're learning the basics
- You have simple discrete action spaces
- Computational efficiency is critical
- You want a simple baseline

### Use Double DQN when:
- Q-values are being overestimated
- You want more stable learning
- DQN shows high variance

### Use Dueling DQN when:
- Many actions have similar values
- The state value is more important than action advantages
- You want better generalization

### Use C51 when:
- You need to model uncertainty
- Returns are multi-modal
- Risk-sensitive decision making is important

### Use QR-DQN when:
- You want distributional RL with flexibility
- Reward ranges are unknown
- Tail distributions are important

### Use Rainbow when:
- Maximum performance is needed
- Computational resources are available
- You've mastered the individual components

## Implementation Details

All implementations in Nexus follow a consistent API:

```python
from nexus.models.rl.dqn import DQNAgent

# Configure agent
config = {
    "state_dim": 4,
    "action_dim": 2,
    "hidden_dim": 128,
    "gamma": 0.99,
    "learning_rate": 1e-3
}

# Initialize
agent = DQNAgent(config)

# Select action
action = agent.select_action(state, training=True)

# Update
batch = replay_buffer.sample(batch_size)
metrics = agent.update(batch)

# Update target network (DQN/Dueling)
agent.update_target_network()
```

## Common Components

### Experience Replay Buffer

All algorithms use experience replay to break temporal correlations:

```python
from nexus.data.replay_buffer import ReplayBuffer

buffer = ReplayBuffer(capacity=100000)
buffer.push(state, action, reward, next_state, done)
batch = buffer.sample(batch_size)
```

### Exploration Strategies

- **ε-greedy**: DQN, Double DQN, Dueling DQN, C51, QR-DQN
- **Noisy Networks**: Rainbow (learned, state-dependent)

### Target Network Updates

- **Hard Update**: Copy weights every N steps (DQN, Dueling)
- **Soft Update**: Polyak averaging (Double DQN, C51, QR-DQN, Rainbow)

## Key Papers

1. **DQN**: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013)
2. **DQN Nature**: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (Mnih et al., 2015)
3. **Double DQN**: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (van Hasselt et al., 2015)
4. **Dueling DQN**: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) (Wang et al., 2016)
5. **Prioritized Replay**: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) (Schaul et al., 2016)
6. **C51**: [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887) (Bellemare et al., 2017)
7. **Noisy Nets**: [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295) (Fortunato et al., 2018)
8. **QR-DQN**: [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044) (Dabney et al., 2018)
9. **Rainbow**: [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) (Hessel et al., 2018)

## Additional Resources

### Tutorials
- [Deep RL Course by Hugging Face](https://huggingface.co/deep-rl-course)
- [Spinning Up in Deep RL by OpenAI](https://spinningup.openai.com/)
- [DQN Tutorial by PyTorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

### Implementations
- [Dopamine](https://github.com/google/dopamine): Google's RL framework
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3): PyTorch RL implementations
- [CleanRL](https://github.com/vwxyzjn/cleanrl): Single-file RL implementations

### Benchmarks
- [Atari 100k Benchmark](https://paperswithcode.com/sota/atari-games-100k-on-atari-100k)
- [OpenAI Gym](https://www.gymlibrary.dev/)

## File Structure

```
value_based/
├── README.md              # This file
├── dqn.md                # Deep Q-Network
├── double_dqn.md         # Double DQN
├── dueling_dqn.md        # Dueling DQN
├── c51.md                # Categorical DQN
├── qrdqn.md              # Quantile Regression DQN
└── rainbow.md            # Rainbow DQN
```

## Getting Started

1. Start with [DQN](dqn.md) to understand the core concepts
2. Read about the specific improvements in [Double DQN](double_dqn.md) and [Dueling DQN](dueling_dqn.md)
3. Explore distributional RL with [C51](c51.md) and [QR-DQN](qrdqn.md)
4. Study [Rainbow](rainbow.md) to see how everything comes together

Each algorithm documentation includes theory, implementation details, code walkthroughs, and practical tips.
