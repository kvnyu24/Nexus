# Policy Gradient and Actor-Critic Methods

This directory contains comprehensive documentation for policy gradient and actor-critic methods, which are fundamental approaches in modern reinforcement learning for continuous and discrete control tasks.

## Overview

Policy gradient methods directly optimize the policy by computing gradients of the expected return with respect to policy parameters. Unlike value-based methods that learn action-value functions, policy gradient methods learn a parameterized policy that can naturally handle:

- **Continuous action spaces**: Essential for robotics, control systems
- **Stochastic policies**: Naturally explore and handle partial observability
- **High-dimensional action spaces**: Scale better than discretization approaches
- **Direct policy optimization**: No need to derive policy from value function

## Algorithm Categories

### 1. Basic Policy Gradient
- **REINFORCE**: The foundational Monte Carlo policy gradient algorithm

### 2. Actor-Critic Methods (Discrete Actions)
- **A2C (Advantage Actor-Critic)**: Synchronous advantage-based actor-critic
- **PPO (Proximal Policy Optimization)**: Clipped surrogate objective for stable updates

### 3. Deterministic Policy Gradient (Continuous Actions)
- **DDPG (Deep Deterministic Policy Gradient)**: Actor-critic for continuous control
- **TD3 (Twin Delayed DDPG)**: Improved DDPG with twin critics and delayed updates

### 4. Stochastic Actor-Critic (Continuous Actions)
- **SAC (Soft Actor-Critic)**: Maximum entropy RL for sample-efficient learning

### 5. Trust Region Methods
- **TRPO (Trust Region Policy Optimization)**: Constrained optimization with guaranteed improvement

## Learning Path

### Beginner Path
1. **Start with REINFORCE** (`reinforce.md`)
   - Understand basic policy gradients
   - Learn Monte Carlo returns
   - Grasp variance reduction with baselines

2. **Progress to A2C** (`a2c.md`)
   - Understand actor-critic architecture
   - Learn bootstrapping with TD learning
   - Master advantage estimation

3. **Study PPO** (`ppo.md`)
   - Learn clipped surrogate objectives
   - Understand trust region concepts (simplified)
   - See production-ready implementation

### Intermediate Path
4. **Explore DDPG** (`ddpg.md`)
   - Move to continuous action spaces
   - Understand deterministic policy gradients
   - Learn target networks and replay buffers

5. **Advance to TD3** (`td3.md`)
   - Master twin critics for overestimation bias
   - Learn delayed policy updates
   - Understand target policy smoothing

6. **Study SAC** (`sac.md`)
   - Learn maximum entropy RL
   - Understand stochastic policies for continuous actions
   - Master automatic temperature tuning

### Advanced Path
7. **Master TRPO** (`trpo.md`)
   - Understand natural policy gradients
   - Learn constrained optimization
   - Study conjugate gradient methods
   - Grasp theoretical guarantees

## Quick Comparison

| Algorithm | Action Space | Key Innovation | Complexity | Sample Efficiency | Stability |
|-----------|--------------|----------------|------------|-------------------|-----------|
| REINFORCE | Both | Monte Carlo PG | Low | Low | Low |
| A2C | Discrete | Advantage estimation | Medium | Medium | Medium |
| PPO | Both | Clipped objective | Medium | Medium | High |
| DDPG | Continuous | Deterministic PG | Medium | Medium | Medium |
| TD3 | Continuous | Twin critics | Medium | High | High |
| SAC | Continuous | Maximum entropy | Medium | Very High | Very High |
| TRPO | Both | Trust region | High | Medium | Very High |

## Key Concepts

### Policy Gradient Theorem
The foundation of all policy gradient methods:
```
∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) Q^π(s,a)]
```

### Variance Reduction Techniques
1. **Baseline subtraction**: Reduce variance without bias
2. **Advantage functions**: Use A(s,a) = Q(s,a) - V(s)
3. **GAE (Generalized Advantage Estimation)**: Bias-variance trade-off
4. **Entropy regularization**: Encourage exploration

### Actor-Critic Architecture
- **Actor**: Policy network π_θ(a|s)
- **Critic**: Value network V_φ(s) or Q_φ(s,a)
- **Advantage**: A(s,a) = Q(s,a) - V(s) or TD error

### Continuous Action Spaces
Two main approaches:
1. **Deterministic policies**: DDPG, TD3 (use μ_θ(s))
2. **Stochastic policies**: SAC (use Gaussian π_θ(a|s))

## Implementation Guide

### Code Structure in Nexus
All implementations are located in `/nexus/models/rl/`:
- `reinforce.py`: REINFORCE implementation
- `a2c.py`: A2C implementation
- `ppo.py`: PPO implementation
- `ddpg.py`: DDPG implementation
- `td3.py`: TD3 implementation
- `sac.py`: SAC implementation
- `trpo.py`: TRPO implementation

### Common Patterns
All implementations follow the Nexus design:
```python
from nexus.models.rl import PPOAgent

config = {
    "state_dim": 8,
    "action_dim": 4,
    "hidden_dim": 256,
    "learning_rate": 3e-4,
    "gamma": 0.99,
}

agent = PPOAgent(config)
action, info = agent.select_action(state)
metrics = agent.update(batch)
```

## When to Use Each Algorithm

### Choose REINFORCE when:
- Learning about policy gradients
- Simple environments
- Episodic tasks
- Educational purposes

### Choose A2C when:
- Discrete action spaces
- Need faster learning than REINFORCE
- Want simple actor-critic
- Atari games, discrete control

### Choose PPO when:
- Need robust, stable training
- Both discrete/continuous actions
- Production deployments
- Robotics, complex control
- **Most recommended for general use**

### Choose DDPG when:
- Continuous control tasks
- Deterministic policies
- Physical simulations
- Lower sample count

### Choose TD3 when:
- Continuous control tasks
- Need more stability than DDPG
- Willing to trade complexity for performance
- Robotics, manipulation

### Choose SAC when:
- Continuous control tasks
- Need best sample efficiency
- Want automatic exploration tuning
- Complex continuous control
- **Recommended for continuous control**

### Choose TRPO when:
- Need guaranteed monotonic improvement
- Stability is critical
- Can afford computational cost
- Theoretical guarantees required
- Research on trust regions

## Common Pitfalls

### General Issues
1. **Reward scaling**: Always normalize rewards
2. **Network initialization**: Use small weights for policy output
3. **Learning rates**: Policy and value may need different rates
4. **Gradient clipping**: Essential for stability
5. **Hyperparameter tuning**: Critical for performance

### Algorithm-Specific
- **REINFORCE**: High variance, needs many episodes
- **A2C**: Sensitive to hyperparameters
- **PPO**: Clip range needs tuning
- **DDPG**: Exploration noise scheduling
- **TD3**: Policy delay parameter important
- **SAC**: Temperature tuning critical
- **TRPO**: Computationally expensive

## Mathematical Foundations

### Expected Return
```
J(θ) = E_τ~π_θ[∑_{t=0}^T γ^t r_t]
```

### Policy Gradient with Baseline
```
∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) (Q^π(s,a) - b(s))]
```

### Actor-Critic Update
```
Critic: minimize (R_t - V_θ(s_t))^2
Actor: maximize E[log π_θ(a|s) A_θ(s,a)]
```

### Advantage Estimation (GAE)
```
A_t = ∑_{l=0}^∞ (γλ)^l δ_{t+l}
where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

## References

### Foundational Papers
1. **Policy Gradient**: Williams (1992) - "Simple Statistical Gradient-Following Algorithms"
2. **Actor-Critic**: Sutton et al. (1999) - "Policy Gradient Methods for RL"
3. **Natural Gradients**: Kakade (2002) - "Natural Policy Gradient"

### Modern Methods
4. **A3C/A2C**: Mnih et al. (2016) - "Asynchronous Methods for Deep RL"
5. **TRPO**: Schulman et al. (2015) - "Trust Region Policy Optimization"
6. **PPO**: Schulman et al. (2017) - "Proximal Policy Optimization"
7. **DDPG**: Lillicrap et al. (2015) - "Continuous Control with Deep RL"
8. **TD3**: Fujimoto et al. (2018) - "Addressing Function Approximation Error"
9. **SAC**: Haarnoja et al. (2018) - "Soft Actor-Critic"

### Survey Papers
- Schulman (2016) - "Optimizing Expectations: From Deep RL to Stochastic Computation Graphs"
- Arulkumaran et al. (2017) - "Deep Reinforcement Learning: A Brief Survey"
- Peters & Schaal (2008) - "Reinforcement Learning of Motor Skills"

## Additional Resources

### Books
- Sutton & Barto (2018) - "Reinforcement Learning: An Introduction" (Chapter 13)
- Algorithms for Reinforcement Learning (Szepesvári, 2010)

### Courses
- CS 285 (Berkeley) - Deep Reinforcement Learning
- CS 234 (Stanford) - Reinforcement Learning
- DeepMind x UCL - Advanced Deep Learning & RL

### Code Repositories
- OpenAI Spinning Up: https://spinningup.openai.com/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- CleanRL: https://github.com/vwxyzjn/cleanrl

## Contributing

When adding new policy gradient algorithms:
1. Follow the 10-section documentation structure
2. Include mathematical derivations
3. Reference Nexus implementations
4. Add practical examples
5. Document common pitfalls
6. Update this README

## Navigation

- [REINFORCE](./reinforce.md) - Monte Carlo policy gradient
- [A2C](./a2c.md) - Advantage Actor-Critic
- [PPO](./ppo.md) - Proximal Policy Optimization
- [DDPG](./ddpg.md) - Deep Deterministic Policy Gradient
- [TD3](./td3.md) - Twin Delayed DDPG
- [SAC](./sac.md) - Soft Actor-Critic
- [TRPO](./trpo.md) - Trust Region Policy Optimization
