# Model-Based Reinforcement Learning

Model-Based RL (MBRL) learns a dynamics model of the environment and uses it for planning or policy optimization. By learning "how the world works," MBRL agents can achieve superior sample efficiency compared to model-free methods, though at the cost of potential model errors.

## Key Concepts

### World Models

A world model predicts environment dynamics:
```
ŝ_{t+1}, r̂_t = f(s_t, a_t | θ)
```

Types:
- **Forward models**: Predict next state and reward
- **Inverse models**: Predict action from state transition
- **Latent models**: Learn compressed representations (DreamerV3)
- **Ensemble models**: Multiple models for uncertainty (MBPO)

### Planning vs Learning

**Planning**: Use model to simulate trajectories and select actions
- Model Predictive Control (MPC): Optimize action sequence online
- Monte Carlo Tree Search (MCTS): Build search tree
- Cross-Entropy Method (CEM): Sample-based optimization

**Learning**: Use model to generate synthetic data for policy training
- Dyna-style: Mix real and imagined transitions
- Pure imagination: Train policy entirely in latent space (DreamerV3)
- Branched rollouts: Short model rollouts from real states (MBPO)

### Challenges

1. **Model Error**: Compounding errors over long horizons
2. **Partial Observability**: Must maintain belief states
3. **Computational Cost**: Model learning + planning overhead
4. **Exploration**: Model-based exploration strategies

## Algorithms Covered

- **[DreamerV3](./dreamerv3.md)**: Universal world model with latent imagination
- **[TD-MPC2](./td_mpc2.md)**: Scalable MPC with learned latent dynamics (NOT YET IMPLEMENTED)
- **[MBPO](./mbpo.md)**: Model-based policy optimization with branched rollouts

## Comparison

| Algorithm | Model Type | Planning | Sample Efficiency | Domains |
|-----------|------------|----------|-------------------|---------|
| DreamerV3 | Latent (RSSM) | Actor-Critic in imagination | Very High | Vision, control, diverse |
| TD-MPC2 | Latent (dynamics) | MPC with learned objective | High | Continuous control |
| MBPO | Ensemble dynamics | Model-free (SAC) on synthetic data | High | Low-dim state spaces |

## When to Use Model-Based RL

**Best for**:
- Sample-limited domains (robotics, expensive simulations)
- Need for interpretability (understand system dynamics)
- Transfer learning (model generalizes across tasks)
- Safety-critical applications (predict outcomes)

**Avoid when**:
- Abundant samples available (Atari games)
- Stochastic/chaotic dynamics (hard to model)
- Very high-dimensional observations without structure
- Real-time performance critical (planning overhead)

## References

1. **DreamerV3**: Hafner et al., "Mastering Diverse Domains through World Models", 2023
2. **TD-MPC2**: Hansen et al., "TD-MPC2: Scalable, Robust World Models for Continuous Control", ICLR 2024
3. **MBPO**: Janner et al., "When to Trust Your Model: Model-Based Policy Optimization", NeurIPS 2019
