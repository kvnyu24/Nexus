# Exploration in Reinforcement Learning

Exploration is critical in RL to discover rewarding states and actions. Without effective exploration, agents may get stuck in local optima or fail to find rewards in sparse-reward environments.

## Key Concepts

### Exploration vs Exploitation

The fundamental trade-off:
- **Exploitation**: Use current knowledge to maximize reward
- **Exploration**: Try new actions to gain information

### Exploration Strategies

**Random Exploration**:
- Epsilon-greedy: Random actions with probability ε
- Boltzmann: Sample from softmax over Q-values
- Action noise: Add Gaussian noise to continuous actions

**Intrinsic Motivation**:
- Curiosity-driven: Bonus for novel/surprising states
- Count-based: Explore less-visited states
- Prediction error: Bonus for states the agent can't predict well

**Structured Exploration**:
- Go-Explore: Remember and return to promising states
- Novelty search: Ignore rewards, maximize behavioral diversity
- Quality diversity: Maintain archive of diverse high-performing behaviors

## Algorithms Covered

- **[ICM](./icm.md)**: Intrinsic Curiosity Module using prediction error
- **[RND](./rnd.md)**: Random Network Distillation for exploration bonus
- **[Go-Explore](./go_explore.md)**: Archive-based exploration with robustification (NOT YET IMPLEMENTED)

## Comparison

| Algorithm | Type | Intrinsic Reward | Best For | Overhead |
|-----------|------|------------------|----------|----------|
| ICM | Prediction error | Forward model error | Continuous/discrete | Moderate |
| RND | Prediction error | Random target error | Hard exploration | Low |
| Go-Explore | Archive-based | N/A (deterministic return) | Extremely sparse rewards | High |

## When to Use Exploration Methods

**ICM**:
- Environments with moderate sparsity
- When dynamics are learnable
- Avoid "noisy TV" problem (unpredictable distractors)

**RND**:
- Very sparse rewards (Montezuma's Revenge)
- Deterministic environments
- Need simple, scalable method

**Go-Explore**:
- Extremely hard exploration (Pitfall, Montezuma's Revenge)
- Deterministic or resettable environments
- Can afford memory overhead
- Need guaranteed exploration of promising areas

## Common Exploration Challenges

1. **Sparse Rewards**: Reward is rare, exploration critical
2. **Deceptive Rewards**: Local optima mislead agent
3. **High-Dimensional State Space**: Curse of dimensionality
4. **Partial Observability**: Can't distinguish novel from seen
5. **Stochastic Dynamics**: Hard to predict, noisyintrinsic signals

## References

1. **ICM**: Pathak et al., "Curiosity-driven Exploration by Self-Supervised Prediction", ICML 2017
2. **RND**: Burda et al., "Exploration by Random Network Distillation", ICLR 2019
3. **Go-Explore**: Ecoffet et al., "Go-Explore: a New Approach for Hard-Exploration Problems", Nature 2021
