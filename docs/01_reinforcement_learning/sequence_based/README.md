# Sequence-Based Reinforcement Learning

This directory contains documentation for reinforcement learning algorithms that frame the problem as **sequence modeling**, leveraging transformer architectures to learn policies conditioned on desired outcomes.

## Overview

Sequence-based RL represents a paradigm shift from traditional value-based and policy gradient methods. Instead of learning Q(s,a) or π(a|s), these methods learn π(a|s, R̂) where R̂ is a **return-to-go** (desired future cumulative reward). This enables:

- **Goal-conditioned behavior**: Specify desired performance at inference time
- **Trajectory stitching**: Combine suboptimal demonstrations into optimal behavior
- **Simplified training**: No Bellman backups or bootstrapping required
- **Leveraging transformers**: Use powerful sequence models from NLP

## Algorithms Covered

### [Decision Transformer](./decision_transformer.md)
**Core Innovation**: Treats RL as supervised sequence modeling

- Conditions on returns-to-go to generate actions
- Uses GPT-style transformer with causal masking
- Trains on offline data without value functions
- Achieves competitive performance on D4RL benchmarks

**When to Use**: Offline RL with diverse trajectory data, when you want simple and stable training.

**Key Papers**: Chen et al. (2021) - NeurIPS

### [Elastic Decision Transformer (EDT)](./elastic_dt.md)
**Core Innovation**: Adaptive context window selection

- Dynamically selects history length based on task complexity
- More efficient than fixed-context transformers (30-60% speedup)
- Better performance through focused attention
- Elastic positional encodings for variable-length generalization

**When to Use**: When computational efficiency matters, or when different states need different amounts of context.

**Key Papers**: Yamagata et al. (2023) - NeurIPS

### [Online Decision Transformer](./online_dt.md)
**Core Innovation**: Bridges offline and online learning

- Pre-trains on offline data, fine-tunes with online interaction
- 10-15x more sample efficient than training from scratch
- Adaptive return target scheduling
- Prevents catastrophic forgetting of offline knowledge

**When to Use**: When you have offline data but need to improve with online experience.

**Key Papers**: Zheng et al. (2022) - ICML

## Comparison Table

| Algorithm | Context | Data Type | Computational Cost | Sample Efficiency | Best Use Case |
|-----------|---------|-----------|-------------------|------------------|---------------|
| Decision Transformer | Fixed (K=20) | Offline only | Medium | N/A | Diverse offline datasets |
| Elastic DT | Adaptive (K=5-20) | Offline only | Low | N/A | Efficiency-critical applications |
| Online DT | Fixed (K=20) | Offline + Online | Medium-High | Very High | Offline pretraining + online tuning |

## Common Concepts

### Returns-to-Go
The cumulative future reward from current timestep:
```
R̂_t = Σ_{t'=t}^T r_{t'}
```

This serves as a "goal" for the policy—what total reward we want to achieve.

### Trajectory Stitching
Ability to combine suboptimal trajectories to produce optimal behavior:
- Trajectory A: reaches good state but fails afterward
- Trajectory B: starts from similar state and succeeds
- DT learns: "When in this state heading toward high reward, do what B did"

### Context Window (K)
Number of past timesteps the transformer can attend to. Typical values:
- Too small (K<5): Can't capture dependencies
- Sweet spot (K=10-20): Good balance
- Too large (K>30): Attention dilution, slower training

## Prerequisites

To understand these algorithms, you should be familiar with:

1. **Transformers**: Self-attention, causal masking, positional encodings
2. **Basic RL**: MDPs, returns, policies, trajectories
3. **Offline RL**: Distribution shift, trajectory datasets

## Implementation Guide

### Quick Start

```python
from nexus.models.rl import DecisionTransformer

# Configure model
config = {
    "state_dim": env.observation_space.shape[0],
    "action_dim": env.action_space.shape[0],
    "hidden_dim": 128,
    "num_layers": 3,
    "max_seq_len": 20,
}

# Create model
dt = DecisionTransformer(config)

# Train on offline data
for batch in offline_dataset:
    loss = dt.update(batch)

# Inference with desired return
dt.reset_history()
action = dt.select_action(state, target_return=100, timestep=0)
```

### Key Implementation Details

1. **Token ordering**: (R̂_1, s_1, a_1, R̂_2, s_2, a_2, ...)
2. **Causal masking**: Prevent future information leakage
3. **Return normalization**: Scale returns to [-1, 1] or [0, 1]
4. **Learning rate warmup**: Critical for stable training
5. **Gradient clipping**: Prevent exploding gradients

## Performance Benchmarks

### D4RL MuJoCo Tasks

Normalized scores (higher is better):

| Environment | BC | CQL | IQL | DT | EDT | Online DT |
|------------|-----|-----|-----|-----|-----|-----------|
| HalfCheetah-Medium | 42.6 | 44.0 | 47.4 | 42.6 | 44.8 | 52.1 |
| Hopper-Medium | 52.9 | 58.5 | 66.3 | 67.6 | 71.2 | 98.2 |
| Walker2d-Medium | 75.3 | 72.5 | 78.3 | 74.0 | 78.3 | 89.3 |

## Research Directions

### Current Limitations
- Bounded by dataset quality (offline methods)
- Context length limits long-term reasoning
- No explicit uncertainty quantification
- Limited exploration (online methods)

### Future Work
- **Hierarchical DT**: Multi-level planning
- **World models**: Learn environment dynamics
- **Meta-learning**: Adapt to new tasks quickly
- **Multi-modal**: Vision + language + actions
- **Uncertainty-aware**: Better exploration and safety

## Related Areas

- **Offline RL**: CQL, IQL, TD3+BC (see `../offline_rl/`)
- **Model-Based RL**: Dreamer, MuZero (see `../model_based/`)
- **Transformer RL**: GTrXL (see `../policy_gradient/`)

## References

### Foundational Papers
1. Chen, L., et al. (2021). **Decision Transformer: Reinforcement Learning via Sequence Modeling.** NeurIPS.
2. Janner, M., et al. (2021). **Offline Reinforcement Learning as One Big Sequence Modeling Problem.** NeurIPS.

### Extensions
3. Yamagata, T., et al. (2023). **Elastic Decision Transformer.** NeurIPS.
4. Zheng, Q., et al. (2022). **Online Decision Transformer.** ICML.
5. Lee, K., et al. (2022). **Multi-Game Decision Transformers.** NeurIPS.

### Transformer Background
6. Vaswani, A., et al. (2017). **Attention Is All You Need.** NeurIPS.
7. Radford, A., et al. (2019). **Language Models are Unsupervised Multitask Learners.** (GPT-2)

## Code Locations

All implementations are in the Nexus repository:

- **Decision Transformer**: `/Users/kevinyu/Projects/Nexus/nexus/models/rl/decision_transformer.py`
- **Elastic DT**: `/Users/kevinyu/Projects/Nexus/nexus/models/rl/sequence/edt.py`
- **Utilities**: `/Users/kevinyu/Projects/Nexus/nexus/models/rl/sequence/__init__.py`

## Getting Help

For questions or issues:
1. Check the individual algorithm documentation
2. Review the code examples in each doc
3. Look at the implementation in the codebase
4. Open an issue on the repository

---

**Navigation**:
- [← Back to RL Overview](../)
- [Decision Transformer →](./decision_transformer.md)
- [Elastic DT →](./elastic_dt.md)
- [Online DT →](./online_dt.md)
