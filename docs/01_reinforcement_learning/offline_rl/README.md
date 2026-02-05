# Offline Reinforcement Learning

Offline RL (also called batch RL) learns policies from pre-collected datasets without interacting with the environment during training. This is crucial for domains where online interaction is expensive, dangerous, or impossible (robotics, healthcare, autonomous driving).

## The Offline RL Challenge

The fundamental challenge in offline RL is **distributional shift**: the agent must learn from a fixed dataset but generalize to states and actions not well-represented in that dataset. Standard RL algorithms fail catastrophically when they query out-of-distribution (OOD) actions due to:

1. **Extrapolation Error**: Q-functions overestimate values for unseen state-action pairs
2. **Policy Degeneracy**: Policies exploit Q-function errors by selecting OOD actions
3. **Bootstrapping Amplification**: Errors compound through temporal difference learning

## Algorithm Taxonomy

Offline RL algorithms can be categorized by their approach to handling distributional shift:

### 1. **Policy Constraint Methods**
Keep the learned policy close to the behavior policy that generated the data:
- **AWR** (Advantage-Weighted Regression): Weighted behavior cloning
- **TD3+BC**: Adds behavior cloning penalty to TD3

### 2. **Value Regularization Methods**
Explicitly regularize Q-values to be conservative on OOD actions:
- **CQL** (Conservative Q-Learning): Minimizes Q-values for OOD actions
- **EDAC** (Ensemble Diversified Actor-Critic): Uses ensemble disagreement for conservatism
- **Cal-QL**: Calibrates CQL's conservatism based on dataset quality

### 3. **In-Sample Learning Methods**
Only learn from actions present in the dataset:
- **IQL** (Implicit Q-Learning): Uses expectile regression to avoid OOD queries
- **IDQL** (Implicit Diffusion Q-Learning): Combines IQL with diffusion policies

### 4. **Hybrid Methods**
Combine multiple techniques:
- **ReBRAC** (Randomized Ensembled Behavior Regularized Actor-Critic): Combines behavior cloning with ensembling

## Algorithm Comparison

| Algorithm | Year | Key Innovation | Pros | Cons | Best Use Case |
|-----------|------|----------------|------|------|---------------|
| **AWR** | 2019 | Advantage-weighted BC | Simple, stable | Sensitive to advantage estimation | High-quality datasets |
| **CQL** | 2020 | Conservative Q-penalties | Strong theoretical guarantees | Requires hyperparameter tuning (α) | General purpose |
| **IQL** | 2021 | Expectile regression | No OOD queries, simple | Less sample efficient | Diverse, suboptimal data |
| **TD3+BC** | 2021 | Normalized BC penalty | Extremely simple | Limited to continuous actions | Quick baseline |
| **EDAC** | 2022 | Ensemble diversity | Handles stochastic environments | Higher compute (ensemble) | Noisy dynamics |
| **Cal-QL** | 2023 | Adaptive conservatism | Robust across dataset qualities | More complex | Mixed-quality datasets |
| **IDQL** | 2023 | Diffusion policies | Multimodal policies | Slower inference | Complex multi-modal behavior |
| **ReBRAC** | 2023 | Layernorm + ensembles | SOTA performance | Many components | Benchmark competitions |

## When to Use Each Algorithm

### High-Quality Expert Data
- **AWR**: Simple and effective when data is near-optimal
- **IQL**: More robust if data quality varies

### Mixed-Quality Data
- **CQL**: Good general-purpose choice with tuned α
- **Cal-QL**: Automatically adapts conservatism to dataset

### Suboptimal/Random Data
- **IQL**: Handles diverse data without OOD issues
- **CQL**: With high α for strong conservatism

### Multi-Modal Behavior
- **IDQL**: Diffusion policies can represent complex action distributions
- **IQL**: With Gaussian policies for simpler multi-modality

### Need Simplicity
- **TD3+BC**: Single hyperparameter, easy to implement
- **AWR**: Clean advantage-weighted formulation

### Need SOTA Performance
- **ReBRAC**: Current top performer on D4RL benchmarks
- **EDAC**: Strong on stochastic environments

## Implementation Notes

All algorithms in this directory follow the NexusModule interface:

```python
from nexus.models.rl.offline import AWRAgent, TD3BCAgent
from nexus.models.rl import IQLAgent, CQLAgent

# Configure agent
config = {
    "state_dim": env.observation_space.shape[0],
    "action_dim": env.action_space.shape[0],
    "hidden_dim": 256,
    # Algorithm-specific params
}

agent = IQLAgent(config)

# Training loop
for batch in dataset:
    metrics = agent.update(batch)
```

## Common Hyperparameters

### Learning Rates
- **Conservative algorithms** (CQL, Cal-QL): 3e-4 for all networks
- **Simple algorithms** (AWR, TD3+BC): 3e-4 for policy, 3e-4 for critic
- **IQL**: Often benefits from lower rates (1e-4)

### Network Architecture
- **Hidden dimensions**: 256 (default), 512 (complex tasks)
- **Number of layers**: 2-3 MLP layers
- **Activation**: ReLU (standard), LayerNorm (ReBRAC)

### Algorithm-Specific
- **CQL α**: 1.0-10.0 (higher = more conservative)
- **IQL expectile**: 0.7-0.9 (higher = more aggressive)
- **TD3+BC α**: 2.5 (normalized by Q-value)
- **AWR β (temperature)**: 0.05-0.5 (lower = sharper weighting)

## Datasets

### D4RL Benchmark
The de facto standard for offline RL evaluation:
- **MuJoCo**: Locomotion tasks (halfcheetah, hopper, walker2d)
- **AntMaze**: Sparse reward navigation
- **Adroit**: Dexterous manipulation

Dataset qualities:
- `random`: Random policy (worst)
- `medium-replay`: Mix of training data
- `medium`: Partially trained policy
- `medium-expert`: Mix of medium + expert
- `expert`: Near-optimal policy (best)

### Evaluation Protocol
- Train on offline dataset (no environment interaction)
- Evaluate on environment with deterministic policy
- Report normalized score: `100 * (score - random) / (expert - random)`
- Average over 10 evaluation episodes
- Report mean ± std over 5 random seeds

## References

### Foundational Papers
- AWR: [Peng et al., 2019](https://arxiv.org/abs/1910.00177)
- CQL: [Kumar et al., 2020](https://arxiv.org/abs/2006.04779)
- IQL: [Kostrikov et al., 2022](https://arxiv.org/abs/2110.06169)
- TD3+BC: [Fujimoto & Gu, 2021](https://arxiv.org/abs/2106.06860)

### Advanced Methods
- EDAC: [An et al., 2021](https://arxiv.org/abs/2110.01548)
- Cal-QL: [Nakamoto et al., 2023](https://arxiv.org/abs/2303.05479)
- IDQL: [Hansen-Estruch et al., 2023](https://arxiv.org/abs/2304.10573)
- ReBRAC: [Tarasov et al., 2023](https://arxiv.org/abs/2305.09836)

### Surveys
- [Offline RL Tutorial (Levine et al.)](https://sites.google.com/view/offlinerltutorial-neurips2020)
- [Decision Transformer (Chen et al., 2021)](https://arxiv.org/abs/2106.01345) - Alternative sequence modeling approach

## Algorithm Details

See individual algorithm documentation:
- [IQL - Implicit Q-Learning](./iql.md)
- [CQL - Conservative Q-Learning](./cql.md)
- [Cal-QL - Calibrated Q-Learning](./cal_ql.md)
- [IDQL - Implicit Diffusion Q-Learning](./idql.md)
- [ReBRAC - Randomized Ensembled Behavior Regularized Actor-Critic](./rebrac.md)
- [EDAC - Ensemble Diversified Actor-Critic](./edac.md)
- [TD3+BC - Twin Delayed DDPG with Behavior Cloning](./td3_bc.md)
- [AWR - Advantage-Weighted Regression](./awr.md)
