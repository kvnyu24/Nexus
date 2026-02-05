# ReBRAC: Randomized Ensembled Behavior Regularized Actor-Critic

**Paper**: [The In-Sample Softmax for Offline Reinforcement Learning](https://arxiv.org/abs/2305.09836) (Tarasov et al., ICLR 2024)

**Status**: Reference documentation (implementation pending)

## Overview

ReBRAC achieves SOTA offline RL performance by combining:
1. **LayerNorm** in networks
2. **Ensemble** of Q-networks
3. **Behavior regularization** (BC)
4. **In-sample learning** (only dataset actions)

### Core Insight

Many "algorithmic" improvements are actually **implementation details**. ReBRAC shows that careful engineering (LayerNorm, ensemble size, BC strength) matters more than complex objectives.

## Key Components

### 1. LayerNorm in Networks

```python
class MLPWithLayerNorm(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Key: LayerNorm after each layer
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
```

**Why**: Stabilizes training, especially with large ensemble.

### 2. Large Q-Network Ensemble

```python
num_critics = 10  # Not 2 like TD3, but 10!
```

Uses random subset for each update:
```python
# Sample 2 critics from 10
critics = random.sample(self.critic_ensemble, k=2)
q1, q2 = critics[0](s, a), critics[1](s, a)
```

**Why**: Reduces overestimation, improves robustness.

### 3. Behavior Regularization

```python
# Similar to TD3+BC, but with ensemble
bc_loss = F.mse_loss(policy_actions, dataset_actions)
actor_loss = -q_value.mean() + lambda_bc * bc_loss
```

**Why**: Prevents OOD extrapolation.

### 4. In-Sample Softmax

Instead of hard min over ensemble, use softmax:
```python
# Standard: min(Q1, Q2, ..., Q10)
# ReBRAC: softmax with temperature
weights = F.softmax(-q_values / temp, dim=0)
q_weighted = (weights * q_values).sum()
```

**Why**: Softer aggregation, less pessimistic than hard min.

## Algorithm Pseudocode

```python
class ReBRACAgent:
    def __init__(self, config):
        # 10 Q-networks with LayerNorm
        self.critics = [
            MLPWithLayerNorm(...) for _ in range(10)
        ]
        self.actor = MLPWithLayerNorm(...)

    def update(self, batch):
        # Update critics
        for critic in self.critics:
            target = reward + gamma * self.soft_q_target(next_state)
            critic_loss = mse(critic(s, a), target)
            optimize(critic, critic_loss)

        # Update actor with BC
        if step % policy_freq == 0:
            policy_actions = self.actor(states)
            q_value = self.soft_q(states, policy_actions)
            bc_loss = mse(policy_actions, actions)
            actor_loss = -q_value + lambda_bc * bc_loss
            optimize(self.actor, actor_loss)

    def soft_q_target(self, state):
        """In-sample softmax over ensemble."""
        # Sample 2 critics randomly
        q_values = [critic(state, next_action) for critic in random.sample(self.critics, 2)]
        weights = softmax(-q_values / temp)
        return (weights * q_values).sum()
```

## Key Hyperparameters

```python
config = {
    "num_critics": 10,         # Ensemble size
    "num_critics_sample": 2,   # Subset size per update
    "lambda_bc": 2.5,          # BC regularization
    "layer_norm": True,        # Use LayerNorm
    "softmax_temp": 1.0,       # In-sample softmax temperature
}
```

## Performance

**D4RL Average Normalized Score**:
- ReBRAC: **73.1**
- IQL: 64.0
- CQL: 60.6
- TD3+BC: 63.8

Currently the best offline RL algorithm on D4RL.

## Why It Works

1. **LayerNorm**: Stabilizes gradients with large ensembles
2. **Large ensemble**: Better uncertainty estimation
3. **Random subsampling**: Efficient, adds diversity
4. **In-sample softmax**: Less pessimistic than hard min
5. **BC regularization**: Prevents OOD issues

## When to Use

**Best for**:
- Need SOTA performance
- Have computational resources for ensemble
- Benchmark competitions

**Avoid when**:
- Limited compute (10x model cost)
- Need simplicity (many hyperparameters)
- Resource-constrained deployment

## References

```bibtex
@inproceedings{tarasov2024rebrac,
  title={The In-Sample Softmax for Offline Reinforcement Learning},
  author={Tarasov, Denis and Kurenkov, Vladislav and Nikulin, Alexander and Akimov, Dmitry and Kolesnikov, Sergey},
  booktitle={ICLR},
  year={2024}
}
```
