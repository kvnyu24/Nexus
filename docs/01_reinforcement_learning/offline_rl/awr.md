# Advantage-Weighted Regression (AWR)

**Paper**: [Advantage-Weighted Regression: Simple and Scalable Off-Policy RL](https://arxiv.org/abs/1910.00177) (Peng et al., NeurIPS 2019)

**Code**: `nexus/models/rl/offline/awr.py`

## Overview

AWR is a simple, stable offline RL algorithm that combines:
1. **Value function learning** (standard TD)
2. **Advantage-weighted behavior cloning** (exponentially weighted imitation)

### Core Idea

Instead of imitating all actions equally (behavior cloning), imitate actions **proportional to their advantage**:

```
BC:  π* = argmax E[log π(a|s)]
AWR: π* = argmax E[exp(A(s,a)/β) · log π(a|s)]

where A(s,a) = Q(s,a) - V(s)
```

High-advantage actions get high weight, low-advantage actions get ignored.

## Mathematical Formulation

### Advantage Weights

```
w(s,a) = exp(A(s,a) / β)

where:
  A(s,a) = r + γV(s') - V(s)  (TD advantage)
  β > 0: temperature controlling sharpness
```

### Policy Loss

```
L_π = -E_{(s,a)~D}[ w(s,a) · log π(a|s) ]

Intuitively:
  - w(s,a) large → strongly imitate action a
  - w(s,a) small → ignore action a
  - w(s,a) ≈ 1 → neutral
```

### Value Loss

```
L_V = E_{(s,r,s')~D}[ (V(s) - (r + γV(s')))² ]
```

Standard TD learning for the value function.

## Implementation

From `nexus/models/rl/offline/awr.py`:

### Advantage Computation (Lines 182-209)

```python
def compute_advantages(self, states, actions, rewards, next_states, dones):
    """Compute TD advantages."""
    with torch.no_grad():
        values = self.critic(states).squeeze(-1)
        next_values = self.critic(next_states).squeeze(-1)
        target_values = rewards + gamma * (1 - dones) * next_values
        advantages = target_values - values

    return advantages
```

### AWR Policy Update (Lines 262-274)

```python
# Compute advantages
advantages = self.compute_advantages(states, actions, rewards, next_states, dones)

# Exponential weighting
weights = torch.exp(advantages / self.beta)
weights = torch.clamp(weights, max=self.max_weight)  # Clip for stability

# Weighted BC loss
log_probs = self.actor.get_log_prob(states, actions)
actor_loss = -(weights * log_probs).mean()

self.actor_optimizer.zero_grad()
actor_loss.backward()
self.actor_optimizer.step()
```

## Hyperparameters

### Temperature (β)

**Range**: 0.05 - 1.0
- **0.05**: Sharp weighting, focus on best actions only
- **0.3**: Default, balanced weighting
- **1.0**: Smooth weighting, less discriminative

**Formula**: `weight = exp(advantage / β)`

Lower β = more aggressive (only imitate top actions).
Higher β = more conservative (imitate more actions).

### Max Weight Clipping

```python
max_weight = 20.0  # Default
```

Prevents numerical instability from extreme advantages. Always clip!

### Value Iterations

```python
value_iters = 5  # Update V multiple times per batch
actor_iters = 1  # Update π once per batch
```

More value updates → more accurate advantages → better policy learning.

## Advantages

1. **Simplicity**: Easy to understand and implement
2. **Stability**: No complex regularization or constraints
3. **Natural weighting**: Exponential advantage is intuitive
4. **No target networks**: Simpler than Q-learning methods

## Limitations

1. **Advantage estimation**: Requires accurate value function
2. **Single-step advantages**: Less effective than n-step returns
3. **Hyperparameter sensitive**: β needs tuning per task
4. **Data quality**: Struggles with very low-quality data

## Common Pitfalls

### 1. Not Clipping Weights

**Mistake**:
```python
weights = torch.exp(advantages / beta)  # Can explode!
```

**Fix**:
```python
weights = torch.clamp(torch.exp(advantages / beta), max=20.0)
```

### 2. Wrong Temperature Scale

Too small (0.01) → numerical instability
Too large (10.0) → all weights ≈ 1, no learning

**Fix**: Start with β=0.3, tune if needed

### 3. Insufficient Value Training

**Mistake**: One value update per batch
```python
value_iters = 1  # Not enough!
```

**Fix**: Multiple value updates
```python
value_iters = 5  # Better advantage estimates
```

## Usage Example

```python
from nexus.models.rl.offline import AWRAgent

config = {
    "state_dim": 17,
    "action_dim": 6,
    "hidden_dim": 256,
    "beta": 0.3,               # Temperature
    "max_weight": 20.0,        # Weight clipping
    "value_iters": 5,          # V updates per batch
    "actor_iters": 1,          # π updates per batch
    "gamma": 0.99,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
}

agent = AWRAgent(config)

for batch in dataset:
    metrics = agent.update(batch)
    print(f"Advantage: {metrics['mean_advantage']:.3f}, "
          f"Weight: {metrics['mean_weight']:.3f}")
```

## When to Use AWR

**Best for**:
- High-quality datasets (expert or near-expert)
- Need simple, interpretable algorithm
- Quick prototyping

**Avoid when**:
- Very mixed quality data → use IQL or CQL
- Need theoretical guarantees → use CQL
- Complex multi-modal behavior → use IDQL

## References

```bibtex
@inproceedings{peng2019awr,
  title={Advantage-Weighted Regression: Simple and Scalable Off-Policy RL},
  author={Peng, Xue Bin and Kumar, Aviral and Zhang, Grace and Levine, Sergey},
  booktitle={NeurIPS},
  year={2019}
}
```

**Extensions**:
- AWAC (Kumar et al., 2020): AWR with actor-critic
- IQL (Kostrikov et al., 2022): Uses AWR for policy extraction
