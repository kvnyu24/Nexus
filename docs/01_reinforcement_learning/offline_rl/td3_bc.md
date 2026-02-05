# TD3+BC: Twin Delayed DDPG with Behavior Cloning

**Paper**: [A Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/abs/2106.06860) (Fujimoto & Gu, NeurIPS 2021)

**Code**: `nexus/models/rl/offline/td3_bc.py`

## Overview

TD3+BC is remarkably simple: add a **single weighted behavior cloning term** to TD3's actor loss. Despite its simplicity, it's competitive with complex offline RL methods.

### Core Modification

**TD3 Actor**: `max E[Q(s, π(s))]`

**TD3+BC Actor**: `max E[Q(s, π(s)) - λ·||π(s) - a_data||²]`

That's it. One hyperparameter (λ), minimal code changes.

## Mathematical Formulation

```
L_actor = -Q(s, π(s)) + λ · ||π(s) - a_data||²

where:
  - Q(s, π(s)): maximize Q-value (standard TD3)
  - ||π(s) - a_data||²: stay close to dataset actions (BC)
  - λ = α / E[|Q(s,a)|]: normalized weight

Normalization:
  λ = α / Q_mean  ensures balance regardless of Q-value scale
  α = 2.5 (default) is the only hyperparameter to tune
```

### Intuition

The BC term acts as a **safety net**:
- When Q is accurate: policy improves beyond dataset
- When Q overestimates: BC pulls policy back to safe actions
- The normalization adapts the trade-off automatically

## Implementation

From `nexus/models/rl/offline/td3_bc.py`:

### Actor Loss (Lines 254-277)

```python
def update(self, batch):
    # ... critic update (standard TD3) ...

    if self.total_updates % self.policy_freq == 0:
        # Policy actions
        policy_actions = self.actor(states)

        # Q-value (use Q1)
        q_values = self.critic.q1_forward(states, policy_actions)

        # BC loss
        bc_loss = F.mse_loss(policy_actions, actions)

        # Normalize lambda by Q-value magnitude
        if self.normalize_q:
            lmbda = self.alpha / q_values.abs().mean().detach()
        else:
            lmbda = self.alpha

        # TD3+BC loss
        actor_loss = -q_values.mean() + lmbda * bc_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
```

### Critic Update (Standard TD3)

```python
# Target with noise smoothing
next_actions = (self.actor_target(next_states) + noise).clamp(...)
target_q = rewards + gamma * (1 - dones) * min(Q1_target, Q2_target)

# Twin Q update
critic_loss = mse_loss(Q1, target_q) + mse_loss(Q2, target_q)
```

## Hyperparameters

### Alpha (α)

**Range**: 1.0 - 5.0
- **2.5**: Default, works for most tasks
- **1.0**: Less conservative (high-quality data)
- **5.0**: More conservative (low-quality data)

**Rarely needs tuning** - 2.5 is robust across datasets.

### Normalize Q (normalize_q)

```python
normalize_q = True  # Default, recommended
```

When True: `λ = α / E[|Q|]` adapts to Q-value scale.
When False: `λ = α` is fixed.

Always use normalization unless you have specific reasons not to.

### TD3 Hyperparamaters

Standard TD3 settings work:
- `policy_noise = 0.2`: target action noise
- `noise_clip = 0.5`: clip noise range
- `policy_freq = 2`: delayed policy updates
- `tau = 0.005`: soft target update

## Experiments

### D4RL Results

| Dataset | TD3+BC | CQL | IQL |
|---------|--------|-----|-----|
| halfcheetah-medium | 48.3 | 44.0 | 47.4 |
| hopper-medium | 59.3 | 58.5 | 66.3 |
| walker2d-medium | 83.7 | 79.2 | 78.3 |
| **Average** | **63.8** | **60.6** | **64.0** |

**Observation**: Competitive with complex methods while being much simpler.

## Advantages

1. **Simplicity**: 5 lines of code vs. 100+ for CQL
2. **Robust**: α=2.5 works across diverse tasks
3. **Fast**: No expensive regularization computation
4. **Memory efficient**: Same as TD3

## Limitations

1. **Continuous actions only**: Not applicable to discrete action spaces
2. **Less theoretical**: No formal guarantees unlike CQL/IQL
3. **Modest improvements**: Not always best, but never terrible

## Common Pitfalls

### 1. Not Normalizing λ

**Mistake**: Using fixed λ across different Q-scales
```python
actor_loss = -q_values.mean() + 2.5 * bc_loss  # Breaks if Q~1000
```

**Fix**: Normalize by Q magnitude
```python
lmbda = 2.5 / q_values.abs().mean().detach()
actor_loss = -q_values.mean() + lmbda * bc_loss
```

### 2. Wrong BC Loss

**Mistake**: Using cross-entropy for continuous actions
```python
bc_loss = F.cross_entropy(policy_actions, actions)  # Wrong!
```

**Fix**: MSE for continuous actions
```python
bc_loss = F.mse_loss(policy_actions, actions)
```

### 3. Forgetting Delayed Updates

**Mistake**: Updating actor every step
```python
# Missing: if self.total_updates % policy_freq == 0:
actor_loss.backward()  # Too frequent!
```

**Fix**: Use delayed updates (policy_freq=2)

## Usage Example

```python
from nexus.models.rl.offline import TD3BCAgent

config = {
    "state_dim": 17,
    "action_dim": 6,
    "hidden_dim": 256,
    "alpha": 2.5,              # BC weight (rarely needs tuning!)
    "normalize_q": True,       # Normalize by Q magnitude
    "policy_noise": 0.2,       # Target smoothing
    "noise_clip": 0.5,
    "policy_freq": 2,          # Delayed updates
    "gamma": 0.99,
    "tau": 0.005,
}

agent = TD3BCAgent(config)

for batch in dataset:
    metrics = agent.update(batch)
```

## When to Use TD3+BC

**Best for**:
- Quick baseline
- Continuous control
- Medium-to-high quality data

**Avoid when**:
- Need theoretical guarantees → use CQL
- Very low quality data → use IQL
- Discrete actions → use DQN-based methods

## References

```bibtex
@inproceedings{fujimoto2021td3bc,
  title={A Minimalist Approach to Offline Reinforcement Learning},
  author={Fujimoto, Scott and Gu, Shixiang Shane},
  booktitle={NeurIPS},
  year={2021}
}
```

**Related**: TD3 (Fujimoto et al., 2018) - Base online algorithm
