# Conservative Q-Learning (CQL)

**Paper**: [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779) (Kumar et al., NeurIPS 2020)

**Code**: `nexus/models/rl/cql.py`

## Overview & Motivation

Conservative Q-Learning (CQL) learns a **conservative** Q-function that lower-bounds the true Q-value, particularly for out-of-distribution (OOD) actions. This prevents the value overestimation that causes offline RL failure.

### Core Idea

Standard Q-learning maximizes Q-values:
```
max E[Q(s,a)] → overestimates OOD actions
```

CQL adds a regularizer that minimizes Q-values for OOD actions while maintaining accuracy for in-dataset actions:
```
min E_{π}[Q(s,a)] - E_{data}[Q(s,a)] + standard_TD_loss
```

This creates a **conservative** Q-function: better to underestimate than overestimate.

## Mathematical Formulation

### CQL Objective

```
L_CQL(θ) = α · (E_{a~π(·|s)}[Q_θ(s,a)] - E_{a~D}[Q_θ(s,a)]) + L_TD(θ)

where:
  - First term: penalize Q-values for policy actions (likely OOD)
  - Second term: push up Q-values for dataset actions
  - L_TD: standard Bellman error
  - α > 0: regularization strength
```

### Practical Implementation

Instead of sampling from policy, CQL uses:
1. **Random actions**: uniform samples from action space
2. **Policy actions**: sampled from current policy

```
Q_penalty = logsumexp([Q(s, a_random), Q(s, a_policy)]) - Q(s, a_data)
```

The logsumexp provides a soft maximum over OOD actions.

## Implementation Details

### CQL Loss Computation

From `nexus/models/rl/cql.py` (lines 215-268):

```python
def compute_cql_loss(self, states, actions):
    batch_size = states.shape[0]

    # Sample random actions
    random_actions = torch.FloatTensor(
        batch_size * n_actions, action_dim
    ).uniform_(-max_action, max_action)

    # Sample policy actions
    policy_actions, policy_log_probs = self.policy.sample(states)

    # Q-values for random and policy actions
    q1_rand, q2_rand = self.critic(states_repeated, random_actions)
    q1_policy, q2_policy = self.critic(states_repeated, policy_actions)

    # Q-values for dataset actions
    q1_data, q2_data = self.critic(states, actions)

    # Importance sampling correction
    q1_policy = q1_policy - policy_log_probs.detach()
    q2_policy = q2_policy - policy_log_probs.detach()

    # CQL penalty: logsumexp - E_data[Q]
    q1_cat = torch.cat([q1_rand, q1_policy], dim=1)
    q2_cat = torch.cat([q2_rand, q2_policy], dim=1)

    cql_loss_q1 = torch.logsumexp(q1_cat, dim=1).mean() - q1_data.mean()
    cql_loss_q2 = torch.logsumexp(q2_cat, dim=1).mean() - q2_data.mean()

    return cql_loss_q1, cql_loss_q2
```

### Full Update

```python
def update(self, batch):
    # Standard TD loss
    current_q1, current_q2 = self.critic(states, actions)
    td_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

    # CQL regularization
    cql_loss = compute_cql_loss(states, actions)

    # Total critic loss
    critic_loss = td_loss + alpha * cql_loss

    # Update policy (standard SAC)
    policy_loss = (alpha_sac * log_probs - q_value).mean()
```

## Hyperparameters

### CQL Alpha (α)

**Critical hyperparameter** controlling conservatism:
- **α = 0**: Reduces to SAC (no conservatism)
- **α = 1.0**: Light conservatism (high-quality data)
- **α = 5.0**: Moderate conservatism (default)
- **α = 10.0**: Heavy conservatism (low-quality data)

**Rule of thumb**:
- Expert data: α = 1-2
- Medium data: α = 5
- Random/mixed data: α = 10-20

### Lagrangian CQL (Advanced)

Auto-tune α to maintain target action gap:

```python
if cql_lagrange:
    alpha_loss = -log_alpha * (cql_loss - target_gap).detach()
    # Optimize log_alpha
```

Sets α dynamically to keep `E_π[Q] - E_data[Q] ≈ target_gap`.

## Optimization Tricks

### 1. Number of CQL Actions

```python
cql_n_actions = 10  # Default
```

More actions = better OOD coverage but slower training. 10 is usually sufficient.

### 2. Importance Sampling Correction

```python
q_policy = q_policy - log_probs.detach()
```

Corrects for sampling from policy instead of uniform distribution.

### 3. Twin Q-Networks

Always use twin Q-networks (Q1, Q2) to reduce overestimation:
```python
target_q = min(Q1_target, Q2_target)
```

### 4. Automatic Alpha Tuning (SAC)

```python
alpha_loss = -(log_alpha * (log_probs + target_entropy)).mean()
```

Auto-tunes SAC temperature for better exploration-exploitation balance.

## Experiments & Results

### D4RL Benchmark

| Task | CQL | IQL | BC |
|------|-----|-----|-----|
| halfcheetah-medium | 44.0 | 47.4 | 42.6 |
| hopper-medium-expert | 98.7 | 91.5 | 52.5 |
| walker2d-medium-replay | 77.3 | 73.9 | 26.7 |

### Strengths
- **Theoretical guarantees**: Provably safe under certain assumptions
- **Versatile**: Works across different data qualities with proper α
- **Well-studied**: Extensive ablations and extensions

### Weaknesses
- **Hyperparameter sensitive**: Requires tuning α per task/dataset
- **Computational cost**: Multiple forward passes for CQL penalty
- **Conservative bias**: Can be overly cautious, missing good OOD actions

## Common Pitfalls

### 1. Wrong Alpha Value

Too small → overestimation errors
Too large → overly conservative, no learning

**Fix**: Start with α=5, increase if unstable, decrease if too conservative.

### 2. Not Using Importance Sampling

**Mistake**:
```python
cql_loss = logsumexp(q_policy) - q_data.mean()  # Missing correction
```

**Fix**:
```python
q_policy_corrected = q_policy - log_probs.detach()
cql_loss = logsumexp(q_policy_corrected) - q_data.mean()
```

### 3. Forgetting to Sample Multiple Actions

**Mistake**: Only using one random action per state

**Fix**: Sample `cql_n_actions` (default 10) per state

### 4. Incorrect Logsumexp

**Mistake**:
```python
cql_loss = torch.max(q_cat).mean()  # Hard max, not soft
```

**Fix**:
```python
cql_loss = torch.logsumexp(q_cat, dim=1).mean()  # Soft max
```

## Usage Example

```python
from nexus.models.rl import CQLAgent

config = {
    "state_dim": 17,
    "action_dim": 6,
    "hidden_dim": 256,
    "cql_alpha": 5.0,           # Conservatism strength
    "cql_n_actions": 10,        # Actions to sample
    "cql_lagrange": False,      # Auto-tune alpha
    "cql_target_action_gap": 5.0,
    "alpha": 0.2,               # SAC temperature
    "auto_alpha": True,         # Auto-tune SAC alpha
    "discount": 0.99,
    "tau": 0.005,
}

agent = CQLAgent(config)

for batch in dataset:
    metrics = agent.update(batch)
    print(f"CQL Loss: {metrics['cql_loss']:.3f}, Q: {metrics['q_value']:.3f}")
```

## References

```bibtex
@inproceedings{kumar2020cql,
  title={Conservative Q-Learning for Offline Reinforcement Learning},
  author={Kumar, Aviral and Zhou, Aurick and Tucker, George and Levine, Sergey},
  booktitle={NeurIPS},
  year={2020}
}
```

**Extensions**:
- Cal-QL: Calibrates CQL for dataset quality
- CQL-REDQ: Combines CQL with randomized ensembles
- SQL: Soft Q-learning variant
