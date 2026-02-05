# Implicit Q-Learning (IQL)

**Paper**: [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169) (Kostrikov et al., ICLR 2022)

**Code**: `nexus/models/rl/iql.py`

## Overview & Motivation

Implicit Q-Learning (IQL) is an offline RL algorithm that avoids the extrapolation error problem by never querying the Q-function on out-of-distribution (OOD) actions. Instead of explicitly maximizing Q-values (which requires evaluating Q(s, a) for potentially unseen actions), IQL uses **expectile regression** to implicitly learn the maximum Q-value.

### Key Insight
Traditional offline RL:
```
V(s) = max_a Q(s, a)  ← requires querying Q on all actions, including OOD
```

IQL:
```
V(s) ≈ τ-expectile of Q(s, a) where a ~ behavior policy
      ← only uses in-dataset actions!
```

By setting τ close to 1 (e.g., 0.7-0.9), the expectile approximates the maximum without explicit maximization.

##

 Theoretical Background

### Expectile Regression

An **expectile** is an asymmetric generalization of the mean. The τ-expectile minimizes:

```
L_τ(x) = |τ - 1{x < expectile}| · (x - expectile)²
```

For τ = 0.5, this is the mean. For τ → 1, it approaches the maximum.

### Why Expectiles for Offline RL?

1. **No explicit max**: Avoids querying Q on OOD actions
2. **Smooth approximation**: More stable than hard max
3. **Theoretical guarantees**: Converges to optimal policy under realizability

### The IQL Objective

IQL learns three networks:
1. **V(s)**: Value function via expectile regression on Q-values
2. **Q(s,a)**: Twin Q-networks via standard TD learning
3. **π(a|s)**: Policy via advantage-weighted regression (AWR)

The key is that V and π only use actions from the dataset.

## Mathematical Formulation

### 1. Value Network Update (Expectile Regression)

```
L_V(θ) = E_{(s,a)~D} [ L_τ(Q_target(s,a) - V_θ(s)) ]

where:
  L_τ(δ) = |τ - 1{δ < 0}| · δ²
  Q_target(s,a) = min(Q₁(s,a), Q₂(s,a))  (twin Q minimum)
  τ ∈ (0.5, 1.0) is the expectile parameter
```

**Intuition**: V(s) learns to approximate the upper quantile of Q(s,a) for actions in the dataset.

### 2. Q-Network Update (Standard TD)

```
L_Q(θ) = E_{(s,a,r,s')~D} [ (Q_θ(s,a) - (r + γ V(s')))² ]
```

**Key difference from SAC/TD3**: The target uses V(s') instead of max_a' Q(s',a'), avoiding OOD queries.

### 3. Policy Update (Advantage-Weighted Regression)

```
L_π(θ) = E_{(s,a)~D} [ exp(β · A(s,a)) · -log π_θ(a|s) ]

where:
  A(s,a) = Q_target(s,a) - V(s)  (advantage)
  β > 0 is the inverse temperature
```

**Intuition**: The policy imitates actions with high advantage (Q > V), ignoring low-advantage actions.

## High-Level Intuition

Think of IQL as "smart behavior cloning":

1. **Standard BC**: Clone all actions equally
2. **AWR**: Clone actions weighted by advantage
3. **IQL**: Use expectile V to identify high-value actions without OOD queries

### Visual Analogy

Imagine a dataset with actions and Q-values:

```
Actions:  [a₁, a₂, a₃, a₄, a₅]
Q-values: [3,  1,  4,  2,  5]
```

- **Mean (τ=0.5)**: 3.0
- **0.7-expectile**: ≈ 4.2  ← weights toward high values
- **0.9-expectile**: ≈ 4.8  ← even closer to max
- **Max**: 5.0

IQL uses the expectile to approximate the max without explicitly computing it.

## Implementation Details

### Network Architecture

```python
class IQLAgent:
    def __init__(self, config):
        # Twin Q-networks
        self.q_network = IQLQNetwork(state_dim, action_dim, hidden_dim)
        self.q_target = copy(self.q_network)  # Target for stability

        # Value network
        self.v_network = IQLValueNetwork(state_dim, hidden_dim)

        # Gaussian policy
        self.policy = IQLGaussianPolicy(state_dim, action_dim, hidden_dim)
```

### Expectile Loss Implementation

```python
def expectile_loss(pred, target, expectile):
    """
    Asymmetric L2 loss for expectile regression.

    Args:
        pred: V(s) predictions
        target: Q(s,a) targets
        expectile: τ parameter (e.g., 0.7)
    """
    diff = target - pred
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * diff.pow(2)).mean()
```

When diff > 0 (target exceeds prediction), use weight τ (e.g., 0.7).
When diff < 0 (target below prediction), use weight (1-τ) (e.g., 0.3).
This makes the loss asymmetric, pulling predictions toward high values.

### Training Loop

```python
for batch in dataset:
    # 1. Update V using expectile regression
    v_loss = update_value(batch)

    # 2. Update Q using TD learning with V targets
    q_loss = update_q(batch)

    # 3. Update policy using AWR
    policy_loss = update_policy(batch)

    # 4. Soft update target Q-network
    soft_update(q_network, q_target, tau=0.005)
```

## Code Walkthrough

### Key Components from `nexus/models/rl/iql.py`

#### 1. Expectile Regression (Lines 151-165)

```python
def expectile_loss(pred: torch.Tensor, target: torch.Tensor, expectile: float):
    """Asymmetric L2 loss for expectile regression."""
    diff = target - pred
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * diff.pow(2)).mean()
```

This is the core innovation. The asymmetric weighting makes V(s) track high Q-values.

#### 2. Value Update (Lines 243-258)

```python
def update_value(self, batch):
    """Update value network using expectile regression."""
    states = batch["states"]
    actions = batch["actions"]

    with torch.no_grad():
        q_value = self.q_target.q_min(states, actions)  # Twin Q minimum

    v_value = self.v_network(states)
    v_loss = expectile_loss(v_value, q_value, self.expectile)

    self.v_optimizer.zero_grad()
    v_loss.backward()
    self.v_optimizer.step()

    return v_loss.item()
```

Uses **only dataset actions** - no max over actions needed!

#### 3. Q Update (Lines 260-279)

```python
def update_q(self, batch):
    """Update Q-networks using TD learning."""
    states, actions, rewards, next_states, dones = batch[...]

    with torch.no_grad():
        next_v = self.v_network(next_states)  # Use V, not max_a Q
        target_q = rewards + gamma * (1 - dones) * next_v

    q1, q2 = self.q_network(states, actions)
    q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

    [... optimize ...]
    return q_loss.item()
```

Standard TD learning, but targets use V(s') instead of max_a' Q(s',a').

#### 4. Policy Update (Lines 281-303)

```python
def update_policy(self, batch):
    """Update policy using advantage-weighted regression."""
    states = batch["states"]
    actions = batch["actions"]

    with torch.no_grad():
        v_value = self.v_network(states)
        q_value = self.q_target.q_min(states, actions)
        advantage = q_value - v_value

        # Exponential weighting with temperature
        weights = torch.exp(self.temperature * advantage)
        weights = torch.clamp(weights, max=100.0)  # Clip for stability

    # Weighted behavior cloning
    log_prob = self.policy.log_prob(states, actions)
    policy_loss = -(weights * log_prob).mean()

    [... optimize ...]
    return policy_loss.item()
```

AWR: imitate actions proportional to exp(advantage). High-advantage actions get high weight.

## Optimization Tricks

### 1. Expectile Parameter (τ)

**Range**: 0.7 - 0.9
- **0.7**: More conservative, better for diverse/suboptimal data
- **0.9**: More aggressive, better for near-expert data
- **Default**: 0.7 works well across datasets

**Tuning**: If policy is too conservative, increase τ. If unstable, decrease τ.

### 2. Temperature (β)

**Range**: 0.5 - 10.0
- **Low (0.5)**: Smoother weighting, more robust
- **High (10.0)**: Sharp weighting, focuses on best actions
- **Default**: 3.0 is a good starting point

**Formula**: `weight = exp(β · advantage)`

### 3. Weight Clipping

Always clip advantage weights to prevent numerical instability:
```python
weights = torch.clamp(weights, max=100.0)
```

Without clipping, exp(advantage) can explode for outlier high-advantage samples.

### 4. Network Architecture

- **Hidden dim**: 256 (standard), 512 (complex tasks)
- **Layers**: 2-3 MLP layers
- **Layer norm**: Helps stabilization (used in IQL paper)
- **Dropout**: Not typically used, but can help with overfitting

### 5. Learning Rates

- **V network**: 3e-4 (can be lower like 1e-4 for stability)
- **Q networks**: 3e-4
- **Policy**: 3e-4 (can be higher like 1e-3 if training is slow)

### 6. Soft Update (τ_target)

```python
tau = 0.005  # Standard value
target_param = tau * param + (1 - tau) * target_param
```

Lower tau = more stable but slower learning.

## Experiments & Results

### D4RL Benchmark Performance

| Dataset | IQL Score | CQL Score | BC Score |
|---------|-----------|-----------|----------|
| halfcheetah-medium | 47.4 | 44.0 | 42.6 |
| halfcheetah-medium-expert | 86.7 | 62.4 | 59.9 |
| walker2d-medium | 78.3 | 79.2 | 75.3 |
| walker2d-medium-expert | 109.6 | 98.7 | 107.5 |
| hopper-medium | 66.3 | 58.5 | 52.9 |
| hopper-medium-expert | 91.5 | 98.7 | 52.5 |

**Observations**:
- IQL excels on medium-expert (mixed quality) data
- Comparable to CQL but simpler (no hyperparameter α to tune)
- Significantly better than behavior cloning

### Advantages
1. **Simplicity**: No complex regularization terms
2. **Robustness**: Works across diverse dataset qualities
3. **Stability**: No OOD queries means fewer failure modes
4. **Speed**: Faster than methods with explicit policy constraints

### Limitations
1. **Sample efficiency**: Can be less sample-efficient than CQL on some tasks
2. **Hyperparameter sensitivity**: τ and β need tuning
3. **Gaussian policies**: Struggles with highly multi-modal behavior (use IDQL instead)

## Common Pitfalls

### 1. Wrong Expectile Direction

**Mistake**: Using τ < 0.5
```python
expectile = 0.3  # WRONG! This learns the minimum, not maximum
```

**Fix**: Always use τ > 0.5, typically 0.7-0.9.

### 2. Not Clipping Weights

**Mistake**: Unbounded advantage weights
```python
weights = torch.exp(temperature * advantage)  # Can explode!
```

**Fix**: Always clip
```python
weights = torch.clamp(torch.exp(temperature * advantage), max=100.0)
```

### 3. Using V(s) for Policy Targets

**Mistake**: Training Q to match V instead of vice versa
```python
target = v_network(next_states)  # WRONG direction
q_loss = mse_loss(q_network(s, a), target)
```

**Fix**: Train V to match Q (via expectile), then use V in Q targets.

### 4. Too Large Temperature

**Mistake**: β = 50 causes numerical issues
```python
weights = torch.exp(50 * advantage)  # Explodes even with small advantages
```

**Fix**: Start with β = 3.0, increase gradually if needed.

### 5. Forgetting Target Network Updates

**Mistake**: Not updating Q target network
```python
# Missing: soft_update(q_network, q_target)
```

**Fix**: Always soft-update after each Q update:
```python
for param, target_param in zip(q_network.parameters(), q_target.parameters()):
    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

### 6. Incorrect Advantage Calculation

**Mistake**: Not detaching advantage for policy loss
```python
policy_loss = -(advantage * log_prob).mean()  # Gradients flow through advantage!
```

**Fix**: Detach or compute advantage in no_grad
```python
with torch.no_grad():
    advantage = q_value - v_value
policy_loss = -(advantage * log_prob).mean()
```

## Usage Example

```python
from nexus.models.rl import IQLAgent
import torch

# Configure
config = {
    "state_dim": 17,
    "action_dim": 6,
    "hidden_dim": 256,
    "expectile": 0.7,          # Expectile for V learning
    "temperature": 3.0,        # AWR temperature
    "discount": 0.99,
    "tau": 0.005,              # Soft update rate
    "q_lr": 3e-4,
    "v_lr": 3e-4,
    "policy_lr": 3e-4,
}

agent = IQLAgent(config)

# Training
for epoch in range(1000):
    for batch in dataloader:
        # batch: {states, actions, rewards, next_states, dones}
        metrics = agent.update(batch)

    if epoch % 10 == 0:
        eval_return = evaluate(agent, env, episodes=10)
        print(f"Epoch {epoch}: Return = {eval_return}")

# Inference
state = env.reset()
action = agent.select_action(state, deterministic=True)
```

## References

### Primary Paper
```bibtex
@inproceedings{kostrikov2022iql,
  title={Offline Reinforcement Learning with Implicit Q-Learning},
  author={Kostrikov, Ilya and Nair, Ashvin and Levine, Sergey},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}
}
```

### Related Work
- **Expectile Regression**: Newey & Powell, 1987 (original asymmetric least squares)
- **AWR**: Peng et al., 2019 (advantage-weighted policy learning)
- **CQL**: Kumar et al., 2020 (alternative offline RL approach)
- **IDQL**: Hansen-Estruch et al., 2023 (extends IQL with diffusion policies)

### Code & Resources
- [Original Implementation](https://github.com/ikostrikov/implicit_q_learning)
- [D4RL Benchmark](https://github.com/Farama-Foundation/D4RL)
- [CleanRL IQL](https://github.com/vwxyzjn/cleanrl) - Clean reference implementation
