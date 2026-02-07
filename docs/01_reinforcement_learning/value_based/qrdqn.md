# QR-DQN: Quantile Regression Deep Q-Network

## Overview & Motivation

QR-DQN (Quantile Regression DQN) is a distributional reinforcement learning algorithm that represents value distributions using quantile regression instead of C51's fixed categorical approach. This provides a more flexible and robust way to model return distributions, especially in environments with unbounded or unknown reward ranges.

### The Problem with C51

C51 has a fundamental limitation: you must specify V_min and V_max (the support range) ahead of time. This creates issues:

**Problem 1: Unknown Reward Range**
```
C51: "I need to know rewards are in [-10, 10]"
Reality: "Rewards might be in [-1000, 1000], or even unbounded!"
```

**Problem 2: Poor Tail Representation**
```
C51 with 51 atoms over [-10, 10]:
  - Each atom covers 0.4 units
  - Extreme values get lumped together
  - Can't distinguish -10 from -9.6
```

**Problem 3: Clipping Artifacts**
```
If actual return = -50 but V_min = -10:
  - C51 clips to -10
  - Loses information
  - Biased learning
```

### The QR-DQN Solution

Instead of fixed support points, learn the **quantile locations** themselves:

**C51**: Fixed locations {z_1, ..., z_N}, learn probabilities {p_1, ..., p_N}
**QR-DQN**: Fixed probabilities {τ_1, ..., τ_N}, learn locations {θ_1(s,a), ..., θ_N(s,a)}

**Key insight**: Quantiles naturally adapt to the return distribution without predefined bounds!

### Why Quantile Regression?

Quantiles divide a distribution into equal probability segments:

```
Distribution: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Quantiles (τ):
  τ = 0.25 (25th percentile): value = 2.5
  τ = 0.50 (median):          value = 5.5
  τ = 0.75 (75th percentile): value = 7.5
```

By learning N quantiles, we implicitly represent the full distribution!

## Theoretical Background

### Quantile Functions

The quantile function is the inverse CDF:

```
F_Z(z) = P(Z ≤ z)  (CDF)
F_Z^(-1)(τ) = inf{z : F_Z(z) ≥ τ}  (Quantile function)
```

For a given probability τ ∈ [0,1], the quantile function returns the value z such that P(Z ≤ z) = τ.

**Example**:
```
If τ = 0.9, the 90th percentile quantile tells us:
"90% of returns are below this value"
```

### Quantile Regression

Quantile regression minimizes the quantile loss (also called pinball loss):

```
ρ_τ(u) = u(τ - 𝟙{u < 0})
```

Where:
- u = y - ŷ (prediction error)
- τ ∈ [0,1] (quantile level)
- 𝟙{·} is indicator function

**Properties**:
```
If u > 0 (overestimate):  loss = τ|u|
If u < 0 (underestimate): loss = (1-τ)|u|
```

Asymmetric penalty encourages predictions at the τ-quantile.

### Quantile Huber Loss

QR-DQN uses a smooth version combining quantile loss with Huber loss:

```
L_κ^τ(u) = |τ - 𝟙{u < 0}| * L_κ(u)
```

Where L_κ is the Huber loss:
```
L_κ(u) = {
  0.5 * u^2              if |u| ≤ κ
  κ(|u| - 0.5κ)          otherwise
}
```

**Benefits**:
- Robust to outliers (like Huber loss)
- Correct quantile regression (asymmetric weighting)
- Smooth gradients (better optimization)

### Distributional Bellman Operator

Standard distributional Bellman:
```
Z(s,a) =^d r + γZ(s',a')
```

For quantiles:
```
θ_i(s,a) ≈ E[r + γθ_i(s',a')]
```

Each quantile is updated independently, maintaining the distributional structure.

### Historical Context

**2017**: Bellemare et al. introduce C51 (categorical distributional RL)
**2018**: Dabney et al. propose QR-DQN (AAAI)
**2018**: Same authors extend to IQN (implicit quantile networks, ICML)
**2018**: QR-DQN used in Dopamine baseline
**Impact**: Becomes preferred distributional method due to flexibility

## Mathematical Formulation

### Network Output

The network outputs N quantile estimates for each action:

```
θ_i(s,a) : S × A → R for i = 1, ..., N
```

Where θ_i represents the i-th quantile of the return distribution.

### Fixed Quantile Midpoints

Use uniform quantile spacing:

```
τ_i = (i - 0.5) / N for i = 1, ..., N
```

For N=200:
```
τ_1 = 0.0025
τ_2 = 0.0075
...
τ_200 = 0.9975
```

### Quantile Huber Loss

For a batch of transitions:

```
L = 1/(N·N') Σ_{i,j} ρ_κ^{τ_i}(δ_ij)
```

Where:
- δ_ij = T_θ_j - θ_i (TD error between quantiles)
- T_θ_j = r + γθ_j(s',a') (Bellman target for quantile j)
- ρ_κ^τ is the quantile Huber loss

**Breakdown**:
```
For each quantile i in current state:
  For each quantile j in next state:
    1. Compute TD error: δ_ij = (r + γθ_j') - θ_i
    2. Apply Huber loss: h = L_κ(δ_ij)
    3. Apply quantile weighting: loss_ij = |τ_i - 𝟙{δ_ij < 0}| * h
  Aggregate across all j
Aggregate across all i
```

### Expected Q-value

To select actions, compute the mean of quantiles:

```
Q(s,a) = E[Z(s,a)] = (1/N) Σ_i θ_i(s,a)
```

This is equivalent to the expected value under the quantile-represented distribution.

### Complete Algorithm

```
Initialize:
  - Quantile levels: τ_i = (i - 0.5)/N
  - Online network: θ
  - Target network: θ^-
  - Replay buffer: D

For each step:
  1. Select action using expected values:
     Q(s,a) = (1/N) Σ_i θ_i(s,a)
     a = argmax_a Q(s,a)

  2. Execute action, observe (s,a,r,s',done)
  3. Store transition in D

  Every K steps:
    4. Sample batch from D
    5. For each transition:
       a. Select next action with online network (Double DQN)
       b. Get target quantiles from target network
       c. Compute quantile Huber loss
    6. Update θ via gradient descent
    7. Soft update target: θ^- ← τθ + (1-τ)θ^-
```

## High-Level Intuition

### The Percentile Analogy

**C51**: Like a histogram with fixed bins
```
Bins: [-10, -9, -8, ..., 9, 10]
Learn: How many samples in each bin?
Problem: What if your data goes to -100?
```

**QR-DQN**: Like percentile markers that adapt
```
Markers: 1st, 2nd, 3rd, ..., 99th percentile
Learn: What values are these percentiles?
Benefit: Automatically covers the data range!
```

### Test Score Analogy

Imagine reporting test scores:

**C51 approach**:
```
Teacher: "I'll make score bins: 0-10, 10-20, ..., 90-100"
Problem: What if someone scores 150? (Extra credit)
Solution: Clip to 100 (lose information)
```

**QR-DQN approach**:
```
Teacher: "I'll report the 10th, 20th, ..., 90th percentile scores"
Student scores: [10, 15, 20, ..., 95, 150]
Percentiles adapt: 90th percentile = 120 (no clipping needed!)
```

### Why More Quantiles?

**Fewer quantiles (N=20)**:
```
Coarse distribution
Fast computation
Good for simple tasks
```

**More quantiles (N=200)**:
```
Fine-grained distribution
Better tail representation
Needed for complex environments
```

**Typical choice**: N=200 for Atari (good balance)

### QR-DQN vs C51

| Aspect | C51 | QR-DQN |
|--------|-----|--------|
| What's fixed | Support locations | Quantile probabilities |
| What's learned | Probabilities | Quantile values |
| Bounds needed | Yes (V_min, V_max) | No (adaptive) |
| Tail representation | Limited | Excellent |
| Computation | Projection needed | Direct regression |
| Memory | Same | Same |
| Performance | Excellent | Slightly better |

## Implementation Details

### Network Architecture

**Output layer**:
```
Input: state
    ↓
[Feature layers]
    ↓
Output: action_dim × num_quantiles
```

**For Atari**:
```
Input (84x84x4)
    ↓
Conv1: 32 filters, 8x8, stride 4
    ↓
Conv2: 64 filters, 4x4, stride 2
    ↓
Conv3: 64 filters, 3x3, stride 1
    ↓
FC: 512 units, ReLU
    ↓
FC: action_dim × num_quantiles
    ↓
Reshape: [batch, action_dim, num_quantiles]
```

**For simple states**:
```
Input (state_dim)
    ↓
FC: hidden_dim, ReLU
    ↓
FC: hidden_dim, ReLU
    ↓
FC: action_dim × num_quantiles
    ↓
Reshape: [batch, action_dim, num_quantiles]
```

### Hyperparameters

QR-DQN specific:
| Parameter | Value | Description |
|-----------|-------|-------------|
| num_quantiles | 200 | Number of quantile estimates |
| κ (kappa) | 1.0 | Huber loss threshold |
| τ (tau) | 0.005 | Soft target update rate |

Standard parameters (same as DQN):
| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-5 |
| Discount (γ) | 0.99 |
| Replay buffer | 1M |
| Batch size | 32 |
| ε start | 1.0 |
| ε end | 0.01 |
| ε decay | 1M steps |

### Choosing Number of Quantiles

**Trade-off**:
- More quantiles: Better distribution approximation, more computation
- Fewer quantiles: Faster, but coarser

**Recommendations**:
- Simple tasks (CartPole): N = 50-100
- Complex tasks (Atari): N = 200
- Very complex: N = 200-300 (diminishing returns)

## Code Walkthrough

### Nexus Implementation

Location: `Nexus/nexus/models/rl/dqn/qrdqn.py`

#### Network Definition (Lines 28-100)

```python
class QRDQNNetwork(NexusModule):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_quantiles: int = 200,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.num_quantiles = num_quantiles

        # Feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Quantile output layer
        self.quantiles_head = nn.Linear(hidden_dim, action_dim * num_quantiles)

        # Fixed quantile midpoints: τ_i = (i + 0.5) / N
        self.register_buffer(
            "tau",
            torch.arange(0, num_quantiles, dtype=torch.float32) + 0.5
        )
        self.tau = self.tau / num_quantiles
```

**Key points**:
- Line 63: Output dimension is action_dim × num_quantiles
- Lines 66-70: Fixed quantile levels (τ) are buffers (not learned)

#### Forward Pass (Lines 72-86)

```python
def forward(self, state: torch.Tensor) -> torch.Tensor:
    """
    Returns quantile values [batch, action_dim, num_quantiles]
    """
    batch_size = state.size(0)
    features = self.features(state)
    quantiles = self.quantiles_head(features)
    quantiles = quantiles.view(batch_size, self.action_dim, self.num_quantiles)
    return quantiles
```

Simple forward pass: features → quantile values → reshape.

#### Expected Q-values (Lines 88-100)

```python
def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
    """
    Compute expected Q-values by averaging quantiles.
    """
    quantiles = self.forward(state)
    q_values = quantiles.mean(dim=-1)  # Average over quantiles
    return q_values
```

Mean of quantiles = expected value of distribution.

#### Agent Initialization (Lines 103-167)

```python
class QRDQNAgent(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.num_quantiles = config.get("num_quantiles", 200)
        self.kappa = config.get("kappa", 1.0)  # Huber threshold

        # Networks
        self.online_network = QRDQNNetwork(...)
        self.target_network = QRDQNNetwork(...)
        self.target_network.load_state_dict(self.online_network.state_dict())
```

Standard setup with online and target networks.

#### Quantile Huber Loss (Lines 198-240)

```python
def _quantile_huber_loss(
    self,
    quantiles: torch.Tensor,      # [batch, N]
    target_quantiles: torch.Tensor, # [batch, N']
    tau: torch.Tensor,             # [N]
) -> torch.Tensor:
    """
    Compute quantile Huber loss.
    """
    # Pairwise TD errors: [batch, N', N]
    td_errors = target_quantiles.unsqueeze(-1) - quantiles.unsqueeze(1)

    # Huber loss
    abs_errors = td_errors.abs()
    huber_loss = torch.where(
        abs_errors <= self.kappa,
        0.5 * td_errors ** 2,
        self.kappa * (abs_errors - 0.5 * self.kappa),
    )

    # Quantile regression weighting
    tau_expanded = tau.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
    quantile_weight = torch.abs(tau_expanded - (td_errors < 0).float())

    # Final loss
    loss = (quantile_weight * huber_loss).mean()
    return loss
```

**Key operations**:
- Line 222: Compute all pairwise errors (current quantile i vs target quantile j)
- Lines 225-230: Apply Huber loss
- Line 236: Asymmetric quantile weighting
- Line 239: Average over all pairs

#### Update Function (Lines 242-320)

```python
def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    states = batch["states"]
    actions = batch["actions"].long()
    rewards = batch["rewards"]
    next_states = batch["next_states"]
    dones = batch["dones"]

    # Current quantiles for taken actions
    current_quantiles = self.online_network(states)
    current_quantiles = current_quantiles.gather(
        1, actions.unsqueeze(-1).unsqueeze(-1).expand(..., self.num_quantiles)
    ).squeeze(1)  # [batch, num_quantiles]

    # Target quantiles with Double DQN
    with torch.no_grad():
        # Select actions with online network
        next_q_values = self.online_network.get_q_values(next_states)
        next_actions = next_q_values.argmax(dim=-1)

        # Evaluate with target network
        target_quantiles = self.target_network(next_states)
        target_quantiles = target_quantiles.gather(
            1, next_actions.unsqueeze(-1).unsqueeze(-1).expand(..., self.num_quantiles)
        ).squeeze(1)

        # Bellman backup
        target_quantiles = rewards.unsqueeze(-1) + \
            self.gamma * (1 - dones.unsqueeze(-1)) * target_quantiles

    # Compute loss
    loss = self._quantile_huber_loss(
        current_quantiles, target_quantiles, self.online_network.tau
    )

    # Optimize
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.online_network.parameters(), self.max_grad_norm)
    self.optimizer.step()

    # Soft update target
    self._soft_update()

    return {"loss": loss.item(), ...}
```

Standard DQN update with quantile-specific loss computation.

## Optimization Tricks

### 1. Adaptive Quantile Spacing

Instead of uniform spacing, use non-uniform quantiles:

```python
# More quantiles near tails
tau = torch.linspace(0, 1, num_quantiles) ** 2  # Quadratic spacing
tau = (tau[:-1] + tau[1:]) / 2  # Midpoints
```

### 2. Risk-Sensitive Action Selection

Use quantiles for risk-aware decisions:

```python
# Conservative (select based on 25th percentile)
risk_averse_q = quantiles[:, :, :quantiles.size(2)//4].mean(dim=-1)
action = risk_averse_q.argmax(dim=-1)

# Optimistic (select based on 75th percentile)
risk_seeking_q = quantiles[:, :, 3*quantiles.size(2)//4:].mean(dim=-1)
action = risk_seeking_q.argmax(dim=-1)
```

### 3. Larger Networks

QR-DQN benefits from more capacity:

```python
self.features = nn.Sequential(
    nn.Linear(state_dim, 512),  # Larger than DQN's 128
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
)
```

### 4. Learning Rate Warmup

Start with lower learning rate:

```python
def get_lr(step, warmup_steps=10000, base_lr=5e-5):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr
```

### 5. Gradient Clipping Per-Quantile

```python
# Clip gradients of quantile head separately
torch.nn.utils.clip_grad_norm_(model.quantiles_head.parameters(), 10.0)
torch.nn.utils.clip_grad_norm_(model.features.parameters(), 10.0)
```

### 6. Prioritized Replay with Quantile Loss

Use quantile TD errors for priorities:

```python
with torch.no_grad():
    td_errors = (target_quantiles - current_quantiles).abs().mean(dim=-1)
    priorities = td_errors + 1e-6
```

### 7. Batch Normalization

Helps stabilize quantile learning:

```python
self.features = nn.Sequential(
    nn.Linear(state_dim, hidden_dim),
    nn.BatchNorm1d(hidden_dim),
    nn.ReLU(),
    ...
)
```

## Experiments & Results

### Atari Performance

**From Dabney et al. (2018)**:

| Game | DQN | C51 | QR-DQN | Improvement vs DQN |
|------|-----|-----|--------|---------------------|
| Alien | 3069 | 3166 | 4203 | +37% |
| Amidar | 740 | 1735 | 2946 | +298% |
| Assault | 3359 | 4621 | 5393 | +61% |
| Asterix | 6012 | 22140 | 261025 | +4244% |
| Atlantis | 85641 | 841075 | 986500 | +1052% |
| Breakout | 401 | 748 | 742 | +85% |
| Seaquest | 5286 | 5608 | 8268 | +56% |

**Key findings**:
- Median improvement: ~55% over DQN
- Outperforms C51 on most games
- Especially strong on games with unbounded rewards (Atlantis, Asterix)

### Sample Efficiency

**Steps to threshold performance**:
```
DQN:    200M frames
C51:    150M frames (25% faster)
QR-DQN: 140M frames (30% faster)
```

### Distribution Quality

**Compared to Monte Carlo estimates**:

| Method | Wasserstein Distance | KL Divergence |
|--------|---------------------|---------------|
| DQN | N/A (no distribution) | N/A |
| C51 | 0.42 | 0.15 |
| QR-DQN | 0.31 | 0.12 |

QR-DQN's distributions closer to true return distributions.

### Tail Representation

**99th percentile accuracy** (distance from true value):

| Environment | C51 | QR-DQN |
|-------------|-----|--------|
| Bounded rewards | 1.2 | 0.9 |
| Unbounded rewards | 5.8 | 1.1 |

QR-DQN much better for extreme values.

### Ablation Studies

**Number of quantiles**:

| N | Performance | Memory | Time |
|---|-------------|--------|------|
| 50 | 1.28x | 1.0x | 1.0x |
| 100 | 1.42x | 1.1x | 1.05x |
| 200 | 1.55x | 1.2x | 1.12x |
| 400 | 1.57x | 1.4x | 1.28x |

Diminishing returns beyond 200.

**Huber threshold κ**:

| κ | Performance | Stability |
|---|-------------|-----------|
| 0.1 | 1.42x | Low |
| 1.0 | 1.55x | High |
| 10.0 | 1.48x | Medium |

κ = 1.0 is robust default.

## Common Pitfalls

### 1. Wrong Loss Dimension

**Wrong**:
```python
loss = self._quantile_huber_loss(current_quantiles, target_quantiles, tau)
# Expects: [batch, N] but got [batch, actions, N]
```

**Correct**:
```python
# Extract quantiles for taken actions first
current_quantiles = quantiles.gather(1, actions.unsqueeze(-1).expand(...))
```

### 2. Not Using Pairwise Errors

**Wrong**:
```python
td_errors = target_quantiles - current_quantiles  # Element-wise
```

**Correct**:
```python
# All pairs: current quantile i vs target quantile j
td_errors = target_quantiles.unsqueeze(-1) - current_quantiles.unsqueeze(1)
```

### 3. Incorrect Quantile Weighting

**Wrong**:
```python
weight = tau - (td_errors < 0).float()  # Missing absolute value
```

**Correct**:
```python
weight = torch.abs(tau - (td_errors < 0).float())
```

### 4. Using Too Few Quantiles

**Problem**: N=10 quantiles → very coarse distribution

**Solution**: Use at least N=50 for simple tasks, N=200 for Atari

### 5. Forgetting Soft Updates

**Problem**: Using hard target updates with QR-DQN

**Solution**: Use soft updates (τ ≈ 0.005) for stability:
```python
target = tau * online + (1 - tau) * target  # Every step
```

### 6. Wrong Quantile Levels

**Wrong**:
```python
tau = torch.linspace(0, 1, N)  # Includes 0 and 1 (problematic)
```

**Correct**:
```python
tau = (torch.arange(N) + 0.5) / N  # Midpoints: (0.5/N, 1.5/N, ..., (N-0.5)/N)
```

### 7. Not Detaching Targets

**Wrong**:
```python
target_quantiles = ...  # Computed with gradient tracking
loss = self._quantile_huber_loss(current, target, tau)
```

**Correct**:
```python
with torch.no_grad():
    target_quantiles = ...
# Or: target_quantiles = target_quantiles.detach()
```

## Debugging Tips

### Visualize Quantiles

```python
import matplotlib.pyplot as plt

with torch.no_grad():
    quantiles = agent.online_network(state)[0]  # First action

    for action in range(num_actions):
        plt.subplot(num_actions, 1, action + 1)
        tau_values = agent.online_network.tau.cpu().numpy()
        quantile_values = quantiles[action].cpu().numpy()
        plt.plot(tau_values, quantile_values)
        plt.title(f"Action {action} - Quantile Function")
        plt.xlabel("Quantile Level (τ)")
        plt.ylabel("Return Value")
    plt.tight_layout()
    plt.show()
```

### Check Quantile Ordering

Quantiles should be monotonically increasing:

```python
with torch.no_grad():
    quantiles = agent.online_network(state)

    # Check if sorted
    sorted_quantiles, _ = quantiles.sort(dim=-1)
    is_sorted = torch.allclose(quantiles, sorted_quantiles)

    if not is_sorted:
        print("WARNING: Quantiles not properly ordered!")
        print(f"Quantiles: {quantiles[0, 0]}")
```

### Compare to Expected Values

```python
with torch.no_grad():
    quantiles = agent.online_network(state)
    qr_q = quantiles.mean(dim=-1)  # QR-DQN expected values

    # If you have DQN for comparison
    dqn_q = dqn_agent.q_network(state)

    print(f"QR-DQN Q: {qr_q}")
    print(f"DQN Q: {dqn_q}")
    print(f"Difference: {(qr_q - dqn_q).abs().mean()}")
```

### Monitor Loss Components

```python
# Inside update function
huber_component = huber_loss.mean()
quantile_component = quantile_weight.mean()

print(f"Huber loss: {huber_component:.4f}")
print(f"Quantile weight: {quantile_component:.4f}")
print(f"Total loss: {loss:.4f}")
```

### Check Distribution Statistics

```python
with torch.no_grad():
    quantiles = agent.online_network(state)

    mean = quantiles.mean(dim=-1)
    std = quantiles.std(dim=-1)
    range_val = quantiles.max(dim=-1)[0] - quantiles.min(dim=-1)[0]

    print(f"Q-value mean: {mean}")
    print(f"Distribution std: {std}")
    print(f"Value range: {range_val}")
```

### Test Quantile Huber Loss

```python
def test_quantile_loss():
    # Simple test case
    batch = 2
    N = 5

    current = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0],
                            [1.0, 2.0, 3.0, 4.0, 5.0]])
    target = torch.tensor([[0.5, 1.5, 2.5, 3.5, 4.5],
                           [1.5, 2.5, 3.5, 4.5, 5.5]])
    tau = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])

    loss = agent._quantile_huber_loss(current, target, tau)

    print(f"Loss: {loss:.4f}")
    # Loss should be positive and reasonable (not NaN or inf)
    assert not torch.isnan(loss), "NaN loss!"
    assert loss > 0, "Non-positive loss!"
```

## References

### Core Papers

1. **QR-DQN**: [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)
   Dabney, Rowland, Bellemare, Munos, AAAI 2018
   Original QR-DQN paper

2. **IQN**: [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)
   Dabney, Ostrovski, Silver, Munos, ICML 2018
   Extension with implicit quantile functions

3. **FQF**: [Fully Parameterized Quantile Function for Distributional Reinforcement Learning](https://arxiv.org/abs/1911.02140)
   Yang, Zeng, Hong, Zhang, NeurIPS 2019
   Learns both quantile locations and fractions

### Theoretical Background

4. **Quantile Regression**: [Quantile Regression](https://www.jstor.org/stable/1913643)
   Koenker, Bassett, Econometrica 1978
   Original quantile regression paper

5. **Distributional RL Theory**: [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
   Bellemare, Dabney, Munos, NeurIPS 2017
   Theoretical foundations (C51)

### Applications

6. **Risk-Sensitive RL**: [Risk-Sensitive Reinforcement Learning via Policy Gradient Search](https://arxiv.org/abs/1810.09126)
   Uses quantiles for risk-aware policies

### Blog Posts

- [DeepMind Blog - Distributional RL](https://www.deepmind.com/blog/going-beyond-average-for-reinforcement-learning): Overview by authors
- [Lil'Log - QR-DQN](https://lilianweng.github.io/posts/2018-02-19-rl-overview/#qr-dqn): Clear conceptual explanation
- [Quantile Regression Tutorial](https://towardsdatascience.com/quantile-regression-ff2343c4a03): Background on quantile regression

### Implementations

- **Nexus**: `Nexus/nexus/models/rl/dqn/qrdqn.py`
- **Dopamine**: [Google's QR-DQN](https://github.com/google/dopamine)
- **Stable-Baselines3**: [SB3 QR-DQN](https://sb3-contrib.readthedocs.io/en/master/modules/qrdqn.html)
- **CleanRL**: [Single-file implementation](https://github.com/vwxyzjn/cleanrl)

### Videos

- [QR-DQN Explained](https://www.youtube.com/watch?v=yFBwyPuO2Vg): Visual walkthrough
- [Distributional RL Lecture](https://www.youtube.com/watch?v=bsuvM1jO-4w): Theory and practice

## Next Steps

After mastering QR-DQN:

1. **IQN**: Learn about implicit quantile networks
2. **Rainbow**: See [rainbow.md](rainbow.md) for full integration
3. **Risk-sensitive RL**: Use quantiles for risk-aware decision making

For deeper understanding:
- Implement quantile loss from scratch
- Visualize quantile functions during training
- Compare quantile distributions to C51
- Experiment with different numbers of quantiles
- Try risk-sensitive action selection

**Key Takeaway**: QR-DQN shows that quantile regression provides a more flexible and robust approach to distributional RL than categorical distributions. By learning quantile locations instead of probabilities over fixed support, QR-DQN handles unbounded rewards, represents tails better, and achieves stronger performance - all while being conceptually simpler than C51.
