# Sophia Optimizer: Second-Order Clipped Stochastic Optimization

## Table of Contents
- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Algorithm Details](#algorithm-details)
- [Implementation](#implementation)
- [Performance Analysis](#performance-analysis)
- [Usage Guide](#usage-guide)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Comparison with Other Optimizers](#comparison-with-other-optimizers)
- [Advanced Topics](#advanced-topics)
- [References](#references)

## Overview

Sophia (Second-order Clipped Stochastic Optimization) is a lightweight second-order optimizer specifically designed for large language model pre-training. Developed by Liu et al. (2023), Sophia achieves 2x faster convergence compared to Adam while maintaining minimal computational overhead.

### Key Features

- **Second-Order Information**: Uses diagonal Hessian estimates for element-wise adaptive learning rates
- **Minimal Overhead**: Hessian updates amortized over k steps (typically 10)
- **Robust Clipping**: Element-wise gradient clipping prevents large updates in high-curvature directions
- **Two Estimation Methods**:
  - Gauss-Newton: Simple but biased
  - Hutchinson: Unbiased stochastic estimate
- **Proven Performance**: 2x faster wall-clock training time on GPT-2 scale models

### When to Use Sophia

**Best for:**
- Large language model pre-training (>100M parameters)
- Training from scratch (not fine-tuning)
- Long training runs where convergence speed matters
- Transformer architectures

**Not recommended for:**
- Small models (<10M parameters) - overhead outweighs benefits
- Fine-tuning tasks - Adam typically sufficient
- Computer vision models - benefits less pronounced
- Very small batch sizes (<256) - Hessian estimates less reliable

## Mathematical Foundation

### Core Optimization Problem

We want to minimize a loss function $L(\theta)$ with respect to parameters $\theta$. Standard first-order optimizers like SGD and Adam use only gradient information:

$$\theta_{t+1} = \theta_t - \alpha_t \cdot m_t$$

where $m_t$ is some function of the gradient (e.g., momentum-smoothed gradient).

### Second-Order Methods

Second-order methods incorporate curvature information via the Hessian matrix $H = \nabla^2 L(\theta)$:

$$\theta_{t+1} = \theta_t - \alpha_t \cdot H^{-1} \nabla L(\theta_t)$$

**Benefits:**
- Adapts step size to local curvature
- Faster convergence in many cases

**Challenges:**
- Computing full Hessian: $O(d^2)$ memory, $O(d^3)$ inversion
- For LLMs with billions of parameters: completely infeasible

### Sophia's Approximation

Sophia uses a **diagonal Hessian approximation** $h \in \mathbb{R}^d$:

$$\theta_{t+1} = \theta_t - \alpha_t \cdot \frac{m_t}{\max(h_t, \rho)}$$

Where:
- $m_t$: First moment (EMA of gradients)
- $h_t$: Diagonal Hessian estimate
- $\rho$: Clipping threshold (typically 0.04)

This provides element-wise adaptive learning rates while requiring only $O(d)$ memory.

### Diagonal Hessian Estimation

#### Method 1: Gauss-Newton Approximation

The simplest approach approximates the diagonal Hessian as:

$$h_i \approx (\nabla_{\theta_i} L)^2$$

**Update rule:**
$$h_t = \beta_2 h_{t-1} + (1-\beta_2) g_t^2$$

where $g_t = \nabla L(\theta_t)$.

**Pros:**
- Simple to implement
- No additional backward passes
- Low overhead

**Cons:**
- Biased estimate
- Only approximates for certain loss functions

#### Method 2: Hutchinson Estimator

For a more accurate (unbiased) estimate, Hutchinson's method uses random projections:

$$\text{diag}(H) = \mathbb{E}_v[v \odot (H v)]$$

where $v \sim \text{Rademacher}(\pm 1)$ and $\odot$ denotes element-wise product.

**Update rule:**
1. Sample $v$ with each element $\pm 1$ with equal probability
2. Compute $H v$ via automatic differentiation
3. Update: $h_t = \beta_2 h_{t-1} + (1-\beta_2) v \odot (H v)$

**Pros:**
- Unbiased estimate
- Theoretically sound

**Cons:**
- Requires additional backward pass
- Higher computational cost (still much cheaper than full Hessian)

### Clipping Mechanism

The key innovation in Sophia is **element-wise clipping** via the denominator $\max(h_t, \rho)$:

$$\text{update}_i = \frac{m_{t,i}}{\max(h_{t,i}, \rho)}$$

**Intuition:**
- **High curvature** ($h_i$ large): Divide by large value → small step
- **Low curvature** ($h_i$ small): Clipped by $\rho$ → larger step (but bounded)
- **Flat regions** ($h_i \approx 0$): Clipped by $\rho$ → prevents explosion

This provides a smooth, element-wise adaptive learning rate that's robust to varying curvatures.

### Complete Algorithm

**Parameters:**
- Learning rate: $\alpha$ (typically $1e-4$ to $3e-4$)
- Beta coefficients: $\beta_1, \beta_2$ (typically $0.965, 0.99$)
- Clipping threshold: $\rho$ (typically $0.04$)
- Hessian update interval: $k$ (typically $10$)

**Pseudocode:**
```
Initialize: m₀ = 0, h₀ = 0

For t = 1, 2, ... do:
    1. Compute gradient: gₜ = ∇L(θₜ)

    2. Update first moment:
       mₜ = β₁ mₜ₋₁ + (1 - β₁) gₜ

    3. Update Hessian estimate (every k steps):
       If t mod k == 0:
           hₜ = β₂ hₜ₋₁ + (1 - β₂) gₜ²  # Gauss-Newton
           # OR
           Sample v ~ Rademacher
           hₜ = β₂ hₜ₋₁ + (1 - β₂) v ⊙ (H v)  # Hutchinson
       Else:
           hₜ = hₜ₋₁

    4. Compute update:
       updateₜ = mₜ / max(hₜ, ρ)

    5. Apply weight decay (decoupled):
       θₜ₊₁ = (1 - α · λ) θₜ - α · updateₜ
```

## Algorithm Details

### Momentum vs. Hessian Update Frequencies

**Momentum** ($m_t$): Updated every step
- Tracks recent gradient direction
- Low-pass filter for gradient noise
- Critical for convergence stability

**Hessian** ($h_t$): Updated every $k$ steps (typically $k=10$)
- Curvature changes slowly in practice
- Amortizes overhead across steps
- Still provides effective adaptive learning rates

This design achieves near-Adam computational cost with second-order benefits.

### Bias Correction

Unlike Adam, Sophia typically doesn't use bias correction for $m_t$ and $h_t$ in the original paper. However, implementations may add it for early training stability:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{h}_t = \frac{h_t}{1 - \beta_2^t}$$

This is optional and depends on your training setup.

### Memory Requirements

**Per parameter** (float32):
- Momentum $m$: 4 bytes
- Hessian diagonal $h$: 4 bytes
- **Total**: 8 bytes per parameter

**Comparison:**
- Adam: 8 bytes per parameter (same!)
- SGD with momentum: 4 bytes per parameter
- Full second-order: $O(d^2)$ bytes (infeasible)

**Conclusion**: Sophia has identical memory footprint to Adam.

### Computational Overhead

**Per step (when Hessian not updated):**
- Gradient computation: Same as Adam
- Momentum update: Same as Adam
- Adaptive step: Division instead of sqrt+division (similar cost)
- **Overhead**: ~0%

**Per Hessian update (every $k$ steps):**
- Gauss-Newton: Element-wise square (negligible)
- Hutchinson: One additional backward pass (~100% overhead for that step)

**Amortized overhead:**
- Gauss-Newton: ~0%
- Hutchinson: ~10% (100% / 10 steps)

### Choosing Between Gauss-Newton and Hutchinson

| Aspect | Gauss-Newton | Hutchinson |
|--------|--------------|------------|
| **Accuracy** | Biased approximation | Unbiased estimate |
| **Overhead** | Negligible | ~10% amortized |
| **Variance** | Low (deterministic) | Higher (stochastic) |
| **Use Case** | Default choice | When accuracy critical |

**Recommendation**: Start with Gauss-Newton. Switch to Hutchinson only if convergence is suboptimal.

## Implementation

### Basic Usage

```python
from nexus.training.optimizers import Sophia

# Create optimizer
optimizer = Sophia(
    model.parameters(),
    lr=1e-4,
    betas=(0.965, 0.99),
    rho=0.04,
    weight_decay=0.0,
    hessian_update_interval=10,
    estimator="gauss_newton"  # or "hutchinson"
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        loss = model(batch)

        # Backward pass
        loss.backward()

        # Update Hessian (automatic based on interval)
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad()
```

### With Gradient Clipping

```python
from torch.nn.utils import clip_grad_norm_

optimizer = Sophia(model.parameters(), lr=1e-4)

for batch in dataloader:
    loss = model(batch)
    loss.backward()

    # Clip gradients (optional, but recommended)
    clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    optimizer.zero_grad()
```

### With Learning Rate Schedule

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = Sophia(model.parameters(), lr=3e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)

for step, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    # Update learning rate
    scheduler.step()
```

### Full Training Example

```python
import torch
from nexus.training.optimizers import Sophia
from nexus.models.transformer import GPT

# Setup
device = torch.device("cuda")
model = GPT(vocab_size=50257, n_layer=12, n_head=12, n_embd=768).to(device)

# Optimizer with recommended settings for GPT
optimizer = Sophia(
    model.parameters(),
    lr=2e-4,
    betas=(0.965, 0.99),
    rho=0.04,
    weight_decay=0.1,  # Decoupled weight decay
    hessian_update_interval=10,
    estimator="gauss_newton"
)

# Learning rate schedule
from torch.optim.lr_scheduler import OneCycleLR
scheduler = OneCycleLR(
    optimizer,
    max_lr=2e-4,
    total_steps=num_training_steps,
    pct_start=0.01  # 1% warmup
)

# Training
model.train()
for step, (input_ids, labels) in enumerate(train_dataloader):
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    # Forward
    logits = model(input_ids)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )

    # Backward
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    # Logging
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
```

### Advanced: Custom Hessian Computation

For research or specialized architectures, you can manually trigger Hessian updates:

```python
optimizer = Sophia(model.parameters(), lr=1e-4)

for step, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()

    # Manual Hessian update (overrides automatic interval)
    if step % custom_interval == 0:
        optimizer.update_hessian()

    optimizer.step()
    optimizer.zero_grad()
```

### Mixed Precision Training

Sophia works seamlessly with automatic mixed precision (AMP):

```python
from torch.cuda.amp import autocast, GradScaler

optimizer = Sophia(model.parameters(), lr=1e-4)
scaler = GradScaler()

for batch in dataloader:
    with autocast():
        loss = model(batch)

    # Scaled backward
    scaler.scale(loss).backward()

    # Unscale before gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Optimizer step with scaler
    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad()
```

## Performance Analysis

### Convergence Speed

Based on the original paper's experiments on GPT-2 scale models:

**Training Time to Target Loss:**
- Adam: 100% (baseline)
- Sophia-H (Hutchinson): 52% (1.9x faster)
- Sophia-G (Gauss-Newton): 50% (2.0x faster)

**Tokens Processed to Target Loss:**
- Adam: 100% (baseline)
- Sophia: ~50% (2x fewer tokens)

**Key Insight**: Sophia achieves better convergence both in wall-clock time AND sample efficiency.

### Memory Overhead

**Memory per parameter:**
- Adam: 8 bytes (m, v)
- Sophia: 8 bytes (m, h)
- **Difference**: 0 bytes

**Peak memory during Hessian update (Hutchinson only):**
- One additional backward pass: Temporary activations
- Typically <5% total memory increase
- Cleared immediately after update

**Conclusion**: Sophia is nearly memory-neutral compared to Adam.

### Computational Cost

**Per-step FLOPs (excluding Hessian update):**
- Adam: $3d$ operations (update m, v, parameter)
- Sophia: $3d$ operations (update m, h, parameter)
- **Overhead**: ~0%

**Amortized FLOPs (including Hessian updates):**
- Gauss-Newton: +0% (negligible element-wise operations)
- Hutchinson: +10% (one backward every 10 steps)

**Wall-clock overhead (measured):**
- Gauss-Newton: <2%
- Hutchinson: ~8-12%

**Conclusion**: Sophia's small overhead is dwarfed by its 2x convergence speedup.

### Scaling Behavior

**Model size scaling:**
- Tested on 125M to 770M parameters
- Benefits increase with model size
- Likely extends to multi-billion parameter models

**Batch size sensitivity:**
- Works well with batch sizes 256+
- Smaller batches: Hessian estimates noisier (increase update interval)
- Larger batches: More reliable estimates, can decrease interval

**Sequence length:**
- No inherent limitations
- Benefits observed for sequences 512-2048 tokens
- Should extend to longer contexts

## Hyperparameter Tuning

### Learning Rate ($\alpha$)

**Recommended range**: $1e-4$ to $3e-4$

**How to tune:**
1. Start with $2e-4$ (good default)
2. If loss diverges: Decrease to $1e-4$
3. If training too slow: Increase to $3e-4$
4. Monitor initial loss decrease rate

**Compared to Adam:**
- Sophia typically uses slightly lower LR than Adam
- If Adam uses $3e-4$, try Sophia at $2e-4$

### Beta Coefficients ($\beta_1, \beta_2$)

**Recommended**: $(0.965, 0.99)$ (paper's default)

**Compared to Adam's $(0.9, 0.999)$:**
- $\beta_1 = 0.965$: Slightly longer momentum memory
- $\beta_2 = 0.99$: Faster Hessian adaptation

**When to adjust:**
- Very noisy gradients: Increase $\beta_1$ to $0.98$
- Fast-changing curvature: Decrease $\beta_2$ to $0.95$
- Generally: Keep at defaults unless you have a specific reason

### Clipping Threshold ($\rho$)

**Recommended**: $0.04$ (paper's default)

**Effect:**
- **Lower $\rho$**: More aggressive clipping, more conservative updates
- **Higher $\rho$**: Less clipping, faster adaptation (risk of instability)

**How to tune:**
1. Start with $0.04$
2. If loss spikes: Decrease to $0.02$ or $0.03$
3. If training too slow: Increase to $0.05$ or $0.06$
4. Rarely go beyond $[0.02, 0.08]$

**Intuition**: $\rho$ sets the maximum effective learning rate per parameter. Lower values are more conservative.

### Hessian Update Interval ($k$)

**Recommended**: $10$ (paper's default)

**Trade-offs:**
- **Smaller $k$ (e.g., 5)**: More frequent updates, more accurate curvature, higher overhead
- **Larger $k$ (e.g., 20)**: Less overhead, potentially stale curvature information

**When to adjust:**
- Fast-changing loss landscape: Decrease to 5-7
- Very stable training: Increase to 15-20
- Hutchinson estimator: Can use larger $k$ due to better estimates

**Rule of thumb**: Overhead from interval $k$ is roughly $1/k$ of one training step.

### Weight Decay

**Recommended**: $0.1$ for LLMs (similar to Adam)

Sophia uses **decoupled weight decay** (like AdamW):

$$\theta_{t+1} = (1 - \alpha \lambda) \theta_t - \alpha \cdot \text{update}_t$$

**How to set:**
- Start with $0.1$ for language models
- Adjust based on validation performance
- Higher weight decay: More regularization, may slow convergence

### Full Hyperparameter Recipe for GPT-2

```python
# GPT-2 125M
optimizer = Sophia(
    model.parameters(),
    lr=2e-4,
    betas=(0.965, 0.99),
    rho=0.04,
    weight_decay=0.1,
    hessian_update_interval=10,
    estimator="gauss_newton"
)

# GPT-2 350M
optimizer = Sophia(
    model.parameters(),
    lr=1.5e-4,  # Slightly lower for larger model
    betas=(0.965, 0.99),
    rho=0.04,
    weight_decay=0.1,
    hessian_update_interval=10,
    estimator="gauss_newton"
)
```

## Comparison with Other Optimizers

### Sophia vs. Adam/AdamW

| Aspect | Adam/AdamW | Sophia |
|--------|------------|--------|
| **Convergence Speed** | Baseline | 2x faster |
| **Memory** | 8 bytes/param | 8 bytes/param |
| **Computation** | Baseline | +0-10% |
| **Hyperparameter Sensitivity** | Moderate | Similar |
| **Implementation Complexity** | Simple | Moderate |
| **Best Use Case** | General purpose | LLM pre-training |

**When to choose Sophia:**
- Long pre-training runs (>100K steps)
- Large models (>100M parameters)
- Wall-clock time is critical

**When to stick with Adam:**
- Fine-tuning (short runs)
- Small models
- Simplicity preferred

### Sophia vs. SGD with Momentum

| Aspect | SGD + Momentum | Sophia |
|--------|----------------|--------|
| **Convergence Speed** | Slowest | Fastest |
| **Memory** | 4 bytes/param | 8 bytes/param |
| **Hyperparameter Tuning** | Difficult | Moderate |
| **Stability** | Requires careful LR scheduling | More robust |

**Verdict**: Sophia dominates SGD for large-scale training. SGD only competitive with heavy tuning.

### Sophia vs. Lion

| Aspect | Lion | Sophia |
|--------|------|--------|
| **Convergence Speed** | Similar to Sophia | 2x faster than Adam |
| **Memory** | 4 bytes/param | 8 bytes/param |
| **Update Type** | Sign-based | Curvature-based |
| **Best Use Case** | General, memory-critical | LLM pre-training |

**Key Difference**: Lion uses sign(momentum) for updates (memory-efficient), while Sophia uses curvature information (sample-efficient).

### Sophia vs. Adafactor

| Aspect | Adafactor | Sophia |
|--------|-----------|--------|
| **Convergence Speed** | Similar to Adam | 2x faster than Adam |
| **Memory** | $O(\sqrt{d})$ factored | 8 bytes/param |
| **Hyperparameter Tuning** | Automatic LR | Manual LR |
| **Use Case** | Extreme memory constraints | Fast convergence |

**When to choose Adafactor**: Only when memory is extremely tight. Otherwise, Sophia's speed advantage wins.

## Advanced Topics

### Sophia with Distributed Training

Sophia works seamlessly with data-parallel and model-parallel training:

```python
from torch.nn.parallel import DistributedDataParallel as DDP

# Wrap model
model = DDP(model, device_ids=[local_rank])

# Create optimizer (same as single-GPU)
optimizer = Sophia(model.parameters(), lr=1e-4)

# Training loop unchanged
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # Gradients averaged across GPUs automatically
    optimizer.step()
    optimizer.zero_grad()
```

**Note**: Hessian diagonal is computed locally per GPU, which is fine since we only need diagonal elements.

### Combining with FSDP

Fully Sharded Data Parallel (FSDP) also compatible:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model)
optimizer = Sophia(model.parameters(), lr=1e-4)

# Training proceeds normally
```

**Caveat**: Hessian computation may require some care with parameter sharding. Gauss-Newton estimator recommended.

### Layer-wise Learning Rates

Different layers can have different learning rates:

```python
optimizer = Sophia([
    {'params': model.embeddings.parameters(), 'lr': 1e-4},
    {'params': model.layers.parameters(), 'lr': 2e-4},
    {'params': model.head.parameters(), 'lr': 3e-4}
], betas=(0.965, 0.99), rho=0.04)
```

Sophia's adaptive nature makes this less critical than for SGD, but can still help.

### Sophia for Fine-Tuning

While designed for pre-training, Sophia can be used for fine-tuning:

```python
# Lower learning rate for fine-tuning
optimizer = Sophia(
    model.parameters(),
    lr=5e-5,  # Much lower than pre-training
    betas=(0.965, 0.999),  # Slower adaptation
    rho=0.04,
    weight_decay=0.01,  # Lower weight decay
    hessian_update_interval=5  # More frequent updates for short runs
)
```

**Recommendation**: For fine-tuning, Adam is often simpler and sufficient unless training is very long.

### Debugging Sophia Training

**Loss spikes:**
1. Check learning rate (decrease to 1e-4)
2. Decrease $\rho$ to 0.03 or 0.02
3. Enable gradient clipping
4. Verify Hessian isn't accumulating NaNs

**Slow convergence:**
1. Increase learning rate to 3e-4
2. Check if Hessian estimates are reasonable (`optimizer.state[param]['hessian']`)
3. Try Hutchinson estimator for better accuracy
4. Ensure batch size is adequate (>256 recommended)

**Memory issues:**
1. Sophia's memory footprint equals Adam
2. If OOM with Adam → OOM with Sophia
3. Use gradient checkpointing or mixed precision

### Visualization and Monitoring

```python
# Log Hessian statistics
for param_name, param in model.named_parameters():
    if param in optimizer.state:
        state = optimizer.state[param]
        h = state['hessian']

        # Log statistics
        print(f"{param_name}:")
        print(f"  Hessian mean: {h.mean().item():.6f}")
        print(f"  Hessian std:  {h.std().item():.6f}")
        print(f"  Hessian max:  {h.max().item():.6f}")
        print(f"  Hessian min:  {h.min().item():.6f}")

        # Check for clipping
        clipped_ratio = (h < rho).float().mean()
        print(f"  Clipped:      {clipped_ratio.item():.2%}")
```

This helps diagnose if $\rho$ needs adjustment.

## References

### Primary Paper

**Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training**
- Authors: Hong Liu, Zhiyuan Li, David Hall, Percy Liang, Tengyu Ma
- Conference: arXiv 2023
- Link: https://arxiv.org/abs/2305.14342

### Key Insights from Paper

1. **2x speedup** on GPT-2 models (125M to 770M parameters)
2. **Hessian diagonal sufficient** for effective second-order optimization
3. **Element-wise clipping** provides robustness without expensive line search
4. **Minimal overhead** makes it practical for large-scale training

### Related Work

**Second-Order Optimization:**
- K-FAC: Kronecker-factored approximate curvature
- Shampoo: Matrix preconditioning
- Adahessian: Hessian-based adaptive learning rates

**Why Sophia is Better:**
- K-FAC/Shampoo: High memory/compute overhead
- Adahessian: Similar idea, but Sophia's clipping more stable

**Sophia's Position:**
- Best convergence speed for LLM pre-training
- Practical overhead (unlike full second-order methods)
- Simple implementation (unlike K-FAC/Shampoo)

### Implementation Notes

The Nexus implementation (`nexus/training/optimizers/sophia.py`) provides:
- Both Gauss-Newton and Hutchinson estimators
- Decoupled weight decay
- Automatic Hessian update scheduling
- Full PyTorch optimizer API compatibility

For the most up-to-date implementation details, see the source code.

---

**Last Updated**: February 2026
**Nexus Version**: 1.0+
**Status**: Production Ready
