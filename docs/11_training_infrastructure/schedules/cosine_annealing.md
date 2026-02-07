# Cosine Annealing Learning Rate Schedule

## Overview

Cosine annealing (Loshchilov & Hutter, 2016) smoothly decreases learning rate following a cosine curve. One of the most popular and effective LR schedules for deep learning.

## Mathematical Formula

$$\\eta_t = \\eta_{\\text{min}} + \\frac{1}{2}(\\eta_{\\text{max}} - \\eta_{\\text{min}})\\left(1 + \\cos\\left(\\frac{t}{T}\\pi\\right)\\right)$$

Where:
- $\\eta_t$: Learning rate at step $t$
- $\\eta_{\\text{max}}$: Maximum (initial) learning rate
- $\\eta_{\\text{min}}$: Minimum (final) learning rate
- $T$: Total training steps

### Properties

1. **Smooth decay**: No abrupt changes
2. **Fast initial decay**: Aggressive early in training
3. **Slow final decay**: Fine-tuning at the end
4. **Reaches minimum**: Exactly $\\eta_{\\text{min}}$ at step $T$

## Implementation

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_training_steps,
    eta_min=1e-6
)

for step in range(num_training_steps):
    train_step()
    scheduler.step()  # Update LR
```

### With Warmup

```python
from nexus.training.scheduler import CosineWarmupScheduler

scheduler = CosineWarmupScheduler(
    optimizer,
    warmup_steps=1000,
    max_steps=100000,
    min_lr=1e-7
)

for step in range(max_steps):
    train_step()
    scheduler.step()
```

## When to Use

**Best for**:
- Long training runs (>10K steps)
- Training from scratch
- Image classification, language modeling

**Not ideal for**:
- Very short training
- When early stopping is likely
- Unknown training duration

## Performance

**Typical improvement** over constant LR: 1-5% final accuracy  
**Overhead**: Negligible (just LR computation)

## References

**SGDR: Stochastic Gradient Descent with Warm Restarts**  
Loshchilov & Hutter, ICLR 2017
