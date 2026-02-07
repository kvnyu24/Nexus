# Cosine Annealing with Warm Restarts (SGDR)

## Overview

Cosine annealing with warm restarts (Loshchilov & Hutter, 2017) periodically resets the learning rate to its maximum value, allowing the optimizer to explore multiple loss basins during training.

## Mathematical Formula

Within cycle $i$ at local step $t$:
$$\\eta_t = \\eta_{\\text{min}} + \\frac{1}{2}(\\eta_{\\text{max}} - \\eta_{\\text{min}})\\left(1 + \\cos\\left(\\frac{t}{T_i}\\pi\\right)\\right)$$

Cycle length: $T_i = T_0 \\cdot M^i$

Where:
- $T_0$: Initial cycle length
- $M$: Cycle length multiplier (typically 1.0 or 2.0)
- $i$: Cycle index

## Key Features

1. **Restarts**: LR jumps back to max at start of each cycle
2. **Multiple Basins**: Can escape local minima
3. **Ensemble Effect**: Different cycles explore different solutions
4. **Variable Cycles**: Each cycle can be longer than previous

## Implementation

```python
from nexus.training.schedulers import CosineRestartScheduler

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = CosineRestartScheduler(
    optimizer,
    peak_lr=0.1,
    min_lr=1e-6,
    cycle_length=10000,
    cycle_mult=2.0,  # Each cycle 2x longer
    warmup_steps=500
)

for step in range(total_steps):
    train_step()
    scheduler.step()
    
    # Get cycle info
    info = scheduler.get_cycle_info()
    print(f"Cycle {info['cycle_idx']}, Step {info['step_in_cycle']}/{info['cycle_length']}")
```

## Cycle Length Strategies

**Constant** (`cycle_mult=1.0`):
- All cycles same length
- More restarts
- Good for exploration

**Growing** (`cycle_mult=2.0`):
- Later cycles longer
- Fewer restarts at end
- Good for convergence

## Snapshot Ensembling

Save model at end of each cycle to create an ensemble:

```python
for step in range(total_steps):
    train_step()
    scheduler.step()
    
    info = scheduler.get_cycle_info()
    if info['step_in_cycle'] == info['cycle_length'] - 1:
        # End of cycle: save snapshot
        torch.save(model.state_dict(), f"snapshot_cycle_{info['cycle_idx']}.pt")
```

**Inference**: Average predictions from all snapshots.

## When to Use

**Best for**:
- Finding robust solutions
- Avoiding sharp minima
- Uncertainty estimation (via ensemble)
- Long training runs

**Not ideal for**:
- Very short training
- When restarts disruptive to convergence
- Stable, well-understood problems

## Performance

**Typical results**:
- 0.5-2% accuracy improvement from ensemble
- More robust to hyperparameters
- Better calibration (uncertainty estimates)

## References

**SGDR: Stochastic Gradient Descent with Warm Restarts**  
Loshchilov & Hutter, ICLR 2017
