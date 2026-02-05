# Learning Rate Schedules

Learning rate scheduling strategies for stable and efficient training.

## Overview

Learning rate scheduling is crucial for training modern deep learning models. The right schedule can:
- Accelerate convergence
- Improve final performance
- Stabilize training
- Enable training without knowing total steps upfront

## Scheduler Comparison

| Scheduler | Requires Total Steps | Exploration | Best Use Case |
|-----------|---------------------|-------------|---------------|
| **Cosine Annealing** | Yes | Single phase | Standard transformer training |
| **SGDR (Cosine Restarts)** | No | Multiple basins | Exploratory training, ensembling |
| **WSD** | No | Controlled phases | Open-ended training |

## Quick Start

```python
from nexus.training.schedulers import WSDScheduler, CosineRestartScheduler

# Warmup-Stable-Decay (WSD)
scheduler = WSDScheduler(
    optimizer,
    warmup_steps=1000,
    stable_steps=50000,
    decay_steps=10000,
    peak_lr=1e-3,
    min_lr=1e-5,
)

# Cosine Annealing with Restarts (SGDR)
scheduler = CosineRestartScheduler(
    optimizer,
    peak_lr=1e-3,
    min_lr=1e-6,
    cycle_length=10000,
    cycle_mult=2.0,
    warmup_steps=500,
)

# Training loop
for step, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

## Detailed Documentation

- [Warmup-Stable-Decay (WSD)](wsd.md) - Three-phase schedule for open-ended training
- [Cosine Annealing with Restarts (SGDR)](sgdr.md) - Periodic restarts for exploration

## Schedule Visualization

```
WSD:
lr |     ____________________
   |   /                      \
   |  /                         \___
   |_/_____________________________|
     warmup  stable      decay

SGDR:
lr | /\    /\      /\
   |/  \  /  \    /  \
   |    \/    \  /    \
   |          \/       \____
   |________________________|
     cycle1  cycle2   cycle3
```

## References

See individual scheduler documentation for detailed references.
