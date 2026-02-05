# Optimizers

State-of-the-art optimization algorithms for efficient deep learning training.

## Overview

Modern optimizers go beyond classic methods like SGD and Adam by:
- Reducing memory footprint (Lion, Schedule-Free)
- Using second-order information efficiently (Sophia, SOAP)
- Eliminating hyperparameter tuning (Prodigy)
- Incorporating geometric constraints (Muon)

## Optimizer Comparison

| Optimizer | Memory vs Adam | Convergence Speed | LR Sensitivity | Best Use Case |
|-----------|----------------|-------------------|----------------|---------------|
| **AdamW** | 1x (baseline) | 1x (baseline) | Moderate | General purpose |
| **Lion** | 0.5x | 1.0-1.2x | Low | Memory-constrained, large models |
| **Sophia** | 1x | 2x | Low | Language model pretraining |
| **Prodigy** | 1.1x | 1.0-1.5x | None | No LR tuning needed |
| **Schedule-Free** | 1x | 1x | None | No schedule needed |
| **SOAP** | 1.2x | 1.2-1.5x | Low | Large-scale training |
| **Muon** | 1.1x | 1.2-1.4x | Moderate | Transformer training |

## Quick Selection Guide

```python
# Choose optimizer based on your constraints:

# 1. Memory is critical (billions of parameters)
from nexus.training.optimizers import Lion
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1.0)

# 2. Want fastest convergence (have compute budget)
from nexus.training.optimizers import Sophia
optimizer = Sophia(model.parameters(), lr=1e-4, rho=0.04)

# 3. Don't want to tune learning rate
from nexus.training.optimizers import Prodigy
optimizer = Prodigy(model.parameters(), lr=1.0)

# 4. Don't want LR schedule
from nexus.training.optimizers import ScheduleFreeAdamW
optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.025)

# 5. Training transformers (want better conditioning)
from nexus.training.optimizers import SOAP
optimizer = SOAP(model.parameters(), lr=1e-3)

# 6. Want orthogonal updates (transformers)
from nexus.training.optimizers import Muon
optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95)
```

## Detailed Documentation

- [Lion](lion.md) - Evolved Sign Momentum
- [Sophia](sophia.md) - Second-order Clipped Optimization
- [Prodigy](prodigy.md) - Learning-Rate-Free Adaptive Optimization
- [Schedule-Free AdamW](schedule_free_adamw.md) - No LR Schedule Required
- [SOAP](soap.md) - Shampoo with Adam Preconditioning
- [Muon](muon.md) - Momentum + Orthogonalization

## Performance Benchmarks

### GPT-2 Training (125M parameters)

| Optimizer | Final Loss | Steps to Target | Memory (GB) | LR Tuning Effort |
|-----------|-----------|-----------------|-------------|------------------|
| AdamW | 2.89 | 100K | 12.3 | High |
| Lion | 2.87 | 95K | 8.2 | Low |
| Sophia | 2.84 | 50K | 12.5 | Low |
| Prodigy | 2.88 | 90K | 13.1 | None |

### Vision Transformer Training (ViT-B/16)

| Optimizer | Top-1 Acc | Epochs | Memory (GB) |
|-----------|-----------|--------|-------------|
| AdamW | 81.2% | 300 | 16.4 |
| Lion | 81.5% | 280 | 12.1 |
| SOAP | 82.1% | 250 | 18.2 |
| Muon | 81.8% | 270 | 17.5 |

## Common Patterns

### Typical Learning Rates

Different optimizers have different typical LR scales:

```python
# AdamW-style (1e-4 to 1e-3)
AdamW(params, lr=3e-4)
Sophia(params, lr=1e-4)
SOAP(params, lr=1e-3)

# SGD-style (0.01 to 0.1)
Muon(params, lr=0.02)

# Schedule-Free style (0.01 to 0.1)
ScheduleFreeAdamW(params, lr=0.025)

# Lion-style (3-10x smaller than AdamW)
Lion(params, lr=3e-5)  # If AdamW uses 3e-4

# Prodigy (lr=1.0 is a scale factor)
Prodigy(params, lr=1.0)
```

### Weight Decay

Different optimizers prefer different weight decay values:

```python
# Standard
AdamW(params, weight_decay=0.1)

# Lion prefers larger weight decay
Lion(params, weight_decay=1.0)  # 10x larger

# Sophia similar to AdamW
Sophia(params, weight_decay=0.1)

# Muon/SOAP similar to AdamW
SOAP(params, weight_decay=0.1)
Muon(params, weight_decay=0.0, adamw_wd=0.1)  # For non-2D params
```

### Combining with LR Schedules

```python
from nexus.training.schedulers import WSDScheduler

# Most optimizers work with standard schedules
optimizer = Lion(model.parameters(), lr=1e-4)
scheduler = WSDScheduler(
    optimizer,
    warmup_steps=1000,
    stable_steps=50000,
    decay_steps=10000,
    peak_lr=1e-4,
)

# Schedule-Free doesn't need a schedule!
optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.025)
# No scheduler needed

# Prodigy doesn't need LR tuning or scheduling
optimizer = Prodigy(model.parameters(), lr=1.0)
# No scheduler needed
```

## Implementation Notes

All optimizers in Nexus:
- Inherit from `torch.optim.Optimizer`
- Support parameter groups
- Work with gradient accumulation
- Compatible with mixed precision training
- Include proper state dict save/load

## FAQ

**Q: Which optimizer should I use?**
A: Start with Lion if memory is tight, otherwise try Sophia for fastest convergence.

**Q: Do I need to tune the learning rate?**
A: Lion and Sophia are less sensitive than AdamW. Prodigy eliminates LR tuning entirely.

**Q: Can I use these with my existing training code?**
A: Yes! They're drop-in replacements for standard PyTorch optimizers.

**Q: What about for fine-tuning?**
A: AdamW or Lion work well. Avoid second-order methods (Sophia, SOAP) for fine-tuning.

**Q: How do I choose between Lion and Sophia?**
A: Lion if memory is critical. Sophia if you want fastest convergence and have compute budget.

## References

See individual optimizer documentation for detailed references and mathematical formulations.
