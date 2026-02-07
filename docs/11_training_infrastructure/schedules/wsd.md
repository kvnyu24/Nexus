# WSD: Warmup-Stable-Decay Learning Rate Schedule

## Overview

The WSD schedule (Hu et al., 2024) divides training into three distinct phases: linear warmup, stable plateau, and controlled decay. Unlike cosine schedules, WSD doesn't require knowing total training steps upfront.

## Three Phases

### 1. Warmup (0 to `warmup_steps`)
Linear increase from `min_lr` to `peak_lr`:
$$\\eta_t = \\eta_{\\text{min}} + \\frac{t}{T_{\\text{warmup}}} (\\eta_{\\text{peak}} - \\eta_{\\text{min}})$$

### 2. Stable (`warmup_steps` to `warmup_steps + stable_steps`)
Constant at peak learning rate:
$$\\eta_t = \\eta_{\\text{peak}}$$

### 3. Decay (remaining steps)
Controlled decrease (linear, cosine, or sqrt):

**Linear**:
$$\\eta_t = \\eta_{\\text{min}} + (1 - p)(\\eta_{\\text{peak}} - \\eta_{\\text{min}})$$

**Cosine**:
$$\\eta_t = \\eta_{\\text{min}} + \\frac{1}{2}(1 + \\cos(\\pi p))(\\eta_{\\text{peak}} - \\eta_{\\text{min}})$$

**Sqrt**:
$$\\eta_t = \\eta_{\\text{min}} + (1 - \\sqrt{p})(\\eta_{\\text{peak}} - \\eta_{\\text{min}})$$

Where $p = \\frac{t - T_{\\text{warmup}} - T_{\\text{stable}}}{T_{\\text{decay}}}$

## Implementation

```python
from nexus.training.schedulers import WSDScheduler

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = WSDScheduler(
    optimizer,
    warmup_steps=1000,
    stable_steps=50000,
    decay_steps=10000,
    peak_lr=1e-3,
    min_lr=1e-5,
    decay_type='cosine'  # 'linear', 'cosine', or 'sqrt'
)

for step in range(total_steps):
    train_step()
    scheduler.step()
    
    # Check current phase
    phase = scheduler.get_phase()  # 'warmup', 'stable', or 'decay'
```

## Key Advantages

1. **Flexible Duration**: Can extend stable phase indefinitely
2. **No Total Steps Required**: Don't need to know training length upfront
3. **Clear Phases**: Easy to understand and debug
4. **Decoupled Design**: Warmup, stable, decay independently configured

## When to Use

**Best for**:
- Unknown training duration
- Large-scale pre-training
- Projects where compute budget may change
- When you want explicit control over training phases

**Typical Configuration** (GPT-style):
- Warmup: 1% of expected training
- Stable: 90% of expected training  
- Decay: 9% of expected training

## Performance

**Empirical Results** (MiniCPM paper):
- Matches cosine annealing performance
- More robust to early stopping
- Easier to extend training if needed

## References

**MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies**  
Hu et al., 2024
