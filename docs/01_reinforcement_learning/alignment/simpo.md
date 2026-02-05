# SimPO: Simple Preference Optimization

**Paper**: SimPO: Simple Preference Optimization with Reference-Free Reward (Meng et al., 2024)

**Code**: `nexus/models/rl/preference/simpo.py`

## Overview

SimPO eliminates the reference model by using **length-normalized log probabilities** as implicit rewards.

### Key Innovation

**DPO**: Needs reference model π_ref
**SimPO**: Reference-free, uses avg log prob as reward

```
r(y|x) = (1/|y|) · sum_t log π(y_t|x,y_{<t})
L = -log σ(β · (r_chosen - r_rejected - γ))
```

## Hyperparameters

- **beta**: 2.0 (higher than DPO)
- **gamma**: 0.5 (reward margin)
- **length_normalization**: True (essential)

From `nexus/models/rl/preference/simpo.py` lines 95-132.

## Advantages

1. No reference model (50% memory savings)
2. Simpler training
3. Competitive performance

## Usage

```python
agent = SimPOAgent(config={"policy": model, "beta": 2.0, "gamma": 0.5})
batch = {"chosen_input_ids": ..., "rejected_input_ids": ..., ...}
metrics = agent.update(batch)
```

## References

```bibtex
@article{meng2024simpo,
  title={SimPO: Simple Preference Optimization with Reference-Free Reward},
  author={Meng, Yu and others},
  year={2024}
}
```
