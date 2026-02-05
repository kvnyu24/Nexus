# RLOO: REINFORCE Leave-One-Out

**Paper**: Back to Basics: Revisiting REINFORCE Style Optimization for RLHF (Ahmadian et al., 2024)

**Code**: `nexus/models/rl/preference/rloo.py`

## Overview

RLOO uses **leave-one-out baselines** for variance reduction without a critic network.

### Key Innovation

**REINFORCE**: High variance, needs critic
**PPO**: Needs critic network
**RLOO**: Leave-one-out baseline (no critic!)

```
For K samples {y_1, ..., y_K} from prompt x:
  baseline_i = (sum_j r_j - r_i) / (K-1)
  advantage_i = r_i - baseline_i
```

From `nexus/models/rl/preference/rloo.py` lines 98-119.

## Mathematical Formulation

```
L_RLOO = -(1/K) sum_i advantage_i · log π(y_i|x)

where advantage_i is computed using other samples as baseline
```

**Unbiased**: E[baseline_i] = E[r] without learned model bias.

## Hyperparameters

- **num_samples**: 4 (K ≥ 2 required for leave-one-out)
- **temperature**: 1.0 (sampling temperature)
- **max_grad_norm**: 1.0

## Advantages

1. No critic network (memory efficient)
2. Unbiased baseline
3. Low variance
4. Theoretically sound

## Usage

```python
agent = RLOOAgent(config={"policy": model, "num_samples": 4})
samples, _, action_mask = agent.generate_samples(prompts, mask, max_new_tokens=256)
rewards = reward_model(samples)
metrics = agent.update({"input_ids": samples, "rewards": rewards, "action_mask": action_mask})
```

## References

```bibtex
@article{ahmadian2024rloo,
  title={Back to Basics: Revisiting REINFORCE for LLM Training},
  author={Ahmadian, Arash and others},
  year={2024}
}
```
