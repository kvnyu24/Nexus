# ReMax: REINFORCE with Maximum Baseline

**Paper**: ReMax: A Simple, Effective, and Efficient Method for Aligning LLMs (Li et al., 2024)

**Code**: `nexus/models/rl/preference/remax.py`

## Overview

ReMax uses the **greedy (argmax) action** as baseline - simplest possible variance reduction.

### Key Innovation

**REINFORCE**: No baseline, high variance
**RLOO**: K samples, leave-one-out
**ReMax**: 1 sample + 1 greedy = 2 total!

```
y_sample ~ π(·|x)  (stochastic)
y_greedy = argmax π(·|x)  (deterministic)
advantage = r(y_sample) - r(y_greedy)
```

From `nexus/models/rl/preference/remax.py` lines 90-137.

## Mathematical Formulation

```
L_ReMax = -(r_sample - r_greedy) · log π(y_sample|x)
```

**Insight**: Greedy action is highly correlated with expected reward but requires no sampling variance.

## Hyperparameters

- **temperature**: 1.0 (for sampling, greedy uses argmax)
- **max_grad_norm**: 1.0

## Advantages

1. Simplest: only 2 generations per prompt
2. No critic network
3. Memory efficient (1 sample + 1 greedy)
4. Fast training

## Usage

```python
agent = ReMaxAgent(config={"policy": model})
sample_ids, _, sample_mask, greedy_ids, _, _ = agent.generate_sample_and_greedy(
    prompts, mask, max_new_tokens=256
)
sample_rewards = reward_model(sample_ids)
greedy_rewards = reward_model(greedy_ids)
metrics = agent.update({
    "sample_input_ids": sample_ids,
    "sample_attention_mask": sample_mask,
    "sample_action_mask": sample_mask,
    "sample_rewards": sample_rewards,
    "greedy_rewards": greedy_rewards
})
```

## References

```bibtex
@article{li2024remax,
  title={ReMax: A Simple, Effective, and Efficient RL Method for Aligning LLMs},
  author={Li, Ziniu and others},
  year={2024}
}
```
