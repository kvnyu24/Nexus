# Group Relative Policy Optimization (GRPO)

**Paper**: DeepSeekMath: Pushing the Limits of Mathematical Reasoning (DeepSeek, 2024)

**Code**: `nexus/models/rl/grpo.py`

## Overview

GRPO eliminates the critic network by using **group-level baselines**: sample K completions per prompt, use their mean as baseline.

### Key Innovation

**PPO**: Needs value network V(s) as baseline
**GRPO**: Uses mean(rewards_in_group) as baseline

```
advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
```

## Mathematical Formulation

```
For prompt x with K samples {y_1, ..., y_K}:
  baseline = mean([r_1, r_2, ..., r_K])
  advantage_i = (r_i - baseline) / std([r_1, ..., r_K])
  L = -(1/K) sum_i clip(ratio_i, 1±ε) · advantage_i
```

From `nexus/models/rl/grpo.py` lines 69-99.

## Hyperparameters

- **group_size**: 4-8 samples per prompt
- **clip_range**: 0.2 (PPO-style clipping)
- **kl_coef**: 0.1 (KL from reference)

## Advantages

1. No critic network (memory efficient)
2. PPO-style stability
3. Natural baseline from sampling

## Usage

```python
agent = GRPOAgent(config={"policy": model, "group_size": 8})
samples, log_probs = agent.generate_samples(prompts, mask)
rewards = reward_model(samples)
metrics = agent.update({"input_ids": samples, "rewards": rewards, "old_log_probs": log_probs})
```

## References

DeepSeekMath paper (2024) - Part of DeepSeek's GRPO training pipeline
