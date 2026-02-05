# Cal-QL: Calibrated Q-Learning

**Paper**: [Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning](https://arxiv.org/abs/2303.05479) (Nakamoto et al., NeurIPS 2023)

**Status**: Reference documentation (implementation pending)

## Overview

Cal-QL automatically calibrates CQL's conservatism based on dataset quality, eliminating the need to manually tune the α hyperparameter.

### Key Innovation

**CQL**: Requires manual tuning of α (conservatism strength)
**Cal-QL**: Learns α automatically based on calibration error

```
Standard CQL: α is fixed (e.g., 5.0)
Cal-QL:       α_t = f(calibration_error_t)
```

## Mathematical Formulation

### Calibration Objective

```
Calibration Error: ε_cal = |E_π[V(s)] - E_data[V(s)]|

Target: min_α |E_π[Q(s,a)] - E_π[r + γV(s')]|

where:
  - E_π[V(s)]: expected value under current policy
  - E_data[V(s)]: empirical value in dataset
  - α adjusted to minimize this gap
```

### Adaptive α Update

```
α_t+1 = α_t + η · ∇_α L_cal(α_t)

where L_cal measures calibration quality
```

## Algorithm Components

### 1. CQL Base

Uses standard CQL as foundation:
```
L_CQL = α_t · (E_π[Q] - E_data[Q]) + L_TD
```

### 2. Calibration Module

Computes calibration error by:
1. Rolling out policy in offline dataset
2. Comparing predicted vs. actual returns
3. Adjusting α to minimize error

### 3. Adaptive α Scheduler

```python
def update_alpha(self, cal_error):
    if cal_error > threshold:
        alpha *= 1.1  # Increase conservatism
    elif cal_error < threshold:
        alpha *= 0.9  # Decrease conservatism
    return alpha
```

## Implementation Sketch

```python
class CalQLAgent:
    def __init__(self, config):
        self.cql_agent = CQLAgent(config)
        self.alpha = config.get("initial_alpha", 5.0)
        self.cal_threshold = config.get("cal_threshold", 0.1)

    def update(self, batch):
        # Standard CQL update with current alpha
        cql_loss = self.cql_agent.compute_cql_loss(batch, alpha=self.alpha)

        # Compute calibration error
        cal_error = self.compute_calibration_error(batch)

        # Adjust alpha
        if cal_error > self.cal_threshold:
            self.alpha *= 1.1  # More conservative
        else:
            self.alpha *= 0.9  # Less conservative

        return {"cql_loss": cql_loss, "alpha": self.alpha, "cal_error": cal_error}

    def compute_calibration_error(self, batch):
        # Rollout policy and compare to actual returns
        with torch.no_grad():
            policy_values = self.value_network(batch["states"])
            actual_returns = compute_mc_returns(batch)
            cal_error = (policy_values - actual_returns).abs().mean()
        return cal_error
```

## Advantages

1. **No hyperparameter tuning**: α adapts automatically
2. **Dataset-aware**: Adjusts conservatism to data quality
3. **Robust**: Works across diverse datasets without retuning

## Key Hyperparameters

### Initial Alpha

```python
initial_alpha = 5.0  # Starting conservatism
```

### Calibration Threshold

```python
cal_threshold = 0.1  # Target calibration error
```

### Alpha Adaptation Rate

```python
alpha_lr = 0.01  # How fast alpha changes
```

## When to Use

**Best for**:
- Unknown dataset quality
- Need robust performance without tuning
- Deploying across multiple tasks

**Avoid when**:
- Dataset quality is known (use fixed CQL)
- Need simplicity (use TD3+BC or IQL)

## References

```bibtex
@inproceedings{nakamoto2023calql,
  title={Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning},
  author={Nakamoto, Mitsuhiko and Zhai, Yuexiang and Singh, Anikait and Mark, Edward and Chebotar, Yevgen and Levine, Sergey and Finn, Chelsea},
  booktitle={NeurIPS},
  year={2023}
}
```
