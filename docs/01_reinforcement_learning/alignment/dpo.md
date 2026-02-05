# Direct Preference Optimization (DPO)

**Paper**: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (Rafailov et al., NeurIPS 2023)

**Code**: `nexus/models/rl/preference/dpo.py`

## Overview

DPO simplifies RLHF by eliminating the explicit reward model. Instead of training a reward model then using PPO, DPO directly optimizes the policy on preference data.

### Key Insight

The optimal RLHF policy satisfies π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β). Rearranging: r(x,y) = β · log(π*(y|x) / π_ref(y|x)). The policy IS the reward model!

## Mathematical Formulation

### DPO Objective

```
L_DPO = -E[ log σ(β · (log_ratio_chosen - log_ratio_rejected)) ]

where:
  log_ratio_chosen = log π_θ(y_w|x) - log π_ref(y_w|x)
  log_ratio_rejected = log π_θ(y_l|x) - log π_ref(y_l|x)
  β: controls deviation from reference
```

Increase likelihood of chosen responses relative to reference, decrease rejected responses.

## Hyperparameters

**Beta**: 0.01 - 0.5 (default 0.1)
- Low: flexible, potential reward hacking
- High: conservative, stable

**Learning Rate**: 1e-6 (smaller than SFT)

## Common Pitfalls

1. Not freezing reference model
2. Wrong logits shift (predict t+1 from t, not t from t)
3. Including prompt in loss (mask to response only)
4. Beta too large

## References

```bibtex
@article{rafailov2023dpo,
  title={Direct Preference Optimization},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and others},
  journal={NeurIPS},
  year={2023}
}
```
