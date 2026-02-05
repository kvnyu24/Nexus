# IPO: Identity Preference Optimization

**Paper**: A General Theoretical Paradigm to Understand Learning from Human Feedback (Azar et al., 2023)

**Code**: `nexus/models/rl/preference/ipo.py`

## Overview

IPO uses **squared loss** instead of DPO's log-sigmoid, providing bounded optimization.

### Key Innovation

**DPO**: Unbounded log-sigmoid loss
**IPO**: Squared loss with target margin

```
L_IPO = E[(log_ratio_diff - 1/(2β))²]

where log_ratio_diff = (log π/π_ref)_chosen - (log π/π_ref)_rejected
```

Prevents DPO's overfitting by setting finite optimum at 1/(2β).

From `nexus/models/rl/preference/ipo.py` lines 138-178.

## Hyperparameters

- **beta**: 0.1 (regularization strength)

## Advantages

1. Bounded loss (prevents overfitting)
2. Theoretical guarantees
3. More stable than DPO

## Usage

```python
agent = IPOAgent(config={"policy": model, "reference_policy": ref_model, "beta": 0.1})
batch = {"chosen_input_ids": ..., "rejected_input_ids": ..., ...}
metrics = agent.update(batch)
```

## References

```bibtex
@article{azar2023ipo,
  title={A General Theoretical Paradigm to Understand RLHF},
  author={Azar, Mohammad Gheshlaghi and others},
  journal={arXiv},
  year={2023}
}
```
