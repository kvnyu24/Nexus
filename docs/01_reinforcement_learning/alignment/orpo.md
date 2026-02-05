# ORPO: Odds Ratio Preference Optimization

**Paper**: ORPO: Monolithic Preference Optimization without Reference Model (Hong et al., 2024)

**Code**: `nexus/models/rl/preference/orpo.py`

## Overview

ORPO combines **SFT + alignment** in one loss, no reference model needed.

### Key Innovation

**Standard**: SFT then DPO (two stages)
**ORPO**: SFT + odds-ratio penalty (one stage)

```
L_SFT = -E[log π(y_chosen|x)]
L_OR = -E[log σ(log_odds_chosen - log_odds_rejected)]
L_ORPO = L_SFT + λ · L_OR
```

From `nexus/models/rl/preference/orpo.py` lines 182-238.

## Hyperparameters

- **lambda_weight**: 1.0 (odds-ratio penalty weight)

## Advantages

1. Monolithic (one stage)
2. No reference model
3. Simpler pipeline

## Usage

```python
agent = ORPOAgent(config={"policy": model, "lambda_weight": 1.0})
batch = {"chosen_input_ids": ..., "rejected_input_ids": ..., ...}
metrics = agent.update(batch)
```

## References

```bibtex
@article{hong2024orpo,
  title={ORPO: Monolithic Preference Optimization without Reference Model},
  author={Hong, Jiwoo and others},
  year={2024}
}
```
