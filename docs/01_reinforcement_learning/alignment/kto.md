# Kahneman-Tversky Optimization (KTO)

**Paper**: KTO: Model Alignment as Prospect Theoretic Optimization (Ethayarajh et al., 2024)

**Code**: `nexus/models/rl/preference/kto.py`

## Overview

KTO uses **binary feedback** (good/bad) instead of pairwise comparisons, based on Kahneman & Tversky's prospect theory.

### Key Innovation

**DPO**: Needs preference pairs (y_chosen, y_rejected)
**KTO**: Needs binary labels (y is good: 1, y is bad: 0)

More data-efficient - collecting "is this good?" is easier than "which is better?"

## Mathematical Formulation

```
For desirable (good) examples:
  L_good = λ_good · (1 - σ(β · (log_ratio - z_ref)))

For undesirable (bad) examples:
  L_bad = λ_bad · (1 - σ(-β · (log_ratio - z_ref)))

where:
  log_ratio = log π(y|x) - log π_ref(y|x)
  z_ref = E[KL(π || π_ref)] (reference point)
  λ_bad > λ_good models loss aversion
```

From `nexus/models/rl/preference/kto.py` lines 181-231.

## Hyperparameters

- **beta**: 0.1 (inverse temperature)
- **lambda_good**: 1.0 (weight for good examples)
- **lambda_bad**: 1.0 (set >1.0 for loss aversion)

## Usage

```python
agent = KTOAgent(config={"policy": model, "reference_policy": ref_model, "beta": 0.1})
batch = {"input_ids": ..., "is_desirable": torch.tensor([1,0,1,0]), "attention_mask": ...}
metrics = agent.update(batch)
```

## References

```bibtex
@article{ethayarajh2024kto,
  title={KTO: Model Alignment as Prospect Theoretic Optimization},
  author={Ethayarajh, Kawin and Xu, Winnie and others},
  year={2024}
}
```
