# VICReg: Variance-Invariance-Covariance Regularization

## Overview & Motivation

VICReg is a non-contrastive self-supervised learning method that prevents representational collapse through three explicit regularization terms: variance (maintains spread), invariance (matches augmented views), and covariance (decorrelates dimensions). Unlike contrastive methods, VICReg doesn't require negative pairs, momentum encoders, or large batch sizes.

### Key Innovation

**Explicit regularization** replaces complex architectural tricks:
- Variance term: Prevents dimension collapse
- Invariance term: Matches augmented views
- Covariance term: Decorrelates features
- Simple, stable, works with small batches

## Mathematical Formulation

### VICReg Loss

```
L_VICReg = λ·L_inv + μ·L_var + ν·L_cov
```

### 1. Invariance Loss (Similarity)

MSE between embeddings of two views:

```
L_inv = MSE(z₁, z₂) = (1/B) Σᵢ ||z₁ⁱ - z₂ⁱ||²
```

### 2. Variance Loss (Spread)

Hinge loss to maintain variance above threshold:

```
L_var = (1/D) Σⱼ max(0, γ - σ(zⱼ))
```

Where σ(zⱼ) is std of dimension j across batch, γ=1 is threshold.

### 3. Covariance Loss (Decorrelation)

Penalize off-diagonal covariance:

```
L_cov = (1/D) Σᵢ≠ⱼ Cov(z)ᵢⱼ²
```

Where Cov(z) is the covariance matrix.

## Implementation Details

### Network Architecture

**Encoder**: Any backbone (ResNet, ViT, etc.)

**Projector (Expander)**:
- Input: encoder_dim (e.g., 2048)
- Hidden: 8192 (expand!)
- Output: 8192
- 3 layers with BatchNorm + ReLU

### Code Reference

```python
from nexus.models.ssl import VICRegModel

config = {
    "encoder_dim": 768,
    "projector_hidden_dim": 8192,
    "projector_output_dim": 8192,
    "lambda_param": 25.0,  # Invariance
    "mu_param": 25.0,      # Variance
    "nu_param": 1.0,       # Covariance
}

model = VICRegModel(config)

# Two augmented views
view1, view2 = augment(images), augment(images)
loss, metrics = model(view1, view2)
```

See `nexus/models/ssl/vicreg.py` for full implementation.

## Optimization Tricks

### 1. Hyperparameter Values

Standard values that work well:
```python
lambda_param = 25.0  # Invariance weight
mu_param = 25.0      # Variance weight  
nu_param = 1.0       # Covariance weight
variance_threshold = 1.0
```

### 2. Projection Dimension

Use high-dimensional projector:
```python
proj_dim = 8192  # Higher is better
```

### 3. Batch Size

VICReg works with smaller batches than contrastive methods:
```python
batch_size = 256  # vs 4096 for SimCLR
```

### 4. Augmentations

Strong augmentations are important:
- Random crop
- Color jitter
- Gaussian blur
- Solarization

## Experiments & Results

### ImageNet-1K Linear Probing

| Method | ViT-B | ResNet-50 | Batch Size |
|--------|-------|-----------|------------|
| SimCLR | 75.3% | 69.3% | 4096 |
| VICReg | **75.5%** | **69.7%** | **256** |

VICReg achieves similar performance with 16× smaller batches!

### Ablation: Loss Components

| Variant | Top-1 Acc |
|---------|-----------|
| Invariance only | 45.2% (collapse) |
| Inv + Var | 67.3% |
| Inv + Cov | 68.9% |
| VICReg (all three) | **75.5%** |

All three terms are necessary.

## Common Pitfalls

### 1. Variance Collapse

**Problem**: Variance loss not strong enough, dimensions collapse
**Solution**: Increase `mu_param` (variance weight)

### 2. Large Batch Requirements

**Problem**: Covariance/variance unstable with tiny batches
**Solution**: Use batch size ≥ 64

### 3. Weak Augmentations

**Problem**: Poor invariance learning
**Solution**: Use strong augmentation pipeline

## References

```bibtex
@inproceedings{bardes2022vicreg,
  title={VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning},
  author={Bardes, Adrien and Ponce, Jean and LeCun, Yann},
  booktitle={ICLR},
  year={2022}
}
```

**Official Code**: https://github.com/facebookresearch/vicreg
**Nexus Implementation**: `nexus/models/ssl/vicreg.py`
