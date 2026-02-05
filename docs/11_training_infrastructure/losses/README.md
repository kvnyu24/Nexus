# Loss Functions

Specialized loss functions for contrastive learning and self-supervised training.

## Overview

Modern loss functions beyond cross-entropy:
- Contrastive losses for representation learning
- Self-supervised losses without labels
- Metric learning for embeddings

## Loss Function Comparison

| Loss | Type | Key Feature | Best Use Case |
|------|------|-------------|---------------|
| **InfoNCE** | Contrastive | Softmax over negatives | CLIP-style training |
| **SigLIP** | Contrastive | Per-pair sigmoid | Scalable contrastive learning |
| **VICReg** | Self-supervised | Explicit regularization | SSL without negatives |
| **Barlow Twins** | Self-supervised | Cross-correlation | SSL without negatives |

## Quick Start

### InfoNCE (Contrastive Loss)

```python
from nexus.training.losses import InfoNCELoss

loss_fn = InfoNCELoss(temperature=0.07)

# For image-text pairs
loss = loss_fn(
    anchors=image_features,      # (B, D)
    positives=text_features,     # (B, D)
    negatives=negative_features, # (N, D)
)
```

### SigLIP Loss (Scalable Contrastive)

```python
from nexus.training.losses import SigmoidContrastiveLoss

loss_fn = SigmoidContrastiveLoss(temperature=0.1)

# More scalable than InfoNCE (no global normalization)
loss = loss_fn(
    features_a=image_features,  # (B, D)
    features_b=text_features,   # (B, D)
)
```

### VICReg (Self-Supervised)

```python
from nexus.training.losses import VICRegLoss

loss_fn = VICRegLoss(
    sim_coeff=25.0,   # Invariance weight
    std_coeff=25.0,   # Variance weight
    cov_coeff=1.0,    # Covariance weight
)

# For two augmented views
loss = loss_fn(z1=view1_embeddings, z2=view2_embeddings)
```

## Detailed Documentation

- [InfoNCE](infonce.md) - Contrastive loss for representation learning
- [SigLIP Loss](siglip.md) - Sigmoid-based contrastive loss
- [VICReg](vicreg.md) - Variance-invariance-covariance regularization

## Performance Comparison

### Contrastive Learning (ImageNet)

| Loss | Batch Size | Linear Probe Acc | Comm Cost |
|------|------------|------------------|-----------|
| InfoNCE | 4096 | 72.3% | High (all-gather) |
| SigLIP | 4096 | 72.8% | Low (no all-gather) |
| VICReg | 256 | 71.5% | None (no negatives) |

## References

See individual loss documentation for detailed references.
