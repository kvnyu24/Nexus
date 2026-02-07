# VICReg: Variance-Invariance-Covariance Regularization

## Overview

VICReg (Bardes et al., 2022) is a self-supervised learning objective that avoids representational collapse through three explicit regularization terms, without requiring negative samples or asymmetric architectures.

## Three Regularization Terms

### 1. Variance (V)
**Goal**: Prevent collapse to constant embeddings

$$\\mathcal{L}_v = \\frac{1}{d} \\sum_{j=1}^d \\max(0, \\gamma - \\sqrt{\\text{Var}(z_j^1) + \\epsilon}) + \\max(0, \\gamma - \\sqrt{\\text{Var}(z_j^2) + \\epsilon})$$

Where:
- $z^1, z^2$: Embeddings from two augmented views
- $\\gamma$: Target variance (typically 1.0)
- Ensures each dimension has minimum variance $\\gamma$

**Intuition**: Keep embeddings "spread out" in each dimension.

### 2. Invariance (I)
**Goal**: Make embeddings invariant to augmentations

$$\\mathcal{L}_i = \\frac{1}{n} \\sum_{i=1}^n \\|z_i^1 - z_i^2\\|^2$$

Simple MSE between paired embeddings.

**Intuition**: Same image (different augmentation) → same embedding.

### 3. Covariance (C)
**Goal**: Decorrelate embedding dimensions

$$\\mathcal{L}_c = \\frac{1}{d} \\sum_{i \\neq j} [\\text{Cov}(z^1)]_{ij}^2 + [\\text{Cov}(z^2)]_{ij}^2$$

Penalize off-diagonal covariance.

**Intuition**: Dimensions should be independent (no redundancy).

### Combined Loss

$$\\mathcal{L} = \\lambda_i \\mathcal{L}_i + \\lambda_v \\mathcal{L}_v + \\lambda_c \\mathcal{L}_c$$

**Default Weights**: $\\lambda_i = 25.0$, $\\lambda_v = 25.0$, $\\lambda_c = 1.0$

## Mathematical Details

### Variance Calculation

$$\\text{Var}(z_j) = \\frac{1}{n-1} \\sum_{i=1}^n (z_{ij} - \\bar{z}_j)^2$$

**Hinge loss** on standard deviation:
- If $\\text{std}(z_j) < \\gamma$: Penalize
- If $\\text{std}(z_j) \\geq \\gamma$: No penalty

### Covariance Matrix

$$\\text{Cov}(z) = \\frac{1}{n-1} (z - \\bar{z})^T (z - \\bar{z})$$

**Penalize off-diagonal**:
$$\\mathcal{L}_c = \\frac{1}{d} \\sum_{i \\neq j} \\text{Cov}(z)_{ij}^2$$

Diagonal (variance of each dim) not penalized.

## Implementation

### Basic Usage

```python
from nexus.training.losses import VICRegLoss

loss_fn = VICRegLoss(
    sim_coeff=25.0,  # Invariance weight
    std_coeff=25.0,  # Variance weight
    cov_coeff=1.0,  # Covariance weight
    variance_target=1.0
)

# Two augmented views
z1 = encoder(augment1(images))  # (batch, dim)
z2 = encoder(augment2(images))  # (batch, dim)

loss = loss_fn(z1, z2)
```

### Self-Supervised Training

```python
class VICRegModel(nn.Module):
    def __init__(self, encoder_dim=2048, proj_dim=8192):
        super().__init__()
        # Encoder (e.g., ResNet)
        self.encoder = ResNet50()
        
        # Expander (projector)
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, proj_dim)
        )
        
        self.loss_fn = VICRegLoss()
    
    def forward(self, x1, x2):
        # Encode
        repr1 = self.encoder(x1)
        repr2 = self.encoder(x2)
        
        # Project to high-dim space
        z1 = self.projector(repr1)
        z2 = self.projector(repr2)
        
        # VICReg loss
        loss = self.loss_fn(z1, z2)
        
        return loss

# Training
model = VICRegModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)

for batch in dataloader:
    x1, x2 = augment(batch)  # Two augmented views
    loss = model(x1, x2)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Hyperparameters

### Loss Coefficients

**Invariance ($\\lambda_i$)**: 25.0
- **Higher**: Stronger invariance to augmentations
- **Lower**: More tolerance for augmentation differences

**Variance ($\\lambda_v$)**: 25.0
- **Higher**: Stronger anti-collapse
- **Lower**: More flexibility in embedding distribution

**Covariance ($\\lambda_c$)**: 1.0
- **Higher**: Stronger decorrelation
- **Lower**: Allow some correlation between dimensions

### Variance Target ($\\gamma$)

**Default**: 1.0

**Effect**: Minimum standard deviation per dimension
- **Higher**: Forces more spread-out embeddings
- **Lower**: Allows more compact embeddings

### Projector Dimension

**Recommended**: 8192 (large!)

**Why Large?**:
- More dimensions → easier to decorrelate
- Prevents collapse by providing more "room"
- Only used during training (not inference)

## Advantages

### 1. No Negative Samples

Unlike InfoNCE/SimCLR:
- **No large batch needed**
- **No memory bank**
- **Simpler implementation**

### 2. Symmetric Architecture

Unlike BYOL/SimSiam:
- **No momentum encoder**
- **No predictor network**
- **Both views treated equally**

### 3. Explicit Regularization

- **Interpretable**: Three clear objectives
- **Debuggable**: Can monitor each term separately
- **Flexible**: Easy to adjust weights

## Performance

**ImageNet-1K** (ResNet-50, 1000 epochs):
- VICReg: 73.2% top-1 accuracy
- SimCLR (baseline): 69.3%
- Barlow Twins: 73.2%

**Key Findings**:
- Matches state-of-the-art
- Works with small batches (256)
- Stable training (explicit collapse prevention)

## Monitoring

```python
# During training, monitor each component
for batch in dataloader:
    z1, z2 = model(x1, x2)
    
    # Compute individual losses
    sim_loss = F.mse_loss(z1, z2)
    std_loss = compute_std_loss(z1, z2)
    cov_loss = compute_cov_loss(z1, z2)
    
    total_loss = 25.0 * sim_loss + 25.0 * std_loss + 1.0 * cov_loss
    
    # Log
    print(f"Sim: {sim_loss:.4f}, Std: {std_loss:.4f}, Cov: {cov_loss:.4f}")
    
    # Check for collapse
    if std_loss > 10.0:  # Warning: variance collapsing
        print("WARNING: Variance loss high, possible collapse!")
```

## Comparison to Other Methods

| Method | Negatives | Asymmetry | Batch Size | Stability |
|--------|-----------|-----------|------------|-----------|
| **SimCLR** | Yes | No | Large (>1K) | Good |
| **BYOL** | No | Yes (momentum) | Any | Good |
| **Barlow Twins** | No | No | Any | Excellent |
| **VICReg** | No | No | Any | Excellent |

**VICReg vs Barlow Twins**: Very similar, both push cross-correlation to identity
- Barlow Twins: One loss (cross-correlation)
- VICReg: Three losses (variance, invariance, covariance)
- VICReg: Slightly more explicit and interpretable

## When to Use

**Best for**:
- Self-supervised pre-training without negatives
- Small-to-medium batch sizes
- When you want interpretable objectives
- Stable training without tricks

**Alternatives**:
- **SimCLR**: If you have large batch sizes
- **BYOL**: If you prefer momentum encoders
- **Barlow Twins**: Similar approach, slightly simpler

## References

**VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning**  
Adrien Bardes, Jean Ponce, Yann LeCun, ICLR 2022  
https://arxiv.org/abs/2105.04906

**Key Insight**: Explicit regularization on variance, invariance, and covariance prevents collapse without negatives or asymmetry.

**Implementation**: `nexus/training/losses.py` (VICRegLoss class)
