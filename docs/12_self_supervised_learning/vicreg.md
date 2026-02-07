# VICReg: Variance-Invariance-Covariance Regularization

## Overview & Motivation

VICReg is a non-contrastive self-supervised learning method that prevents representational collapse through three explicit regularization terms: variance (maintains spread), invariance (matches augmented views), and covariance (decorrelates dimensions). Unlike contrastive methods, VICReg doesn't require negative pairs, momentum encoders, or large batch sizes.

### Key Innovation

**Explicit regularization** replaces complex architectural tricks:
- Variance term: Prevents dimension collapse
- Invariance term: Matches augmented views
- Covariance term: Decorrelates features
- Simple, stable, works with small batches

### Historical Context

VICReg emerged as a refinement of ideas from Barlow Twins, making the collapse prevention mechanisms more explicit and interpretable. While Barlow Twins uses a cross-correlation matrix, VICReg decomposes the objective into three interpretable components:

1. **Invariance**: Explicit distance minimization (simple MSE)
2. **Variance**: Explicit variance maintenance (hinge loss)
3. **Covariance**: Explicit decorrelation (covariance minimization)

This decomposition provides:
- **Better interpretability**: Each term has clear meaning
- **Independent control**: Tune each objective separately
- **Diagnostic clarity**: Easy to identify which component is failing
- **Flexible design**: Can adapt terms for specific use cases

## Theoretical Background

### Three Pillars of VICReg

VICReg prevents collapse through three complementary mechanisms:

**1. Invariance**: Ensures augmented views have similar representations
- Without this: Model could ignore input entirely
- Mechanism: MSE loss between view embeddings

**2. Variance**: Ensures each dimension has sufficient spread
- Without this: All embeddings collapse to a single point
- Mechanism: Hinge loss on standard deviation

**3. Covariance**: Ensures dimensions are decorrelated
- Without this: All dimensions encode the same information
- Mechanism: Penalty on off-diagonal covariance

### Why All Three Are Needed

**Missing Invariance**:
```
Result: Random, unstructured representations
Problem: No connection between augmented views
```

**Missing Variance**:
```
Result: Complete collapse (all outputs identical)
Problem: No diversity in representations
```

**Missing Covariance**:
```
Result: Redundant features (all dimensions correlated)
Problem: Inefficient use of representation capacity
```

### Information Theory Perspective

VICReg can be viewed as:

```
maximize I(Z; X)  subject to  H(Z) ≥ threshold
```

Where:
- I(Z; X): Mutual information between representations and inputs
- H(Z): Entropy of representations (maintained by variance)
- Decorrelation: Ensures entropy is spread across all dimensions

This differs from Barlow Twins which implicitly enforces these constraints through the cross-correlation matrix structure.

## Mathematical Formulation

### VICReg Loss

```
L_VICReg = λ·L_inv + μ·L_var + ν·L_cov
```

Standard hyperparameters: λ=25, μ=25, ν=1

### 1. Invariance Loss (Similarity)

MSE between embeddings of two views:

```
L_inv = (1/B) Σ_i ||z¹_i - z²_i||²
```

Where:
- B: Batch size
- z¹_i, z²_i: Embeddings of two views for sample i
- ||·||²: L2 norm squared

**Properties**:
- Simple and interpretable
- Differentiable everywhere
- Scale-invariant when combined with variance term

### 2. Variance Loss (Spread)

Hinge loss to maintain variance above threshold:

```
L_var = (1/D) Σ_d max(0, γ - σ(z_d))
```

Where:
- D: Embedding dimension
- σ(z_d): Standard deviation of dimension d across batch
- γ: Variance threshold (typically 1.0)

**Standard deviation computation**:
```
σ(z_d) = sqrt((1/B) Σ_i (z_i,d - μ_d)²)
μ_d = (1/B) Σ_i z_i,d
```

**Why hinge loss**:
- Allows variance to exceed threshold (no penalty)
- Only activates when variance is too low
- Prevents over-regularization

### 3. Covariance Loss (Decorrelation)

Penalize off-diagonal covariance:

```
L_cov = (1/D) Σ_{d≠d'} Cov(z)_{d,d'}²
```

**Covariance matrix computation**:
```
Cov(z)_{d,d'} = (1/B) Σ_i (z_i,d - μ_d)(z_i,d' - μ_d')
```

Where:
- Cov(z): D × D covariance matrix
- Only off-diagonal elements are penalized
- Encourages independence between dimensions

### Complete Loss Derivation

For two views z¹ and z², the full loss is:

```
L = λ · (1/B) Σ_i ||z¹_i - z²_i||²
  + μ · (1/D) [Σ_d max(0, γ - σ(z¹_d)) + Σ_d max(0, γ - σ(z²_d))]
  + ν · (1/D) [Σ_{d≠d'} Cov(z¹)²_{d,d'} + Σ_{d≠d'} Cov(z²)²_{d,d'}]
```

Note: Variance and covariance are computed separately for each view and averaged.

### Gradient Analysis

**Invariance gradient**:
```
∂L_inv/∂z¹_i = (2λ/B)(z¹_i - z²_i)
```
Pulls views together.

**Variance gradient**:
```
∂L_var/∂z_i,d ∝ -(z_i,d - μ_d)  if σ(z_d) < γ
              = 0              otherwise
```
Pushes samples away from mean when variance is low.

**Covariance gradient**:
```
∂L_cov/∂z_i,d ∝ Σ_{d'≠d} Cov(z)_{d,d'}(z_i,d' - μ_d')
```
Decorrelates dimensions.

## High-Level Intuition

Think of VICReg like organizing a team of specialists:

1. **Invariance**: All team members agree on the main task (augmented views = same input)
2. **Variance**: Each specialist is actively engaged (not idle/collapsed)
3. **Covariance**: Each specialist has a unique role (no redundancy)

### Real-World Analogy

Imagine hiring employees for a company:

**Bad hiring (without VICReg)**:
- Everyone does the same job (high covariance)
- Some employees do nothing (low variance)
- No one understands the company mission (no invariance)

**Good hiring (with VICReg)**:
- Everyone understands company goals (invariance)
- Everyone actively contributes (variance)
- Each has unique responsibilities (low covariance)

## Implementation Details

### Network Architecture

**Encoder**: Any backbone (ResNet, ViT, etc.)
- ResNet-50: Standard for ImageNet
- ViT-B/16: For transformer-based learning
- Output: 2048 (ResNet-50) or 768 (ViT-B)

**Projector (Expander)**:
```
Input: encoder_dim (e.g., 2048)
Hidden Layer 1: Linear(encoder_dim, 8192) + BatchNorm + ReLU
Hidden Layer 2: Linear(8192, 8192) + BatchNorm + ReLU
Output Layer: Linear(8192, 8192) + BatchNorm + ReLU
Final dimension: 8192
```

**Why "Expander"**:
- Increases dimensionality (2048 → 8192)
- Provides more capacity for decorrelation
- Higher dimensions reduce spurious correlations

### Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VICReg(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder_dim = config["encoder_dim"]
        self.proj_hidden_dim = config.get("projector_hidden_dim", 8192)
        self.proj_output_dim = config.get("projector_output_dim", 8192)

        # Loss weights
        self.lambda_param = config.get("lambda_param", 25.0)  # Invariance
        self.mu_param = config.get("mu_param", 25.0)          # Variance
        self.nu_param = config.get("nu_param", 1.0)           # Covariance
        self.gamma = config.get("variance_threshold", 1.0)

        # Encoder
        self.encoder = self._build_encoder(config)

        # Projector (expander)
        self.projector = nn.Sequential(
            nn.Linear(self.encoder_dim, self.proj_hidden_dim),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(self.proj_hidden_dim, self.proj_hidden_dim),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
            nn.BatchNorm1d(self.proj_output_dim),
            nn.ReLU(inplace=True)
        )

    def _build_encoder(self, config):
        """Build encoder backbone"""
        if config.get("backbone") == "resnet50":
            from torchvision.models import resnet50
            encoder = resnet50(pretrained=False)
            encoder.fc = nn.Identity()
            return encoder
        elif config.get("backbone") == "vit":
            from timm import create_model
            encoder = create_model('vit_base_patch16_224', pretrained=False)
            encoder.head = nn.Identity()
            return encoder
        else:
            raise ValueError(f"Unknown backbone: {config.get('backbone')}")

    def forward(self, x1, x2):
        """
        Args:
            x1, x2: Two augmented views (B, C, H, W)

        Returns:
            loss: Total VICReg loss
            metrics: Dictionary of individual loss components
        """
        # Get representations
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        # Compute loss
        loss, metrics = self.vicreg_loss(z1, z2)

        return loss, metrics

    def vicreg_loss(self, z1, z2):
        """
        Compute VICReg loss

        Args:
            z1, z2: Embeddings (B, D)

        Returns:
            loss: Total loss
            metrics: Dictionary of loss components
        """
        # Invariance loss
        inv_loss = F.mse_loss(z1, z2)

        # Variance loss
        var_loss = self.variance_loss(z1) + self.variance_loss(z2)
        var_loss = var_loss / 2  # Average over both views

        # Covariance loss
        cov_loss = self.covariance_loss(z1) + self.covariance_loss(z2)
        cov_loss = cov_loss / 2  # Average over both views

        # Total loss
        loss = (self.lambda_param * inv_loss +
                self.mu_param * var_loss +
                self.nu_param * cov_loss)

        metrics = {
            'loss': loss.item(),
            'inv_loss': inv_loss.item(),
            'var_loss': var_loss.item(),
            'cov_loss': cov_loss.item(),
            'std_mean': z1.std(dim=0).mean().item(),
        }

        return loss, metrics

    def variance_loss(self, z):
        """
        Variance loss: encourages std > gamma for each dimension

        Args:
            z: Embeddings (B, D)

        Returns:
            loss: Variance loss
        """
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        loss = torch.mean(F.relu(self.gamma - std))
        return loss

    def covariance_loss(self, z):
        """
        Covariance loss: penalizes off-diagonal covariance

        Args:
            z: Embeddings (B, D)

        Returns:
            loss: Covariance loss
        """
        B, D = z.shape

        # Center the embeddings
        z = z - z.mean(dim=0)

        # Compute covariance matrix
        cov = (z.T @ z) / (B - 1)

        # Off-diagonal penalty
        off_diag = cov.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
        loss = off_diag.pow_(2).sum() / D

        return loss


def off_diagonal_alternative(cov):
    """Alternative way to get off-diagonal elements"""
    D = cov.shape[0]
    mask = ~torch.eye(D, dtype=bool, device=cov.device)
    return cov[mask]
```

### Augmentation Pipeline

VICReg uses strong augmentations similar to other SSL methods:

```python
from torchvision import transforms

def get_vicreg_augmentation(img_size=224):
    """
    Augmentation pipeline for VICReg
    """
    # Color jitter
    color_jitter = transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.2,
        hue=0.1
    )

    # Gaussian blur
    gaussian_blur = transforms.GaussianBlur(
        kernel_size=23,
        sigma=(0.1, 2.0)
    )

    # Solarization
    solarize = transforms.RandomSolarize(threshold=128, p=0.2)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.08, 1.0),
            ratio=(3./4., 4./3.)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([gaussian_blur], p=0.5),
        solarize,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform
```

### Code Reference

```python
from nexus.models.ssl import VICRegModel

config = {
    "backbone": "resnet50",
    "encoder_dim": 2048,
    "projector_hidden_dim": 8192,
    "projector_output_dim": 8192,
    "lambda_param": 25.0,  # Invariance
    "mu_param": 25.0,      # Variance
    "nu_param": 1.0,       # Covariance
    "variance_threshold": 1.0,
}

model = VICRegModel(config)

# Two augmented views
augment = get_vicreg_augmentation()
view1, view2 = augment(images), augment(images)

# Training
loss, metrics = model(view1, view2)
```

See `nexus/models/ssl/vicreg.py` for full implementation.

## Training Procedures

### Pre-training Setup

**Dataset**: ImageNet-1K

**Batch Size**: 2048 (can work with smaller, even 256)

**Epochs**: 1000 for ResNet-50

**Optimizer**:
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-3,  # Base learning rate
    weight_decay=0.04,
    betas=(0.9, 0.999)
)
```

**Learning Rate Schedule**:
```python
# Cosine decay with linear warmup
warmup_epochs = 10

def lr_schedule(epoch, total_epochs=1000):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + cos(pi * progress))
```

### Complete Training Loop

```python
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def train_vicreg(model, train_loader, epochs=1000):
    model = model.cuda()
    model = nn.DataParallel(model)

    # AdamW optimizer
    base_lr = 2e-3
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr * (batch_size / 256),  # Linear scaling
        weight_decay=0.04,
        betas=(0.9, 0.999)
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_inv = 0
        total_var = 0
        total_cov = 0

        # Update learning rate
        lr = base_lr * lr_schedule(epoch, epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for batch_idx, (images, _) in enumerate(train_loader):
            # Create two augmented views
            x1 = augment(images).cuda()
            x2 = augment(images).cuda()

            # Forward pass
            loss, metrics = model(x1, x2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += metrics['loss']
            total_inv += metrics['inv_loss']
            total_var += metrics['var_loss']
            total_cov += metrics['cov_loss']

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {metrics['loss']:.4f} "
                      f"Inv: {metrics['inv_loss']:.4f} "
                      f"Var: {metrics['var_loss']:.4f} "
                      f"Cov: {metrics['cov_loss']:.4f} "
                      f"StdMean: {metrics['std_mean']:.4f}")

        # Epoch statistics
        n = len(train_loader)
        print(f"Epoch {epoch} - "
              f"Loss: {total_loss/n:.4f} "
              f"Inv: {total_inv/n:.4f} "
              f"Var: {total_var/n:.4f} "
              f"Cov: {total_cov/n:.4f}")

        # Save checkpoint
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / n,
            }, f'vicreg_epoch_{epoch}.pt')
```

## Hyperparameter Guidelines

### Standard Configuration

```python
# Standard values that work well across datasets
config = {
    "lambda_param": 25.0,  # Invariance weight
    "mu_param": 25.0,      # Variance weight
    "nu_param": 1.0,       # Covariance weight
    "variance_threshold": 1.0,
    "projector_hidden_dim": 8192,
    "projector_output_dim": 8192,
}
```

### Lambda (λ) - Invariance Weight

Controls how strongly views should match:

```python
lambda_param = 25.0  # Standard

# Effect of different values:
# λ = 10:  Weaker invariance, more diverse features
# λ = 25:  Standard, balanced
# λ = 50:  Stronger invariance, risk of over-alignment
```

**How to tune**:
- Monitor invariance loss (should be low, ~0.1-0.5)
- If inv_loss is high: Increase λ
- If features are too similar: Decrease λ

### Mu (μ) - Variance Weight

Controls dimension spread:

```python
mu_param = 25.0  # Standard

# Effect of different values:
# μ = 10:  Weaker variance enforcement
# μ = 25:  Standard, prevents collapse
# μ = 50:  Stronger variance, may conflict with invariance
```

**How to tune**:
- Monitor std_mean (should be > 1.0)
- If std_mean < 0.5: Increase μ (variance collapsing)
- If std_mean > 2.0: Can decrease μ

### Nu (ν) - Covariance Weight

Controls feature decorrelation:

```python
nu_param = 1.0  # Standard (lower than λ, μ)

# Effect of different values:
# ν = 0.5:  Weaker decorrelation
# ν = 1.0:  Standard
# ν = 2.0:  Stronger decorrelation
```

**Why ν is smaller**:
- Covariance term is naturally larger in magnitude
- Lower weight provides balance
- Prevents over-penalization of correlations

### Projection Dimension

Use high-dimensional projector:

```python
proj_dim = 8192  # Standard

# Why high-dimensional?
# - More capacity for decorrelation
# - Reduces spurious correlations
# - Better representation learning

# Options:
# 4096: Minimum recommended
# 8192: Standard, best results
# 16384: Can improve but memory-intensive
```

### Batch Size Flexibility

VICReg works with smaller batches than other methods:

```python
# Recommended batch sizes:
batch_size = 256   # Minimum, works well
batch_size = 512   # Good
batch_size = 1024  # Better
batch_size = 2048  # Best (but not required)

# Scale learning rate:
lr = base_lr * (batch_size / 256)
```

**Advantage over SimCLR/MoCo**:
- SimCLR needs 4096+ batch size
- VICReg works well with 256

## Optimization Tricks

### 1. AdamW Optimizer

VICReg works well with AdamW (unlike Barlow Twins which needs LARS):

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-3,
    weight_decay=0.04,
    betas=(0.9, 0.999)
)
```

**Why AdamW works**:
- Explicit variance term stabilizes training
- Don't need LARS's per-layer adaptation
- Simpler and more widely available

### 2. Weight Decay

Moderate weight decay helps:

```python
weight_decay = 0.04  # Standard for ViT
weight_decay = 0.05  # Alternative for ResNet
```

### 3. Learning Rate Warmup

Linear warmup for 10 epochs:

```python
warmup_epochs = 10

def get_lr(epoch):
    if epoch < warmup_epochs:
        return base_lr * (epoch / warmup_epochs)
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + cos(pi * progress))
```

### 4. Mixed Precision Training

Significant speedup with AMP:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss, metrics = model(x1, x2)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 5. Gradient Clipping

For stability with large models:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 6. Numerical Stability

Small epsilon in variance computation:

```python
std = torch.sqrt(z.var(dim=0) + 1e-4)  # Avoid sqrt(0)
```

### 7. BatchNorm in Projector

Include BatchNorm for stability:

```python
projector = nn.Sequential(
    nn.Linear(dim_in, dim_hidden),
    nn.BatchNorm1d(dim_hidden),  # Important!
    nn.ReLU(),
    ...
)
```

## Experiments & Results

### ImageNet-1K Linear Probing

**ResNet-50**:
| Method | Batch Size | Epochs | Top-1 Acc | Top-5 Acc |
|--------|------------|--------|-----------|-----------|
| Supervised | 256 | 100 | 76.5% | 93.0% |
| SimCLR | 4096 | 1000 | 69.3% | 89.0% |
| BYOL | 4096 | 1000 | 71.2% | 90.1% |
| Barlow Twins | 2048 | 1000 | 71.5% | 90.4% |
| VICReg | 256 | 1000 | **69.7%** | **89.5%** |
| VICReg | 2048 | 1000 | **71.8%** | **90.6%** |

**Key Observation**: VICReg achieves competitive results even with small batch sizes!

**ViT-Base/16**:
| Method | Pre-train Epochs | Top-1 Acc | Top-5 Acc |
|--------|------------------|-----------|-----------|
| MAE | 800 | 67.8% | - |
| MoCo v3 | 300 | 76.7% | - |
| DINO | 400 | 78.2% | - |
| VICReg | 300 | **75.5%** | **92.8%** |
| VICReg | 1000 | **76.3%** | **93.1%** |

### Comparison with Other Methods

**ViT-Large/16**:
| Method | Params | Top-1 Acc | Training Time |
|--------|--------|-----------|---------------|
| Supervised | 304M | 82.6% | Fast |
| SimCLR | 304M | 78.3% | Slow (large batches) |
| VICReg | 304M | **79.1%** | Medium |

### Ablation Studies

**Loss Components** (ResNet-50, 1000 epochs):
| Variant | Top-1 Acc | Notes |
|---------|-----------|-------|
| Invariance only | 45.2% | Complete collapse |
| Inv + Var | 67.3% | Prevents collapse but redundant |
| Inv + Cov | 68.9% | Decorrelated but can collapse |
| VICReg (all three) | **71.8%** | All components needed |

**Hyperparameter Sensitivity**:

**Lambda (λ)**:
| λ | Inv Loss | Var Loss | Cov Loss | Top-1 Acc |
|---|----------|----------|----------|-----------|
| 10 | 0.8 | 0.05 | 0.03 | 69.5% |
| 25 | 0.3 | 0.05 | 0.03 | **71.8%** |
| 50 | 0.1 | 0.08 | 0.05 | 70.2% |

**Mu (μ)**:
| μ | StdMean | Collapse? | Top-1 Acc |
|---|---------|-----------|-----------|
| 10 | 0.6 | Some dims | 68.9% |
| 25 | 1.2 | No | **71.8%** |
| 50 | 1.8 | No | 71.3% |

**Nu (ν)**:
| ν | Cov Loss | Redundancy | Top-1 Acc |
|---|----------|------------|-----------|
| 0.5 | 0.08 | High | 70.5% |
| 1.0 | 0.03 | Low | **71.8%** |
| 2.0 | 0.01 | Very Low | 71.2% |

**Projection Dimension**:
| proj_dim | Params | Memory | Top-1 Acc |
|----------|--------|--------|-----------|
| 2048 | Low | 12GB | 68.3% |
| 4096 | Medium | 16GB | 70.1% |
| 8192 | High | 24GB | **71.8%** |
| 16384 | Very High | 40GB | 72.0% |

**Batch Size**:
| Batch Size | Top-1 Acc | Notes |
|------------|-----------|-------|
| 64 | 66.2% | Unstable |
| 128 | 68.5% | Usable |
| 256 | 69.7% | Good |
| 512 | 70.9% | Better |
| 2048 | **71.8%** | Best |

VICReg is much less sensitive to batch size than SimCLR!

### Transfer Learning Results

**COCO Object Detection** (Mask R-CNN, 1x schedule):
| Pre-training | APbox | APmask | APbox50 |
|--------------|-------|--------|---------|
| Supervised | 39.8 | 35.8 | 60.1 |
| MoCo v2 | 40.9 | 36.5 | 61.3 |
| VICReg | **41.5** | **37.0** | **61.8** |

**ADE20K Semantic Segmentation** (UperNet):
| Pre-training | mIoU | mAcc |
|--------------|------|------|
| Supervised | 47.4 | 58.1 |
| BEiT | 48.1 | 59.0 |
| VICReg | **48.3** | **59.2** |

**iNaturalist Fine-grained Classification**:
| Pre-training | Top-1 Acc |
|--------------|-----------|
| Supervised | 69.2% |
| SimCLR | 71.5% |
| VICReg | **72.3%** |

### Data Efficiency

**ImageNet with Limited Labels**:

**1% labels** (~13 images/class):
| Method | Top-1 Acc |
|--------|-----------|
| Supervised | 25.4% |
| SimCLR | 48.3% |
| VICReg | **51.7%** |

**10% labels** (~130 images/class):
| Method | Top-1 Acc |
|--------|-----------|
| Supervised | 56.4% |
| SimCLR | 65.6% |
| VICReg | **67.2%** |

## Common Pitfalls

### 1. Variance Collapse

**Problem**: Variance loss not strong enough, dimensions collapse

**Symptoms**:
- std_mean < 0.5
- Variance loss stays high
- Poor downstream performance
- Some dimensions have near-zero std

**Solution**: Increase `mu_param` (variance weight)

**Debugging**:
```python
# Check per-dimension std
stds = z.std(dim=0)
print(f"Min std: {stds.min():.4f}, Max std: {stds.max():.4f}")
print(f"Num collapsed dims (std < 0.1): {(stds < 0.1).sum()}")

# If many dims collapsed, increase mu
if (stds < 0.1).sum() > 0.1 * len(stds):
    print("Warning: Many dimensions collapsed, increase mu_param")
```

### 2. Unbalanced Loss Components

**Problem**: One loss term dominates others

**Symptoms**:
- One loss is 100x larger than others
- Training is unstable
- Poor convergence

**Solution**: Rebalance hyperparameters

**Debugging**:
```python
print(f"Inv: {inv_loss:.4f}, Var: {var_loss:.4f}, Cov: {cov_loss:.4f}")

# All three should be similar order of magnitude
# If not, adjust weights
```

**Good balance example**:
```
Inv: 0.3, Var: 0.05, Cov: 0.03  ✓
```

**Bad balance example**:
```
Inv: 10.0, Var: 0.001, Cov: 0.001  ✗ (inv dominates)
```

### 3. Weak Augmentations

**Problem**: Poor invariance learning with weak augmentations

**Symptoms**:
- Very low invariance loss quickly
- Poor downstream accuracy
- Model hasn't learned robust features

**Solution**: Use strong augmentation pipeline

**Code Fix**:
```python
# Bad
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
])

# Good
transform = get_vicreg_augmentation()  # Includes jitter, blur, etc.
```

### 4. Small Batch Size with Unstable Statistics

**Problem**: Batch statistics unreliable with very small batches

**Symptoms**:
- Noisy loss curves
- Unstable covariance estimates
- Poor batch normalization

**Solution**: Use batch_size ≥ 128 (preferably 256+)

### 5. Forgetting Numerical Stability

**Problem**: sqrt(0) in variance computation causes NaN

**Symptoms**:
- NaN losses during training
- Sudden divergence

**Solution**: Add epsilon

**Code Fix**:
```python
# Bad
std = torch.sqrt(z.var(dim=0))

# Good
std = torch.sqrt(z.var(dim=0) + 1e-4)
```

### 6. Wrong Covariance Normalization

**Problem**: Not centering embeddings before covariance

**Symptoms**:
- Covariance loss very large
- Poor decorrelation

**Solution**: Center embeddings

**Code Fix**:
```python
# Bad
cov = (z.T @ z) / B

# Good
z_centered = z - z.mean(dim=0)
cov = (z_centered.T @ z_centered) / (B - 1)
```

### 7. Not Monitoring Individual Losses

**Problem**: Only tracking total loss, missing component failures

**Symptoms**:
- Total loss decreases but performance poor
- Don't know which component is failing

**Solution**: Log all loss components

```python
wandb.log({
    'loss/total': loss,
    'loss/invariance': inv_loss,
    'loss/variance': var_loss,
    'loss/covariance': cov_loss,
    'stats/std_mean': std_mean,
})
```

### 8. Using Wrong Optimizer

**Problem**: Using SGD instead of AdamW

**Symptoms**:
- Need very careful LR tuning
- Slower convergence
- More sensitive to hyperparameters

**Solution**: Use AdamW

```python
# Bad (requires more tuning)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Good (more robust)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
```

## Advanced Topics

### VICReg for Vision Transformers

Adapt for ViT with proper feature extraction:

```python
class VICRegViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # ViT encoder
        self.encoder = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12
        )

        # Projector
        self.projector = build_projector(
            input_dim=768,
            hidden_dim=8192,
            output_dim=8192
        )

    def forward(self, x1, x2):
        # Extract CLS tokens or mean pooling
        feat1 = self.encoder(x1)[:, 0]  # CLS token
        feat2 = self.encoder(x2)[:, 0]

        # Project
        z1 = self.projector(feat1)
        z2 = self.projector(feat2)

        return self.vicreg_loss(z1, z2)
```

### VICReg with Local and Global Views

Multi-crop training for better efficiency:

```python
def multi_crop_vicreg(model, images):
    # Create global and local crops
    global_crops = [global_augment(img) for img in images[:2]]
    local_crops = [local_augment(img) for img in images[2:]]

    # Get embeddings
    global_z = [model.forward_one(crop) for crop in global_crops]
    local_z = [model.forward_one(crop) for crop in local_crops]

    # VICReg loss between globals
    loss = model.vicreg_loss(global_z[0], global_z[1])

    # Additional loss between global and local
    for local in local_z:
        loss += 0.5 * model.vicreg_loss(global_z[0], local)
        loss += 0.5 * model.vicreg_loss(global_z[1], local)

    return loss / (1 + len(local_z))
```

### VICReg for Multimodal Learning

Extend to vision-language learning:

```python
class VICRegMultimodal(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = VisionEncoder()
        self.text_encoder = TextEncoder()
        self.vision_proj = build_projector(2048, 8192, 8192)
        self.text_proj = build_projector(768, 8192, 8192)

    def forward(self, images, texts):
        # Single modality losses
        img1, img2 = augment(images), augment(images)
        txt1, txt2 = augment(texts), augment(texts)

        v1, v2 = self.vision_proj(self.vision_encoder(img1)), \
                 self.vision_proj(self.vision_encoder(img2))
        t1, t2 = self.text_proj(self.text_encoder(txt1)), \
                 self.text_proj(self.text_encoder(txt2))

        loss_vision = self.vicreg_loss(v1, v2)
        loss_text = self.vicreg_loss(t1, t2)

        # Cross-modal alignment
        loss_cross = self.vicreg_loss(v1, t1)

        return loss_vision + loss_text + loss_cross
```

### VICReg with Learnable Temperature

Adaptive weighting of loss components:

```python
class VICRegAdaptive(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... encoder and projector ...

        # Learnable loss weights (log scale)
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(25.0)))
        self.log_mu = nn.Parameter(torch.log(torch.tensor(25.0)))
        self.log_nu = nn.Parameter(torch.log(torch.tensor(1.0)))

    def vicreg_loss(self, z1, z2):
        inv_loss = F.mse_loss(z1, z2)
        var_loss = self.variance_loss(z1) + self.variance_loss(z2)
        cov_loss = self.covariance_loss(z1) + self.covariance_loss(z2)

        # Adaptive weights
        lambda_param = torch.exp(self.log_lambda)
        mu_param = torch.exp(self.log_mu)
        nu_param = torch.exp(self.log_nu)

        loss = (lambda_param * inv_loss +
                mu_param * var_loss / 2 +
                nu_param * cov_loss / 2)

        return loss
```

## Comparison with Related Methods

### VICReg vs Barlow Twins

| Aspect | VICReg | Barlow Twins |
|--------|--------|--------------|
| Invariance | Explicit MSE | Cross-correlation diagonal |
| Decorrelation | Covariance penalty | Cross-correlation off-diag |
| Variance | Explicit hinge loss | Implicit (via BN) |
| Hyperparameters | 3 weights | 1 weight (lambda) |
| Interpretability | High (separate terms) | Medium (unified matrix) |
| Batch size | Small OK (256+) | Larger needed (512+) |
| Optimizer | AdamW works | Needs LARS |
| Complexity | Slightly more | Simpler |

### VICReg vs SimCLR

| Aspect | VICReg | SimCLR |
|--------|--------|--------|
| Negative pairs | No | Yes |
| Batch size | 256+ | 4096+ |
| Memory | Low | High |
| Loss | Three explicit terms | Contrastive |
| Temperature | Not needed | Critical |
| Stability | High | Medium |

### VICReg vs BYOL

| Aspect | VICReg | BYOL |
|--------|--------|------|
| Momentum encoder | No | Yes |
| Predictor | No | Yes |
| Collapse prevention | Explicit (variance) | Implicit (architecture) |
| Simplicity | High | Medium |
| Training speed | Fast | Medium (EMA updates) |

## References

### Primary Paper

```bibtex
@inproceedings{bardes2022vicreg,
  title={VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning},
  author={Bardes, Adrien and Ponce, Jean and LeCun, Yann},
  booktitle={ICLR},
  year={2022},
  url={https://arxiv.org/abs/2105.04906}
}
```

### Related Work

```bibtex
@inproceedings{zbontar2021barlow,
  title={Barlow twins: Self-supervised learning via redundancy reduction},
  author={Zbontar, Jure and Jing, Li and Misra, Ishan and LeCun, Yann and Deny, St{\'e}phane},
  booktitle={ICML},
  year={2021},
  url={https://arxiv.org/abs/2103.03230}
}

@inproceedings{chen2020simple,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={ICML},
  year={2020},
  url={https://arxiv.org/abs/2002.05709}
}

@inproceedings{grill2020bootstrap,
  title={Bootstrap your own latent: A new approach to self-supervised learning},
  author={Grill, Jean-Bastien and Strub, Florian and Altch{\'e}, Florent and Tallec, Corentin and Richemond, Pierre H and Buchatskaya, Elena and Doersch, Carl and Pires, Bernardo Avila and Guo, Zhaohan Daniel and Azar, Mohammad Gheshlaghi and others},
  booktitle={NeurIPS},
  year={2020},
  url={https://arxiv.org/abs/2006.07733}
}

@inproceedings{caron2021emerging,
  title={Emerging properties in self-supervised vision transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J{\'e}gou, Herv{\'e} and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={ICCV},
  year={2021},
  url={https://arxiv.org/abs/2104.14294}
}

@article{bardes2023vcreg,
  title={VICRegL: Self-Supervised Learning of Local Visual Features},
  author={Bardes, Adrien and Ponce, Jean and LeCun, Yann},
  journal={NeurIPS},
  year={2022},
  url={https://arxiv.org/abs/2210.01571}
}

@inproceedings{balestriero2022vicreg,
  title={VICReg: Learning representations by maximizing information and minimizing variance across views},
  author={Balestriero, Randall and LeCun, Yann},
  booktitle={ICLR Workshop},
  year={2022}
}

@article{garrido2023vicreg,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023},
  url={https://arxiv.org/abs/2304.07193}
}

@inproceedings{he2020momentum,
  title={Momentum contrast for unsupervised visual representation learning},
  author={He, Kaiming and Fan, Haoqi and Wu, Yuxin and Xie, Saining and Girshick, Ross},
  booktitle={CVPR},
  year={2020},
  url={https://arxiv.org/abs/1911.05722}
}
```

**Official Code**: https://github.com/facebookresearch/vicreg

**Nexus Implementation**: `nexus/models/ssl/vicreg.py`

**Additional Resources**:
- [VICReg Explained (Blog)](https://ai.facebook.com/blog/vicreg-self-supervised-learning)
- [VICReg Paper Walkthrough](https://www.youtube.com/watch?v=xKQ8v8XG-iM)
- [VICRegL (Local Features)](https://arxiv.org/abs/2210.01571)
