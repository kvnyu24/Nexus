# Barlow Twins: Self-Supervised Learning via Redundancy Reduction

## Overview & Motivation

Barlow Twins learns representations by making the cross-correlation matrix between embeddings of augmented views approach the identity matrix. This simple objective naturally prevents collapse through redundancy reduction, inspired by neuroscience principles proposed by Horace Barlow (1961).

### Key Innovation

**Cross-correlation matrix objective**:
- On-diagonal: Should be 1 (invariance to augmentation)
- Off-diagonal: Should be 0 (decorrelation between features)
- No negative pairs, no momentum encoder needed
- Simple, elegant, and stable

### Historical Context

Barlow Twins draws inspiration from neuroscience, specifically Horace Barlow's redundancy reduction principle (1961), which suggests that the brain learns efficient representations by removing redundancy in neural responses. This principle has profound implications:

1. **Biological Plausibility**: Mimics how biological neurons decorrelate their responses
2. **Information Theory**: Maximizes information content by reducing redundancy
3. **Efficient Coding**: Learns compact, non-redundant representations
4. **Natural Collapse Prevention**: Decorrelation inherently prevents representational collapse

Unlike contrastive methods (SimCLR, MoCo) that require negative pairs, Barlow Twins achieves similar results through a simpler mechanism: making the cross-correlation matrix between twin networks approach the identity matrix.

## Theoretical Background

### Redundancy Reduction Principle

The core idea is that good representations should:
1. **Be invariant**: Similar inputs produce similar representations (on-diagonal = 1)
2. **Be decorrelated**: Different features capture different information (off-diagonal = 0)

This is formalized through the cross-correlation matrix:

```
C_ij = (Σ_b z^A_{b,i} z^B_{b,j}) / sqrt((Σ_b (z^A_{b,i})²)(Σ_b (z^B_{b,j})²))
```

Where:
- z^A, z^B: Embeddings from two augmented views
- b: Batch index
- i, j: Feature dimensions
- C: Cross-correlation matrix (D × D)

### Why This Prevents Collapse

**Collapse scenario**: All embeddings become identical
- Cross-correlation matrix: All entries = 1
- Loss: Both on-diagonal and off-diagonal terms penalized
- Contradiction: Cannot satisfy both objectives simultaneously

**Optimal solution**:
- On-diagonal = 1 (invariance achieved)
- Off-diagonal = 0 (decorrelation achieved)
- Result: Rich, diverse representations

### Information Theory Perspective

Barlow Twins can be viewed as maximizing:

```
I(Z^A; X) + I(Z^B; X) - I(Z^A; Z^B|X)
```

Where:
- I(Z^A; X): Information about input in view A
- I(Z^B; X): Information about input in view B
- I(Z^A; Z^B|X): Redundant information between views

This encourages learning all information about X while minimizing redundancy between features.

## Mathematical Formulation

### Barlow Twins Loss

```
L = Σᵢ (1 - Cᵢᵢ)² + λ Σᵢ Σⱼ≠ᵢ Cᵢⱼ²
     \_________/      \__________/
     Invariance      Redundancy reduction
```

Where:
- First term: Encourages on-diagonal elements to be 1
- Second term: Encourages off-diagonal elements to be 0
- λ: Trade-off parameter (typically 0.005)

### Cross-Correlation Matrix Computation

Step-by-step computation:

**1. Batch Normalization**:
```
z̃^A_i = (z^A_i - μ^A_i) / σ^A_i
z̃^B_i = (z^B_i - μ^B_i) / σ^B_i

Where:
μ^A_i = (1/B) Σ_b z^A_{b,i}
σ^A_i = sqrt((1/B) Σ_b (z^A_{b,i} - μ^A_i)²)
```

**2. Cross-Correlation**:
```
C_ij = (1/B) Σ_b z̃^A_{b,i} z̃^B_{b,j}
```

**3. Loss Computation**:
```
L_inv = Σᵢ (1 - C_ii)²
L_red = Σᵢ Σⱼ≠ᵢ C_ij²
L_total = L_inv + λ·L_red
```

### Gradient Analysis

The gradient with respect to embeddings:

```
∂L/∂z^A_i ∝ Σ_j [(2(1-C_ii)·∂C_ii/∂z^A_i) + 2λ·C_ij·∂C_ij/∂z^A_i]
```

This gradient:
- Pulls on-diagonal correlations toward 1
- Pushes off-diagonal correlations toward 0
- Naturally balances through λ

## High-Level Intuition

Think of Barlow Twins like teaching twins to be:
1. **Similar in essence** (both recognize "this is a cat")
2. **Different in details** (one notices fur, another notices eyes)

The cross-correlation matrix measures:
- **Diagonal**: How well the twins agree on each concept
- **Off-diagonal**: Whether they're learning redundant features

**Goal**: Perfect agreement on concepts (diagonal = 1) with no redundancy (off-diagonal = 0).

### Real-World Analogy

Imagine two students studying the same material:
- **Bad strategy**: Both memorize identical notes (high redundancy)
- **Good strategy**: One focuses on theory, another on examples (decorrelated but both understand)

Barlow Twins encourages the "good strategy" - learn everything but don't be redundant.

## Implementation Details

### Network Architecture

**Encoder**: Any backbone (ResNet, ViT)
- ResNet-50: Standard choice for ImageNet
- ViT-B/16: For transformer-based learning
- Output dim: 2048 (ResNet-50), 768 (ViT-B)

**Projection Head (3-layer MLP)**:
```
Input: encoder_dim (e.g., 2048)
Layer 1: Linear(2048, 8192) + BatchNorm + ReLU
Layer 2: Linear(8192, 8192) + BatchNorm + ReLU
Layer 3: Linear(8192, 8192) + BatchNorm (no ReLU, no affine)
Output: 8192-dim normalized embeddings
```

**Architecture Rationale**:
- 3 layers: Sufficient non-linearity
- 8192 dim: High-dimensional space reduces correlation by chance
- Final BatchNorm without affine: Ensures normalized outputs
- No final ReLU: Allows negative correlations

### Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BarlowTwins(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder_dim = config["encoder_dim"]
        self.proj_dim = config.get("proj_dim", 8192)
        self.lambd = config.get("lambd", 0.005)

        # Encoder (e.g., ResNet-50)
        self.encoder = self._build_encoder(config)

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(self.encoder_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.ReLU(inplace=True),

            nn.Linear(self.proj_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.ReLU(inplace=True),

            nn.Linear(self.proj_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim, affine=False)  # No affine
        )

    def _build_encoder(self, config):
        """Build encoder backbone"""
        if config.get("backbone") == "resnet50":
            from torchvision.models import resnet50
            encoder = resnet50(pretrained=False)
            encoder.fc = nn.Identity()  # Remove classification head
            return encoder
        elif config.get("backbone") == "vit":
            from timm import create_model
            encoder = create_model('vit_base_patch16_224', pretrained=False)
            encoder.head = nn.Identity()
            return encoder
        else:
            raise ValueError(f"Unknown backbone: {config.get('backbone')}")

    def forward(self, y1, y2):
        """
        Args:
            y1, y2: Two augmented views (B, C, H, W)

        Returns:
            loss: Barlow Twins loss
            metrics: Dictionary of metrics
        """
        # Get representations
        z1 = self.projector(self.encoder(y1))
        z2 = self.projector(self.encoder(y2))

        # Compute loss
        loss, metrics = self.barlow_twins_loss(z1, z2)

        return loss, metrics

    def barlow_twins_loss(self, z1, z2):
        """
        Compute Barlow Twins loss

        Args:
            z1, z2: Embeddings (B, D)

        Returns:
            loss: Scalar loss
            metrics: Dictionary containing on-diag and off-diag losses
        """
        batch_size = z1.shape[0]

        # Normalize by batch statistics
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)

        # Cross-correlation matrix
        c = torch.mm(z1_norm.T, z2_norm) / batch_size

        # Loss computation
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.lambd * off_diag

        metrics = {
            'loss': loss.item(),
            'on_diag_loss': on_diag.item(),
            'off_diag_loss': off_diag.item(),
            'cross_corr_mean': c.mean().item(),
            'cross_corr_std': c.std().item(),
        }

        return loss, metrics

    def off_diagonal(self, x):
        """Return off-diagonal elements of a square matrix"""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def off_diagonal_alternative(x):
    """Alternative implementation using masking"""
    n = x.shape[0]
    mask = ~torch.eye(n, dtype=bool, device=x.device)
    return x[mask]
```

### Augmentation Pipeline

Barlow Twins requires strong augmentations:

```python
from torchvision import transforms

def get_barlow_twins_augmentation(img_size=224):
    """
    Augmentation pipeline for Barlow Twins
    """
    # Color jitter parameters
    color_jitter = transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.2,
        hue=0.1
    )

    # Gaussian blur
    gaussian_blur = transforms.GaussianBlur(
        kernel_size=23,  # Large kernel
        sigma=(0.1, 2.0)
    )

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
from nexus.models.ssl import BarlowTwinsModel

config = {
    "backbone": "resnet50",
    "encoder_dim": 2048,
    "proj_dim": 8192,
    "lambd": 0.005,  # Off-diagonal weight
}

model = BarlowTwinsModel(config)

# Two augmented views
augment = get_barlow_twins_augmentation()
view1, view2 = augment(images), augment(images)

# Training
loss, metrics = model(view1, view2)
```

See `nexus/models/ssl/barlow_twins.py` for full implementation.

## Training Procedures

### Pre-training Setup

**Dataset**: ImageNet-1K

**Batch Size**: 2048 (distributed across GPUs)

**Epochs**: 1000 for ResNet-50

**Optimizer**:
```python
optimizer = torch.optim.LARS(
    model.parameters(),
    lr=0.2,  # Base LR
    weight_decay=1e-6,
    momentum=0.9
)
```

**Learning Rate Schedule**:
```python
# Linear warmup + cosine decay
warmup_epochs = 10

def lr_schedule(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        return 0.5 * (1 + cos(pi * (epoch - warmup_epochs) /
                              (total_epochs - warmup_epochs)))
```

### Complete Training Loop

```python
import torch
from torch.utils.data import DataLoader

def train_barlow_twins(model, train_loader, epochs=1000):
    model = model.cuda()
    model = nn.DataParallel(model)

    # LARS optimizer
    optimizer = torch.optim.LARS(
        model.parameters(),
        lr=0.2 * (batch_size / 256),  # Linear scaling
        weight_decay=1e-6,
        momentum=0.9
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_on_diag = 0
        total_off_diag = 0

        # Update learning rate
        lr = 0.2 * lr_schedule(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for batch_idx, (images, _) in enumerate(train_loader):
            # Create two augmented views
            y1 = augment(images).cuda()
            y2 = augment(images).cuda()

            # Forward pass
            loss, metrics = model(y1, y2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += metrics['loss']
            total_on_diag += metrics['on_diag_loss']
            total_off_diag += metrics['off_diag_loss']

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {metrics['loss']:.4f} "
                      f"OnDiag: {metrics['on_diag_loss']:.4f} "
                      f"OffDiag: {metrics['off_diag_loss']:.4f}")

        # Epoch statistics
        avg_loss = total_loss / len(train_loader)
        avg_on_diag = total_on_diag / len(train_loader)
        avg_off_diag = total_off_diag / len(train_loader)

        print(f"Epoch {epoch} - Loss: {avg_loss:.4f} "
              f"OnDiag: {avg_on_diag:.4f} OffDiag: {avg_off_diag:.4f}")

        # Save checkpoint
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'barlow_twins_epoch_{epoch}.pt')
```

## Hyperparameter Guidelines

### Lambda (λ) Parameter

Controls trade-off between invariance and decorrelation:

```python
lambd = 0.005  # Standard value

# Effect of different values:
# λ = 0.001: More invariance emphasis (higher on-diag loss tolerance)
# λ = 0.005: Balanced (optimal for most cases)
# λ = 0.01:  More decorrelation emphasis (higher off-diag penalty)
```

**How to tune**:
- Monitor on-diagonal and off-diagonal losses
- On-diagonal should be small (< 0.1)
- Off-diagonal should be small (< 0.01)
- If on-diagonal dominates: Decrease λ
- If off-diagonal dominates: Increase λ

### Projection Dimension

High-dimensional projections work better:

```python
proj_dim = 8192  # Standard

# Why high-dimensional?
# - More dimensions = less spurious correlation
# - Allows richer representations
# - Better decorrelation in high-D space

# Options:
# 2048: Fast but suboptimal
# 8192: Standard, best trade-off
# 16384: Better but memory-intensive
```

### Batch Size

Larger batches give better statistics:

```python
batch_size = 2048  # For ImageNet with ResNet-50

# Minimum recommended: 256
# Optimal: 2048-4096
# Scale learning rate linearly: lr = base_lr * (batch_size / 256)
```

### Learning Rate

Use LARS optimizer with high learning rate:

```python
base_lr = 0.2
lr = base_lr * (batch_size / 256)

# LARS is crucial for high learning rates
# Adam/SGD won't work well with lr=0.2
```

## Optimization Tricks

### 1. LARS Optimizer

Layer-wise Adaptive Rate Scaling (LARS) is crucial:

```python
from torch.optim import LARS  # Custom implementation needed

optimizer = LARS(
    model.parameters(),
    lr=0.2,
    weight_decay=1e-6,
    momentum=0.9,
    eta=0.001,  # Trust coefficient
)
```

**Why LARS**:
- Handles large batch training
- Adapts learning rate per layer
- Stable with high learning rates

### 2. Batch Normalization in Projector

Use BatchNorm in projector, not LayerNorm:
```python
# Correct
nn.BatchNorm1d(proj_dim)

# Wrong
nn.LayerNorm(proj_dim)
```

**Reason**: BN provides implicit normalization needed for cross-correlation

### 3. Final Layer BatchNorm

Last layer should have BN without affine transformation:
```python
nn.BatchNorm1d(proj_dim, affine=False)
```

**Why**: Prevents scale/shift parameters from interfering with correlation

### 4. Gradient Checkpointing

For large models, save memory:
```python
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(self, x):
    return checkpoint(self.projector, self.encoder(x))
```

### 5. Mixed Precision Training

Speeds up training significantly:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss, metrics = model(y1, y2)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 6. Learning Rate Warmup

Critical for stability:
```python
warmup_epochs = 10

for epoch in range(warmup_epochs):
    lr = base_lr * (epoch / warmup_epochs)
```

## Experiments & Results

### ImageNet-1K Pre-training

**ResNet-50 Linear Probing**:
| Method | Epochs | Batch Size | Top-1 Acc | Top-5 Acc |
|--------|--------|------------|-----------|-----------|
| Supervised | 100 | 256 | 76.5% | 93.0% |
| SimCLR | 1000 | 4096 | 69.3% | 89.0% |
| BYOL | 1000 | 4096 | 71.2% | 90.1% |
| SwAV | 800 | 4096 | 71.8% | 90.5% |
| Barlow Twins | 1000 | 2048 | **71.5%** | **90.4%** |

**Key Observations**:
- Competitive with BYOL/SwAV
- Simpler than both (no momentum encoder)
- Smaller batch size than SimCLR

### Comparison with Other Methods

**ViT-Base/16 Linear Probing**:
| Method | Negative Pairs | Momentum Encoder | Top-1 Acc |
|--------|----------------|------------------|-----------|
| SimCLR | ✅ | ❌ | 75.3% |
| MoCo v3 | ✅ | ✅ | 76.7% |
| DINO | ❌ | ✅ | 78.2% |
| BYOL | ❌ | ✅ | 74.3% |
| Barlow Twins | ❌ | ❌ | **75.8%** |

Barlow Twins is simpler yet achieves competitive performance!

### Ablation Studies

**Effect of Lambda (λ)**:
| λ | On-Diag Loss | Off-Diag Loss | Top-1 Acc |
|---|--------------|---------------|-----------|
| 0.001 | 0.08 | 0.15 | 69.2% |
| 0.005 | 0.05 | 0.05 | **71.5%** |
| 0.01 | 0.12 | 0.02 | 70.3% |
| 0.05 | 0.25 | 0.001 | 67.8% |

**Sweet spot**: λ = 0.005 balances both objectives

**Projection Dimension**:
| proj_dim | Params | Memory | Top-1 Acc | Training Speed |
|----------|--------|--------|-----------|----------------|
| 2048 | Low | Low | 68.9% | Fast |
| 4096 | Medium | Medium | 70.7% | Medium |
| 8192 | High | High | **71.5%** | Slow |
| 16384 | Very High | Very High | 71.7% | Very Slow |

**Trade-off**: 8192 is optimal for performance/efficiency

**Batch Size**:
| Batch Size | Top-1 Acc | Notes |
|------------|-----------|-------|
| 256 | 68.5% | Unstable statistics |
| 512 | 69.8% | Better |
| 1024 | 70.9% | Good |
| 2048 | **71.5%** | Optimal |
| 4096 | 71.6% | Marginal improvement |

**Minimum**: 512 for stable training

**Number of Projector Layers**:
| Layers | Top-1 Acc |
|--------|-----------|
| 2 | 69.8% |
| 3 | **71.5%** |
| 4 | 71.3% |

3 layers is sufficient.

### Transfer Learning

**COCO Object Detection (Faster R-CNN)**:
| Pre-training | APbox | APmask |
|--------------|-------|--------|
| Supervised | 39.8 | 35.8 |
| MoCo v2 | 40.9 | 36.5 |
| Barlow Twins | **41.3** | **36.9** |

**Places205 Classification**:
| Pre-training | Top-1 Acc |
|--------------|-----------|
| Supervised | 53.2% |
| SimCLR | 54.6% |
| Barlow Twins | **55.1%** |

### Efficiency Comparison

**Training Time** (100 epochs on 8x V100):
| Method | Time | Memory/GPU |
|--------|------|------------|
| SimCLR | 42h | 28GB |
| BYOL | 38h | 26GB |
| Barlow Twins | **35h** | **24GB** |

**Why faster**:
- No momentum encoder
- No negative pair computation
- Simple loss computation

## Common Pitfalls

### 1. Wrong Lambda

**Problem**: λ too high causes poor invariance, too low causes poor decorrelation

**Symptoms**:
- λ too high: High on-diagonal loss, poor representations
- λ too low: High off-diagonal loss, correlated features

**Solution**: Use λ = 0.005 (don't tune much)

**Debugging**:
```python
# Monitor both losses
print(f"On-diag: {on_diag_loss:.4f}, Off-diag: {off_diag_loss:.4f}")

# They should be balanced
if on_diag_loss > 0.1:
    print("Warning: High on-diag loss, consider decreasing lambda")
if off_diag_loss > 0.1:
    print("Warning: High off-diag loss, consider increasing lambda")
```

### 2. Low Projection Dimension

**Problem**: proj_dim < 4096 causes collapse or poor decorrelation

**Symptoms**:
- Off-diagonal correlations remain high
- Poor linear probing accuracy
- Features are redundant

**Solution**: Use proj_dim ≥ 8192

**Code Fix**:
```python
# Bad
config["proj_dim"] = 2048  # Too low

# Good
config["proj_dim"] = 8192  # Sufficient
```

### 3. Missing Batch Normalization

**Problem**: Cross-correlation unstable without BN in projector

**Symptoms**:
- Loss becomes NaN
- Training diverges
- Unstable cross-correlation values

**Solution**: Use BN in projector head

**Code Fix**:
```python
# Bad
self.projector = nn.Sequential(
    nn.Linear(encoder_dim, proj_dim),
    nn.ReLU(),  # No BN!
)

# Good
self.projector = nn.Sequential(
    nn.Linear(encoder_dim, proj_dim),
    nn.BatchNorm1d(proj_dim),
    nn.ReLU(),
)
```

### 4. Using Adam Instead of LARS

**Problem**: Adam doesn't work well with large batch sizes and high LR

**Symptoms**:
- Need to use very low learning rate
- Slow convergence
- Suboptimal performance

**Solution**: Use LARS optimizer

**Code Fix**:
```python
# Bad
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Good
optimizer = torch.optim.LARS(model.parameters(), lr=0.2)
```

### 5. Insufficient Augmentation

**Problem**: Weak augmentations lead to trivial solutions

**Symptoms**:
- Very low loss quickly
- Poor downstream performance
- Model hasn't learned useful features

**Solution**: Use strong augmentation pipeline

**Code Fix**:
```python
# Bad
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

# Good
transform = get_barlow_twins_augmentation()  # Strong augmentations
```

### 6. Small Batch Size

**Problem**: Batch statistics unreliable with batch_size < 256

**Symptoms**:
- Noisy loss
- Unstable training
- Poor batch normalization

**Solution**: Use batch_size ≥ 512

### 7. Final Layer Has Affine BN

**Problem**: Scale/shift in final BN interferes with correlation

**Symptoms**:
- Suboptimal performance
- Cross-correlation matrix not centered

**Solution**: Use `affine=False` in final BN

```python
# Bad
nn.BatchNorm1d(proj_dim)  # affine=True by default

# Good
nn.BatchNorm1d(proj_dim, affine=False)
```

### 8. Not Using Linear Warmup

**Problem**: High initial LR causes instability

**Symptoms**:
- Loss spikes early
- NaN values
- Divergence

**Solution**: Use 10-epoch linear warmup

```python
if epoch < 10:
    lr = base_lr * (epoch / 10)
```

## Advanced Topics

### Barlow Twins for Vision Transformers

Adapt for ViT architecture:

```python
class BarlowTwinsViT(nn.Module):
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

        # Same projector as ResNet version
        self.projector = build_projector(768, 8192)

    def forward(self, y1, y2):
        # Extract CLS tokens
        z1 = self.encoder(y1)[:, 0]  # CLS token
        z2 = self.encoder(y2)[:, 0]

        # Project
        z1 = self.projector(z1)
        z2 = self.projector(z2)

        return self.barlow_twins_loss(z1, z2)
```

### Multi-Crop Barlow Twins

Use multiple crops for better efficiency:

```python
def multi_crop_forward(self, images):
    # Global crops
    global_crops = [global_transform(img) for img in images]

    # Local crops
    local_crops = [local_transform(img) for img in images for _ in range(4)]

    # Get embeddings
    global_z = [self.forward_one(crop) for crop in global_crops]
    local_z = [self.forward_one(crop) for crop in local_crops]

    # Compute losses between all pairs
    loss = 0
    loss += barlow_loss(global_z[0], global_z[1])
    for local in local_z:
        loss += barlow_loss(global_z[0], local)
        loss += barlow_loss(global_z[1], local)

    return loss
```

### Barlow Twins for Other Modalities

Extend to audio, text, or multimodal:

```python
class BarlowTwinsMultimodal(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = ResNet50()
        self.text_encoder = BERT()
        self.projector = build_projector(768, 8192)

    def forward(self, images, captions):
        # Augment both modalities
        img1, img2 = augment_image(images)
        text1, text2 = augment_text(captions)

        # Encode
        v1, v2 = self.vision_encoder(img1), self.vision_encoder(img2)
        t1, t2 = self.text_encoder(text1), self.text_encoder(text2)

        # Project
        z_v1, z_v2 = self.projector(v1), self.projector(v2)
        z_t1, z_t2 = self.projector(t1), self.projector(t2)

        # Losses
        loss_vision = barlow_loss(z_v1, z_v2)
        loss_text = barlow_loss(z_t1, z_t2)
        loss_cross = barlow_loss(z_v1, z_t1)  # Align modalities

        return loss_vision + loss_text + loss_cross
```

## Comparison with Related Methods

### Barlow Twins vs VICReg

| Aspect | Barlow Twins | VICReg |
|--------|--------------|---------|
| Invariance | Cross-correlation diagonal | MSE loss |
| Decorrelation | Cross-correlation off-diagonal | Covariance matrix |
| Variance | Implicit (through BN) | Explicit hinge loss |
| Hyperparameters | 1 (lambda) | 3 (lambda, mu, nu) |
| Complexity | Simpler | More explicit |
| Performance | Similar | Similar |

### Barlow Twins vs SimCLR

| Aspect | Barlow Twins | SimCLR |
|--------|--------------|--------|
| Negative pairs | No | Yes |
| Batch size | Medium (2048) | Large (4096+) |
| Memory | Lower | Higher |
| Simplicity | High | Medium |
| Performance | Better | Good |

### Barlow Twins vs BYOL

| Aspect | Barlow Twins | BYOL |
|--------|--------------|------|
| Momentum encoder | No | Yes |
| Complexity | Simple | Complex |
| Predictor | No | Yes |
| Stability | High | Medium (needs predictor) |
| Performance | Similar | Similar |

## References

### Primary Paper

```bibtex
@inproceedings{zbontar2021barlow,
  title={Barlow twins: Self-supervised learning via redundancy reduction},
  author={Zbontar, Jure and Jing, Li and Misra, Ishan and LeCun, Yann and Deny, St{\'e}phane},
  booktitle={ICML},
  pages={12310--12320},
  year={2021},
  url={https://arxiv.org/abs/2103.03230}
}
```

### Related Work

```bibtex
@article{barlow1961possible,
  title={Possible principles underlying the transformation of sensory messages},
  author={Barlow, Horace B},
  journal={Sensory communication},
  pages={217--234},
  year={1961}
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

@inproceedings{caron2020unsupervised,
  title={Unsupervised learning of visual features by contrasting cluster assignments},
  author={Caron, Mathilde and Misra, Ishan and Mairal, Julien and Goyal, Priya and Bojanowski, Piotr and Joulin, Armand},
  booktitle={NeurIPS},
  year={2020},
  url={https://arxiv.org/abs/2006.09882}
}

@inproceedings{bardes2022vicreg,
  title={VICReg: Variance-invariance-covariance regularization for self-supervised learning},
  author={Bardes, Adrien and Ponce, Jean and LeCun, Yann},
  booktitle={ICLR},
  year={2022},
  url={https://arxiv.org/abs/2105.04906}
}

@inproceedings{you2020large,
  title={Large batch optimization for deep learning: Training BERT in 76 minutes},
  author={You, Yang and Li, Jing and Reddi, Sashank and Hseu, Jonathan and Kumar, Sanjiv and Bhojanapalli, Srinadh and Song, Xiaodan and Demmel, James and Keutzer, Kurt and Hsieh, Cho-Jui},
  booktitle={ICLR},
  year={2020},
  url={https://arxiv.org/abs/1904.00962}
}

@article{richemond2020byol,
  title={BYOL works even without batch statistics},
  author={Richemond, Pierre H and Grill, Jean-Bastien and Altch{\'e}, Florent and Tallec, Corentin and Strub, Florian and Brock, Andrew and Smith, Samuel and De, Soham and Pascanu, Razvan and Piot, Bilal and others},
  journal={arXiv preprint arXiv:2010.10241},
  year={2020},
  url={https://arxiv.org/abs/2010.10241}
}

@inproceedings{ermolov2021whitening,
  title={Whitening for self-supervised representation learning},
  author={Ermolov, Aleksandr and Siarohin, Aliaksandr and Sangineto, Enver and Sebe, Nicu},
  booktitle={ICML},
  year={2021},
  url={https://arxiv.org/abs/2007.06346}
}
```

**Official Code**: https://github.com/facebookresearch/barlowtwins

**Nexus Implementation**: `nexus/models/ssl/barlow_twins.py`

**Additional Resources**:
- [Barlow Twins Explained (Blog)](https://ai.facebook.com/blog/barlow-twins)
- [Paper Walkthrough (YouTube)](https://www.youtube.com/watch?v=zbOntYKJ5Wc)
- [Code Tutorial](https://github.com/facebookresearch/barlowtwins/blob/main/main.py)
