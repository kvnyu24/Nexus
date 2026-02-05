# MAE: Masked Autoencoder

## Overview & Motivation

Masked Autoencoders (MAE) is a simple yet powerful self-supervised learning approach that learns visual representations by masking random patches of images and reconstructing the missing pixels. Inspired by masked language modeling in NLP (BERT), MAE demonstrates that a high mask ratio (75%) combined with an asymmetric encoder-decoder architecture can learn excellent visual representations.

### Key Innovation

**Asymmetric encoder-decoder design**: 
- Large encoder processes only visible patches (computational efficiency)
- Lightweight decoder reconstructs full image from encoded visible patches + mask tokens
- Mask ratio of 75% makes pre-training fast and effective

## Theoretical Background

### Learning Paradigm: Masked Reconstruction

MAE learns by solving a denoising autoencoding task in pixel space:

```
minimize ||reconstruct(encode(mask(x))) - x||²
```

### Why High Mask Ratio Works

1. **Information Asymmetry**: Masking 75% creates a difficult task requiring semantic understanding
2. **Computational Efficiency**: Encoder processes only 25% of patches
3. **Redundancy Removal**: Forces model to learn high-level representations, not just copy pixels
4. **Better Generalization**: Prevents overfitting to low-level statistics

### Autoencoding vs MAE

Traditional autoencoding compresses all information. MAE is different:
- **Random sampling**: Removes random patches (not compression)
- **High mask ratio**: 75% vs typical 15-30%
- **Asymmetric**: Encoder much larger than decoder
- **Normalized pixels**: Reconstruct normalized patch statistics

## Mathematical Formulation

### Loss Function

Mean squared error on masked patches only:

```
L = (1/|M|) Σ_{i∈M} ||x_i - x̂_i||²
```

Where:
- M: Set of masked patch indices
- x_i: Original pixel values for patch i (normalized)
- x̂_i: Reconstructed pixel values

### Pixel Normalization

Normalize each patch independently:

```
x_norm = (x - μ_patch) / √(σ²_patch + ε)
```

This prevents shortcuts based on overall image statistics.

### Encoder Forward Pass

```
1. Patchify image: x → patches (B, N, P²C)
2. Linear projection: patches → embeddings (B, N, D)
3. Add positional embedding: embeddings + pos_embed
4. Remove masked patches: embeddings[:, visible_mask]
5. Prepend CLS token
6. Transform: ViT encoder
7. Output: encoded_visible (B, N_vis+1, D)
```

### Decoder Forward Pass

```
1. Project encoder output to decoder dim
2. Add mask tokens at masked positions
3. Add positional embeddings for all positions
4. Transform: lightweight transformer decoder
5. Remove CLS token
6. Linear head to pixel space: (B, N, P²C)
7. Reshape to patches: (B, N, patch_dim)
```

## High-Level Intuition

Think of MAE like a jigsaw puzzle solver:

1. **Masking**: Someone removes 75% of puzzle pieces
2. **Encoder**: You look at the remaining 25% and understand the scene
3. **Decoder**: You imagine what the missing pieces look like
4. **Learning**: You get better by comparing your imagined pieces to the real ones

**Key insight**: With only 25% of pieces, you can't just memorize pixel patterns. You must understand the semantic content (e.g., "this is a forest scene, so the missing parts probably have trees and leaves").

## Implementation Details

### Network Architecture

**Encoder (ViT-Base)**:
- Patch size: 16×16
- Embedding dim: 768
- Layers: 12
- Attention heads: 12
- Processes only visible patches (25%)
- Has CLS token

**Decoder (Lightweight)**:
- Embedding dim: 512 (smaller than encoder)
- Layers: 8 (fewer than encoder)
- Attention heads: 16
- Processes all patches (visible + masked)

### Masking Implementation

```python
def random_masking(x, mask_ratio=0.75):
    """
    Random masking by per-sample shuffling.
    
    Args:
        x: Input sequence (B, N, D)
        mask_ratio: Fraction of patches to mask
        
    Returns:
        x_masked: Visible patches (B, N*(1-mask_ratio), D)
        mask: Binary mask (B, N), 1 is keep, 0 is remove
        ids_restore: Indices to restore original order
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))
    
    # Random permutation for each sample
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    # Keep visible patches
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    
    # Generate binary mask: 1 is keep, 0 is remove
    mask = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, mask, ids_restore
```

### Code Reference

See `nexus/models/ssl/mae.py` for full implementation:

```python
from nexus.models.ssl import MAE

config = {
    "img_size": 224,
    "patch_size": 16,
    "encoder_dim": 768,
    "decoder_dim": 512,
    "encoder_layers": 12,
    "decoder_layers": 8,
    "mask_ratio": 0.75,
    "norm_pix_loss": True
}

model = MAE(config)
loss, reconstructed, mask = model(images)
```

## Optimization Tricks

### 1. Normalized Pixel Loss

Normalize targets by patch-wise mean and variance:

```python
if norm_pix_loss:
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    target = (target - mean) / (var + 1e-6).sqrt()
```

This improves representation quality by preventing shortcuts.

### 2. Learning Rate Scaling

Scale learning rate with batch size:

```python
base_lr = 1.5e-4
actual_lr = base_lr * batch_size / 256
```

### 3. Warmup Schedule

```python
warmup_epochs = 40
# Linear warmup then cosine decay
```

### 4. Weight Decay

High weight decay works well:

```python
weight_decay = 0.05  # Higher than typical supervised learning
```

### 5. Mixed Precision Training

MAE benefits greatly from mixed precision:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss, _, _ = model(images)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Experiments & Results

### ImageNet-1K Pre-training

| Model | Encoder Params | Pre-train Epochs | Linear Probe Acc | Fine-tune Acc |
|-------|----------------|------------------|------------------|---------------|
| ViT-B/16 | 86M | 800 | 67.8% | 83.6% |
| ViT-L/16 | 304M | 800 | 75.5% | 85.9% |
| ViT-H/14 | 632M | 1600 | 76.6% | 86.9% |

### Ablation Studies

**Mask Ratio**:
| Mask Ratio | Linear Probe Acc |
|------------|------------------|
| 15% | 61.2% |
| 50% | 65.7% |
| 75% | **67.8%** |
| 90% | 64.3% |

**Decoder Depth**:
| Decoder Layers | Linear Probe Acc | Training Time |
|----------------|------------------|---------------|
| 1 | 65.2% | Fast |
| 4 | 66.9% | Medium |
| 8 | **67.8%** | Slow |

## Common Pitfalls

### 1. Low Mask Ratio
**Problem**: Mask ratio < 50% leads to poor representations
**Solution**: Use 75% mask ratio (optimal in most cases)

### 2. Decoder Too Large
**Problem**: Large decoder makes training slow without much benefit
**Solution**: Use lightweight decoder (8 layers, 512 dim for ViT-B)

### 3. No Pixel Normalization
**Problem**: Model learns to predict mean color without understanding semantics
**Solution**: Enable `norm_pix_loss=True`

### 4. Evaluating on Training Task
**Problem**: Reconstruction quality ≠ representation quality
**Solution**: Evaluate via linear probing or fine-tuning on downstream tasks

### 5. Wrong Masking Order
**Problem**: Masking after positional encoding leaks information
**Solution**: Mask before adding positional embeddings

## References

```bibtex
@article{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={CVPR},
  year={2022}
}
```

**Official Code**: https://github.com/facebookresearch/mae
**Nexus Implementation**: `nexus/models/ssl/mae.py`
