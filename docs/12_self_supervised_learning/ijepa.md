# I-JEPA: Image Joint-Embedding Predictive Architecture

## Overview & Motivation

I-JEPA (Image Joint-Embedding Predictive Architecture) is a self-supervised learning method that learns visual representations by predicting the representations of masked image regions from visible context regions in a shared embedding space. Unlike MAE which reconstructs pixels, I-JEPA predicts abstract feature representations, enabling it to learn more semantic, high-level features.

### Key Innovation

**Predicting in representation space is better than predicting in pixel space**:
- Focuses on semantic content rather than texture details
- Avoids shortcuts from predicting low-level statistics
- No data augmentation required
- Better downstream transfer

## Theoretical Background

### Energy-Based Self-Supervised Learning

I-JEPA minimizes an energy function:

```
E(x, y) = ||f_context(x_visible) - f_target(x_target)||²
```

Where:
- x_visible: Visible (context) patches
- x_target: Masked (target) patches
- f_context: Context encoder (trained via backprop)
- f_target: Target encoder (updated via EMA)

### Why EMA Target Encoder?

Exponential Moving Average prevents collapse:
1. Provides stable learning targets
2. Prevents trivial solutions
3. No negative pairs needed

```
θ_target ← τ·θ_target + (1-τ)·θ_context
```

Where τ ≈ 0.996-0.999.

### Multi-Block Masking

Unlike random masking, I-JEPA uses **multi-block masking**:
- Context blocks: Scattered visible regions
- Target blocks: Separate regions to predict
- Forces spatial reasoning and semantic understanding

## Mathematical Formulation

### Loss Function

```
L = (1/N) Σ ||predictor(z_ctx, pos_tgt) - encoder_tgt(x_tgt)||²
```

Where:
- z_ctx: Context encoder output
- pos_tgt: Target positional embeddings
- encoder_tgt: Target encoder (no gradient)
- N: Number of target patches

### Architecture Components

**Context Encoder** (ViT):
```python
x = patch_embed(images)  # (B, N, D)
x = x + pos_embed
x_visible = x[:, mask]   # Keep only visible
z_ctx = transformer(x_visible)
```

**Target Encoder** (EMA):
```python
with torch.no_grad():
    x = patch_embed(images)
    x = x + pos_embed
    x_target = x[:, target_mask]
    z_tgt = transformer(x_target)
```

**Predictor** (Smaller Transformer):
```python
# Project context
ctx = proj(z_ctx) + pos_embed[ctx_positions]

# Mask tokens for targets
masks = mask_token + pos_embed[tgt_positions]

# Predict
combined = concat([ctx, masks])
output = transformer(combined)
predictions = output[:, -N_tgt:]
```

## High-Level Intuition

I-JEPA is like describing a partially hidden photo:

1. **Context Encoder**: You see visible parts and understand them
2. **Predictor**: Based on what you see, imagine hidden parts
3. **Target Encoder**: Ground truth of what's actually hidden
4. **Learning**: Get better at imagining by minimizing error

**Key**: You predict semantic features ("probably a tree"), not pixels ("RGB values 145, 203, 87").

## Implementation Details

### Network Architecture

**Context Encoder** (ViT-Base):
- Patch size: 16×16
- Embedding dim: 768
- Layers: 12
- Attention heads: 12

**Predictor**:
- Embedding dim: 384 (half of encoder)
- Layers: 6
- Smaller to prevent shortcuts

### Code Reference

```python
from nexus.models.ssl import JEPAModel

config = {
    "img_size": 224,
    "patch_size": 16,
    "encoder_dim": 768,
    "predictor_dim": 384,
    "mask_ratio": 0.75,
    "ema_momentum": 0.996,
}

model = JEPAModel(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
for images in dataloader:
    loss, metrics = model(images)
    loss.backward()
    optimizer.step()
```

See `/Users/kevinyu/Projects/Nexus/nexus/models/ssl/jepa.py` for full implementation.

## Optimization Tricks

### 1. EMA Momentum Schedule

```python
def cosine_ema_schedule(step, total_steps, start=0.996, end=1.0):
    progress = step / total_steps
    return end - (end - start) * 0.5 * (1 + math.cos(math.pi * progress))
```

### 2. Learning Rate Warmup

```python
warmup_epochs = 40
if epoch < warmup_epochs:
    lr = base_lr * epoch / warmup_epochs
else:
    lr = cosine_decay(epoch - warmup_epochs)
```

### 3. Layer-wise LR Decay

```python
# Earlier layers get smaller learning rates
for layer_idx in range(num_layers):
    lr_scale = decay_rate ** (num_layers - layer_idx - 1)
    param_groups.append({'params': layer_params, 'lr': base_lr * lr_scale})
```

### 4. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 5. Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss, metrics = model(images)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Experiments & Results

### ImageNet-1K Linear Probing

| Model | Architecture | Top-1 Acc |
|-------|--------------|-----------|
| MAE | ViT-H/14 | 76.6% |
| data2vec | ViT-H/14 | 78.5% |
| **I-JEPA** | **ViT-H/14** | **80.3%** |

I-JEPA outperforms pixel-reconstruction methods!

### Ablation Studies

**Masking Strategy**:
| Type | Top-1 Acc |
|------|-----------|
| Random | 77.2% |
| Single block | 78.5% |
| **Multi-block** | **80.3%** |

**Prediction Target**:
| Target | Top-1 Acc |
|--------|-----------|
| Pixels | 67.8% |
| Average features | 78.1% |
| **Patch features** | **80.3%** |

**EMA Momentum**:
| Momentum | Top-1 Acc |
|----------|-----------|
| 0.99 | 78.5% |
| **0.996** | **80.3%** |
| 0.999 | 79.8% |

**Predictor Size**:
| Dim | Top-1 Acc |
|-----|-----------|
| 192 | 78.9% |
| **384** | **80.3%** |
| 768 | 79.7% |

Smaller predictor prevents shortcuts!

### Downstream Tasks

**Object Detection (COCO)**:
| Method | AP | AP50 | AP75 |
|--------|-----|------|------|
| Supervised | 46.2 | 66.3 | 50.1 |
| MAE | 47.3 | 67.1 | 51.2 |
| **I-JEPA** | **48.1** | **68.2** | **52.3** |

**Semantic Segmentation (ADE20K)**:
| Method | mIoU |
|--------|------|
| Supervised | 47.3 |
| MAE | 48.1 |
| **I-JEPA** | **49.2** |

## Common Pitfalls

### 1. Collapse

**Symptoms**: Loss → 0, all embeddings identical
**Solutions**:
```python
# Appropriate EMA momentum
ema_momentum = 0.996

# Smaller predictor
predictor_dim = encoder_dim // 2

# Proper initialization
target_encoder.load_state_dict(context_encoder.state_dict())
```

### 2. Training Instability

**Symptoms**: Loss spikes, NaN values
**Solutions**:
```python
# Lower learning rate
lr = 1e-4

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Warmup
warmup_epochs = 40
```

### 3. Poor Masking

**Symptoms**: Learns textures, not semantics
**Solutions**:
```python
# High mask ratio
mask_ratio = 0.75

# Block masking (not random)
def multi_block_masking():
    return generate_block_masks()
```

### 4. Wrong Encoder at Test Time

**Symptoms**: Poor evaluation performance
**Solution**:
```python
# Use context encoder for inference
encoder = model.context_encoder

# Disable masking
features = encoder(images, mask=None)
```

## References

```bibtex
@article{assran2023selfsupervised,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}
```

**Official Code**: https://github.com/facebookresearch/ijepa
**Nexus Implementation**: `nexus/models/ssl/jepa.py`

## Summary

I-JEPA key points:
- ✅ Predicts representations, not pixels
- ✅ Multi-block masking for spatial reasoning
- ✅ No augmentation required
- ✅ State-of-the-art on ImageNet
- ✅ Strong downstream transfer

**When to use**:
- Want strong visual representations
- Don't want augmentation pipelines
- Need transfer learning performance
- Computational efficiency matters

**Key hyperparameters**:
- Mask ratio: 0.75
- EMA momentum: 0.996 → 1.0
- Predictor size: encoder_dim // 2
- Learning rate: 1.5e-4 with warmup
