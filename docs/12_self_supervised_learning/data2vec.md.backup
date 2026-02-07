# data2vec 2.0: Efficient Multimodal Self-Supervised Learning

## Overview & Motivation

data2vec 2.0 is a unified self-supervised learning framework for vision, speech, and text that predicts contextualized latent representations from a teacher model. Version 2.0 introduces key efficiency improvements: inverse block masking, fast convolutional decoder, and multi-masking, achieving 2× training speedup over v1.

### Key Innovation

**Multimodal framework with efficiency improvements**:
- Inverse masking: Process only visible tokens
- Fast conv decoder: Replace transformer decoder
- Multi-masking: Multiple masks per sample
- Works across vision, audio, and text

## Mathematical Formulation

### Loss Function

Smooth L1 (Huber) loss on masked positions:

```
L = (1/|M|) Σᵢ∈M SmoothL1(student(xᵢ), teacher(xᵢ))
```

### Inverse Masking

Traditional masking processes all tokens:
```python
# Old: Process N tokens
x_all = encoder(all_tokens)  # Expensive!
loss = loss_fn(x_all[masked_positions])
```

Inverse masking processes only visible tokens:
```python
# New: Process only visible tokens
x_visible = encoder(visible_tokens)  # Efficient!
x_full = reconstruct_with_mask_tokens(x_visible)
loss = loss_fn(decoder(x_full)[masked_positions])
```

### EMA Teacher Update

```
θ_teacher ← τ·θ_teacher + (1-τ)·θ_student
```

Where τ increases from 0.999 → 0.9999 via cosine schedule.

## Implementation Details

### Architecture

**Student Encoder**:
- ViT for vision / Transformer for text/audio
- Processes only visible tokens (inverse masking)
- Output: visible representations + mask tokens at masked positions

**Teacher Encoder**:
- Same architecture as student
- EMA updated, no gradients
- Processes full input (no masking)

**Contextualized Decoder**:
- Fast depthwise separable convolutions
- Much faster than transformer decoder
- Projects to teacher's representation space

### Modality-Specific Input

**Vision**:
```python
x = patch_embed(images)  # Conv2d projection
```

**Text**:
```python
x = token_embed(token_ids)  # Embedding layer
```

**Audio**:
```python
x = spectrogram_embed(audio)  # Conv2d on spectrogram
```

### Code Reference

```python
from nexus.models.ssl import Data2VecModel

config = {
    "encoder_dim": 768,
    "decoder_dim": 384,
    "modality": "vision",  # or "audio", "text"
    "mask_ratio": 0.6,
    "ema_momentum": 0.999,
    "multi_mask": 2,  # Multiple masks per sample
    "loss_beta": 2.0,  # Smooth L1 beta
}

model = Data2VecModel(config)
loss, metrics = model(images)
```

See `nexus/models/ssl/data2vec.py` for full implementation.

## Optimization Tricks

### 1. Multi-Masking

Generate multiple masks per sample for better efficiency:

```python
for _ in range(num_masks):
    mask = generate_random_mask()
    loss += compute_loss(student(x, mask), teacher(x))
loss /= num_masks
```

### 2. EMA Momentum Schedule

Increase momentum over training:

```python
# Cosine schedule: 0.999 → 0.9999
tau = tau_end - (tau_end - tau_start) * 0.5 * (1 + cos(π * progress))
```

### 3. Target Normalization

Normalize teacher targets for stability:

```python
target = layer_norm(teacher_output)
```

### 4. Loss Function

Use Smooth L1 (less sensitive to outliers than MSE):

```python
loss = F.smooth_l1_loss(pred, target, beta=2.0)
```

## Experiments & Results

### ImageNet-1K (ViT-Base)

| Method | Top-1 Acc | Multimodal | Training Speed |
|--------|-----------|------------|----------------|
| MAE | 67.8% | ❌ | Fast |
| data2vec 1.0 | 74.2% | ✅ | Slow |
| data2vec 2.0 | **74.2%** | ✅ | **2× faster** |

Same performance, half the training time!

### Multimodal Results

| Modality | Task | Performance |
|----------|------|-------------|
| Vision | ImageNet | 74.2% |
| Speech | Librispeech WER | 1.9% |
| Text | GLUE avg | 83.4 |

Unified framework works across modalities!

### Efficiency Gains

| Component | Speedup |
|-----------|---------|
| Inverse masking | 1.5× |
| Fast conv decoder | 1.3× |
| **Total** | **2×** |

## Common Pitfalls

### 1. Wrong Mask Ratio

**Problem**: Mask ratio should vary by modality
**Solution**: 
- Vision: 0.6-0.7
- Audio: 0.5-0.6
- Text: 0.15-0.3

### 2. Teacher Divergence

**Problem**: Teacher diverges from student
**Solution**: Use high EMA momentum (0.999-0.9999)

### 3. Slow Training

**Problem**: Not using efficiency tricks
**Solution**: Enable inverse masking + conv decoder

## References

```bibtex
@inproceedings{baevski2022data2vec,
  title={data2vec 2.0: Efficient self-supervised learning with contextualized target representations for vision, speech and NLP},
  author={Baevski, Alexei and Hsu, Wei-Ning and Xu, Qiantong and Babu, Arun and Gu, Jiatao and Auli, Michael},
  booktitle={ICML},
  year={2022}
}
```

**Official Code**: https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec
**Nexus Implementation**: `nexus/models/ssl/data2vec.py`
