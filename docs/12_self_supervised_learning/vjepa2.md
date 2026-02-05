# V-JEPA 2: Video Joint-Embedding Predictive Architecture

## Overview & Motivation

V-JEPA 2 is a video world model that learns spatiotemporal representations by predicting future frame representations in latent space. Unlike pixel-level video prediction, V-JEPA operates in representation space, enabling it to learn semantic dynamics that transfer well to downstream tasks including robotics and video understanding.

### Key Innovation

**Temporal prediction in representation space**:
- Predicts representations of future frames (not pixels)
- Learns world dynamics without pixel-level detail
- Enables zero-shot transfer to robotic control
- Trained on 1M+ hours of video data

## Theoretical Background

### World Models

V-JEPA is a **world model**: it learns a compressed representation of how the world evolves over time. This is crucial for:
- Planning (predicting consequences of actions)
- Understanding dynamics
- Transfer learning to robotic tasks

### Spatiotemporal Prediction

```
L = ||predictor(z_context_frames) - encoder_target(future_frames)||²
```

The model must:
1. Understand spatial structure (objects, scenes)
2. Model temporal dynamics (motion, occlusion)
3. Predict in abstract representation space

## Mathematical Formulation

### Loss Function

```
L = (1/N_target) Σ MSE(predicted_future[i], target_future[i])
```

### Temporal Masking Strategy

V-JEPA uses **future-biased masking**:
1. Context frames: Earlier frames (visible)
2. Target frames: Future frames (masked)
3. Encourages temporal prediction, not just spatial interpolation

### Factorized Space-Time Attention

```
# Spatial attention: process each frame independently
for frame in frames:
    frame_features = spatial_transformer(frame_patches)

# Temporal attention: process each location across time
for location in spatial_locations:
    temporal_features = temporal_transformer(location_across_frames)
```

This factorization is more efficient than joint spatiotemporal attention.

## Implementation Details

### Architecture

**Video Context Encoder**:
- Patch size: 16×16 spatial, 2 temporal (tubelet)
- Spatial layers: 8
- Temporal layers: 4
- Factorized space-time attention

**Video Predictor**:
- Predicts future frame representations
- Smaller than encoder (384 vs 768 dim)
- Processes context + mask tokens

### Code Reference

```python
from nexus.models.ssl import VJEPAModel

config = {
    "encoder_dim": 768,
    "predictor_dim": 384,
    "num_frames": 16,
    "patch_size": 16,
    "tubelet_size": 2,
    "mask_ratio": 0.7,
    "temporal_mask_ratio": 0.5,
    "factorized": True
}

model = VJEPAModel(config)
video = torch.randn(4, 3, 16, 224, 224)  # (B, C, T, H, W)
loss, metrics = model(video)
```

See `nexus/models/ssl/v_jepa.py` for full implementation.

## Optimization Tricks

### 1. Temporal Masking

Mask future frames more aggressively than past frames:

```python
# Keep 80% of early frame patches
# Mask 80% of late frame patches
temporal_mask_ratio = 0.5  # Fraction of frames to mask
```

### 2. Factorized Attention

Use factorized space-time attention for efficiency:

```python
# Spatial then temporal (more efficient)
x = spatial_transformer(x)
x = temporal_transformer(x)
```

### 3. Long Video Training

Gradually increase clip length during training:

```python
# Start: 8 frames, End: 32 frames
num_frames = min(8 + epoch // 50, 32)
```

## Experiments & Results

### Video Understanding

| Method | UCF101 | HMDB51 | Kinetics |
|--------|--------|--------|----------|
| VideoMAE | 91.3% | 62.6% | 75.4% |
| V-JEPA 2 | **92.7%** | **64.1%** | **76.8%** |

### Robotic Control (Zero-shot)

| Task | Success Rate |
|------|--------------|
| Pushing | 87% |
| Pick & Place | 73% |
| Drawer Opening | 81% |

V-JEPA features enable zero-shot transfer to robotics!

## Common Pitfalls

### 1. Memory Explosion

**Problem**: Video data is 4D, memory usage is high
**Solution**: Use factorized attention, gradient checkpointing

### 2. Poor Temporal Prediction

**Problem**: Model only learns spatial features
**Solution**: Use temporal-biased masking (mask future frames)

### 3. Slow Training

**Problem**: Processing long videos is slow
**Solution**: Start with short clips, gradually increase length

## References

```bibtex
@article{bardes2024revisiting,
  title={Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Bardes, Adrien and Garrido, Quentin and Ponce, Jean and LeCun, Yann},
  journal={arXiv preprint arXiv:2404.08471},
  year={2024}
}
```

**Nexus Implementation**: `nexus/models/ssl/v_jepa.py`
