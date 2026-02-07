# V-JEPA 2: Video Joint-Embedding Predictive Architecture

## Overview & Motivation

V-JEPA 2 is a video world model that learns spatiotemporal representations by predicting future frame representations in latent space. Unlike pixel-level video prediction, V-JEPA operates in representation space, enabling it to learn semantic dynamics that transfer well to downstream tasks including robotics and video understanding.

### Key Innovation

**Temporal prediction in representation space**:
- Predicts representations of future frames (not pixels)
- Learns world dynamics without pixel-level detail
- Enables zero-shot transfer to robotic control
- Trained on 1M+ hours of video data
- Factorized space-time attention for efficiency
- Future-biased temporal masking strategy

### Why V-JEPA 2 Matters

Traditional video models either:
1. **Pixel prediction**: Computationally expensive, focuses on low-level details
2. **Frame-by-frame**: Ignores temporal dependencies
3. **Contrastive**: Requires careful negative sampling

V-JEPA 2 learns **abstract spatiotemporal representations** that capture:
- Object motion and interactions
- Scene dynamics and physics
- Causal relationships over time
- Generalizable world knowledge

This makes it ideal for:
- Video understanding tasks (action recognition, temporal grounding)
- Robotic vision (zero-shot policy transfer)
- Video generation (world modeling)
- Predictive control (model-based RL)

## Theoretical Background

### World Models

V-JEPA is a **world model**: it learns a compressed representation of how the world evolves over time. This is crucial for:
- Planning (predicting consequences of actions)
- Understanding dynamics
- Transfer learning to robotic tasks

A world model learns the transition function:
```
z_{t+1} = f(z_t, a_t)
```

Where:
- z_t: State representation at time t
- a_t: Action (implicit in video prediction)
- f: Learned dynamics model

### Why Predict Representations, Not Pixels?

Pixel-level prediction suffers from:
1. **High-dimensional output**: Images are large (224×224×3)
2. **Low-level details**: Must predict exact colors, textures
3. **Multimodal uncertainty**: Many valid futures for same context
4. **Computational cost**: Expensive to generate and evaluate

Representation prediction is better:
1. **Low-dimensional**: Latent space is compact (e.g., 768-d)
2. **Semantic**: Captures meaning, not appearance
3. **Deterministic**: Single target representation
4. **Efficient**: Fast to compute and compare

### Spatiotemporal Prediction

```
L = ||predictor(z_context_frames) - encoder_target(future_frames)||²
```

The model must:
1. Understand spatial structure (objects, scenes)
2. Model temporal dynamics (motion, occlusion)
3. Predict in abstract representation space

This forces learning of:
- **Object permanence**: Objects persist across frames
- **Physics intuition**: Motion follows natural laws
- **Occlusion reasoning**: Predict hidden objects
- **Causal understanding**: Cause precedes effect

### Joint-Embedding Predictive Architecture (JEPA)

JEPA is Yann LeCun's framework for self-supervised learning:

```
Context → Predictor → Predicted Representation
Input → Encoder → Target Representation
```

Key principles:
1. **No reconstruction**: Don't generate pixels
2. **Latent prediction**: Predict in representation space
3. **Collapse prevention**: Use stop-gradient on targets
4. **Information asymmetry**: Predict from partial context

Benefits:
- Learns invariances naturally (lighting, viewpoint)
- Focuses on content, not details
- Computationally efficient
- Avoids trivial solutions

## Mathematical Formulation

### Loss Function

The primary loss is mean squared error on predicted representations:

```
L_pred = (1/N_target) Σ_{i=1}^{N_target} ||predictor(z_context) - sg(target_encoder(x_i))||²
```

Where:
- N_target: Number of target (future) frames
- z_context: Encoded context (past) frames
- sg(): Stop-gradient operation
- x_i: Target frame i

### Temporal Masking Strategy

V-JEPA uses **future-biased masking**:
1. Context frames: Earlier frames (visible)
2. Target frames: Future frames (masked)
3. Encourages temporal prediction, not just spatial interpolation

Masking schedule:
```
Context: frames [1, 2, 3, ..., t_context]
Target: frames [t_context+1, ..., T]
```

Typical split:
- Context: 8 frames (first half)
- Target: 8 frames (second half)
- Total: 16 frames

### Factorized Space-Time Attention

Standard joint spatiotemporal attention is O(N²) where N = H×W×T (spatial×temporal).

Factorized attention reduces complexity:

```
# Spatial attention: process each frame independently
for frame in frames:
    frame_features = spatial_transformer(frame_patches)

# Temporal attention: process each location across time
for location in spatial_locations:
    temporal_features = temporal_transformer(location_across_frames)
```

Complexity:
- Joint: O((HWT)²) = O(N²)
- Factorized: O(HWT·HW + HWT·T) = O(N·HW + N·T)

For 16×16 patches on 224×224 images with 16 frames:
- H = W = 14, T = 16
- Joint: (14×14×16)² ≈ 123M operations
- Factorized: (14×14×16)×(14×14) + (14×14×16)×16 ≈ 8M operations
- **15× speedup!**

### Tubelet Tokenization

Videos are patchified into 3D "tubelets":

```
Tubelet: (patch_size, patch_size, tubelet_size)
Example: (16, 16, 2) - 16×16 spatial, 2 temporal
```

For video of shape (B, C, T, H, W):
```python
num_patches_h = H // patch_size
num_patches_w = W // patch_size
num_patches_t = T // tubelet_size
total_patches = num_patches_h × num_patches_w × num_patches_t
```

Each tubelet is projected to embedding dimension:
```python
tubelet_embed = Conv3d(
    in_channels=3,
    out_channels=embed_dim,
    kernel_size=(tubelet_size, patch_size, patch_size),
    stride=(tubelet_size, patch_size, patch_size)
)
```

### Target Encoder (EMA)

Target encoder uses Exponential Moving Average updates:

```
θ_target ← τ·θ_target + (1-τ)·θ_online
```

Where:
- τ: Momentum coefficient (typically 0.996-0.999)
- θ_target: Target encoder parameters (no gradients)
- θ_online: Online encoder parameters (trained)

Momentum schedule (cosine):
```python
tau_base = 0.996
tau_end = 1.0
tau = tau_end - (tau_end - tau_base) * 0.5 * (1 + cos(π * progress))
```

Why EMA?
1. **Stable targets**: Slowly changing representations
2. **Prevents collapse**: Can't trivially match moving target
3. **Better features**: Averaging improves generalization
4. **No extra backprop**: Only forward pass through target

## Implementation Details

### Architecture

**Video Context Encoder**:
- Patch size: 16×16 spatial, 2 temporal (tubelet)
- Spatial layers: 8 transformer blocks
- Temporal layers: 4 transformer blocks
- Factorized space-time attention
- Embedding dimension: 768
- Attention heads: 12
- MLP ratio: 4.0

**Video Predictor**:
- Predicts future frame representations
- Smaller than encoder (384 vs 768 dim)
- Processes context + mask tokens
- 6 transformer layers
- 8 attention heads
- Projects to encoder dimension for loss

**Target Encoder**:
- Identical architecture to context encoder
- EMA-updated, no gradients
- Processes all frames (no masking)
- Provides stable prediction targets

### Architecture Diagram

```
Context Frames [1...8]              Target Frames [9...16]
       ↓                                    ↓
  Context Encoder                    Target Encoder (EMA)
  (768-d, 12 layers)                (768-d, 12 layers)
       ↓                                    ↓
  Context Embeddings                 Target Embeddings
       ↓                                    ↓
    Predictor ─────────────────→    Stop Gradient
  (384-d, 6 layers)                        ↓
       ↓                                    ↓
  Predicted Embeddings ←──── MSE Loss ──→  Target Embeddings
```

### Detailed Forward Pass

```python
def forward(self, video):
    """
    Args:
        video: (B, C, T, H, W) - batch of video clips

    Returns:
        loss: prediction loss
        metrics: dict of logging metrics
    """
    B, C, T, H, W = video.shape

    # 1. Split into context and target frames
    context_frames = video[:, :, :T//2]  # First half
    target_frames = video[:, :, T//2:]   # Second half

    # 2. Encode context frames
    context_tokens = self.tubelet_embed(context_frames)  # (B, N_ctx, D)
    context_tokens = self.add_pos_embed(context_tokens)

    # Spatial attention
    for layer in self.spatial_layers:
        context_tokens = layer(context_tokens)

    # Temporal attention
    context_tokens = rearrange(context_tokens,
                               'b (t h w) d -> b (h w) t d')
    for layer in self.temporal_layers:
        context_tokens = layer(context_tokens)
    context_tokens = rearrange(context_tokens,
                               'b (h w) t d -> b (t h w) d')

    # 3. Predictor: predict target representations
    # Add mask tokens for target positions
    mask_tokens = self.mask_token.expand(B, N_target, -1)
    predictor_input = torch.cat([context_tokens, mask_tokens], dim=1)

    predicted = self.predictor(predictor_input)
    predicted_target = predicted[:, -N_target:]  # Extract target predictions

    # 4. Target encoder (EMA, no gradients)
    with torch.no_grad():
        target_tokens = self.target_tubelet_embed(target_frames)
        target_tokens = self.target_add_pos_embed(target_tokens)

        for layer in self.target_spatial_layers:
            target_tokens = layer(target_tokens)

        target_tokens = rearrange(target_tokens,
                                  'b (t h w) d -> b (h w) t d')
        for layer in self.target_temporal_layers:
            target_tokens = layer(target_tokens)
        target_tokens = rearrange(target_tokens,
                                  'b (h w) t d -> b (t h w) d')

    # 5. Compute loss
    loss = F.mse_loss(predicted_target, target_tokens)

    # 6. Update EMA target encoder
    self._update_target_encoder()

    return loss, {"loss": loss.item()}
```

### Nexus Code Reference

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
    "factorized": True,
    "num_spatial_layers": 8,
    "num_temporal_layers": 4,
    "num_predictor_layers": 6,
    "ema_momentum": 0.996
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

# Alternative: predict k frames ahead
context_end = total_frames - k
context = video[:, :, :context_end]
target = video[:, :, context_end:]
```

**Why it works**:
- Forces temporal reasoning (can't just copy)
- Learns dynamics (how state evolves)
- Prevents spatial-only solutions

### 2. Factorized Attention

Use factorized space-time attention for efficiency:

```python
# Spatial then temporal (more efficient)
x = spatial_transformer(x)
x = temporal_transformer(x)

# vs. joint attention (slower)
x = spatiotemporal_transformer(x)  # O(N²) where N=HWT
```

**Implementation**:
```python
# Spatial: attend within each frame
x = rearrange(x, 'b (t h w) d -> (b t) (h w) d')
x = spatial_attn(x)
x = rearrange(x, '(b t) (h w) d -> b (t h w) d')

# Temporal: attend across time for each location
x = rearrange(x, 'b (t h w) d -> (b h w) t d')
x = temporal_attn(x)
x = rearrange(x, '(b h w) t d -> b (t h w) d')
```

### 3. Long Video Training

Gradually increase clip length during training:

```python
# Start: 8 frames, End: 32 frames
num_frames = min(8 + epoch // 50, 32)

# Schedule
epoch_ranges = [
    (0, 100): 8 frames,
    (100, 200): 16 frames,
    (200, 300): 24 frames,
    (300, +∞): 32 frames
]
```

**Benefits**:
- Start fast (short clips)
- End with better temporal understanding (long clips)
- Curriculum learning for dynamics

### 4. Multi-Scale Temporal Prediction

Predict at multiple temporal scales:

```python
# Short-term (1 frame ahead)
loss_short = mse(pred_1frame, target_1frame)

# Medium-term (4 frames ahead)
loss_medium = mse(pred_4frames, target_4frames)

# Long-term (8 frames ahead)
loss_long = mse(pred_8frames, target_8frames)

# Combined
loss = loss_short + 0.5*loss_medium + 0.25*loss_long
```

### 5. Gradient Checkpointing

Save memory on long videos:

```python
import torch.utils.checkpoint as checkpoint

def transformer_block_with_checkpoint(x):
    return checkpoint.checkpoint(transformer_block, x)

# Use in forward pass
x = transformer_block_with_checkpoint(x)
```

**Trade-off**:
- Memory: 50% reduction
- Speed: 20% slower (recompute on backward)

### 6. Mixed Precision Training

Essential for video (4D tensors):

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for video in dataloader:
    with autocast():
        loss, metrics = model(video)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Speedup**: 2-3× faster with same accuracy

## Experiments & Results

### Video Understanding

Linear probing on action recognition datasets:

| Method | UCF101 | HMDB51 | Kinetics-400 | SSv2 |
|--------|--------|--------|--------------|------|
| Random Init | 54.2% | 24.8% | 45.6% | 18.3% |
| VideoMAE | 91.3% | 62.6% | 75.4% | 68.7% |
| **V-JEPA 2** | **92.7%** | **64.1%** | **76.8%** | **71.2%** |

### Robotic Control (Zero-shot)

V-JEPA features enable zero-shot transfer to robotics:

| Task | Success Rate | Baseline (Scratch) |
|------|--------------|-------------------|
| Pushing | 87% | 34% |
| Pick & Place | 73% | 19% |
| Drawer Opening | 81% | 28% |
| Button Pressing | 79% | 41% |

**Key insight**: Learned world model transfers without task-specific training!

### Temporal Grounding

Localize actions in untrimmed videos:

| Method | ActivityNet mAP | Charades mAP |
|--------|-----------------|--------------|
| I3D | 34.5% | 39.8% |
| VideoMAE | 43.2% | 45.6% |
| **V-JEPA 2** | **46.8%** | **48.3%** |

### Ablation Studies

**Effect of Factorized Attention**:
| Attention Type | Accuracy | Speed (it/s) | Memory (GB) |
|----------------|----------|--------------|-------------|
| Joint | 76.4% | 1.2 | 38.4 |
| **Factorized** | **76.8%** | **8.7** | **12.1** |

**Effect of Temporal Masking**:
| Context Frames | Target Frames | Accuracy |
|----------------|---------------|----------|
| 16 | 0 | 68.2% (spatial only) |
| 12 | 4 | 74.5% |
| 8 | 8 | **76.8%** |
| 4 | 12 | 75.1% |

**Effect of Prediction Horizon**:
| Frames Ahead | Accuracy | Transfer to Robotics |
|--------------|----------|---------------------|
| 1 | 75.2% | 71% |
| 4 | **76.8%** | **87%** |
| 8 | 74.9% | 82% |

**Effect of Pre-training Data**:
| Dataset | Size | Accuracy |
|---------|------|----------|
| Kinetics-400 | 240K videos | 72.3% |
| Kinetics-700 | 650K videos | 74.8% |
| Webvid (Mixed) | 1M+ videos | **76.8%** |

### Scaling Laws

V-JEPA 2 scales well with model size:

| Model | Params | Accuracy | Training Time |
|-------|--------|----------|---------------|
| Base | 86M | 76.8% | 100 GPU-days |
| Large | 304M | 80.2% | 400 GPU-days |
| Huge | 632M | 82.1% | 1200 GPU-days |

### Comparison with Pixel Prediction

| Method | Prediction Target | Accuracy | Speed |
|--------|------------------|----------|-------|
| Pixel Prediction | RGB values | 71.2% | 2.1 it/s |
| Pixel VAE | Latent pixels | 73.8% | 4.5 it/s |
| **V-JEPA 2** | **Representations** | **76.8%** | **8.7 it/s** |

## Advanced Topics

### Multi-View Prediction

Predict future from multiple camera angles:

```python
# Input: multiple camera views
views = [view1, view2, view3]  # Each (B, C, T, H, W)

# Encode each view
context_embeddings = [encoder(v[:, :, :T//2]) for v in views]

# Fuse multi-view context
fused_context = attention_pooling(context_embeddings)

# Predict each view's future
predictions = [predictor(fused_context, v) for v in views]

# Loss on all views
loss = sum([mse(pred, target) for pred, target in zip(predictions, targets)])
```

### Action-Conditioned Prediction

Predict future given actions (for robotics):

```python
# Input: video + action sequence
context_frames = video[:, :, :T//2]
actions = action_sequence[:, :T//2]  # Robot actions

# Encode context + actions
context_embed = encoder(context_frames)
action_embed = action_encoder(actions)
conditioned_context = context_embed + action_embed

# Predict future conditioned on actions
predicted_future = predictor(conditioned_context)
```

### Hierarchical Temporal Prediction

Predict at multiple temporal resolutions:

```python
# Coarse-to-fine prediction
# Level 1: Predict every 4th frame (coarse)
coarse_pred = predictor_coarse(context)
loss_coarse = mse(coarse_pred, target_frames[::4])

# Level 2: Predict every 2nd frame (medium)
medium_pred = predictor_medium(coarse_pred)
loss_medium = mse(medium_pred, target_frames[::2])

# Level 3: Predict every frame (fine)
fine_pred = predictor_fine(medium_pred)
loss_fine = mse(fine_pred, target_frames)

# Combined hierarchical loss
loss = loss_coarse + loss_medium + loss_fine
```

## Common Pitfalls

### 1. Memory Explosion

**Problem**: Video data is 4D, memory usage is high
**Symptoms**:
- CUDA out of memory errors
- Can only fit batch size of 1-2
- Training very slow

**Solution**:
```python
# Use gradient checkpointing
model.enable_gradient_checkpointing()

# Factorized attention (not joint)
config["factorized"] = True

# Smaller batch size, accumulate gradients
accumulation_steps = 8
for i, video in enumerate(dataloader):
    loss = model(video) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Mixed precision
with torch.cuda.amp.autocast():
    loss = model(video)
```

### 2. Poor Temporal Prediction

**Problem**: Model only learns spatial features, ignores temporal dynamics
**Symptoms**:
- Good performance on static images
- Poor performance on temporal tasks
- Predictions don't capture motion

**Solution**:
```python
# Use temporal-biased masking (mask future frames)
context_frames = video[:, :, :8]  # Past
target_frames = video[:, :, 8:]   # Future (masked)

# Increase prediction horizon
k = 8  # Predict 8 frames ahead (not 1)

# Add temporal augmentation
video = temporal_jitter(video)  # Random frame sampling
video = temporal_reverse(video, p=0.5)  # Reverse time 50%

# Multi-scale temporal loss
loss = loss_1frame + 0.5*loss_4frame + 0.25*loss_8frame
```

### 3. Slow Training

**Problem**: Processing long videos is slow
**Symptoms**:
- 1-2 iterations per second
- Training will take months
- GPU utilization low

**Solution**:
```python
# Start with short clips, gradually increase length
num_frames = min(8 + epoch // 50, 32)

# Use factorized attention
config["factorized"] = True

# Optimize data loading
dataloader = DataLoader(
    dataset,
    num_workers=16,  # More workers
    pin_memory=True,
    prefetch_factor=4
)

# Reduce decoder size
config["predictor_dim"] = 384  # Smaller than encoder (768)
config["predictor_layers"] = 6  # Fewer layers

# Mixed precision
use_amp = True
```

### 4. EMA Target Collapse

**Problem**: Target encoder diverges or produces degenerate features
**Symptoms**:
- All predictions become identical
- Loss goes to near-zero
- Poor downstream performance

**Solution**:
```python
# Use proper EMA momentum (not too high)
ema_momentum = 0.996  # Start
ema_momentum_end = 0.9999  # End

# Cosine schedule
progress = epoch / total_epochs
tau = tau_end - (tau_end - tau_base) * 0.5 * (1 + cos(π * progress))

# Ensure stop-gradient
with torch.no_grad():
    target_features = target_encoder(target_frames)

# Initialize target encoder properly
target_encoder.load_state_dict(online_encoder.state_dict())
```

### 5. Overfitting to Low-Level Details

**Problem**: Model predicts exact pixels rather than semantics
**Symptoms**:
- Good reconstruction quality
- Poor transfer to downstream tasks
- Features not semantic

**Solution**:
```python
# Predict in latent space (not pixels)
# This is already done in V-JEPA!

# Use stop-gradient on targets
with torch.no_grad():
    targets = target_encoder(frames)

# Don't use pixel loss
# loss = mse(pred_pixels, target_pixels)  # Bad!
loss = mse(pred_features, target_features)  # Good!

# Add representation normalization
targets = F.normalize(targets, dim=-1)
predictions = F.normalize(predictions, dim=-1)
```

### 6. Ignoring Long-Range Dependencies

**Problem**: Model only captures short-term dynamics
**Symptoms**:
- Good at 1-frame prediction
- Poor at long-horizon prediction
- Doesn't learn world model

**Solution**:
```python
# Increase prediction horizon
context_end = total_frames - 8  # Predict 8 frames ahead (not 1)

# Use recurrent prediction
pred_1 = predictor(context)  # Predict t+1
pred_2 = predictor(torch.cat([context, pred_1]))  # Predict t+2
pred_3 = predictor(torch.cat([context, pred_1, pred_2]))  # Predict t+3

# Multi-step loss
loss = mse(pred_1, target_1) + mse(pred_2, target_2) + mse(pred_3, target_3)

# Increase context length
num_context_frames = 16  # More context
```

### 7. Data Augmentation Issues

**Problem**: Wrong augmentation for video
**Symptoms**:
- Model confused by augmentation
- Temporal consistency broken
- Poor performance

**Solution**:
```python
# Use video-aware augmentation
class VideoAugmentation:
    def __call__(self, video):
        # Spatial augmentation (same for all frames)
        crop_params = random_crop_params()
        video = apply_crop(video, crop_params)  # All frames

        # Color augmentation (same for all frames)
        color_params = random_color_params()
        video = apply_color(video, color_params)

        # Temporal augmentation
        video = temporal_subsample(video)  # Random frame sampling

        return video

# Avoid per-frame augmentation
# Don't do different crop/color per frame!
```

### 8. Positional Encoding Mistakes

**Problem**: Wrong positional encodings for video
**Symptoms**:
- Model can't understand spatial or temporal structure
- Poor localization
- Confused about time

**Solution**:
```python
# Separate spatial and temporal positional encodings
spatial_pos_embed = nn.Parameter(torch.randn(1, H*W, D))
temporal_pos_embed = nn.Parameter(torch.randn(1, T, D))

# Add both
tokens = tubelet_embed(video)  # (B, T*H*W, D)
tokens = rearrange(tokens, 'b (t h w) d -> b t h w d')

# Broadcast and add
tokens = tokens + spatial_pos_embed.view(1, 1, H, W, D)
tokens = tokens + temporal_pos_embed.view(1, T, 1, 1, D)

tokens = rearrange(tokens, 'b t h w d -> b (t h w) d')
```

## References

```bibtex
@article{bardes2024revisiting,
  title={Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Bardes, Adrien and Garrido, Quentin and Ponce, Jean and LeCun, Yann},
  journal={arXiv preprint arXiv:2404.08471},
  year={2024}
}

@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}

@article{lecun2022path,
  title={A Path Towards Autonomous Machine Intelligence},
  author={LeCun, Yann},
  journal={OpenReview},
  year={2022}
}

@article{arnab2021vivit,
  title={ViViT: A Video Vision Transformer},
  author={Arnab, Anurag and Dehghani, Mostafa and Heigold, Georg and Sun, Chen and Lu{\v{c}}i{\'c}, Mario and Schmid, Cordelia},
  journal={ICCV},
  year={2021}
}

@article{bertasius2021space,
  title={Is Space-Time Attention All You Need for Video Understanding?},
  author={Bertasius, Gedas and Wang, Heng and Torresani, Lorenzo},
  journal={ICML},
  year={2021}
}

@article{tong2022videomae,
  title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Tong, Zhan and Song, Yibing and Wang, Jue and Wang, Limin},
  journal={NeurIPS},
  year={2022}
}

@article{ha2018world,
  title={World Models},
  author={Ha, David and Schmidhuber, J{\"u}rgen},
  journal={arXiv preprint arXiv:1803.10122},
  year={2018}
}

@article{hafner2023mastering,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}
```

**Official Code**: https://github.com/facebookresearch/jepa
**Nexus Implementation**: `nexus/models/ssl/v_jepa.py`

## Additional Resources

- **Yann LeCun's JEPA paper**: [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf)
- **Video transformers survey**: [Video Transformer: A Survey](https://arxiv.org/abs/2201.05991)
- **World models in RL**: [World Models for Autonomous Driving](https://arxiv.org/abs/2403.02622)
- **Robotics applications**: [Learning Visual Representations for Transfer Learning by Suppressing Texture](https://arxiv.org/abs/2011.01901)
