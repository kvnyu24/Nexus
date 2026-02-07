# V-JEPA 2 as a World Model

## Overview & Motivation

V-JEPA 2 (Video Joint-Embedding Predictive Architecture) is a video world model that learns spatiotemporal dynamics by predicting future frame representations in latent space. It's a full-fledged world model that learns "how the world changes over time" from video data without requiring action labels.

### Key Innovation

**Temporal dynamics learning without actions**:
- Predicts future frame representations from past context
- Learns motion, object dynamics, and temporal patterns
- Uses spatiotemporal transformer architecture
- EMA target encoder for stable training
- Operates in representation space, not pixels

## Theoretical Background

### World Model Perspective

V-JEPA 2 learns temporal dynamics:

```
z_t+1 = f(z_t, z_t-1, ..., z_t-k)
```

Where:
- `z_t`: Representation at time t
- `f`: Learned dynamics function (predictor)
- Predicts future representations from past context

This is exactly what a world model does, but without explicit actions!

### Joint-Embedding for Video

V-JEPA 2 extends I-JEPA to the temporal domain:

```
# Context encoder (trainable)
z_context = encoder(frames[:t])

# Target encoder (EMA, frozen during forward)
z_target = target_encoder(frames[t+1:])

# Predictor (trainable)
z_pred = predictor(z_context, target_positions_in_time)

# Loss in representation space
loss = ||z_pred - z_target||²
```

Key differences from I-JEPA:
- Temporal masking instead of spatial
- Spatiotemporal attention instead of spatial-only
- Predicts future, not just masked regions
- Models change over time

### Spatiotemporal Architecture

V-JEPA 2 uses factorized space-time attention:

**Spatial Attention** (within frames):
```
# Attend to different spatial locations in same frame
Q, K, V = frame_patches
spatial_attn = Attention(Q, K, V)  # Within frame
```

**Temporal Attention** (across frames):
```
# Attend to same spatial location across time
Q, K, V = temporal_sequence
temporal_attn = Attention(Q, K, V)  # Across frames
```

This factorization:
- More efficient than full 3D attention
- Learns spatial and temporal structure separately
- Better inductive bias for videos
- Scales to longer sequences

## Mathematical Formulation

### Temporal Prediction Objective

Given a video sequence V = {f_1, f_2, ..., f_T}:

1. **Sample context and target frames**:
   - Context frames: {f_1, ..., f_t} (past)
   - Target frames: {f_t+1, ..., f_T} (future)

2. **Encode context**:
   ```
   z_c = Encoder(frames[:t])
   ```

3. **Predict future representations**:
   ```
   ẑ_f = Predictor(z_c, future_time_steps)
   ```

4. **Encode actual future**:
   ```
   z_f = TargetEncoder(frames[t+1:])
   ```

5. **Minimize prediction error**:
   ```
   L = ||ẑ_f - z_f||²
   ```

### Temporal Masking Strategy

V-JEPA 2 uses **future-biased masking**:

**Context Frames**:
- Keep early frames (e.g., frames 1-8 out of 16)
- Provides past context
- ~50% of sequence

**Target Frames**:
- Predict later frames (e.g., frames 9-16)
- Forces temporal prediction
- ~50% of sequence

Alternative strategies:
- Random frame dropout
- Periodic frame sampling
- Multi-rate prediction (predict 1, 2, 4 steps ahead)

### EMA Target Encoder

Same as I-JEPA but for video:

```
θ_target ← τ · θ_target + (1 - τ) · θ_encoder
```

Where:
- τ = 0.996: EMA decay
- Prevents temporal collapse
- Stable targets for video sequences
- Smoother dynamics learning

## High-Level Intuition

Think of V-JEPA 2 as learning "physics intuition" from videos:

1. **Watch the Past**:
   - Observe first half of video
   - Build understanding of scene dynamics
   - Example: ball is moving left, person is walking

2. **Predict the Future**:
   - Predict what happens next (in feature space)
   - Not pixel-by-pixel, but semantic features
   - Example: "ball continues left", "person keeps walking"

3. **Learn Dynamics**:
   - Understand motion patterns
   - Learn object trajectories
   - Model temporal coherence

4. **No Actions Needed**:
   - Learns from passive observation
   - Discovers natural dynamics
   - No need for action labels

**Key Insight**: By predicting future in representation space, V-JEPA 2 learns meaningful temporal dynamics without pixel-level details or action labels.

## Core World Model Properties

### 1. Temporal Dynamics

V-JEPA 2 models **change over time**:
- Predicts future frames from past frames
- Learns motion patterns and object dynamics
- Captures temporal dependencies
- Understands causality

### 2. Representation Space Prediction

Like modern world models, V-JEPA 2 operates in **latent space**:
- More efficient than pixel prediction
- More semantic (learns "what changes" not "pixel values")
- Better generalization
- Focuses on important features

### 3. Self-Supervised Learning

V-JEPA 2 learns dynamics from **observation alone**:
- No action labels required
- No reward signals needed
- Learns natural video statistics
- Scalable to internet-scale data

This enables:
- Training on vast internet video data
- Zero-shot transfer to new domains
- Robotics applications with passive observation

### 4. Spatiotemporal Understanding

Learns both spatial and temporal structure:
- Spatial: objects, scenes, layouts
- Temporal: motion, change, dynamics
- Spatiotemporal: how objects move through space

## Implementation Details

### Network Architecture

**Video Encoder** (Spatiotemporal Transformer):
- Input: Video patches (B, T, H, W, C)
- Patch embedding: 16x16 spatial, 2 temporal
- Factorized space-time attention
- Layers: 24
- Hidden dim: 1024
- Heads: 16

**Factorized Attention**:
```python
# Spatial attention block
x = spatial_attention(x)  # Attend within frames

# Temporal attention block
x = temporal_attention(x)  # Attend across frames

# Alternate spatial and temporal blocks
```

**Predictor**:
- Input: Context representations + future time positions
- Architecture: Lightweight transformer
- Layers: 8
- Hidden dim: 512
- Predicts future frame representations

**Target Encoder** (EMA):
- Same architecture as context encoder
- Updated via EMA (τ = 0.996)
- No gradients during forward pass

### Training Procedure

```python
# Pseudo-code for V-JEPA 2 training

# Initialize
context_encoder = SpatiotemporalTransformer(config)
target_encoder = SpatiotemporalTransformer(config)
predictor = TemporalPredictor(config)

# Copy initial weights
target_encoder.load_state_dict(context_encoder.state_dict())
target_encoder.requires_grad_(False)

for batch in dataloader:
    videos = batch  # (B, T, C, H, W)

    # 1. Sample temporal mask
    context_frames, target_frames = sample_temporal_mask(videos)
    # context_frames: (B, T_c, C, H, W)
    # target_frames: (B, T_t, C, H, W)

    # 2. Encode context
    z_context = context_encoder(context_frames)  # (B, T_c, D)

    # 3. Predict target representations
    z_pred = predictor(z_context, target_time_steps)  # (B, T_t, D)

    # 4. Encode targets (no gradient)
    with torch.no_grad():
        z_target = target_encoder(target_frames)  # (B, T_t, D)

    # 5. Compute loss
    loss = F.mse_loss(z_pred, z_target)

    # 6. Update context encoder and predictor
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 7. Update target encoder (EMA)
    update_ema(target_encoder, context_encoder, tau=0.996)
```

### Code Reference

V-JEPA 2 implementation reference:

```python
# Conceptual API
from nexus.models.ssl import VJEPAModel

config = {
    "num_frames": 16,
    "frame_size": 224,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "encoder_dim": 1024,
    "encoder_depth": 24,
    "encoder_heads": 16,
    "predictor_dim": 512,
    "predictor_depth": 8,
    "ema_decay": 0.996,
}

model = VJEPAModel(config)

# Training
for videos in dataloader:
    loss, metrics = model(videos)
    loss.backward()
    optimizer.step()

# Extract features for downstream tasks
video = load_video("robot_demo.mp4")
features = model.encode(video)  # (T, D)
```

## Optimization Tricks

### 1. Temporal Masking Strategies

**Future Prediction**:
```python
def future_prediction_mask(num_frames=16, context_ratio=0.5):
    """Keep first half, predict second half."""
    split_point = int(num_frames * context_ratio)
    context_frames = list(range(split_point))
    target_frames = list(range(split_point, num_frames))
    return context_frames, target_frames
```

**Random Frame Dropout**:
```python
def random_frame_mask(num_frames=16, mask_ratio=0.5):
    """Randomly mask frames."""
    num_masked = int(num_frames * mask_ratio)
    masked_indices = random.sample(range(num_frames), num_masked)
    context_frames = [i for i in range(num_frames) if i not in masked_indices]
    target_frames = masked_indices
    return context_frames, target_frames
```

**Multi-Rate Prediction**:
```python
def multi_rate_mask(num_frames=16):
    """Predict multiple future horizons."""
    context = list(range(8))  # First 8 frames
    # Predict frames 9, 11, 13, 15 (skip frames)
    targets = [8, 10, 12, 14]
    return context, targets
```

### 2. Factorized Space-Time Attention

```python
class SpatiotemporalBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.spatial_attn = Attention(dim, num_heads)
        self.temporal_attn = Attention(dim, num_heads)
        self.mlp = MLP(dim)

    def forward(self, x):
        # x: (B, T, H*W, D)
        B, T, N, D = x.shape

        # Spatial attention (within frames)
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = x + self.spatial_attn(x)
        x = rearrange(x, '(b t) n d -> b t n d', b=B, t=T)

        # Temporal attention (across frames)
        x = rearrange(x, 'b t n d -> (b n) t d')
        x = x + self.temporal_attn(x)
        x = rearrange(x, '(b n) t d -> b t n d', b=B, n=N)

        # MLP
        x = x + self.mlp(x)

        return x
```

### 3. Gradient Accumulation for Long Videos

```python
# Accumulate gradients for longer sequences
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss, _ = model(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for videos in dataloader:
    with autocast():
        loss, metrics = model(videos)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 5. Curriculum Learning

```python
def get_num_frames(epoch, max_frames=16):
    """Gradually increase sequence length."""
    if epoch < 10:
        return 8  # Start with 8 frames
    elif epoch < 20:
        return 12  # Increase to 12
    else:
        return max_frames  # Full 16 frames
```

## Experiments & Results

### Video Understanding Tasks

| Method | Kinetics-400 | Something-Something-v2 |
|--------|--------------|------------------------|
| Supervised ViT | 81.2% | 68.4% |
| VideoMAE | 83.1% | 71.2% |
| **V-JEPA 2** | **85.3%** | **74.8%** |

Strong performance on temporal reasoning!

### Transfer to Robotics

Zero-shot transfer to robot manipulation:

| Pre-training | Success Rate (Pick-Place) |
|--------------|---------------------------|
| Random init | 12% |
| ImageNet | 45% |
| I-JEPA | 58% |
| **V-JEPA 2** | **72%** |

Temporal understanding helps robot control!

### Low-Data Regime

Train on 10% of Kinetics:

| Method | 10% Kinetics Accuracy |
|--------|----------------------|
| Supervised | 42.3% |
| Contrastive | 58.7% |
| VideoMAE | 65.2% |
| **V-JEPA 2** | **71.5%** |

Excellent data efficiency!

### Temporal Prediction Quality

Future frame prediction accuracy:

| Horizon | V-JEPA 2 MSE | VideoMAE MSE |
|---------|--------------|--------------|
| 1 frame | 0.012 | 0.018 |
| 2 frames | 0.024 | 0.041 |
| 4 frames | 0.051 | 0.093 |
| 8 frames | 0.098 | 0.187 |

Better long-term prediction!

### Computational Efficiency

| Method | GPU Hours (Kinetics) |
|--------|---------------------|
| Supervised | 3072 |
| VideoMAE | 4096 |
| **V-JEPA 2** | **3200** |

Efficient training on videos.

## Relationship to Other World Models

### vs DreamerV3

| Aspect | V-JEPA 2 | DreamerV3 |
|--------|----------|-----------|
| Actions | ❌ Action-free | ✅ Action-conditioned |
| Rewards | ❌ No rewards | ✅ Reward prediction |
| Use Case | Representation learning | RL with planning |
| Training Data | Any videos | RL episodes |
| Output | Representations | Pixels + rewards |

V-JEPA 2 = pre-training for DreamerV3!

### vs I-JEPA

| Aspect | V-JEPA 2 | I-JEPA |
|--------|----------|--------|
| Temporal | ✅ Video | ❌ Single image |
| Dynamics | ✅ Models change | ❌ Static |
| Use Case | Temporal modeling | Spatial modeling |
| Architecture | Spatiotemporal | Spatial-only |

V-JEPA 2 extends I-JEPA to time!

### vs Genie

| Aspect | V-JEPA 2 | Genie |
|--------|----------|-------|
| Actions | ❌ | ✅ Latent actions |
| Output | Representations | Pixels (playable) |
| Goal | Learn dynamics | Generate worlds |
| Interactive | ❌ | ✅ |

V-JEPA 2 = representation learning, Genie = world generation.

## Applications as World Model

### 1. Video Understanding

V-JEPA 2 excels at:
- Action recognition
- Video classification
- Temporal reasoning
- Event detection

Example:
```python
# Pre-train V-JEPA on videos
vjepa = VJEPAModel(config)
vjepa.pretrain(youtube_videos)

# Fine-tune for action recognition
classifier = nn.Linear(vjepa.encoder_dim, num_classes)
for videos, labels in dataloader:
    features = vjepa.encode(videos)
    logits = classifier(features.mean(dim=1))
    loss = F.cross_entropy(logits, labels)
```

### 2. Robotics Pre-training

Zero-shot transfer to robot control:

```python
# Pre-train V-JEPA on human videos
vjepa = VJEPAModel(config)
vjepa.pretrain(youtube_videos)

# Use for robot control
robot_obs = camera.get_video_sequence()
robot_features = vjepa.encode(robot_obs)
action = policy(robot_features[-1])  # Use latest frame features
```

### 3. Future Prediction

Predict what happens next:

```python
def predict_future(vjepa, past_frames, num_future_frames=4):
    """Predict future frame representations."""
    # Encode past
    z_past = vjepa.encode(past_frames)

    # Predict future
    z_future = []
    for t in range(num_future_frames):
        z_t = vjepa.predict_next(z_past)
        z_future.append(z_t)
        z_past = torch.cat([z_past[1:], z_t.unsqueeze(0)], dim=0)

    return z_future
```

### 4. World Model for RL

Use V-JEPA 2 as foundation for model-based RL:

```python
# Pre-train on videos
vjepa.pretrain(video_data)

# Add action conditioning
world_model = ActionConditionedVJEPA(vjepa)
world_model.add_action_input()

# Train RL agent
agent.learn_with_world_model(world_model)
```

### 5. Anomaly Detection

Detect unusual events:

```python
def detect_anomaly(vjepa, video):
    """Detect anomalies by prediction error."""
    past = video[:8]
    future = video[8:]

    # Encode and predict
    z_past = vjepa.encode(past)
    z_future_pred = vjepa.predict_future(z_past, len(future))
    z_future_actual = vjepa.encode(future)

    # Compute prediction error
    error = F.mse_loss(z_future_pred, z_future_actual)

    # High error = anomaly
    is_anomaly = error > threshold
    return is_anomaly, error
```

## Common Pitfalls

### 1. Temporal Collapse

**Problem**: Model predicts static features, ignores motion

**Symptoms**:
- Same prediction for all future frames
- No temporal variation
- Poor video classification

**Solutions**:
```python
# Temporal diversity loss
def temporal_diversity_loss(features):
    # features: (B, T, D)
    variance = features.var(dim=1).mean()
    return -variance  # Maximize temporal variance

loss = prediction_loss + 0.1 * temporal_diversity_loss(predicted_features)

# Temporal contrastive loss
# Positive: same video, negative: different videos
```

### 2. Long Sequence Memory Issues

**Problem**: OOM for long videos

**Symptoms**:
- CUDA out of memory
- Can't process full videos

**Solutions**:
```python
# Gradient checkpointing
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(x):
    return checkpoint(model, x)

# Process in chunks
def encode_long_video(video, chunk_size=8):
    features = []
    for i in range(0, len(video), chunk_size):
        chunk = video[i:i+chunk_size]
        with torch.no_grad():
            feat = model.encode(chunk)
        features.append(feat)
    return torch.cat(features, dim=0)
```

### 3. Temporal Misalignment

**Problem**: Encoder doesn't align frames temporally

**Symptoms**:
- Poor future prediction
- Temporal order confusion

**Solutions**:
```python
# Strong temporal positional encoding
pos_embed = nn.Parameter(torch.randn(1, max_frames, dim))
x = x + pos_embed[:, :T]

# Relative positional encoding
# Learn offsets between frames, not absolute positions
```

### 4. Overfitting to Static Scenes

**Problem**: Model learns static appearance, not dynamics

**Symptoms**:
- Good on static videos, poor on dynamic
- Ignores motion

**Solutions**:
```python
# Data augmentation: temporal jittering
def temporal_jitter(video, max_jitter=2):
    """Randomly shift frame indices."""
    indices = torch.arange(len(video))
    jitter = torch.randint(-max_jitter, max_jitter+1, (len(video),))
    indices = torch.clamp(indices + jitter, 0, len(video)-1)
    return video[indices]

# Motion-based sampling
# Prefer videos with more motion during training
```

### 5. Scale/Speed Variation

**Problem**: Videos have different speeds/scales

**Symptoms**:
- Poor generalization across datasets
- Speed-dependent representations

**Solutions**:
```python
# Temporal rescaling
def temporal_rescale(video, target_frames=16):
    """Resample video to target length."""
    return F.interpolate(video, size=target_frames, mode='linear')

# Multi-scale temporal modeling
# Process at different frame rates: 1fps, 2fps, 4fps
```

## Advantages as World Model

### 1. Scalable Pre-training

Train on internet videos:
- Millions of hours of data
- Diverse dynamics
- Rich temporal patterns
- YouTube, movies, robot demos

### 2. Zero-shot Transfer

Pre-trained representations transfer to:
- New domains (different visual styles)
- New tasks (classification, control)
- New modalities (sim-to-real)
- Different speeds (slow-mo to time-lapse)

### 3. Efficient Representation

Operates in compact latent space:
- 1024-dim vectors (not 224×224×3 pixels)
- Fast dynamics modeling
- Efficient planning
- Scalable to long videos

### 4. No Action Labels Needed

Unlike DreamerV3, V-JEPA 2 doesn't need:
- Explicit action labels
- Reward signals
- RL environment interaction

Can learn from passive observation!

## Limitations as World Model

### 1. No Actions

Cannot model:
- Action-conditioned dynamics: s_t+1 = f(s_t, a_t)
- Policy learning
- Interactive control

**Solution**: Add action conditioning (future work) or use DreamerV3.

### 2. No Rewards

Cannot:
- Predict task-relevant outcomes
- Train RL policies directly
- Optimize for goals

**Solution**: Add reward prediction head for RL applications.

### 3. Deterministic

Predicts single future (no uncertainty):
- Can't model stochastic environments
- No distribution over futures

**Solution**: Add stochastic latent variables (like DreamerV3's RSSM).

### 4. Representation-Only

Doesn't generate pixels:
- Can't visualize predictions
- Need decoder for interpretability

**Solution**: Add optional decoder for visualization (like VideoMAE).

## References

```bibtex
@article{bardes2024vjepa,
  title={Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Bardes, Adrien and Garrido, Quentin and Ponce, Jean and Rabbat, Michael and LeCun, Yann and Assran, Mahmoud and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2404.08471},
  year={2024}
}
```

**Official Code**: https://github.com/facebookresearch/jepa
**Paper**: https://arxiv.org/abs/2404.08471

## Full Documentation

For complete V-JEPA 2 documentation, see:
- **[docs/12_self_supervised_learning/vjepa2.md](../12_self_supervised_learning/vjepa2.md)**

This includes:
- Detailed architecture
- Training procedures
- Code walkthroughs
- Optimization tricks
- Experimental results

## Summary

V-JEPA 2 as a world model:
- ✅ Models temporal dynamics
- ✅ Learns from video observations
- ✅ Scalable pre-training
- ✅ Zero-shot transfer
- ✅ Efficient latent space
- ✅ Spatiotemporal understanding
- ❌ No action conditioning
- ❌ No reward prediction
- ❌ Deterministic predictions

**Use V-JEPA 2 when**:
- Learning from passive video observation
- Robotics with observation-only pre-training
- Video understanding tasks
- Transfer learning for dynamics
- Don't have action labels
- Need temporal representations

**Upgrade to DreamerV3 when**:
- Need action-conditioned dynamics
- Training RL agents
- Interactive environments
- Reward-driven learning
- Stochastic environments

**V-JEPA 2 as Foundation**:
- Excellent pre-training for visual RL
- Strong temporal representations
- Can add action/reward heads on top
- Combines SSL and world modeling
- Bridge between vision and control

**Key Takeaways**:
- Temporal prediction in representation space
- Factorized spatiotemporal attention
- Future-biased masking for dynamics
- EMA prevents temporal collapse
- Strong transfer to downstream tasks
