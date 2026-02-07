# I-JEPA as a World Model

## Overview & Motivation

I-JEPA (Image Joint-Embedding Predictive Architecture) functions as a world model by learning to predict representations of masked image regions from visible context. While primarily a self-supervised learning method, I-JEPA embodies world modeling principles by learning a predictive model of visual scenes.

### Key Innovation

**Representation-space prediction without reconstruction**:
- Predicts in abstract representation space, not pixels
- Avoids low-level pixel prediction (which can lead to shortcuts)
- Uses joint-embedding architecture with predictor
- EMA target encoder for stable learning
- Learns semantic spatial structure

## Theoretical Background

### World Model Perspective

From a world model viewpoint, I-JEPA learns:

```
z_target = f(z_context, position)
```

Where:
- `z_context`: Representation of visible image regions (current state)
- `position`: Location information (implicit "action" - where to look)
- `z_target`: Predicted representation of target region (next state)

This is analogous to a world model learning:
```
s_t+1 = f(s_t, a_t)
```

But in the spatial domain rather than temporal domain.

### Joint-Embedding Architecture

I-JEPA uses a joint-embedding predictive architecture:

```
# Context encoder (trainable)
z_context = encoder(x_context)

# Target encoder (EMA, frozen during forward)
z_target = target_encoder(x_target)

# Predictor (trainable)
z_pred = predictor(z_context, position)

# Loss in representation space
loss = ||z_pred - z_target||²
```

This architecture:
- Prevents collapse through EMA updates
- Avoids pixel-level reconstruction shortcuts
- Learns abstract semantic features
- Enables flexible masking strategies

### Why Not Pixel Reconstruction?

Traditional autoencoders reconstruct pixels:
```
x_recon = decoder(encoder(x_masked))
loss = ||x_recon - x||²
```

Problems:
- Model learns low-level textures, not semantics
- Wastes capacity on irrelevant details
- Difficult to predict exact pixel values
- Shortcuts (e.g., blur, color matching)

I-JEPA solution:
- Predict representations, not pixels
- Forces model to learn semantic features
- More efficient and effective
- Better downstream performance

## Mathematical Formulation

### Prediction Objective

Given an image x, I-JEPA:

1. **Samples context and target masks**:
   - Context mask M_c (what's visible)
   - Target mask M_t (what to predict)

2. **Encodes context**:
   ```
   z_c = Encoder(x ⊙ M_c, M_c)
   ```

3. **Predicts target representation**:
   ```
   ẑ_t = Predictor(z_c, M_t)
   ```

4. **Encodes actual target**:
   ```
   z_t = TargetEncoder(x ⊙ M_t, M_t)
   ```

5. **Minimizes prediction error**:
   ```
   L = ||ẑ_t - z_t||²
   ```

### EMA Target Encoder

The target encoder uses exponential moving average:

```
θ_target ← τ · θ_target + (1 - τ) · θ_encoder
```

Where:
- τ ∈ [0.996, 0.999]: EMA decay (typically 0.996)
- θ_encoder: Context encoder parameters
- θ_target: Target encoder parameters

This provides:
- Stable targets during training
- Prevents representational collapse
- Smoother optimization landscape
- Better final representations

### Masking Strategy

I-JEPA uses multi-block masking:

**Context Mask**:
- Keep 4 random blocks (each ~15% of image)
- Total context: ~60% of image
- Non-contiguous regions

**Target Mask**:
- Predict 1-4 blocks (~15% each)
- Must be spatially separated from context
- Forces long-range spatial reasoning

This strategy:
- Prevents trivial local interpolation
- Encourages semantic understanding
- Forces model to learn object-level features
- Better than random pixel masking

## High-Level Intuition

Think of I-JEPA as learning spatial common sense:

1. **Context Understanding**:
   - See parts of an image (e.g., left half of a car)
   - Build an understanding in abstract feature space

2. **Spatial Prediction**:
   - Predict what the right half looks like (in feature space)
   - Not pixel-by-pixel, but semantic features

3. **No Pixel Details**:
   - Don't predict exact colors or textures
   - Predict high-level semantics: "there's a wheel here", "car body continues"

4. **World Knowledge**:
   - Learns: "cars have wheels", "faces are symmetric"
   - Spatial structure: "if top is sky, bottom is likely ground"
   - Object coherence: "parts of objects fit together"

**Key Insight**: By predicting in representation space, I-JEPA learns meaningful world structure without getting distracted by low-level pixel details.

## Implementation Details

### Network Architecture

**Context Encoder** (Vision Transformer):
- Input: Patches from context regions
- Architecture: ViT-H/16 (Huge, 16x16 patches)
- Hidden dim: 1280
- Layers: 32
- Heads: 16
- Output: Patch-level representations

**Predictor**:
- Input: Context representations + target positions
- Architecture: Small transformer
- Layers: 12
- Hidden dim: 768
- Predicts representation for each target patch

**Target Encoder** (EMA):
- Same architecture as context encoder
- Updated via EMA (τ = 0.996)
- No gradients during forward pass

### Training Procedure

```python
# Pseudo-code for I-JEPA training

# Initialize
context_encoder = VisionTransformer(config)
target_encoder = VisionTransformer(config)  # Copy of context encoder
predictor = PredictorNetwork(config)

# Copy initial weights
target_encoder.load_state_dict(context_encoder.state_dict())
target_encoder.requires_grad_(False)

for batch in dataloader:
    images = batch  # (B, 3, 224, 224)

    # 1. Sample masks
    context_mask, target_mask = sample_masks(images)

    # 2. Extract patches
    context_patches = extract_patches(images, context_mask)
    target_patches = extract_patches(images, target_mask)

    # 3. Encode context
    z_context = context_encoder(context_patches)  # (B, N_c, D)

    # 4. Predict target representations
    z_pred = predictor(z_context, target_mask.positions)  # (B, N_t, D)

    # 5. Encode targets (no gradient)
    with torch.no_grad():
        z_target = target_encoder(target_patches)  # (B, N_t, D)

    # 6. Compute loss
    loss = F.mse_loss(z_pred, z_target)

    # 7. Update context encoder and predictor
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 8. Update target encoder (EMA)
    update_ema(target_encoder, context_encoder, tau=0.996)
```

### Code Reference

Note: Full I-JEPA implementation in `docs/12_self_supervised_learning/ijepa.md` and `nexus/models/ssl/`:

```python
# Conceptual API
from nexus.models.ssl import IJEPAModel

config = {
    "image_size": 224,
    "patch_size": 16,
    "encoder_dim": 1280,
    "encoder_depth": 32,
    "encoder_heads": 16,
    "predictor_dim": 768,
    "predictor_depth": 12,
    "ema_decay": 0.996,
}

model = IJEPAModel(config)

# Training
for images in dataloader:
    loss, metrics = model(images)
    loss.backward()
    optimizer.step()

# Extract features for downstream tasks
features = model.encode(images)
```

## Optimization Tricks

### 1. Multi-Block Masking

Sample non-overlapping blocks:

```python
def sample_masks(image_size, num_context_blocks=4, num_target_blocks=1):
    """Sample context and target masks."""
    # Context: 4 blocks, ~15% each
    context_blocks = sample_random_blocks(
        image_size,
        num_blocks=4,
        scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5)
    )

    # Target: 1 block, spatially separated
    target_block = sample_random_blocks(
        image_size,
        num_blocks=1,
        scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        exclude_regions=context_blocks  # Don't overlap
    )

    return context_blocks, target_block
```

### 2. EMA Scheduling

Gradually increase EMA momentum:

```python
def get_ema_decay(epoch, base_decay=0.996, max_decay=0.9996):
    """Cosine schedule for EMA decay."""
    progress = epoch / total_epochs
    decay = base_decay + (max_decay - base_decay) * (1 + math.cos(math.pi * progress)) / 2
    return decay
```

### 3. Predictor Design

Use lightweight predictor:

```python
class Predictor(nn.Module):
    def __init__(self, encoder_dim=1280, predictor_dim=768, depth=12):
        super().__init__()

        # Project encoder features
        self.proj = nn.Linear(encoder_dim, predictor_dim)

        # Transformer predictor
        self.transformer = TransformerEncoder(
            dim=predictor_dim,
            depth=depth,
            heads=12
        )

        # Target position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 196, predictor_dim))

        # Output projection
        self.out_proj = nn.Linear(predictor_dim, encoder_dim)

    def forward(self, context_features, target_positions):
        # Project context
        x = self.proj(context_features)

        # Add target position embeddings
        target_pos_embeds = self.pos_embed[:, target_positions]

        # Concatenate context and target positions
        x = torch.cat([x, target_pos_embeds], dim=1)

        # Transform
        x = self.transformer(x)

        # Extract target predictions
        num_targets = target_positions.shape[1]
        predictions = x[:, -num_targets:]

        # Project back to encoder dim
        predictions = self.out_proj(predictions)

        return predictions
```

### 4. Gradient Clipping

Stabilize training:

```python
# Clip gradients
torch.nn.utils.clip_grad_norm_(
    list(context_encoder.parameters()) +
    list(predictor.parameters()),
    max_norm=1.0
)
```

### 5. Learning Rate Warmup

Use warmup + cosine decay:

```python
def get_lr(step, warmup_steps=10000, total_steps=100000, base_lr=1e-3, min_lr=1e-5):
    if step < warmup_steps:
        # Linear warmup
        return base_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
```

## Experiments & Results

### ImageNet Linear Probe

Freeze encoder, train linear classifier:

| Method | ImageNet Top-1 Accuracy |
|--------|-------------------------|
| Supervised ViT-H | 78.3% |
| MAE | 75.4% |
| data2vec | 78.0% |
| **I-JEPA** | **80.3%** |

I-JEPA outperforms supervised learning!

### Transfer Learning

Fine-tune on downstream tasks:

| Task | I-JEPA | Supervised |
|------|--------|------------|
| CIFAR-10 | 98.2% | 97.8% |
| CIFAR-100 | 87.5% | 85.2% |
| Food-101 | 91.3% | 89.7% |
| Flowers-102 | 97.8% | 96.1% |

Consistently better transfer.

### Low-Data Regime

Train on 1% of ImageNet:

| Method | 1% ImageNet Accuracy |
|--------|---------------------|
| Supervised | 35.2% |
| SimCLR | 48.3% |
| MAE | 52.1% |
| **I-JEPA** | **56.7%** |

Huge gains in low-data scenarios!

### Semantic Segmentation

| Method | ADE20k mIoU |
|--------|-------------|
| Supervised ViT-H | 52.3 |
| MAE ViT-H | 53.6 |
| **I-JEPA ViT-H** | **55.1** |

Better dense prediction tasks.

### Computational Efficiency

| Method | GPU Hours (ImageNet) |
|--------|---------------------|
| Supervised | 2048 |
| SimCLR | 4096 |
| MAE | 2560 |
| **I-JEPA** | **2048** |

Efficient as supervised, better results.

## Relevance to World Modeling

### 1. Representation Prediction

I-JEPA predicts in **representation space** rather than pixel space, a key principle of modern world models. This enables:
- More abstract, semantic predictions
- Computational efficiency
- Better generalization
- Focus on task-relevant features

### 2. No Actions Required

I-JEPA is an **action-free world model**:
- Learns natural image statistics
- Models spatial relationships
- No explicit action inputs needed

This makes it suitable for:
- Learning from passive observations
- Understanding static scenes
- Pre-training visual encoders for downstream RL

### 3. EMA Target Encoder

The exponential moving average (EMA) target encoder provides:
- Stable prediction targets
- Prevents collapse (like self-supervised world models)
- Similar to teacher-student approaches in Genie
- Smoother optimization

### 4. Spatial World Model

I-JEPA is a **spatial world model**:
- Models: z_target = f(z_context, position)
- Learns spatial coherence
- Understands object structure
- Predicts missing information

Analogous to temporal world models:
- Models: s_t+1 = f(s_t, a_t)
- Learns temporal dynamics
- Understands action effects
- Predicts future states

## Applications as World Model

### Scene Understanding

I-JEPA learns spatial coherence useful for:
- Object detection and segmentation
- Spatial reasoning tasks
- Scene completion
- Inpainting

### Transfer to Robotics

I-JEPA representations can be used for:
- Visual control policies
- State estimation
- Goal-conditioned RL
- Visual navigation

Example:
```python
# Pre-train I-JEPA on robot observations
ijepa = IJEPAModel(config)
ijepa.pretrain(robot_camera_images)

# Use for robot control
robot_obs = camera.get_frame()
robot_features = ijepa.encode(robot_obs)
action = policy(robot_features)
```

### Pre-training for RL

Use I-JEPA as pre-training for model-based RL:

```python
# Pre-train I-JEPA on images
ijepa = IJEPAModel(config)
ijepa.train(image_dataset)

# Use encoder for RL world model
world_model = WorldModel()
world_model.encoder = ijepa.context_encoder
world_model.train_dynamics(rl_data)

# Train policy in imagination
agent.train_with_world_model(world_model)
```

### Zero-Shot Scene Completion

Complete occluded regions:

```python
def complete_scene(ijepa, image, mask):
    """Complete masked regions using I-JEPA."""
    # Encode visible context
    context_features = ijepa.encode(image, mask=~mask)

    # Predict masked regions
    predicted_features = ijepa.predict(context_features, target_mask=mask)

    # Decode to pixels (optional, for visualization)
    completed_image = ijepa.decode(predicted_features)

    return completed_image
```

## Common Pitfalls

### 1. Representational Collapse

**Problem**: Encoder outputs constant features

**Symptoms**:
- Loss plateaus early
- Zero variance in representations
- No meaningful features learned

**Solutions**:
```python
# EMA target encoder (prevents collapse)
update_ema(target_encoder, context_encoder, tau=0.996)

# Variance regularization
features = encoder(images)
variance = features.var(dim=0).mean()
loss = prediction_loss - 0.01 * variance  # Encourage diversity

# Batch normalization
features = F.batch_norm(features)  # Prevent collapse
```

### 2. Trivial Shortcuts

**Problem**: Model learns low-level patterns, not semantics

**Symptoms**:
- Good reconstruction, poor downstream performance
- Model predicts textures, not objects
- Transfer learning fails

**Solutions**:
```python
# Multi-block masking (prevent local interpolation)
context_blocks = sample_non_overlapping_blocks(num_blocks=4)
target_blocks = sample_spatially_separated_blocks(num_blocks=1, min_distance=context_blocks)

# Large mask ratios
mask_ratio = 0.75  # Hide 75% of image

# Prediction in representation space (not pixels)
loss = F.mse_loss(predicted_features, target_features)  # Not pixels!
```

### 3. EMA Too High or Too Low

**Problem**: Unstable training or poor targets

**Symptoms**:
- Oscillating loss (EMA too low)
- No learning progress (EMA too high)

**Solutions**:
```python
# Start with moderate EMA
ema_decay = 0.996  # Good default

# Gradually increase
ema_decay = 0.996 + (0.9996 - 0.996) * epoch / total_epochs

# Adaptive EMA based on loss
if loss.std() > threshold:
    ema_decay = min(ema_decay + 0.0001, 0.9996)  # Increase stability
```

### 4. Predictor Too Large

**Problem**: Predictor overfits, encoder underfits

**Symptoms**:
- Predictor has more capacity than encoder
- Downstream tasks perform poorly
- Encoder learns trivial features

**Solutions**:
```python
# Lightweight predictor
encoder_dim = 1280  # Large
predictor_dim = 768  # Smaller
predictor_depth = 12  # Shallow

# Regularize predictor
for layer in predictor.layers:
    layer.dropout = 0.1  # Add dropout
```

### 5. Wrong Masking Strategy

**Problem**: Masks too easy or too hard

**Symptoms**:
- Model learns trivial interpolation (masks too easy)
- Loss doesn't decrease (masks too hard)

**Solutions**:
```python
# Balanced masking
context_ratio = 0.60  # Keep 60% as context
target_ratio = 0.15   # Predict 15% targets

# Multi-scale blocks
block_sizes = [(0.1, 0.2), (0.15, 0.25), (0.2, 0.3)]  # Range of scales

# Curriculum learning
mask_ratio = 0.5 + 0.25 * epoch / total_epochs  # Start easy, get harder
```

## Limitations as World Model

I-JEPA differs from full world models:

1. **No Temporal Dynamics**: Single images, not video sequences
2. **No Actions**: Can't model action-conditioned transitions
3. **No Rewards**: Doesn't predict task-relevant outcomes
4. **Static**: Doesn't model change over time

For temporal dynamics, see [V-JEPA 2](vjepa2.md).
For action-conditioned dynamics, see [DreamerV3](dreamerv3.md).

## References

```bibtex
@article{assran2023ijepa,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}
```

**Official Code**: https://github.com/facebookresearch/ijepa
**Paper**: https://arxiv.org/abs/2301.08243

## Full Documentation

For complete I-JEPA documentation, see:
- **[docs/12_self_supervised_learning/ijepa.md](../12_self_supervised_learning/ijepa.md)**

This includes:
- Detailed architecture
- Training procedures
- Code walkthroughs
- Optimization tricks
- Experimental results

## Summary

I-JEPA as a world model:
- ✅ Predicts in representation space
- ✅ Learns spatial structure
- ✅ No action labels needed
- ✅ Strong transfer learning
- ✅ Efficient training
- ❌ No temporal dynamics
- ❌ No action conditioning
- ❌ No reward prediction

**Use I-JEPA when**:
- Learning from static images
- Need visual representations for downstream tasks
- Action-free world modeling is sufficient
- Pre-training for model-based RL
- Transfer learning is important
- Low-data scenarios

**Upgrade to V-JEPA 2 when**:
- Temporal dynamics are needed
- Working with video data
- Modeling change over time is important
- Action-free video understanding required

**Upgrade to DreamerV3 when**:
- Need action-conditioned dynamics
- Training RL agents
- Interactive environments
- Reward prediction needed

**Key Takeaways**:
- Representation prediction > pixel prediction
- EMA target encoder prevents collapse
- Multi-block masking learns semantics
- Excellent pre-training for downstream tasks
- Spatial world model for static scenes
