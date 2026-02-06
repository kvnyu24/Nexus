# V-JEPA 2 as a World Model

V-JEPA 2 (Video Joint-Embedding Predictive Architecture) is a video world model that learns spatiotemporal dynamics by predicting future frame representations in latent space. It's a full-fledged world model that learns "how the world changes over time" from video data.

## World Model Perspective

V-JEPA 2 learns temporal dynamics:

```
z_t+1 = f(z_t, z_t-1, ..., z_t-k)
```

Where:
- `z_t`: Representation at time t
- `f`: Learned dynamics function (predictor)
- Predicts future representations from past context

## Core World Model Properties

### 1. Temporal Dynamics

V-JEPA 2 models **change over time**:
- Predicts future frames from past frames
- Learns motion patterns and object dynamics
- Captures temporal dependencies

### 2. Representation Space Prediction

Like modern world models, V-JEPA 2 operates in **latent space**:
- More efficient than pixel prediction
- More semantic (learns "what changes" not "pixel values")
- Better generalization

### 3. Self-Supervised Learning

V-JEPA 2 learns dynamics from **observation alone**:
- No action labels required
- No reward signals needed
- Learns natural video statistics

This enables:
- Training on vast internet video data
- Zero-shot transfer to new domains
- Robotics applications with passive observation

## Relationship to Other World Models

### vs DreamerV3

| Aspect | V-JEPA 2 | DreamerV3 |
|--------|----------|-----------|
| Actions | ❌ Action-free | ✅ Action-conditioned |
| Rewards | ❌ No rewards | ✅ Reward prediction |
| Use Case | Representation learning | RL with planning |
| Training Data | Any videos | RL episodes |
| Output | Representations | Pixels + rewards |

### vs I-JEPA

| Aspect | V-JEPA 2 | I-JEPA |
|--------|----------|--------|
| Temporal | ✅ Video | ❌ Single image |
| Dynamics | ✅ Models change | ❌ Static |
| Use Case | Temporal modeling | Spatial modeling |

### vs Genie

| Aspect | V-JEPA 2 | Genie |
|--------|----------|-------|
| Actions | ❌ | ✅ Latent actions |
| Output | Representations | Pixels (playable) |
| Goal | Learn dynamics | Generate worlds |
| Interactive | ❌ | ✅ |

## Applications as World Model

### 1. Video Understanding

V-JEPA 2 excels at:
- Action recognition
- Video classification
- Temporal reasoning

### 2. Robotics

Zero-shot transfer to robot control:
- Learn dynamics from human videos
- Transfer to robot visual observations
- Policy learning in representation space

Example:
```python
# Pre-train V-JEPA on human videos
vjepa = VJEPAModel(config)
vjepa.train(youtube_videos)

# Use for robot control
robot_obs = camera.get_frame()
robot_features = vjepa.encode(robot_obs)
action = policy(robot_features)
```

### 3. Future Prediction

Predict what happens next:
- Anticipate object motion
- Predict occlusions
- Model scene dynamics

### 4. World Model for RL

Use V-JEPA 2 as a foundation for model-based RL:

```python
# Pre-train on videos
vjepa.pretrain(video_data)

# Add action conditioning
world_model = ActionConditionedVJEPA(vjepa)
world_model.add_action_input()

# Train RL agent
agent.learn_with_world_model(world_model)
```

## Architecture for World Modeling

### Spatiotemporal Encoder

```python
# Factorized space-time attention
x = spatial_transformer(video_patches)  # Spatial understanding
x = temporal_transformer(x)              # Temporal dynamics
```

Learns:
- Spatial structure (objects, scenes)
- Temporal patterns (motion, change)

### Future Prediction

```python
# Context: past frames
z_context = encoder(frames[:t])

# Predict: future frames
z_future = predictor(z_context)

# Target: actual future (from EMA encoder)
z_target = target_encoder(frames[t+1:])

loss = mse(z_future, z_target)
```

### Temporal Masking

V-JEPA 2 uses **future-biased masking**:
- Keep early frames (past)
- Mask later frames (future)
- Forces model to predict temporal evolution

This is exactly what a world model needs!

## Advantages as World Model

### 1. Scalable Pre-training

Train on internet videos:
- Millions of hours of data
- Diverse dynamics
- Rich temporal patterns

### 2. Zero-shot Transfer

Pre-trained representations transfer to:
- New domains (different visual styles)
- New tasks (classification, control)
- New modalities (sim-to-real)

### 3. Efficient Representation

Operates in compact latent space:
- 768-dim vectors (not 224×224×3 pixels)
- Fast dynamics modeling
- Efficient planning

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

## Full Documentation

For complete V-JEPA 2 documentation, see:
- **[docs/12_self_supervised_learning/vjepa2.md](../12_self_supervised_learning/vjepa2.md)**

This includes:
- Detailed architecture
- Training procedures
- Code walkthroughs  
- Optimization tricks
- Experimental results

## Code Example

```python
from nexus.models.ssl import VJEPAModel

# Configure V-JEPA 2
config = {
    "encoder_dim": 768,
    "predictor_dim": 384,
    "num_frames": 16,
    "mask_ratio": 0.7,
    "temporal_mask_ratio": 0.5,
}

# Pre-train on videos
model = VJEPAModel(config)
for videos in video_dataset:
    loss, metrics = model(videos)
    loss.backward()

# Use as world model encoder
world_model = WorldModel()
world_model.encoder = model.context_encoder

# Extract features for downstream tasks
video = load_video("robot_demo.mp4")
features = model.context_encoder(video)
# features.shape: (batch, num_frames, 768)
```

## Summary

V-JEPA 2 as a world model:
- ✅ Models temporal dynamics
- ✅ Learns from video observations
- ✅ Scalable pre-training
- ✅ Zero-shot transfer
- ✅ Efficient latent space
- ❌ No action conditioning
- ❌ No reward prediction
- ❌ Deterministic predictions

**Use V-JEPA 2 when**:
- Learning from passive video observation
- Robotics with observation-only pre-training
- Video understanding tasks
- Transfer learning for dynamics
- Don't have action labels

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
