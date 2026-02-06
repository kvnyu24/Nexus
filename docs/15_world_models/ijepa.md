# I-JEPA as a World Model

I-JEPA (Image Joint-Embedding Predictive Architecture) functions as a world model by learning to predict representations of masked image regions from visible context. While primarily a self-supervised learning method, I-JEPA embodies world modeling principles by learning a predictive model of visual scenes.

## World Model Perspective

From a world model viewpoint, I-JEPA learns:

```
z_target = f(z_context, position)
```

Where:
- `z_context`: Representation of visible image regions (current state)
- `position`: Location information (implicit "action" - where to look)
- `z_target`: Predicted representation of target region (next state)

## Relevance to World Modeling

### 1. Representation Prediction

I-JEPA predicts in **representation space** rather than pixel space, a key principle of modern world models. This enables:
- More abstract, semantic predictions
- Computational efficiency
- Better generalization

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

## Applications as World Model

### Scene Understanding

I-JEPA learns spatial coherence useful for:
- Object detection and segmentation
- Spatial reasoning tasks
- Scene completion

### Transfer to Robotics

I-JEPA representations can be used for:
- Visual control policies
- State estimation
- Goal-conditioned RL

### Pre-training for RL

Use I-JEPA as pre-training for model-based RL:

```python
# Pre-train I-JEPA on images
ijepa = JEPAModel(config)
ijepa.train(image_dataset)

# Use encoder for RL world model
world_model = WorldModel()
world_model.encoder = ijepa.context_encoder
world_model.train_dynamics(rl_data)
```

## Limitations as World Model

I-JEPA differs from full world models:

1. **No Temporal Dynamics**: Single images, not video sequences
2. **No Actions**: Can't model action-conditioned transitions
3. **No Rewards**: Doesn't predict task-relevant outcomes
4. **Static**: Doesn't model change over time

For temporal dynamics, see [V-JEPA 2](vjepa2.md).
For action-conditioned dynamics, see [DreamerV3](dreamerv3.md).

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
- ❌ No temporal dynamics
- ❌ No action conditioning
- ❌ No reward prediction

**Use I-JEPA when**:
- Learning from static images
- Need visual representations for downstream tasks
- Action-free world modeling is sufficient
- Pre-training for model-based RL

**Upgrade to V-JEPA 2 when**:
- Temporal dynamics are needed
- Working with video data
- Modeling change over time is important
