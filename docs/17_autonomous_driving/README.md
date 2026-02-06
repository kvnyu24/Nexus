# Autonomous Driving

Comprehensive documentation for end-to-end autonomous driving systems, covering perception, prediction, and planning with unified transformer architectures.

## Overview

Autonomous driving systems integrate multiple perception, prediction, and planning tasks into unified frameworks that process multi-modal sensor inputs (cameras, LiDAR, HD maps) and generate safe, interpretable driving behaviors. Modern approaches leverage transformer architectures to jointly learn all driving tasks end-to-end, enabling better reasoning about complex driving scenarios and multi-agent interactions.

**Key Components:**
- **Multi-modal Perception**: 3D object detection, tracking, and HD map construction from cameras and LiDAR
- **Motion Forecasting**: Multi-modal trajectory prediction for all traffic agents considering scene context
- **Motion Planning**: Safe, comfortable ego trajectory generation considering predicted agent behaviors
- **Scene Understanding**: Unified reasoning over geometric and semantic scene representations
- **Sensor Fusion**: Integration of complementary sensor modalities for robust perception

**Modern Paradigms:**
1. **Query-Based Architecture**: Learnable queries represent agents, map elements, and trajectories
2. **Unified Multi-Task Learning**: Shared representations across perception, prediction, and planning
3. **Vectorized Representations**: Structured outputs (polylines, trajectories) instead of rasters
4. **End-to-End Differentiability**: Direct optimization from raw sensors to planning objectives
5. **Temporal Consistency**: Recurrent states for smooth online driving behavior

## Methods Covered

This section covers three state-of-the-art end-to-end autonomous driving frameworks:

### 1. UniAD (Unified Autonomous Driving)
**Planning-Oriented Autonomous Driving** - CVPR 2023 Best Paper

A unified framework that connects perception, prediction, and planning through query-based representations. UniAD introduces agent queries that persist across frames for tracking, are enhanced with motion forecasting, and guide collision-aware planning.

**Key Features:**
- Query-based unified architecture across all tasks
- BEV (Bird's Eye View) feature extraction from multi-camera inputs
- Agent queries with temporal propagation for tracking
- Motion forecasting decoder with multi-modal prediction
- Planning decoder with collision-aware trajectory scoring
- End-to-end trainable with task-specific losses

**Best For:**
- Research on unified perception-prediction-planning
- Systems requiring interpretable intermediate representations
- Applications with strong emphasis on planning safety
- Multi-camera setups without LiDAR

[Read Full Documentation →](uniad.md)

### 2. VAD (Vectorized Autonomous Driving)
**Vectorized Scene Representation** - ICCV 2023

VAD represents all driving scene elements as structured vectors (polylines for maps, bounding boxes for agents, waypoint sequences for trajectories), enabling efficient learning and interpretable outputs.

**Key Features:**
- Fully vectorized representations for all outputs
- Hierarchical queries (scene-level, agent-level, point-level)
- Polyline-based HD map construction
- Vector-based trajectory prediction and planning
- Differentiable vector matching for training
- Efficient structured output generation

**Best For:**
- Systems requiring interpretable, structured outputs
- Applications needing map-agent interaction modeling
- Scenarios with explicit geometric reasoning
- Integration with planning systems expecting vector inputs

[Read Full Documentation →](vad.md)

### 3. DriveTransformer
**Unified Transformer for Scalable Driving** - 2025

DriveTransformer presents a fully unified transformer that scales with model size and data. It tokenizes all sensor modalities into a common representation and uses task-specific decoders for all outputs.

**Key Features:**
- Unified transformer backbone for all tasks
- Multi-modal sensor tokenization (cameras + LiDAR + maps)
- Scalable architecture benefiting from pre-training
- Recurrent memory for temporal consistency
- Task-agnostic feature learning
- Support for imitation and reinforcement learning

**Best For:**
- Large-scale systems with abundant training data
- Multi-modal sensor fusion (camera + LiDAR)
- Applications requiring model scalability
- Research on foundation models for driving
- Transfer learning from vision/language pre-training

[Read Full Documentation →](drive_transformer.md)

## Comparison Table

| Aspect | UniAD | VAD | DriveTransformer |
|--------|-------|-----|------------------|
| **Architecture** | Query-based decoders per task | Vectorized decoders | Unified transformer + task heads |
| **Representation** | Feature queries + BEV | Vectorized (polylines, boxes) | Multi-modal tokens |
| **Sensor Support** | Multi-camera (+ optional map) | Multi-camera | Camera + LiDAR + Map |
| **Output Format** | Dense predictions | Structured vectors | Dense + structured |
| **Temporal Modeling** | Query propagation | Per-frame (can add memory) | Recurrent memory module |
| **Scalability** | Fixed architecture | Fixed architecture | Scales with model size |
| **Planning Method** | Collision-aware scoring | Goal-conditioned waypoints | Multi-modal trajectory search |
| **Training** | Multi-task losses | Hungarian matching | Multi-task + foundation model |
| **Interpretability** | Medium (query attention) | High (explicit vectors) | Lower (end-to-end features) |
| **Computational Cost** | Medium | Low-Medium | Medium-High (scales with size) |
| **Best Dataset** | nuScenes | nuScenes, Argoverse | Large-scale multi-modal datasets |
| **Key Innovation** | Planning-oriented design | Vectorized representations | Unified scalable architecture |

## Key Concepts

### End-to-End Learning

End-to-end autonomous driving learns a direct mapping from raw sensor inputs to driving actions, jointly optimizing all intermediate tasks.

**Advantages:**
- Joint optimization enables better information flow across tasks
- Eliminates hand-designed interfaces between modules
- Can learn implicit representations not captured by hand-crafted features
- Simplifies system architecture and deployment

### Query-Based Architecture

Modern autonomous driving systems use learnable queries as flexible representations for agents, maps, and trajectories. Queries attend to scene features and are refined through transformer decoder layers.

### Bird's Eye View (BEV) Representation

BEV provides a unified spatial reference frame with metric coordinates, natural for driving tasks and spatial reasoning.

### Multi-Task Learning

Joint learning of perception, prediction, and planning enables shared representations and task synergies.

## Quick Start

```python
from nexus.models.autonomous import UniAD, VAD, DriveTransformer

# Initialize a model
config = {
    'embed_dim': 256,
    'num_cameras': 6,
    'bev_height': 200,
    'bev_width': 200,
}

model = UniAD(config)
# or: model = VAD(config)
# or: model = DriveTransformer(config)

# Forward pass
images = get_camera_images()  # (B, 6, 3, H, W)
outputs = model(images)

# Extract predictions
detections = outputs['detections']
trajectories = outputs['trajectories']
ego_plan = outputs['ego_trajectories']
```

## Navigation

- **[UniAD Documentation](uniad.md)** - Planning-oriented unified framework
- **[VAD Documentation](vad.md)** - Vectorized scene representation  
- **[DriveTransformer Documentation](drive_transformer.md)** - Scalable unified transformer

## References

1. **UniAD**: Hu et al., "Planning-oriented Autonomous Driving", CVPR 2023
2. **VAD**: Zhou et al., "Vectorized Scene Representation for Efficient Autonomous Driving", ICCV 2023
3. **BEVFormer**: Li et al., "BEVFormer: Learning Bird's-Eye-View Representation", ECCV 2022
4. **nuScenes**: Caesar et al., "nuScenes: A Multimodal Dataset for Autonomous Driving", CVPR 2020
