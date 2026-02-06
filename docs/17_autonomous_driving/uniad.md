# UniAD: Unified Autonomous Driving

Planning-oriented end-to-end autonomous driving framework that unifies perception, prediction, and planning.

**Paper**: "Planning-oriented Autonomous Driving" (CVPR 2023 Best Paper)  
**Authors**: Yihan Hu et al., OpenDriveLab  
**Code**: https://github.com/OpenDriveLab/UniAD  
**Implementation**: `/nexus/models/autonomous/uniad.py`

---

## 1. Overview & Motivation

### The Problem

Traditional autonomous driving systems use modular pipelines where perception, prediction, and planning are optimized independently:

**Information Loss**: Hand-crafted interfaces between modules discard rich contextual information.

**Error Accumulation**: Mistakes in perception propagate through prediction to planning without correction.

**Suboptimal Planning**: Planning cannot provide feedback to perception about safety-critical elements.

**Limited Interaction Modeling**: Independent modules struggle to reason about agent-map-ego interactions.

### The UniAD Solution

UniAD proposes a **planning-oriented** unified framework with:

1. **Connected Tasks**: All tasks linked through learnable query representations
2. **Information Flow**: Planning gradients guide perception to focus on relevant features  
3. **Interaction Modeling**: Explicit reasoning about agents, maps, and ego planning
4. **Temporal Consistency**: Query propagation maintains smooth tracking and planning

**Key Insight**: Planning is the ultimate goal. By making the system planning-oriented, perception and prediction learn features most relevant for safe driving.

### Why It Matters

UniAD won CVPR 2023 Best Paper Award for:
- State-of-the-art performance across all driving tasks on nuScenes
- First end-to-end system competitive with specialized modular approaches
- Unified architecture simpler than maintaining separate modules
- Interpretable intermediate representations for debugging

### Architecture Overview

```
Multi-Camera Images → BEV Encoder → BEV Features
                                         ↓
                              ┌──────────┴──────────┐
                              ↓                     ↓
                        Agent Queries ←──→ Map Queries
                              ↓
                        Tracking Decoder
                              ↓
                     (Detection + Tracking)
                              ↓
                     Motion Forecasting Decoder
                              ↓
                    (Multi-Modal Trajectories)
                              ↓
                        Planning Decoder
                              ↓
                    (Ego Trajectory + Costs)
```

---

## 2. Theoretical Background

### Query-Based Architecture

UniAD extends DETR (DEtection TRansformer) to autonomous driving.

**DETR Foundation**:
- Learnable object queries attend to image features
- Each query specializes in detecting certain object types/locations
- Queries refined through decoder layers
- Final predictions: bounding boxes + classes

**UniAD Extension**:
- **Agent queries**: Persistent across frames for tracking
- **Map queries**: Detect and localize map elements  
- **Planning queries**: Generate candidate ego trajectories
- **Temporal propagation**: Queries maintain state for online inference

### Bird's Eye View (BEV) Representation

BEV provides unified spatial reference for driving.

**Why BEV?**
- Natural top-down view (like maps)
- Metric space: 1 pixel = 0.5m (typical)
- Facilitates geometric reasoning
- Aligns camera and map coordinate systems

**BEV Construction**:
1. Extract multi-camera image features
2. Estimate depth distribution per pixel
3. Project features to 3D frustum
4. Pool into BEV grid via learned attention

### Multi-Task Learning Framework

UniAD jointly optimizes related tasks in a hierarchy:

```
Perception (Detection, Tracking)
    ↓
Prediction (Motion Forecasting)
    ↓
Planning (Ego Trajectory Generation)
```

**Benefits**:
- Shared BEV features improve sample efficiency
- Planning gradients guide perception
- Motion prediction benefits from tracking consistency
- End-to-end optimization of driving objective

### Temporal Modeling

**Query Propagation**:
- Agent queries from frame t → frame t+1
- Maintains object identity for tracking
- Accumulates temporal context
- Enables smooth, stable predictions

---

## 3. Mathematical Formulation

### BEV Feature Extraction

Given multi-camera images $\{I_c\}_{c=1}^{N_{cam}}$, extract BEV features:

$$F_{img}^c = \text{CNN}(I_c) \in \mathbb{R}^{H' \times W' \times C}$$

$$D^c = \text{DepthNet}(F_{img}^c) \in \mathbb{R}^{H' \times W' \times D}$$

$$F_{BEV} = \text{BEVEncoder}(\{F_{img}^c, D^c\}_{c=1}^{N_{cam}})$$

### Tracking with Agent Queries

Initialize agent queries $Q_{agent}^0 \in \mathbb{R}^{N_q \times C}$ (learnable embeddings).

**Temporal Propagation**:
$$Q_{agent}^t = \text{Propagate}(Q_{agent}^{t-1}, \Delta t)$$

**Tracking Decoder**:
$$Q_{agent}^{(l)} = \text{DecoderLayer}(Q_{agent}^{(l-1)}, F_{BEV})$$

**Detection Heads**:
$$B = \text{BBoxHead}(Q_{agent}) \in \mathbb{R}^{N_q \times 10}$$
$$C = \text{ClassHead}(Q_{agent}) \in \mathbb{R}^{N_q \times N_{cls}}$$

**Detection Loss**:
$$\mathcal{L}_{det} = \mathcal{L}_{cls}(C, C^*) + \lambda_1 \mathcal{L}_{l1}(B, B^*) + \lambda_2 \mathcal{L}_{giou}(B, B^*)$$

### Motion Forecasting

Predict multi-modal future trajectories.

**Motion Decoder**:
$$Q_{motion} = \text{MotionDecoder}(Q_{agent}, F_{BEV}, F_{map})$$

**Multi-Modal Prediction**:
$$T = \text{TrajHead}(Q_{motion}) \in \mathbb{R}^{N_q \times K \times T \times 2}$$
$$P = \text{softmax}(\text{ConfHead}(Q_{motion})) \in \mathbb{R}^{N_q \times K}$$

Where K = modes (typically 6), T = future steps (typically 12).

**Winner-Takes-All Loss**:
$$k^* = \arg\min_k \|T_{:,k,:,:} - T^*\|_2$$
$$\mathcal{L}_{motion} = \text{SmoothL1}(T_{:,k^*,:,:}, T^*)$$

### Planning Decoder

Generate ego vehicle trajectory.

**Planning Queries**: $Q_{plan} \in \mathbb{R}^{N_m \times C}$ (typically $N_m = 3$ modes)

**Context Aggregation**:
$$F_{context} = \text{Concat}[Q_{agent}, F_{BEV}, F_{map}]$$

**Planning Decoder**:
$$Q_{plan}^{(l)} = \text{DecoderLayer}(Q_{plan}^{(l-1)}, F_{context})$$

**Ego Trajectory**:
$$\tau = \text{TrajHead}(Q_{plan}) \in \mathbb{R}^{N_m \times T_{plan} \times 3}$$

**Cost Head**:
$$c = \text{CostHead}(Q_{plan}) \in \mathbb{R}^{N_m}$$

**Planning Loss**:
$$m^* = \arg\min_m \|\tau_m - \tau^*\|_2$$
$$\mathcal{L}_{plan} = \text{SmoothL1}(\tau_{m^*}, \tau^*)$$

### Total Loss

$$\mathcal{L}_{total} = \lambda_{det} \mathcal{L}_{det} + \lambda_{mot} \mathcal{L}_{motion} + \lambda_{plan} \mathcal{L}_{plan}$$

Typical weights: $\lambda_{det} = 2.0, \lambda_{mot} = 1.0, \lambda_{plan} = 1.0$

---

## 4. High-Level Intuition

### The Core Idea

Think of UniAD as a **reasoning system** using queries as "thoughts":

1. **Agent Queries** = "Where are the other cars? What are they doing?"
2. **Map Queries** = "What lanes exist? Where are boundaries?"
3. **Planning Queries** = "What trajectories can I take? Which is safest?"

Queries start as random "hunches" and are refined by attending to visual evidence (BEV features).

### Why Queries Work

**Traditional Approach**:
```
Images → CNN → NMS → Track → Predict → Plan
         ↓ (fixed boxes, errors accumulate)
```

**Query Approach**:
```
Images → BEV Features ← Queries (bidirectional attention)
                        ↓
                  Refined Queries → Predictions
```

Queries "ask questions" of visual features and get refined answers through attention.

### Planning-Oriented Design

**Principle**: Make planning the primary objective, let perception/prediction learn what's needed.

**Example**:
- Traditional: Detect all objects equally
- UniAD: Focus on objects affecting planning (nearby agents, traffic lights)

This happens automatically through backpropagation from planning loss.

### Temporal Consistency

**Problem**: Independent per-frame processing causes flickering.

**Solution**: Propagate queries across frames.

Frame t: Agent Query 1 → Detects car at (x=10, y=5)  
Frame t+1: Agent Query 1 (propagated) → Detects same car at (x=10.5, y=5.1)

Query maintains "memory" of the car for stable tracking.

---

## 5. Implementation Details

### Network Architecture

**BEV Encoder**:
- Backbone: ResNet-50/101 for image features
- FPN: Feature Pyramid Network for multi-scale
- Depth Net: 64 bins, 0-60m range
- BEV Pooling: Deformable attention
- Output: (200×200×256) BEV feature map (100m×100m coverage)

**Tracking Decoder**:
- Queries: 300 agent queries, 256-dim
- Layers: 3 transformer decoder layers
- Heads: 8 attention heads
- FFN: 4× expansion (256→1024→256)

**Motion Forecasting Decoder**:
- Input: Agent queries + BEV + map features
- Layers: 3 transformer decoder layers
- Output: 6 modes × 12 steps × 2D coordinates

**Planning Decoder**:
- Queries: 3 planning mode queries
- Layers: 3 transformer decoder layers
- Output: 3 modes × 6 waypoints × (x, y, θ)

### Training Procedure

**Dataset**: nuScenes (1000 scenes, 40K samples)

**Training Stages**:

1. **Stage 1: Perception Pre-training** (20 epochs)
   - Train BEV encoder + tracking decoder only
   - Learning rate: 2e-4

2. **Stage 2: Full Multi-Task** (40 epochs)
   - Unfreeze all components
   - Learning rate: 2e-4 → 2e-5 (cosine decay)

**Hyperparameters**:
- Optimizer: AdamW
- Weight decay: 0.01
- Batch size: 8 (4 GPUs, 2/GPU)
- Gradient clipping: Max norm 35.0
- Input resolution: 640×1600 per camera

**Data Augmentation**:
- Random flip (horizontal only)
- Random scale: [0.9, 1.1]
- Random rotation: [-5°, 5°]
- Color jitter

### Inference Details

**Online Inference**:

```python
model.eval()
prev_agent_queries = None

for frame in driving_sequence:
    images = get_camera_images(frame)

    with torch.no_grad():
        outputs = model(images, prev_agent_queries=prev_agent_queries)

    # Extract results
    ego_plan = outputs['ego_trajectories']
    costs = outputs['planning_costs']

    # Select best mode
    best_mode = costs.argmin()
    selected_plan = ego_plan[0, best_mode]

    # Execute
    execute_control(selected_plan[0])

    # Update state
    prev_agent_queries = outputs['agent_queries']
```

**Latency** (NVIDIA A100):
- BEV Encoding: 35ms
- Tracking: 15ms
- Motion Forecasting: 20ms
- Planning: 10ms
- **Total**: ~80ms (12.5 FPS)

---

## 6. Code Walkthrough

### Model Initialization

```python
from nexus.models.autonomous import UniAD

config = {
    'num_cameras': 6,
    'bev_height': 200,
    'bev_width': 200,
    'embed_dim': 256,
    'num_agent_queries': 300,
    'num_modes': 6,
    'future_steps': 12,
    'planning_modes': 3,
    'planning_steps': 6,
}

model = UniAD(config)
```

### Forward Pass

```python
# Inputs
images = torch.randn(2, 6, 3, 640, 1600).cuda()
prev_agent_queries = None

# Forward
outputs = model(images, prev_agent_queries=prev_agent_queries)

# Outputs
print(outputs.keys())
# ['bev_features', 'detections', 'classes', 'track_embeds',
#  'trajectories', 'traj_confidences', 'ego_trajectories',
#  'planning_costs', 'agent_queries']
```

### Loss Computation

```python
targets = {
    'gt_boxes': torch.randn(2, 300, 10).cuda(),
    'gt_classes': torch.randint(0, 10, (2, 300)).cuda(),
    'gt_future_trajs': torch.randn(2, 300, 12, 2).cuda(),
    'gt_ego_traj': torch.randn(2, 6, 3).cuda(),
}

losses = model.compute_loss(outputs, targets)
loss = losses['total_loss']
loss.backward()
```

### BEV Encoder

From `/nexus/models/autonomous/uniad.py`:

```python
class BEVEncoder(NexusModule):
    def __init__(self, config):
        self.image_encoder = nn.Sequential(...)
        self.depth_net = nn.Sequential(...)
        self.bev_queries = nn.Parameter(...)
        self.bev_layers = nn.ModuleList([...])

    def forward(self, images, camera_params=None):
        # Extract image features
        img_feats = self.image_encoder(images)
        depth = self.depth_net(img_feats)

        # Initialize BEV queries
        bev_feats = self.bev_queries.expand(batch_size, -1, -1)

        # Cross-attention to images
        for layer in self.bev_layers:
            bev_feats = layer(bev_feats, img_feats)

        return bev_feats
```

---

## 7. Optimization Tricks

### 1. Multi-Task Loss Balancing

Use uncertainty-based adaptive weighting:

```python
class UncertaintyWeighting(nn.Module):
    def __init__(self, num_tasks=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        weighted = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted.append(precision * loss + self.log_vars[i])
        return sum(weighted)
```

### 2. Query Initialization

Pre-train queries with detection task before full training.

### 3. Query Propagation Strategies

**Simple**: $Q_t = Q_{t-1}$  
**Velocity-Based**: $Q_t = Q_{t-1} + v_{t-1} \cdot dt$  
**Learned**: $Q_t = \text{MLP}([Q_{t-1}, v_{t-1}, dt])$

### 4. Efficient Attention

Use deformable attention: $O(NK)$ instead of $O(N^2)$ where $K=8$ sample points.

### 5. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        outputs = model(images)
        loss = compute_loss(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

Benefit: 2× faster training, 50% less memory.

---

## 8. Experiments & Results

### nuScenes Benchmark

**Detection (NDS)**:
| Method | NDS ↑ | mAP ↑ |
|--------|-------|-------|
| DETR3D | 0.479 | 0.412 |
| BEVFormer | 0.517 | 0.416 |
| **UniAD** | **0.533** | **0.434** |

**Tracking (AMOTA)**:
| Method | AMOTA ↑ |
|--------|---------|
| QD-3DT | 0.217 |
| PF-Track | 0.298 |
| **UniAD** | **0.359** |

**Motion Forecasting (minADE/minFDE in meters)**:
| Method | minADE ↓ | minFDE ↓ |
|--------|----------|----------|
| MultiPath | 1.82 | 3.91 |
| **UniAD** | **1.31** | **2.85** |

**Planning (L2 Error in meters)**:
| Method | 1s ↓ | 2s ↓ | 3s ↓ |
|--------|------|------|------|
| ST-P3 | 0.68 | 1.31 | 2.09 |
| **UniAD** | **0.30** | **0.71** | **1.22** |

### Ablation Studies

**Impact of Planning-Oriented Training**:
| Config | Planning L2 ↓ | Detection NDS ↑ |
|--------|---------------|-----------------|
| Det Only | - | 0.521 |
| Det + Motion + Plan | **0.74** | **0.533** |

Planning gradients improve detection by +1.2% NDS.

**Impact of Temporal Propagation**:
| Method | AMOTA ↑ | Flickering ↓ |
|--------|---------|--------------|
| Independent | 0.312 | 0.43 |
| Propagation | **0.359** | **0.18** |

---

## 9. Common Pitfalls

### 1. Incorrect Query Propagation

**Wrong**:
```python
for frame in sequence:
    outputs = model(images, prev_agent_queries=None)  # ❌
```

**Correct**:
```python
prev_queries = None
for frame in sequence:
    outputs = model(images, prev_agent_queries=prev_queries)
    prev_queries = outputs['agent_queries']  # ✅
```

### 2. Loss Weight Imbalance

Monitor per-task losses, adjust if one dominates.

### 3. Insufficient BEV Resolution

Use at least 200×200 for 100m×100m (0.5m/pixel) to detect small objects.

### 4. Training Instability

Solutions:
- Gradient clipping: max_norm=35.0
- Lower learning rate: 1e-4 instead of 2e-4
- Warmup learning rate

### 5. Ignoring Camera Calibration

Pass camera parameters for accurate BEV projection.

### 6. Overfitting to Benchmark

Use strong augmentation and train on multiple datasets.

---

## 10. References

### Primary Paper

1. **UniAD**: Yihan Hu et al., "Planning-oriented Autonomous Driving", CVPR 2023 (Best Paper)
   - arXiv: https://arxiv.org/abs/2212.10156
   - Code: https://github.com/OpenDriveLab/UniAD

### Foundational Works

2. **DETR**: Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020
3. **DETR3D**: Wang et al., "DETR3D: 3D Object Detection from Multi-view Images", CoRL 2021
4. **BEVFormer**: Li et al., "BEVFormer: Learning Bird's-Eye-View Representation", ECCV 2022

### Related Work

5. **ST-P3**: Hu et al., "ST-P3: End-to-end Vision-based Autonomous Driving", ECCV 2022
6. **FIERY**: Hu et al., "FIERY: Future Instance Prediction in Bird's-Eye View", ICCV 2021

### Benchmarks

7. **nuScenes**: Caesar et al., "nuScenes: A Multimodal Dataset for Autonomous Driving", CVPR 2020
8. **Waymo**: Sun et al., "Scalability in Perception for Autonomous Driving", CVPR 2020

### Implementation

9. **Official UniAD**: https://github.com/OpenDriveLab/UniAD
10. **Nexus Implementation**: `/nexus/models/autonomous/uniad.py`
