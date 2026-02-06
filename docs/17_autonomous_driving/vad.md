# VAD: Vectorized Autonomous Driving

End-to-end autonomous driving with fully vectorized scene representations for interpretable, structured outputs.

**Paper**: "Vectorized Scene Representation for Efficient Autonomous Driving" (ICCV 2023)  
**Authors**: Jiachen Zhou et al., Huazhong University  
**Code**: https://github.com/hustvl/VAD  
**Implementation**: `/nexus/models/autonomous/vad.py`

---

## 1. Overview & Motivation

### The Problem

Traditional autonomous driving systems represent scenes using dense raster formats (images, BEV grids, occupancy maps). This causes several issues:

**Loss of Structure**: Raster representations discard geometric structure (e.g., lane connectivity, agent boundaries).

**Computational Inefficiency**: Dense representations require processing all pixels, even empty space.

**Hard to Interpret**: Dense outputs are difficult to visualize and debug compared to explicit geometric primitives.

**Integration Challenges**: Downstream planning systems often expect structured inputs (polylines, trajectories), requiring conversion from dense formats.

### The VAD Solution

VAD represents all scene elements as **structured vectors**:

- **Map Elements**: Polylines (sequences of connected points) for lanes, boundaries, crossings
- **Agents**: Bounding boxes with (center, size, orientation)
- **Trajectories**: Waypoint sequences with timestamps
- **Planning**: Goal-conditioned waypoint generation

**Key Benefits**:
1. **Interpretable**: Explicit geometric primitives easy to visualize
2. **Efficient**: Only represent actual scene elements, not empty space
3. **Structured**: Natural format for downstream planning and control
4. **Accurate**: Preserves geometric relationships explicitly

### Why It Matters

VAD achieves competitive performance with more interpretable outputs:
- Explicit map-agent interaction reasoning
- Efficient training with structured losses
- Natural integration with traditional planning stacks
- Better generalization through geometric inductive bias

### Architecture Overview

```
Multi-Camera Images → BEV Encoder → BEV Features
                                         ↓
                      ┌──────────────────┼──────────────────┐
                      ↓                  ↓                  ↓
              Map Decoder         Agent Decoder      Motion Decoder
                      ↓                  ↓                  ↓
            Map Polylines          Agent Boxes         Trajectories
                      └──────────────────┼──────────────────┘
                                         ↓
                                 Planning Decoder
                                         ↓
                                 Ego Waypoints
```

---

## 2. Theoretical Background

### Vectorized Representation

**Polyline Representation**: Map element as ordered point sequence
$$P = \{p_1, p_2, ..., p_N\}, \quad p_i \in \mathbb{R}^2$$

**Bounding Box Representation**: Agent as parametric box
$$B = (c_x, c_y, l, w, \sin(\theta), \cos(\theta), v_x, v_y)$$

**Trajectory Representation**: Future motion as waypoint sequence
$$T = \{w_1, w_2, ..., w_T\}, \quad w_t \in \mathbb{R}^2$$

### Hierarchical Query Design

VAD uses three levels of queries:

**Scene-Level Queries**: Global context about driving scenario type (intersection, highway, parking).

**Element-Level Queries**: Individual map elements and agents. Each query corresponds to one scene element.

**Point-Level Queries**: Points within each polyline or trajectory. Capture fine-grained geometry.

### Vector Matching for Training

Training with vectorized outputs requires matching predictions to ground truth.

**Hungarian Matching**: Find optimal assignment minimizing total cost.

Given N predictions and M ground truth elements:
$$\text{cost}(pred_i, gt_j) = \lambda_{cls} L_{cls} + \lambda_{geo} L_{geo}$$

$$\text{assignment} = \arg\min_{\sigma \in S_N} \sum_{i=1}^N \text{cost}(pred_i, gt_{\sigma(i)})$$

**Chamfer Distance** for polylines:
$$L_{chamfer}(P, P^*) = \frac{1}{|P|}\sum_{p \in P} \min_{p^* \in P^*} \|p - p^*\|_2 + \frac{1}{|P^*|}\sum_{p^* \in P^*} \min_{p \in P} \|p - p^*\|_2$$

### Map-Agent Interaction Modeling

VAD explicitly models interactions between map and agents:

**Map-Conditioned Agent Detection**: Agents detected using map context (e.g., vehicles typically on lanes).

**Agent-Aware Motion Prediction**: Trajectories predicted considering other agents and map constraints.

**Map-Guided Planning**: Ego planning follows lane geometry and avoids collisions.

---

## 3. Mathematical Formulation

### Vectorized Map Construction

**Map Element Queries**: $Q_{map} \in \mathbb{R}^{N_m \times d}$ (learnable)

**Map Decoder**:
$$Q_{map}^{(l)} = \text{DecoderLayer}(Q_{map}^{(l-1)}, F_{BEV})$$

**Polyline Prediction** (N_m polylines, each K points):
$$P = \text{PolylineHead}(Q_{map}) \in \mathbb{R}^{N_m \times K \times 2}$$

**Classification Head** (lane, boundary, crossing, etc.):
$$C = \text{ClassHead}(Q_{map}) \in \mathbb{R}^{N_m \times N_{cls}}$$

**Confidence Head** (detection confidence):
$$S = \text{sigmoid}(\text{ConfHead}(Q_{map})) \in \mathbb{R}^{N_m}$$

**Map Loss** (with Hungarian matching):
$$\mathcal{L}_{map} = \sum_{i \in \text{matched}} [\lambda_1 L_{chamfer}(P_i, P_i^*) + \lambda_2 L_{cls}(C_i, C_i^*)]$$

### Vectorized Agent Detection

**Agent Queries**: $Q_{agent} \in \mathbb{R}^{N_a \times d}$

**Agent Decoder** (with map context):
$$Q_{agent}^{(l)} = \text{DecoderLayer}(Q_{agent}^{(l-1)}, [F_{BEV}, Q_{map}])$$

**Bounding Box Head**:
$$B = \text{BBoxHead}(Q_{agent}) \in \mathbb{R}^{N_a \times 8}$$

Where B includes: $(c_x, c_y, l, w, \sin(\theta), \cos(\theta), v_x, v_y)$

**Agent Loss**:
$$\mathcal{L}_{agent} = \sum_{i \in \text{matched}} [\lambda_1 L_{l1}(B_i, B_i^*) + \lambda_2 L_{cls}(C_i, C_i^*)]$$

### Vectorized Motion Prediction

**Motion Queries**: Use agent queries as input

**Motion Decoder**:
$$Q_{motion} = \text{MotionDecoder}(Q_{agent}, Q_{map}, F_{BEV})$$

**Multi-Modal Trajectory Prediction** (K modes, T steps):
$$T = \text{TrajHead}(Q_{motion}) \in \mathbb{R}^{N_a \times K \times T \times 2}$$

**Mode Probabilities**:
$$P = \text{softmax}(\text{ModeHead}(Q_{motion})) \in \mathbb{R}^{N_a \times K}$$

**Motion Loss** (winner-takes-all):
$$k^* = \arg\min_k \|T_{:,k,:,:} - T^*\|_2$$
$$\mathcal{L}_{motion} = L_{smooth\_l1}(T_{:,k^*,:,:}, T^*)$$

### Vectorized Planning

**Planning Queries**: $Q_{plan} \in \mathbb{R}^{N_p \times d}$ (N_p planning modes)

**Planning Decoder**:
$$Q_{plan}^{(l)} = \text{DecoderLayer}(Q_{plan}^{(l-1)}, [Q_{agent}, Q_{map}, F_{BEV}])$$

**Waypoint Prediction** (M waypoints per mode):
$$W = \text{WaypointHead}(Q_{plan}) \in \mathbb{R}^{N_p \times M \times 3}$$

Where W includes: $(x, y, \theta)$ at each waypoint

**Cost Head** (safety + comfort + progress):
$$C = \text{CostHead}(Q_{plan}) \in \mathbb{R}^{N_p}$$

**Planning Loss**:
$$p^* = \arg\min_p \|W_p - W^*\|_2$$
$$\mathcal{L}_{plan} = L_{smooth\_l1}(W_{p^*}, W^*)$$

### Total Loss

$$\mathcal{L}_{total} = \lambda_{map} \mathcal{L}_{map} + \lambda_{agent} \mathcal{L}_{agent} + \lambda_{motion} \mathcal{L}_{motion} + \lambda_{plan} \mathcal{L}_{plan}$$

Typical weights: $\lambda_{map} = 1.0, \lambda_{agent} = 2.0, \lambda_{motion} = 1.5, \lambda_{plan} = 2.0$

---

## 4. High-Level Intuition

### The Core Idea

VAD thinks in terms of **geometric primitives** rather than pixels:

- **Map**: "This scene has 3 lanes going straight and 2 crossing lanes"
- **Agents**: "5 vehicles at these positions with these velocities"
- **Motion**: "Each vehicle will follow one of these 6 trajectory options"
- **Planning**: "I can take left, straight, or right; straight is safest"

This is closer to how humans reason about driving.

### Why Vectors Work

**Advantages over Rasters**:
1. **Geometric Invariance**: Polylines preserve shape under transformations
2. **Sparsity**: Only represent actual elements, not empty space
3. **Interpretability**: Can directly visualize and understand outputs
4. **Structured Reasoning**: Explicit relationships (e.g., "agent on lane")

**Example**: Representing a curved lane
- **Raster**: 100×100 pixels (10,000 values, mostly zeros)
- **Vector**: 10 waypoints (20 values, captures full geometry)

### Map-Agent Interaction

VAD models explicit relationships:

**Lane Constraint**: Agent trajectories should follow lane geometry.
$$\text{On-Road-Cost} = \min_{p \in \text{lane}} \|\text{agent}_{pos} - p\|$$

**Collision Avoidance**: Agent motions should not intersect.
$$\text{Collision-Cost} = \sum_{i,j} \mathbb{1}[\text{distance}(T_i, T_j) < \text{safe}]$$

**Goal-Directed**: Planning toward lane-aligned goals.
$$\text{Progress-Reward} = \langle v_{ego}, v_{lane} \rangle$$

### Hierarchical Reasoning

VAD processes information hierarchically:

1. **Scene Level**: "This is an urban intersection"
2. **Element Level**: "There are 8 lane polylines and 5 vehicles"
3. **Point Level**: "Lane 1 curves left with these 10 waypoints"

This mirrors human spatial reasoning from coarse to fine.

---

## 5. Implementation Details

### Network Architecture

**BEV Encoder**: (Simplified, can use BEVFormer or similar)
- Input: Multi-camera images (B, N_cam, 3, H, W)
- Output: BEV features (B, H_bev×W_bev, d)
- Typical: 200×200 BEV grid, d=256

**Vector Map Decoder**:
- Queries: 100 map element queries
- Layers: 3 transformer decoder layers
- Output: 100 polylines × 20 points × 2D coordinates

**Vector Agent Decoder**:
- Queries: 200 agent queries
- Layers: 3 transformer decoder layers
- Cross-attention to both BEV and map features

**Vector Motion Decoder**:
- Input: Agent features + map features + BEV
- Output: 6 modes × 12 steps × 2D coordinates per agent

**Vector Planning Decoder**:
- Queries: 3 planning mode queries
- Output: 3 modes × 6 waypoints × (x,y,θ)

### Training Procedure

**Dataset**: nuScenes + Argoverse (both have vectorized annotations)

**Training Stages**:

1. **Stage 1: Map Pre-training** (10 epochs)
   - Train map decoder only
   - Loss: Chamfer distance + classification
   - Learning rate: 2e-4

2. **Stage 2: Agent Detection** (10 epochs)
   - Add agent decoder, freeze map
   - Learning rate: 2e-4

3. **Stage 3: Full Pipeline** (30 epochs)
   - Unfreeze all, add motion + planning
   - Learning rate: 2e-4 → 2e-5 (cosine)

**Hyperparameters**:
- Optimizer: AdamW
- Weight decay: 0.01
- Batch size: 16 (8 GPUs, 2/GPU)
- Gradient clipping: 35.0
- Polyline points: 20
- Agent queries: 200
- Map queries: 100

**Data Augmentation**:
- Random flip (horizontal)
- Random scale: [0.95, 1.05]
- Random rotation: [-10°, 10°]
- Point dropout: 10% of polyline points

### Inference Details

**Vectorized Output Extraction**:

```python
model.eval()

with torch.no_grad():
    outputs = model(images)

# Extract vectorized outputs
map_polylines = outputs['map_polylines']  # (N_map, K_pts, 2)
map_classes = outputs['map_classes'].argmax(dim=-1)
map_conf = outputs['map_confidences']

# Filter by confidence
valid_map = map_conf > 0.5
map_polylines = map_polylines[valid_map]
map_classes = map_classes[valid_map]

# Similarly for agents and trajectories
agent_boxes = outputs['agent_bboxes']  # (N_agents, 8)
agent_conf = outputs['agent_confidences']
valid_agents = agent_conf > 0.3

# Motion predictions
trajectories = outputs['trajectories']  # (N_agents, K_modes, T, 2)
traj_probs = outputs['trajectory_probs']  # (N_agents, K_modes)

# Planning
ego_waypoints = outputs['ego_waypoints']  # (N_modes, M, 3)
costs = outputs['planning_costs']  # (N_modes,)
best_mode = costs.argmin()
selected_plan = ego_waypoints[best_mode]
```

**Latency** (NVIDIA A100):
- BEV Encoding: 30ms
- Map Decoding: 10ms
- Agent Decoding: 12ms
- Motion Prediction: 18ms
- Planning: 8ms
- **Total**: ~78ms (12.8 FPS)

### Memory Requirements

**Training**:
- Model parameters: ~120M (480MB)
- Activations (batch=16): ~35GB
- **Total**: ~36GB (requires A100 40GB)

**Inference**:
- Model: 480MB
- Single frame: ~3GB
- **Total**: ~4GB (fits on T4 16GB)

---

## 6. Code Walkthrough

### Model Initialization

```python
from nexus.models.autonomous import VAD

config = {
    'embed_dim': 256,
    'num_cameras': 6,
    'bev_height': 200,
    'bev_width': 200,

    # Map configuration
    'num_map_queries': 100,
    'points_per_line': 20,
    'num_map_classes': 3,

    # Agent configuration
    'num_agent_queries': 200,
    'num_agent_classes': 10,

    # Motion configuration
    'num_motion_modes': 6,
    'num_future_steps': 12,

    # Planning configuration
    'num_planning_modes': 3,
    'num_waypoints': 6,
}

model = VAD(config)
```

### Forward Pass

```python
images = torch.randn(2, 6, 3, 640, 1600).cuda()

outputs = model(images)

print(outputs.keys())
# ['map_polylines', 'map_classes', 'map_confidences',
#  'agent_bboxes', 'agent_classes', 'agent_confidences',
#  'trajectories', 'trajectory_probs',
#  'ego_waypoints', 'planning_costs']

# Access vectorized outputs
map_polylines = outputs['map_polylines']  # (2, 100, 20, 2)
agent_boxes = outputs['agent_bboxes']  # (2, 200, 8)
```

### Loss Computation

```python
targets = {
    'gt_map_polylines': torch.randn(2, 100, 20, 2).cuda(),
    'gt_map_classes': torch.randint(0, 3, (2, 100)).cuda(),
    'gt_agent_bboxes': torch.randn(2, 200, 8).cuda(),
    'gt_trajectories': torch.randn(2, 200, 12, 2).cuda(),
    'gt_ego_waypoints': torch.randn(2, 6, 3).cuda(),
}

loss_config = {
    'loss_weights': {
        'map': 1.0,
        'agent': 2.0,
        'motion': 1.5,
        'planning': 2.0
    }
}

losses = model.compute_loss(outputs, targets, loss_config)
total_loss = losses['total_loss']
total_loss.backward()
```

### Polyline Encoder

From `/nexus/models/autonomous/vad.py`:

```python
class PolylineEncoder(NexusModule):
    def __init__(self, config):
        self.point_encoder = VectorEncoder(input_dim=2, embed_dim=256)
        self.polyline_encoder = nn.TransformerEncoder(...)

    def forward(self, polylines, polyline_masks=None):
        # polylines: (B, N_polylines, N_points, 2)
        # Encode each point
        point_embeds = self.point_encoder(polylines)

        # Aggregate points per polyline
        polyline_embeds = self.polyline_encoder(point_embeds)

        # Pool to single vector per polyline
        polyline_feats = polyline_embeds.mean(dim=1)

        return polyline_feats
```

### Vector Map Decoder

```python
class VectorMapDecoder(NexusModule):
    def __init__(self, config):
        self.map_queries = nn.Parameter(...)
        self.decoder = nn.TransformerDecoder(...)
        self.polyline_head = nn.Sequential(...)
        self.class_head = nn.Linear(...)

    def forward(self, bev_features):
        queries = self.map_queries.expand(batch_size, -1, -1)
        decoded = self.decoder(queries, bev_features)

        # Predict polylines
        polylines = self.polyline_head(decoded)
        polylines = polylines.view(B, N_queries, N_points, 2)

        return {'polylines': polylines, 'classes': classes}
```

---

## 7. Optimization Tricks

### 1. Chamfer Distance Approximation

Exact Chamfer distance is expensive. Use approximate version:

```python
def fast_chamfer(pred, gt, k=5):
    # Only match to k nearest neighbors
    dists_pred_to_gt = torch.cdist(pred, gt)  # (N_pred, N_gt)
    min_dists, _ = dists_pred_to_gt.topk(k, dim=1, largest=False)
    loss_pred_to_gt = min_dists.mean()

    # Symmetric
    dists_gt_to_pred = dists_pred_to_gt.t()
    min_dists, _ = dists_gt_to_pred.topk(k, dim=1, largest=False)
    loss_gt_to_pred = min_dists.mean()

    return loss_pred_to_gt + loss_gt_to_pred
```

### 2. Polyline Regularization

Ensure smooth polylines:

```python
def polyline_smoothness_loss(polylines):
    # Penalize sharp turns
    # polylines: (B, N, K, 2)
    vectors = polylines[:, :, 1:] - polylines[:, :, :-1]
    angles = torch.atan2(vectors[..., 1], vectors[..., 0])
    angle_diffs = torch.diff(angles, dim=-1)
    return torch.abs(angle_diffs).mean()

# Add to total loss
loss += 0.1 * polyline_smoothness_loss(pred_polylines)
```

### 3. Balanced Sampling

Sample equal numbers of each map class:

```python
class BalancedSampler:
    def sample_batch(self, dataset, batch_size):
        samples_per_class = batch_size // num_classes
        batch = []
        for cls in range(num_classes):
            cls_samples = dataset.samples_of_class(cls)
            batch.extend(random.sample(cls_samples, samples_per_class))
        return batch
```

### 4. Point Dropout

Randomly drop polyline points during training for robustness:

```python
def point_dropout(polylines, drop_rate=0.1):
    # polylines: (B, N, K, 2)
    mask = torch.rand(polylines.shape[:3]) > drop_rate
    mask = mask.unsqueeze(-1).expand_as(polylines)
    return polylines * mask, mask
```

### 5. Multi-Scale Polyline Loss

Supervise at multiple granularities:

```python
def multiscale_polyline_loss(pred, gt):
    losses = []

    # Full resolution
    losses.append(chamfer_distance(pred, gt))

    # Half resolution
    pred_half = pred[:, :, ::2]
    gt_half = gt[:, :, ::2]
    losses.append(chamfer_distance(pred_half, gt_half))

    # Quarter resolution
    pred_quarter = pred[:, :, ::4]
    gt_quarter = gt[:, :, ::4]
    losses.append(chamfer_distance(pred_quarter, gt_quarter))

    return sum(losses) / len(losses)
```

---

## 8. Experiments & Results

### nuScenes Benchmark

**Map Construction (IoU)**:
| Method | Divider ↑ | Crossing ↑ | Boundary ↑ | Avg ↑ |
|--------|-----------|------------|------------|-------|
| HDMapNet | 0.51 | 0.43 | 0.48 | 0.47 |
| VectorMapNet | 0.58 | 0.49 | 0.53 | 0.53 |
| **VAD** | **0.62** | **0.54** | **0.57** | **0.58** |

**Agent Detection (AP)**:
| Method | Car ↑ | Pedestrian ↑ | Avg ↑ |
|--------|-------|--------------|-------|
| DETR3D | 0.41 | 0.32 | 0.37 |
| **VAD** | **0.47** | **0.38** | **0.43** |

**Motion Forecasting (minADE/minFDE)**:
| Method | minADE ↓ | minFDE ↓ |
|--------|----------|----------|
| DenseTNT | 1.45 | 3.26 |
| **VAD** | **1.38** | **3.12** |

**Planning (L2 Error)**:
| Method | 1s ↓ | 2s ↓ | 3s ↓ | Avg ↓ |
|--------|------|------|------|-------|
| ST-P3 | 0.68 | 1.31 | 2.09 | 1.36 |
| **VAD** | **0.46** | **1.00** | **1.63** | **1.03** |

### Ablation Studies

**Impact of Vectorized Representation**:
| Representation | Planning L2 ↓ | Inference Speed ↑ |
|----------------|---------------|-------------------|
| Raster BEV | 1.15 | 10 FPS |
| Vectorized | **1.03** | **12.8 FPS** |

**Impact of Map-Agent Interaction**:
| Configuration | Motion minADE ↓ | Planning L2 ↓ |
|--------------|-----------------|---------------|
| Independent | 1.52 | 1.28 |
| Map Context | **1.38** | **1.03** |

**Impact of Hierarchical Queries**:
| Query Design | Map IoU ↑ | Agent AP ↑ |
|-------------|-----------|-----------|
| Flat Queries | 0.52 | 0.39 |
| Hierarchical | **0.58** | **0.43** |

### Qualitative Analysis

**Scenario 1: Complex Intersection**
- **Map Output**: 8 lane polylines, 4 crossing boundaries
- **Agents**: 6 vehicles correctly detected and tracked
- **Planning**: Smooth trajectory following center lane
- **Advantage**: Explicit lane-following constraint in planning

**Scenario 2: Curved Road**
- **Map Output**: Smooth curved lane polylines
- **Motion**: Agent trajectories naturally follow lane curvature
- **Planning**: Ego trajectory respects road geometry
- **Advantage**: Geometric structure preserved in vectors

---

## 9. Common Pitfalls

### 1. Insufficient Polyline Points

**Problem**: Using too few points (K=5) loses geometric detail.

**Solution**: Use at least K=20 points per polyline for smooth curves.

### 2. Ignoring Permutation Invariance

**Problem**: Treating polyline points as ordered sequence when they should be permutation-invariant.

**Solution**: Use set-based losses (Hungarian matching) instead of ordered losses.

### 3. Imbalanced Classes

**Problem**: More lane polylines than crossings, model ignores rare classes.

**Solution**: Use balanced sampling and class-weighted losses.

### 4. Chamfer Distance Asymmetry

**Problem**: One-way Chamfer distance misses outliers.

**Solution**: Always use bidirectional Chamfer distance.

### 5. Polyline Connectivity

**Problem**: Predicted polylines have discontinuities.

**Solution**: Add connectivity constraints between adjacent points.

### 6. Coordinate System Confusion

**Problem**: Mixing ego-centric and world coordinates.

**Solution**: Clearly define coordinate frame for each output.

---

## 10. References

### Primary Paper

1. **VAD**: Jiachen Zhou et al., "Vectorized Scene Representation for Efficient Autonomous Driving", ICCV 2023
   - arXiv: https://arxiv.org/abs/2303.12077
   - Code: https://github.com/hustvl/VAD

### Related Vectorized Methods

2. **VectorMapNet**: Yicheng Liu et al., "VectorMapNet: End-to-end Vectorized HD Map Learning", ICML 2023
3. **MapTR**: Bencheng Liao et al., "MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction", ICLR 2023
4. **PolyLaneNet**: Toms et al., "End-to-end Lane Shape Prediction with Transformers", WACV 2021

### Foundational Work

5. **UniAD**: Hu et al., "Planning-oriented Autonomous Driving", CVPR 2023
6. **BEVFormer**: Li et al., "BEVFormer: Learning Bird's-Eye-View Representation", ECCV 2022
7. **DETR**: Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020

### Benchmarks

8. **nuScenes**: Caesar et al., "nuScenes: A Multimodal Dataset for Autonomous Driving", CVPR 2020
9. **Argoverse**: Chang et al., "Argoverse: 3D Tracking and Forecasting with Rich Maps", CVPR 2019

### Implementation

10. **Official VAD**: https://github.com/hustvl/VAD
11. **Nexus Implementation**: `/nexus/models/autonomous/vad.py`
