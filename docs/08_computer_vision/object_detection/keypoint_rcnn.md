# Keypoint R-CNN

## Overview

Keypoint R-CNN extends Mask R-CNN for human pose estimation by adding a keypoint prediction branch. It predicts person keypoints (joints) as spatial heatmaps in parallel with instance segmentation and bounding box detection, enabling multi-person pose estimation in a single framework.

**Key Innovation**: Per-instance keypoint heatmaps predicted via a small FCN, with person detection and keypoint localization in a unified multi-task framework.

**Architecture Highlights**:
- Parallel keypoint prediction branch
- Per-instance heatmap prediction
- Multi-task learning (detection + segmentation + keypoints)
- OKS-based evaluation
- 17 COCO keypoints support

**Performance**: 64.2 AP (keypoints) on COCO test-dev, state-of-the-art multi-person pose estimation.

## Theory

### Human Pose Estimation

Estimate the spatial locations of key body parts (joints):
- **2D Pose**: (x, y) coordinates in image
- **COCO keypoints**: 17 points (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
- **Multi-person**: Detect and localize keypoints for each person instance

### Top-Down vs. Bottom-Up

**Bottom-Up** (e.g., OpenPose):
- Detect all keypoints first
- Group them into instances
- Faster but less accurate

**Top-Down** (Keypoint R-CNN):
- Detect person instances first
- Predict keypoints per instance
- Slower but more accurate

Keypoint R-CNN follows top-down approach.

### Keypoint Representation

Keypoints are represented as heatmaps:
- **Input**: RoI features
- **Output**: K heatmaps (K keypoints), each of size m×m
- **Peak**: Highest activation indicates keypoint location
- **Loss**: Per-keypoint per-pixel loss

Alternative representations (coordinates, offsets) are less effective.

### Architecture Extension

Keypoint R-CNN = Mask R-CNN + Keypoint Branch:

```
Image → Backbone → FPN → RPN → Proposals
                ↓
        RoI Align → [Box Branch → classes, boxes]
                    [Mask Branch → masks]
                    [Keypoint Branch → keypoints]
```

All branches share the same RoI features.

### Object Keypoint Similarity (OKS)

OKS is the metric for keypoint accuracy, analogous to IoU for boxes:

```
OKS = Σ_i exp(-d_i² / (2s²κ_i²)) δ(v_i > 0) / Σ_i δ(v_i > 0)
```

where:
- d_i: Euclidean distance between predicted and ground truth for keypoint i
- s: Object scale (sqrt(area))
- κ_i: Per-keypoint constant (larger for easier keypoints like eyes, smaller for harder like ankles)
- v_i: Visibility flag

OKS ∈ [0, 1], with 1 being perfect alignment.

## Mathematical Formulation

### Overall Architecture

For each person RoI r:

**Box Branch**:
```
h_box = FC(RoIAlign(F, r))
class_logits = FC_cls(h_box) ∈ ℝ^K
bbox_delta = FC_box(h_box) ∈ ℝ^(4K)
```

**Mask Branch** (optional, can be removed):
```
h_mask = Conv(RoIAlign(F, r))
mask_logits = Conv1x1(h_mask) ∈ ℝ^(1×m×m)
```

**Keypoint Branch** (new):
```
h_kpt = Conv(RoIAlign(F, r))  # Multiple conv layers
kpt_heatmaps = Conv1x1(h_kpt) ∈ ℝ^(K×m×m)
```

where K is number of keypoints (17 for COCO).

### Keypoint Heatmap Generation

Ground truth heatmap for keypoint k at location (x*, y*):

```
H_k(x, y) = exp(-((x - x*)² + (y - y*)²) / (2σ²))
```

This is a 2D Gaussian centered at the ground truth location with standard deviation σ (typically 1-2 pixels).

### Keypoint Loss

Per-pixel cross-entropy loss on heatmaps:

```
L_kpt = -(1/(K·m²)) Σ_k Σ_(i,j) [H_k*(i,j) log H_k(i,j) + (1-H_k*(i,j)) log(1-H_k(i,j))]
```

where:
- H_k: Predicted heatmap for keypoint k
- H_k*: Ground truth heatmap for keypoint k

Only compute loss for visible keypoints (v > 0).

### Multi-Task Loss

Total loss combines four objectives:

```
L = L_rpn + L_cls + λ_box L_box + λ_mask L_mask + λ_kpt L_kpt
```

Standard weights:
- λ_box = 1.0
- λ_mask = 1.0
- λ_kpt = 1.0

### Keypoint Localization

During inference:
1. Predict heatmaps H_k for all keypoints
2. For each keypoint k:
   - Find peak: (x_max, y_max) = argmax H_k
   - Sub-pixel refinement (optional):
     ```
     x_refined = x_max + 0.25 * sign(H[x+1] - H[x-1])
     y_refined = y_max + 0.25 * sign(H[y+1] - H[y-1])
     ```
3. Confidence: c_k = max(H_k)
4. Transform to image coordinates

### Visibility Prediction

Each keypoint has visibility v ∈ {0, 1, 2}:
- 0: Not labeled (ignore)
- 1: Labeled but occluded
- 2: Labeled and visible

Can add visibility head:
```
vis_logits = FC(h_kpt) ∈ ℝ^(K×3)
```

## High-Level Intuition

### Why Heatmaps?

Coordinate regression (direct (x,y) prediction) issues:
- Multimodal (multiple possible locations)
- Hard to optimize
- Poor spatial reasoning

Heatmaps:
- Unimodal peak at target location
- Spatial structure preserved
- Easier to learn with convolutions

### Per-Instance Prediction

Unlike bottom-up methods that predict all keypoints globally, Keypoint R-CNN predicts keypoints for each person separately:

**Advantages**:
- No keypoint grouping problem
- Better handles occlusion
- Higher accuracy

**Disadvantages**:
- Slower (scales with number of people)
- Depends on person detection quality

### Multi-Task Learning Benefits

Training all branches together helps:
- **Mask helps keypoints**: Body shape informs joint locations
- **Keypoints help mask**: Joint positions define body outline
- **Detection helps all**: Good bounding boxes improve both

Shared representations learn complementary features.

### The Gaussian Trick

Why Gaussian heatmaps instead of delta functions?

**Delta function** (single pixel):
- Hard target
- Sensitive to small errors
- Difficult to learn

**Gaussian** (smooth distribution):
- Soft target
- Tolerant to small errors
- Easier to optimize
- Nearby pixels get positive signal

## Implementation Details

### Architecture Configuration

```python
config = {
    'backbone': {
        'type': 'ResNet50',
        'pretrained': True,
        'frozen_stages': 1,
    },
    'neck': {
        'type': 'FPN',
        'in_channels': [256, 512, 1024, 2048],
        'out_channels': 256,
    },
    'rpn': {
        'anchor_scales': [2, 4, 8, 16, 32],
        'anchor_ratios': [0.5, 1.0, 2.0],
    },
    'roi_head': {
        'bbox_roi_extractor': {
            'type': 'SingleRoIExtractor',
            'roi_layer': {'type': 'RoIAlign', 'output_size': 7},
        },
        'bbox_head': {
            'type': 'Shared2FCBBoxHead',
            'num_classes': 2,  # Only 'person' class
        },
        'keypoint_roi_extractor': {
            'type': 'SingleRoIExtractor',
            'roi_layer': {'type': 'RoIAlign', 'output_size': 14},
        },
        'keypoint_head': {
            'type': 'FCNKeypointHead',
            'num_convs': 8,
            'in_channels': 256,
            'conv_out_channels': 512,
            'num_keypoints': 17,
            'heatmap_size': 56,
        },
    },
}
```

### Training Hyperparameters

```python
optimizer = SGD(
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0001
)

scheduler = StepLR(step_size=[8, 11], gamma=0.1)
epochs = 12
batch_size = 16

# Loss weights
rpn_weight = 1.0
cls_weight = 1.0
bbox_weight = 1.0
keypoint_weight = 1.0

# Heatmap parameters
heatmap_size = 56  # Output resolution
gaussian_sigma = 2.0  # For ground truth heatmaps
```

### Data Augmentation

Must handle keypoints consistently with images:

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
    
    # Geometric augmentations
    dict(type='Resize', img_scale=[(1333, 480), (1333, 960)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomAffine', max_translate=0.1, max_scale=0.2),
    
    # Color augmentations
    dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints']),
]
```

Important: When flipping horizontally, swap left/right keypoints:
- Left shoulder ↔ Right shoulder
- Left elbow ↔ Right elbow
- etc.

### COCO Keypoint Format

COCO defines 17 keypoints in specific order:

```python
COCO_KEYPOINTS = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle',    # 16
]

# Keypoint data format: [x, y, v] for each keypoint
# v: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
```

### Skeleton Connections

For visualization, connect keypoints into skeleton:

```python
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # Legs
    [6, 12], [7, 13],  # Torso
    [6, 8], [7, 9], [8, 10], [9, 11],  # Arms
    [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]  # Head
]
```

## Code Walkthrough

Implementation in `/Users/kevinyu/Projects/Nexus/nexus/models/cv/rcnn/keypoint_rcnn.py`.

### Keypoint Head

Small FCN predicting heatmaps:

```python
class KeypointHead(nn.Module):
    def __init__(self, in_channels=256, num_convs=8, num_keypoints=17, heatmap_size=56):
        super().__init__()
        
        # Conv layers
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels if i == 0 else 512, 512, 3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Deconv for upsampling
        self.deconv = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        
        # 1x1 conv for heatmap prediction
        self.conv_logits = nn.Conv2d(512, num_keypoints, 1)
        
        self.heatmap_size = heatmap_size
    
    def forward(self, x):
        # x: [N, C, 14, 14] from RoI Align
        
        # Conv layers
        for conv in self.convs:
            x = conv(x)
        
        # Upsample
        x = F.relu(self.deconv(x))  # [N, 512, 28, 28]
        
        # Another upsample to reach target size
        if x.shape[-1] != self.heatmap_size:
            x = F.interpolate(x, size=self.heatmap_size, mode='bilinear')
        
        # Predict heatmaps
        heatmaps = self.conv_logits(x)  # [N, num_keypoints, heatmap_size, heatmap_size]
        
        return heatmaps
```

### Generate Ground Truth Heatmaps

```python
def generate_target_heatmaps(keypoints, heatmap_size=56, sigma=2.0):
    """
    Generate Gaussian heatmaps for keypoints.
    
    Args:
        keypoints: [N, num_keypoints, 3] (x, y, visibility)
        heatmap_size: Output heatmap size
        sigma: Gaussian standard deviation
        
    Returns:
        heatmaps: [N, num_keypoints, heatmap_size, heatmap_size]
    """
    N, K = keypoints.shape[:2]
    heatmaps = torch.zeros(N, K, heatmap_size, heatmap_size)
    
    for n in range(N):
        for k in range(K):
            x, y, v = keypoints[n, k]
            
            if v == 0:  # Not labeled
                continue
            
            # Scale to heatmap coordinates
            x_hm = x * heatmap_size
            y_hm = y * heatmap_size
            
            # Generate Gaussian
            y_grid, x_grid = torch.meshgrid(
                torch.arange(heatmap_size),
                torch.arange(heatmap_size)
            )
            
            heatmap = torch.exp(-((x_grid - x_hm)**2 + (y_grid - y_hm)**2) / (2 * sigma**2))
            
            heatmaps[n, k] = heatmap
    
    return heatmaps
```

### Keypoint Loss

```python
def keypoint_loss(pred_heatmaps, target_heatmaps, visibility):
    """
    Compute keypoint loss.
    
    Args:
        pred_heatmaps: [N, K, H, W] predicted heatmaps
        target_heatmaps: [N, K, H, W] ground truth heatmaps
        visibility: [N, K] visibility flags
        
    Returns:
        loss: scalar keypoint loss
    """
    # Only compute loss for visible keypoints
    valid_mask = (visibility > 0).float()  # [N, K]
    
    # Expand mask to heatmap size
    valid_mask = valid_mask.unsqueeze(-1).unsqueeze(-1)  # [N, K, 1, 1]
    
    # Per-pixel cross-entropy
    loss = F.binary_cross_entropy_with_logits(
        pred_heatmaps,
        target_heatmaps,
        reduction='none'
    )
    
    # Apply mask and average
    loss = (loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
    
    return loss
```

### Keypoint Decoding

```python
def decode_keypoints(heatmaps, boxes, threshold=0.2):
    """
    Decode keypoints from heatmaps.
    
    Args:
        heatmaps: [N, K, H, W] predicted heatmaps
        boxes: [N, 4] bounding boxes
        threshold: Confidence threshold
        
    Returns:
        keypoints: [N, K, 3] (x, y, confidence)
    """
    N, K, H, W = heatmaps.shape
    
    # Apply sigmoid
    heatmaps = torch.sigmoid(heatmaps)
    
    keypoints = []
    
    for n in range(N):
        person_keypoints = []
        
        for k in range(K):
            heatmap = heatmaps[n, k]
            
            # Find peak
            max_val, max_idx = heatmap.flatten().max(dim=0)
            y_hm = max_idx // W
            x_hm = max_idx % W
            
            if max_val < threshold:
                # Low confidence, mark as not visible
                person_keypoints.append([0, 0, 0])
                continue
            
            # Sub-pixel refinement
            if 0 < x_hm < W-1:
                dx = 0.25 * (heatmap[y_hm, x_hm+1].item() - heatmap[y_hm, x_hm-1].item())
            else:
                dx = 0
            
            if 0 < y_hm < H-1:
                dy = 0.25 * (heatmap[y_hm+1, x_hm].item() - heatmap[y_hm-1, x_hm].item())
            else:
                dy = 0
            
            x_refined = (x_hm.item() + dx) / W
            y_refined = (y_hm.item() + dy) / H
            
            # Transform to image coordinates
            x1, y1, x2, y2 = boxes[n]
            box_w = x2 - x1
            box_h = y2 - y1
            
            x_img = x1 + x_refined * box_w
            y_img = y1 + y_refined * box_h
            
            person_keypoints.append([x_img, y_img, max_val.item()])
        
        keypoints.append(person_keypoints)
    
    return torch.tensor(keypoints)
```

### Full Keypoint R-CNN

```python
class KeypointRCNN(nn.Module):
    def __init__(self, num_keypoints=17):
        super().__init__()
        
        # Backbone + FPN (same as Mask R-CNN)
        self.backbone = ResNet50()
        self.neck = FPN([256, 512, 1024, 2048], 256)
        
        # RPN
        self.rpn = RPN(in_channels=256)
        
        # RoI heads
        self.roi_align_box = RoIAlign(output_size=7)
        self.roi_align_keypoint = RoIAlign(output_size=14)
        
        self.bbox_head = BBoxHead(num_classes=2)  # Person only
        self.keypoint_head = KeypointHead(num_keypoints=num_keypoints)
    
    def forward(self, images, targets=None):
        # Extract features
        features = self.backbone(images)
        features = self.neck(features)
        
        # RPN
        proposals = self.rpn(features, targets)
        
        if self.training:
            # Sample proposals
            proposals, targets = self.sample_proposals(proposals, targets)
            
            # Box branch
            roi_box_feats = self.roi_align_box(features, proposals)
            cls_scores, bbox_preds = self.bbox_head(roi_box_feats)
            
            # Keypoint branch
            roi_kpt_feats = self.roi_align_keypoint(features, proposals)
            kpt_heatmaps = self.keypoint_head(roi_kpt_feats)
            
            # Generate target heatmaps
            target_heatmaps = generate_target_heatmaps(
                targets['keypoints'],
                heatmap_size=kpt_heatmaps.shape[-1]
            )
            
            # Compute losses
            losses = {
                'loss_cls': cls_loss(cls_scores, targets['labels']),
                'loss_bbox': bbox_loss(bbox_preds, targets['boxes']),
                'loss_keypoint': keypoint_loss(
                    kpt_heatmaps,
                    target_heatmaps,
                    targets['keypoints'][:, :, 2]  # Visibility
                )
            }
            
            return losses
        else:
            # Inference
            roi_box_feats = self.roi_align_box(features, proposals)
            cls_scores, bbox_preds = self.bbox_head(roi_box_feats)
            
            # Refine boxes
            boxes = refine_boxes(proposals, bbox_preds)
            
            # Keypoints
            roi_kpt_feats = self.roi_align_keypoint(features, boxes)
            kpt_heatmaps = self.keypoint_head(roi_kpt_feats)
            
            # Decode keypoints
            keypoints = decode_keypoints(kpt_heatmaps, boxes)
            
            return {
                'boxes': boxes,
                'scores': cls_scores,
                'keypoints': keypoints
            }
```

## Optimization Tricks

### 1. Larger Heatmap Resolution

Use 96×96 or 128×128 heatmaps for better precision:

```python
keypoint_head = KeypointHead(heatmap_size=96)  # Instead of 56
```

### 2. Multi-Scale Training

Train with varying image scales:

```python
img_scales = [(640, 640), (800, 800), (960, 960), (1120, 1120)]
```

### 3. Data Augmentation

Strong augmentation helps generalization:

```python
# Rotation
transforms.RandomRotation(degrees=30)

# Scaling
transforms.RandomAffine(scale=(0.75, 1.25))

# Flip with keypoint swapping
RandomHorizontalFlipWithKeypoints()
```

### 4. Focal Loss for Heatmaps

Use focal loss to handle class imbalance (most pixels are background):

```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()
```

### 5. OKS-Based NMS

Use OKS instead of IoU for NMS:

```python
def oks_nms(keypoints, scores, threshold=0.9):
    keep = []
    for i in range(len(keypoints)):
        overlap = False
        for j in keep:
            if compute_oks(keypoints[i], keypoints[j]) > threshold:
                overlap = True
                break
        if not overlap:
            keep.append(i)
    return keep
```

### 6. Integral Pose Regression

Predict coordinates as soft-argmax of heatmaps:

```python
def soft_argmax(heatmaps):
    # Heatmaps: [N, K, H, W]
    N, K, H, W = heatmaps.shape
    
    # Normalize
    heatmaps = heatmaps.softmax(dim=-1).softmax(dim=-2)
    
    # Create coordinate grids
    x_grid = torch.arange(W).reshape(1, 1, 1, W)
    y_grid = torch.arange(H).reshape(1, 1, H, 1)
    
    # Expected coordinates
    x_coord = (heatmaps * x_grid).sum(dim=[2, 3])
    y_coord = (heatmaps * y_grid).sum(dim=[2, 3])
    
    return torch.stack([x_coord, y_coord], dim=-1)
```

### 7. Offset Prediction

Predict sub-pixel offsets for higher precision:

```python
class KeypointHeadWithOffsets(nn.Module):
    def forward(self, x):
        heatmaps = self.heatmap_head(x)  # [N, K, H, W]
        offsets = self.offset_head(x)    # [N, K*2, H, W]
        return heatmaps, offsets

# At peak location, use offset for refinement
x_refined = x_peak + offset_x[y_peak, x_peak]
y_refined = y_peak + offset_y[y_peak, x_peak]
```

## Experiments & Results

### COCO Keypoint Detection

**Keypoint R-CNN ResNet-50-FPN**:
- Keypoint AP: 64.2
- Keypoint AP50: 86.9
- Keypoint AP75: 70.4
- APM: 59.8
- APL: 71.9

**Keypoint R-CNN ResNet-101-FPN**:
- Keypoint AP: 66.1
- Keypoint AP50: 87.8
- Keypoint AP75: 72.5

**Keypoint R-CNN ResNeXt-101-FPN**:
- Keypoint AP: 68.2
- Keypoint AP50: 88.9
- Keypoint AP75: 74.8

### Comparison with Other Methods

| Method | Backbone | AP | AP50 | AP75 |
|--------|----------|-----|------|------|
| OpenPose | VGG-19 | 61.8 | 84.9 | 67.5 |
| Associative Embedding | Hourglass | 56.6 | 81.8 | 61.8 |
| PersonLab | ResNet-152 | 66.5 | 88.0 | 72.6 |
| Keypoint R-CNN | ResNet-50 | 64.2 | 86.9 | 70.4 |
| HRNet | HRNet-W48 | 75.5 | 92.2 | 82.5 |

### Ablation Studies

**Effect of Heatmap Resolution**:
- 28×28: 60.1 AP
- 56×56: 64.2 AP
- 96×96: 65.8 AP

**Effect of Number of Conv Layers**:
- 4 layers: 62.3 AP
- 8 layers: 64.2 AP
- 12 layers: 64.5 AP

**Effect of Training Schedule**:
- 1x (12 epochs): 64.2 AP
- 2x (24 epochs): 65.0 AP
- 3x (36 epochs): 65.4 AP

**Effect of Input Resolution**:
- 640×640: 61.8 AP
- 800×800: 64.2 AP
- 1024×1024: 66.1 AP

### Per-Keypoint AP

| Keypoint | AP |
|----------|-----|
| Nose | 74.2 |
| Eyes | 72.8 |
| Ears | 68.5 |
| Shoulders | 71.3 |
| Elbows | 68.9 |
| Wrists | 64.2 |
| Hips | 69.7 |
| Knees | 67.1 |
| Ankles | 62.3 |

Extremities (wrists, ankles) are harder than torso keypoints.

### Speed-Accuracy Trade-off

| Model | Backbone | Input Size | AP | FPS |
|-------|----------|------------|-----|-----|
| Keypoint R-CNN | ResNet-50 | 800×800 | 64.2 | 4.2 |
| Keypoint R-CNN | ResNet-101 | 800×800 | 66.1 | 3.1 |
| OpenPose | VGG-19 | 368×368 | 61.8 | 8.8 |

Top-down methods (Keypoint R-CNN) are slower but more accurate than bottom-up (OpenPose).

## Common Pitfalls

### 1. Wrong Keypoint Ordering

Problem: COCO keypoint order is fixed, mixing it up breaks evaluation.

Solution: Always use COCO keypoint order (nose, left_eye, right_eye, ...).

```python
# COCO order must be preserved
KEYPOINT_ORDER = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
```

### 2. Not Swapping Left/Right on Flip

Problem: Horizontal flip requires swapping left/right keypoints.

Solution: Implement proper keypoint flipping.

```python
def flip_keypoints(keypoints):
    # Swap left/right pairs
    FLIP_PAIRS = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
    
    flipped = keypoints.copy()
    for left, right in FLIP_PAIRS:
        flipped[:, left], flipped[:, right] = keypoints[:, right].copy(), keypoints[:, left].copy()
    
    # Flip x coordinates
    flipped[:, :, 0] = image_width - flipped[:, :, 0]
    
    return flipped
```

### 3. Ignoring Visibility Flags

Problem: Computing loss on invisible keypoints introduces noise.

Solution: Mask loss by visibility.

```python
# Only compute loss where visibility > 0
valid_mask = (visibility > 0).float()
loss = (loss * valid_mask).sum() / valid_mask.sum()
```

### 4. Too Small Heatmap Resolution

Problem: 28×28 heatmaps lack precision.

Solution: Use at least 56×56, preferably 96×96.

```python
keypoint_head = KeypointHead(heatmap_size=96)  # Better precision
```

### 5. Not Using OKS for Evaluation

Problem: Using IoU or distance for evaluation.

Solution: Always use OKS metric for keypoint evaluation.

```python
# Use OKS, not Euclidean distance
oks = compute_oks(pred_keypoints, gt_keypoints, object_scale)
```

### 6. Forgetting Sub-Pixel Refinement

Problem: Integer coordinates are imprecise.

Solution: Use sub-pixel refinement from heatmap gradients.

```python
# Sub-pixel refinement
x_refined = x_max + 0.25 * sign(heatmap[y, x+1] - heatmap[y, x-1])
y_refined = y_max + 0.25 * sign(heatmap[y+1, x] - heatmap[y-1, x])
```

### 7. Small Gaussian Sigma

Problem: σ=1 makes targets too sharp, hard to learn.

Solution: Use σ=2 or 3 for smoother targets.

```python
target_heatmap = gaussian_2d(center, sigma=2.0)  # Smoother
```

### 8. Not Normalizing by Object Scale in OKS

Problem: OKS doesn't account for object size.

Solution: Always include scale factor s in OKS computation.

```python
oks = exp(-d**2 / (2 * s**2 * kappa**2))
# where s = sqrt(object_area)
```

## References

### Original Paper

- **Mask R-CNN**
  - Authors: Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick
  - Conference: ICCV 2017
  - Paper: https://arxiv.org/abs/1703.06870
  - Keypoint R-CNN is described in Section 5

### Related Work

- **OpenPose: Realtime Multi-Person 2D Pose Estimation**
  - TPAMI 2019
  - https://arxiv.org/abs/1812.08008
  - Bottom-up approach

- **Associative Embedding: End-to-End Learning for Joint Detection and Grouping**
  - NeurIPS 2017
  - https://arxiv.org/abs/1611.05424
  - Bottom-up with learned grouping

- **PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model**
  - ECCV 2018
  - https://arxiv.org/abs/1803.08225

- **Deep High-Resolution Representation Learning for Human Pose Estimation (HRNet)**
  - CVPR 2019
  - https://arxiv.org/abs/1902.09212
  - State-of-the-art top-down method

### COCO Keypoint Dataset

- **Microsoft COCO: Common Objects in Context**
  - ECCV 2014
  - https://arxiv.org/abs/1405.0312
  - Includes keypoint annotations

### Implementation Resources

- Detectron2: https://github.com/facebookresearch/detectron2
- MMPose: https://github.com/open-mmlab/mmpose
- PyTorch Keypoint R-CNN: https://pytorch.org/vision/stable/models.html#keypoint-r-cnn
- Local: `/Users/kevinyu/Projects/Nexus/nexus/models/cv/rcnn/keypoint_rcnn.py`

### Applications

- **Sports Analytics**: Athlete motion analysis
- **Healthcare**: Gait analysis, physical therapy
- **AR/VR**: Motion capture, avatar control
- **Fitness**: Form correction, exercise tracking
- **Sign Language Recognition**: Hand and body pose
- **Animation**: Motion reference for animators

### Tutorials

- "Keypoint R-CNN Explained": https://medium.com/@hirotoschwert/introduction-to-pose-estimation-using-keypoint-rcnn-f0dce81af8a2
- Detectron2 Keypoint Tutorial: https://detectron2.readthedocs.io/en/latest/tutorials/keypoint.html
- MMPose Documentation: https://mmpose.readthedocs.io/

### Metrics

- **Object Keypoint Similarity (OKS)**: Primary metric
  - OKS-based AP, AP50, AP75
  - Per-keypoint κ values defined in COCO
