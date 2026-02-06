# Mask R-CNN

## Overview

Mask R-CNN extends Faster R-CNN by adding a branch for predicting instance segmentation masks in parallel with bounding box detection. It introduces RoI Align for pixel-level alignment and a fully convolutional mask head, enabling state-of-the-art instance segmentation while maintaining real-time detection performance.

**Key Innovation**: Decoupling mask and class prediction, combined with RoI Align for precise pixel alignment, enables high-quality instance segmentation without sacrificing detection accuracy.

**Architecture Highlights**:
- Parallel mask prediction branch
- RoI Align for pixel-perfect feature extraction
- Fully convolutional mask head
- Multi-task loss (classification + bbox + mask)
- Binary masks per class

**Performance**: 41.0 mAP (bbox), 37.1 mAP (mask) on COCO with ResNet-50 at 5 FPS.

## Theory

### Instance Segmentation

Instance segmentation combines:
- **Object Detection**: Where are the objects? (boxes)
- **Semantic Segmentation**: What pixel belongs to what class? (masks)
- **Instance Separation**: Which pixels belong to which instance?

Output: For each detected object, a binary mask indicating its pixels.

### Mask R-CNN Architecture

Mask R-CNN adds a mask branch to Faster R-CNN:

```
Image → Backbone → FPN → RPN → Proposals
                ↓
        RoI Align → [Box Branch → classes, boxes]
                    [Mask Branch → masks]
```

Key insight: Predict masks in parallel with boxes, not sequentially.

### RoI Align

RoI Pooling has quantization artifacts:
- Divides RoI into grid cells with rounding
- Loses sub-pixel precision
- Misalignment accumulates

RoI Align fixes this:
- Uses bilinear interpolation
- No quantization
- Samples at exact floating-point positions

This 1-2 mAP improvement is critical for mask quality.

### Mask Representation

Masks are predicted as binary masks per class:
- Output: K × m × m (K classes, m×m resolution)
- During inference: Select mask for predicted class
- Binary cross-entropy loss per pixel

Alternative approaches (polygon, contour) are more complex.

## Mathematical Formulation

### Overall Architecture

For each RoI r with features F_r:

**Box Branch** (same as Faster R-CNN):
```
h_box = FC(Flatten(F_r))
class_logits = FC_cls(h_box) ∈ ℝ^K
bbox_delta = FC_box(h_box) ∈ ℝ^(4K)
```

**Mask Branch** (new):
```
h_mask = Conv(F_r)  # Multiple conv layers
mask_logits = Conv1x1(h_mask) ∈ ℝ^(K×m×m)
```

where m is mask resolution (typically 28).

### RoI Align Operation

For a RoI with bounds (x1, y1, x2, y2):

1. Divide into grid cells (e.g., 7×7)
2. For each cell, sample at 4 regular locations
3. Use bilinear interpolation for each sample:

```
f(x, y) = Σ_i Σ_j w_ij · F[i, j]

where w_ij = max(0, 1-|x-i|) · max(0, 1-|y-j|)
```

4. Max pool the 4 samples

No rounding → sub-pixel precision.

### Multi-Task Loss

Total loss combines three objectives:

```
L = L_cls + λ_box L_box + λ_mask L_mask
```

**Classification Loss** (cross-entropy):
```
L_cls = -log p_k  where k is true class
```

**Box Loss** (smooth L1):
```
L_box = Σ_i smooth_L1(t_i - t_i*)
```

**Mask Loss** (per-pixel binary cross-entropy):
```
L_mask = -(1/m²) Σ_(i,j) [y_ij log σ(m_ij) + (1-y_ij) log(1-σ(m_ij))]
```

where:
- m_ij: predicted mask logit at pixel (i,j)
- y_ij: ground truth mask at pixel (i,j)
- σ: sigmoid function

Important: Only compute L_mask for the ground truth class (not all classes).

### Loss Weights

Standard weights:
- λ_box = 1.0
- λ_mask = 1.0

The mask loss is automatically normalized by mask area.

### Mask Prediction

During training:
- Predict masks for all K classes
- Compute loss only on ground truth class

During inference:
- Predict masks for all K classes
- Select mask corresponding to predicted class
- Threshold at 0.5

### Mask Post-Processing

1. Get predicted class k for RoI
2. Extract k-th mask: M = sigmoid(mask_logits[k])
3. Resize from m×m to RoI size
4. Paste into image at RoI location
5. Threshold: M_binary = (M > 0.5)

## High-Level Intuition

### Why Parallel Branches?

Sequential approach (predict box, then mask):
- Box errors propagate to mask
- Mask depends on box quality

Parallel approach (predict box and mask together):
- Independent predictions reduce error coupling
- Mask provides additional signal for box
- More robust to imperfect boxes

### The Decoupling Principle

Key innovation: Separate "where" from "what":
- **Classification head**: What class?
- **Box head**: Where is it?
- **Mask head**: What shape is it?

Each head specializes in its task.

### Why FCN for Masks?

Fully convolutional network (FCN) for masks:
- Preserves spatial structure
- Fewer parameters than FC
- Resolution-independent

FC layers would destroy spatial information needed for masks.

### RoI Align Intuition

Analogy: Taking a photo:
- **RoI Pooling**: Digital zoom (blocky, pixelated)
- **RoI Align**: Optical zoom (smooth, precise)

For masks, precision matters at pixel level.

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
        'num_outs': 5,
    },
    'rpn': {
        'anchor_scales': [8],
        'anchor_ratios': [0.5, 1.0, 2.0],
        'anchor_strides': [4, 8, 16, 32, 64],
    },
    'roi_head': {
        'bbox_roi_extractor': {
            'type': 'SingleRoIExtractor',
            'roi_layer': {'type': 'RoIAlign', 'output_size': 7},
            'featmap_strides': [4, 8, 16, 32],
        },
        'bbox_head': {
            'type': 'Shared2FCBBoxHead',
            'in_channels': 256,
            'fc_out_channels': 1024,
            'num_classes': 80,
        },
        'mask_roi_extractor': {
            'type': 'SingleRoIExtractor',
            'roi_layer': {'type': 'RoIAlign', 'output_size': 14},
            'featmap_strides': [4, 8, 16, 32],
        },
        'mask_head': {
            'type': 'FCNMaskHead',
            'num_convs': 4,
            'in_channels': 256,
            'conv_out_channels': 256,
            'num_classes': 80,
            'mask_size': 28,
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
cls_weight = 1.0
bbox_weight = 1.0
mask_weight = 1.0

# Image scales (multi-scale training)
img_scales = [(1333, 480), (1333, 960)]
```

### Data Augmentation

Standard augmentation plus mask handling:

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=[(1333, 480), (1333, 960)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
```

Note: Masks must be transformed consistently with images (flip, resize, etc.).

## Code Walkthrough

Implementation in `/Users/kevinyu/Projects/Nexus/nexus/models/cv/rcnn/mask_rcnn.py`.

### Mask Head

The mask head is a small FCN:

```python
class MaskHead(nn.Module):
    def __init__(self, in_channels=256, num_convs=4, num_classes=80, mask_size=28):
        super().__init__()
        
        # Conv layers
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Deconv for upsampling
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        
        # 1x1 conv for mask prediction
        self.conv_logits = nn.Conv2d(in_channels, num_classes, 1)
        
        self.mask_size = mask_size
    
    def forward(self, x):
        # x: [N, C, 14, 14] from RoI Align
        
        # Apply convolutions
        for conv in self.convs:
            x = conv(x)
        
        # Upsample to mask_size
        x = F.relu(self.deconv(x))  # [N, C, 28, 28]
        
        # Predict masks
        mask_logits = self.conv_logits(x)  # [N, num_classes, 28, 28]
        
        return mask_logits
```

### Mask Loss

```python
def mask_loss(mask_pred, mask_target, labels):
    """
    Compute mask loss.
    
    Args:
        mask_pred: [N, num_classes, 28, 28] predicted mask logits
        mask_target: [N, 28, 28] ground truth masks
        labels: [N] ground truth class labels
        
    Returns:
        loss: scalar mask loss
    """
    N = mask_pred.size(0)
    
    # Select masks for ground truth classes only
    mask_pred_selected = mask_pred[torch.arange(N), labels]  # [N, 28, 28]
    
    # Binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(
        mask_pred_selected,
        mask_target.float(),
        reduction='mean'
    )
    
    return loss
```

### Full Forward Pass

```python
class MaskRCNN(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        
        # Backbone + FPN (same as Faster R-CNN)
        self.backbone = ResNet50()
        self.neck = FPN([256, 512, 1024, 2048], 256)
        
        # RPN (same as Faster R-CNN)
        self.rpn = RPN(in_channels=256)
        
        # RoI heads
        self.roi_align_box = RoIAlign(output_size=7)
        self.roi_align_mask = RoIAlign(output_size=14)
        
        self.bbox_head = BBoxHead(num_classes=num_classes)
        self.mask_head = MaskHead(num_classes=num_classes)
    
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
            
            # Mask branch
            roi_mask_feats = self.roi_align_mask(features, proposals)
            mask_preds = self.mask_head(roi_mask_feats)
            
            # Compute losses
            losses = {
                'loss_cls': cls_loss(cls_scores, targets['labels']),
                'loss_bbox': bbox_loss(bbox_preds, targets['boxes']),
                'loss_mask': mask_loss(mask_preds, targets['masks'], targets['labels'])
            }
            
            return losses
        else:
            # Inference
            roi_box_feats = self.roi_align_box(features, proposals)
            cls_scores, bbox_preds = self.bbox_head(roi_box_feats)
            
            # Refine boxes
            boxes = refine_boxes(proposals, bbox_preds)
            
            # Get masks
            roi_mask_feats = self.roi_align_mask(features, boxes)
            mask_preds = self.mask_head(roi_mask_feats)
            
            return {'boxes': boxes, 'scores': cls_scores, 'masks': mask_preds}
```

### Mask Post-Processing

```python
def postprocess_masks(mask_logits, boxes, image_shape, mask_threshold=0.5):
    """
    Convert mask logits to instance masks.
    
    Args:
        mask_logits: [N, num_classes, 28, 28]
        boxes: [N, 4] detected boxes
        image_shape: (H, W) original image shape
        mask_threshold: threshold for binarization
        
    Returns:
        masks: [N, H, W] binary masks
    """
    N = mask_logits.size(0)
    H, W = image_shape
    
    masks = []
    
    for i in range(N):
        # Get predicted class mask
        class_id = boxes[i].argmax()  # Predicted class
        mask_logit = mask_logits[i, class_id]  # [28, 28]
        
        # Sigmoid
        mask_prob = torch.sigmoid(mask_logit)
        
        # Resize to box size
        x1, y1, x2, y2 = boxes[i].int()
        box_h, box_w = y2 - y1, x2 - x1
        mask_resized = F.interpolate(
            mask_prob.unsqueeze(0).unsqueeze(0),
            size=(box_h, box_w),
            mode='bilinear',
            align_corners=False
        )[0, 0]
        
        # Paste into full image
        mask_full = torch.zeros((H, W), dtype=torch.float32)
        mask_full[y1:y2, x1:x2] = mask_resized
        
        # Threshold
        mask_binary = mask_full > mask_threshold
        
        masks.append(mask_binary)
    
    return torch.stack(masks)
```

## Optimization Tricks

### 1. Larger Mask Resolution

Use 56×56 or higher resolution masks:

```python
mask_head = MaskHead(mask_size=56)  # Instead of 28
```

Improves mask quality at cost of speed.

### 2. GCE Loss

Generalized Cross Entropy loss for noisy labels:

```python
def gce_loss(pred, target, q=0.7):
    pred = pred.sigmoid()
    loss = (1 - (target * pred + (1-target) * (1-pred)) ** q) / q
    return loss.mean()
```

More robust to annotation noise.

### 3. Cascade Mask R-CNN

Add mask branch to Cascade R-CNN:

```python
# Predict masks at each cascade stage
for stage in range(3):
    boxes = cascade_stage(boxes)
    masks = mask_head[stage](boxes)
```

### 4. PointRend

Adaptive point-based rendering for finer boundaries:

```python
# Sample uncertain points on mask boundary
points = sample_uncertain_points(coarse_mask, num_points=1000)

# Predict high-resolution features at these points
point_features = sample_features(features, points)
point_logits = point_head(point_features)

# Combine with coarse mask
final_mask = combine(coarse_mask, point_logits, points)
```

### 5. SOLOv2-style Decoupled Mask Prediction

Predict mask and category separately:

```python
# Mask kernel prediction
mask_kernels = kernel_head(features)

# Mask features
mask_feats = mask_branch(features)

# Generate masks via convolution
masks = conv(mask_feats, mask_kernels)
```

### 6. Soft Mask Loss

Use soft labels for masks:

```python
# Instead of hard 0/1 masks
mask_target = distance_transform(binary_mask)  # Smooth near boundaries

loss = F.l1_loss(mask_pred.sigmoid(), mask_target)
```

### 7. Boundary Refinement

Add explicit boundary prediction:

```python
# Predict boundary
boundary_logits = boundary_head(mask_features)

# Boundary loss
boundary_loss = F.binary_cross_entropy_with_logits(
    boundary_logits,
    extract_boundaries(mask_target)
)
```

## Experiments & Results

### COCO Instance Segmentation

**Mask R-CNN ResNet-50-FPN** (1x schedule):
- bbox mAP: 41.0
- mask mAP: 37.1
- mask AP50: 59.5
- mask AP75: 39.4
- APS: 17.8
- APM: 40.2
- APL: 52.9
- FPS: 5

**Mask R-CNN ResNet-101-FPN**:
- bbox mAP: 42.9
- mask mAP: 38.6

**Mask R-CNN ResNeXt-101-FPN**:
- bbox mAP: 44.3
- mask mAP: 39.8

### Comparison with Faster R-CNN

Box detection:
- Faster R-CNN: 40.2 mAP
- Mask R-CNN: 41.0 mAP (+0.8)

Adding mask branch slightly improves box detection!

### Ablation Studies

**Effect of RoI Align**:
- RoI Pooling: 35.8 mask mAP
- RoI Align: 37.1 mask mAP (+1.3)

**Effect of Mask Resolution**:
- 14×14: 34.2 mask mAP
- 28×28: 37.1 mask mAP
- 56×56: 37.8 mask mAP

**Effect of Number of Conv Layers in Mask Head**:
- 2 layers: 36.1 mask mAP
- 4 layers: 37.1 mask mAP
- 8 layers: 37.3 mask mAP

**Class-Specific vs. Class-Agnostic Masks**:
- Class-agnostic (single mask): 35.9 mask mAP
- Class-specific (K masks): 37.1 mask mAP (+1.2)

### Speed-Accuracy Trade-off

| Model | Backbone | bbox mAP | mask mAP | FPS |
|-------|----------|----------|----------|-----|
| Mask R-CNN | ResNet-50 | 41.0 | 37.1 | 5.0 |
| Mask R-CNN | ResNet-101 | 42.9 | 38.6 | 3.5 |
| Mask R-CNN | ResNeXt-101 | 44.3 | 39.8 | 2.8 |

### Mask Quality Analysis

Average IoU between predicted and ground truth masks:
- Mask R-CNN: 0.62
- FCN (semantic seg): 0.55

Instance-level masks are more accurate than semantic segmentation.

## Common Pitfalls

### 1. Wrong RoI Align Output Size

Problem: Using same output size for box and mask branches.

Solution: Larger output for masks (14×14) than boxes (7×7).

```python
roi_align_box = RoIAlign(output_size=7)   # For box
roi_align_mask = RoIAlign(output_size=14)  # For mask (needs more spatial detail)
```

### 2. Computing Mask Loss for All Classes

Problem: Computing mask loss for all K classes (wasteful, noisy).

Solution: Only compute loss for ground truth class.

```python
# Wrong
loss = F.binary_cross_entropy_with_logits(mask_pred, mask_target)

# Correct
mask_pred_gt_class = mask_pred[torch.arange(N), gt_labels]
loss = F.binary_cross_entropy_with_logits(mask_pred_gt_class, mask_target)
```

### 3. Forgetting to Threshold Masks

Problem: Using soft mask probabilities directly.

Solution: Threshold at 0.5 for binary masks.

```python
mask_prob = torch.sigmoid(mask_logits)
mask_binary = (mask_prob > 0.5).float()
```

### 4. Not Resizing Masks to Box Size

Problem: Directly pasting 28×28 mask into image.

Solution: Resize mask to match RoI size.

```python
# Resize to RoI size first
box_h, box_w = y2 - y1, x2 - x1
mask_resized = F.interpolate(mask_28x28, size=(box_h, box_w))

# Then paste into image
full_mask[y1:y2, x1:x2] = mask_resized
```

### 5. Incorrect Mask Data Augmentation

Problem: Not transforming masks consistently with images.

Solution: Apply same transforms to both.

```python
# Flip both image and masks
if random.random() > 0.5:
    image = image.flip(-1)
    masks = masks.flip(-1)
```

### 6. Using RoI Pooling Instead of RoI Align

Problem: Quantization artifacts hurt mask quality.

Solution: Always use RoI Align for masks.

```python
# Don't use this for masks
roi_pool = torchvision.ops.roi_pool

# Use this
roi_align = torchvision.ops.roi_align  # With align_corners=False
```

### 7. Small Mask Resolution

Problem: 14×14 masks are too coarse.

Solution: Use at least 28×28, preferably 56×56 for high quality.

```python
mask_head = MaskHead(mask_size=56)  # Better quality
```

### 8. Not Handling Overlapping Instances

Problem: Overlapping masks interfere with each other.

Solution: Process instances in score order (high to low), with occlusion handling.

```python
# Sort by score
sorted_indices = scores.argsort(descending=True)

# Paste in order (high score first)
for idx in sorted_indices:
    mask_full[masks[idx] > 0.5] = idx + 1  # Instance ID
```

## References

### Original Paper

- **Mask R-CNN**
  - Authors: Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick
  - Conference: ICCV 2017
  - Paper: https://arxiv.org/abs/1703.06870
  - Code: https://github.com/facebookresearch/Detectron

### Related Work

- **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**
  - NIPS 2015
  - https://arxiv.org/abs/1506.01497

- **Feature Pyramid Networks for Object Detection**
  - CVPR 2017
  - https://arxiv.org/abs/1612.03144

- **Fully Convolutional Networks for Semantic Segmentation**
  - CVPR 2015
  - https://arxiv.org/abs/1411.4038

### Extensions

- **Cascade R-CNN: Delving into High Quality Object Detection**
  - CVPR 2018
  - https://arxiv.org/abs/1712.00726

- **PointRend: Image Segmentation as Rendering**
  - CVPR 2020
  - https://arxiv.org/abs/1912.08193

- **Mask Scoring R-CNN**
  - CVPR 2019
  - https://arxiv.org/abs/1903.00241

- **Hybrid Task Cascade for Instance Segmentation**
  - CVPR 2019
  - https://arxiv.org/abs/1901.07518

### Implementation Resources

- Detectron2: https://github.com/facebookresearch/detectron2
- MMDetection: https://github.com/open-mmlab/mmdetection
- TorchVision: https://pytorch.org/vision/stable/models.html#mask-r-cnn
- Original Detectron: https://github.com/facebookresearch/Detectron
- Local: `/Users/kevinyu/Projects/Nexus/nexus/models/cv/rcnn/mask_rcnn.py`

### Applications

- Medical image segmentation
- Autonomous driving (pedestrian/vehicle segmentation)
- Video instance segmentation
- Agricultural crop detection
- Retail object counting

### Tutorials

- "Mask R-CNN Explained": https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272
- Detectron2 Tutorial: https://detectron2.readthedocs.io/en/latest/tutorials/
- TorchVision Tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
