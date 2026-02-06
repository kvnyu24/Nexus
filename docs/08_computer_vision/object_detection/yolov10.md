# YOLOv10

## Overview

YOLOv10 is the latest iteration in the YOLO series, introducing NMS-free training through consistent dual assignments and achieving state-of-the-art real-time object detection performance. It eliminates post-processing bottlenecks while maintaining or improving accuracy across all model sizes.

**Key Innovation**: NMS-free end-to-end training via dual label assignment (one-to-many for training, one-to-one for inference), removing the post-processing bottleneck while improving accuracy.

**Architecture Highlights**:
- NMS-free training and inference
- Consistent dual assignments
- Spatial-channel decoupled downsampling
- Rank-guided block design
- Large-kernel depth-wise convolutions
- Partial self-attention

**Performance**: 54.4 mAP on COCO at 80 FPS (YOLOv10-L), surpassing YOLOv8 and RT-DETR with lower latency.

## Theory

### The NMS Problem

Traditional YOLO models suffer from NMS (Non-Maximum Suppression) bottlenecks:
- **Extra latency**: NMS adds 2-4ms to inference time
- **Non-differentiable**: Cannot be trained end-to-end
- **Hyperparameter sensitive**: IoU threshold affects performance
- **Not parallelizable**: Sequential processing

YOLOv10 eliminates NMS entirely through end-to-end training.

### One-to-Many vs. One-to-One Assignment

**Traditional YOLO (One-to-Many)**:
- Each ground truth object matches multiple predictions
- Requires NMS to remove duplicates
- Rich supervision during training
- Post-processing needed at inference

**DETR-style (One-to-One)**:
- Each ground truth matches exactly one prediction
- No duplicates, no NMS needed
- Sparse supervision (slower convergence)
- End-to-end at inference

**YOLOv10 (Dual Assignment)**:
- One-to-many during training (rich supervision)
- One-to-one at inference (NMS-free)
- Best of both worlds!

### Consistent Dual Assignments

YOLOv10 uses two parallel assignment branches:

**One-to-Many Branch** (training only):
- Traditional YOLO assignment (3-5 predictions per GT)
- Provides rich supervision
- Helps model convergence
- Discarded at inference

**One-to-One Branch** (training + inference):
- Hungarian matching (like DETR)
- One prediction per GT
- Used for final predictions
- No NMS needed

Both branches share the same backbone and neck, only detection heads differ.

### Architecture Improvements

Beyond NMS-free training, YOLOv10 introduces several architectural improvements:

1. **Spatial-Channel Decoupled Downsampling**: Reduce parameters while maintaining performance
2. **Rank-Guided Block Design**: Adaptive block depth based on stage importance
3. **Large-Kernel Depth-wise Convolutions**: Expand receptive field efficiently
4. **Partial Self-Attention (PSA)**: Efficient global modeling

## Mathematical Formulation

### Overall Architecture

```
Image → Backbone → Neck → [One-to-Many Head] (training only)
                        → [One-to-One Head] (always)
```

### Dual Assignment Strategy

**One-to-Many Assignment** (TAL - Task Alignment Learning):

For each GT box g and anchor a:
```
# Alignment metric
t_align = s^α × u^β

where:
s = class prediction score
u = IoU(predicted_box, gt_box)
α, β = hyperparameters (typically 0.5, 6.0)
```

Select top-k anchors with highest t_align for each GT.

**One-to-One Assignment** (Hungarian Matching):

Cost matrix:
```
C_ij = λ_cls L_cls(p_i, c_j) + λ_box L_box(b_i, g_j)

where:
p_i = predicted class probabilities
b_i = predicted box
c_j = ground truth class
g_j = ground truth box
```

Use Hungarian algorithm to find optimal one-to-one assignment.

### Loss Function

Total loss combines both branches:

```
L = L_o2m + L_o2o

where:
L_o2m = λ_cls L_cls^o2m + λ_box L_box^o2m + λ_dfl L_dfl^o2m  (one-to-many)
L_o2o = λ_cls L_cls^o2o + λ_box L_box^o2o + λ_dfl L_dfl^o2o  (one-to-one)
```

**Classification Loss** (Binary Cross-Entropy):
```
L_cls = -Σ [y log(p) + (1-y) log(1-p)]
```

**Box Loss** (CIoU):
```
L_box = 1 - IoU + ρ²(b, b^gt) / c² + αv

where:
ρ = center distance
c = diagonal of smallest enclosing box
v = aspect ratio consistency
α = trade-off parameter
```

**Distribution Focal Loss** (DFL):
```
L_dfl = -Σ_i ((y_i+1 - y) log(p_i) + (y - y_i) log(p_{i+1}))
```

DFL models bounding box coordinates as a distribution over bins for better localization.

### Spatial-Channel Decoupled Downsampling

Traditional downsampling:
```
x → Conv(3x3, stride=2) → BatchNorm → SiLU
Parameters: C_in × C_out × 3 × 3
```

Decoupled downsampling:
```
x → [DepthWiseConv(3x3, stride=2) → BatchNorm → SiLU]  (spatial)
  → [Conv(1x1) → BatchNorm → SiLU]  (channel)

Parameters: C_in × 3 × 3 + C_in × C_out × 1 × 1
```

This reduces parameters by ~50% with minimal accuracy loss.

### Large-Kernel Depth-wise Conv

Replace 3×3 convs with 7×7 or 9×9 depth-wise convs:

```
# Standard
Conv(3x3) → Parameters: C × C × 3 × 3

# Large-kernel depth-wise
DepthWiseConv(7x7) → Conv(1x1)
Parameters: C × 7 × 7 + C × C × 1 × 1
```

Larger receptive field with fewer parameters.

### Partial Self-Attention (PSA)

Apply self-attention to only part of the channels:

```
x = Concat[x1, x2] where x1 has C//2 channels, x2 has C//2 channels

x1_attn = SelfAttention(x1)  # Apply attention
x2_pass = x2  # Pass through

output = Concat[x1_attn, x2_pass]
```

This reduces computation by 50% while maintaining global modeling.

### Rank-Guided Block Design

Analyze feature redundancy via intrinsic rank:

```
rank(F) = number of significant singular values
```

Stages with low rank (high redundancy) use fewer blocks.

Typical configuration:
- Stage 1: 3 blocks (low rank)
- Stage 2: 6 blocks (medium rank)
- Stage 3: 6 blocks (high rank)
- Stage 4: 3 blocks (medium rank)

## High-Level Intuition

### The NMS-Free Insight

Traditional pipeline:
```
Image → Model → 1000s of boxes → NMS → Final detections
        ↑                           ↑
   Fast, parallel              Slow, sequential
```

YOLOv10:
```
Image → Model → N boxes (one per object) → Done!
        ↑
   Fast, parallel, end-to-end
```

No post-processing needed!

### Why Dual Assignments Work

**Training phase**:
- One-to-many: Rich gradients, faster convergence
- One-to-one: Learns to produce unique predictions

**Inference phase**:
- Only one-to-one branch: Direct predictions, no NMS

Analogy: Train with multiple teachers (one-to-many), but test with independent thinking (one-to-one).

### Decoupled Downsampling Intuition

Spatial downsampling and channel transformation are independent:
- **Spatial**: Reduce resolution (H×W → H/2×W/2)
- **Channel**: Change feature dimension (C_in → C_out)

Why couple them in one 3×3 conv? Decouple for efficiency!

### Large-Kernel Benefits

Larger kernels = larger receptive field = better context:
- 3×3 conv: Sees 9 pixels
- 7×7 conv: Sees 49 pixels

But 7×7 is expensive! Solution: Depth-wise convolution (per-channel, not cross-channel).

### Partial Self-Attention Rationale

Full self-attention:
- Expensive: O(N² × C)
- Overkill for all features

Partial self-attention:
- Apply only to half the channels
- 50% computation savings
- Sufficient for global modeling

## Implementation Details

### Architecture Configuration

```python
# YOLOv10-N (Nano)
config_n = {
    'depth_multiple': 0.33,
    'width_multiple': 0.25,
    'max_channels': 1024,
}

# YOLOv10-S (Small)
config_s = {
    'depth_multiple': 0.33,
    'width_multiple': 0.50,
    'max_channels': 1024,
}

# YOLOv10-M (Medium)
config_m = {
    'depth_multiple': 0.67,
    'width_multiple': 0.75,
    'max_channels': 768,
}

# YOLOv10-L (Large)
config_l = {
    'depth_multiple': 1.00,
    'width_multiple': 1.00,
    'max_channels': 512,
}

# YOLOv10-X (Extra Large)
config_x = {
    'depth_multiple': 1.00,
    'width_multiple': 1.25,
    'max_channels': 512,
}
```

### Training Hyperparameters

```python
# Training setup
optimizer = SGD(
    lr=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    nesterov=True
)

# Cosine annealing with warmup
scheduler = CosineAnnealingLR(T_max=300, eta_min=0.0001)
warmup_epochs = 3
warmup_bias_lr = 0.1

epochs = 300  # 500 for best results
batch_size = 128  # Across GPUs
input_size = 640

# Loss weights
cls_weight = 0.5
box_weight = 7.5
dfl_weight = 1.5
```

### Data Augmentation

YOLOv10 uses heavy augmentation:

```python
# Geometric augmentations
augmentation = [
    Mosaic(p=1.0),  # 4-image mosaic
    MixUp(p=0.15),  # Blend two images
    RandomAffine(
        degrees=0.0,
        translate=0.1,
        scale=0.9,
        shear=0.0,
        perspective=0.0
    ),
    RandomFlip(p=0.5, direction='horizontal'),
]

# Color augmentations
color_aug = [
    HSVAugment(hgain=0.015, sgain=0.7, vgain=0.4),
    RandomGrayscale(p=0.01),
]

# Advanced augmentations
advanced_aug = [
    CopyPaste(p=0.5),  # Copy objects between images
    Cutout(p=0.5, n_holes=1, max_h_size=32, max_w_size=32),
]
```

### Close Mosaic and MixUp

Disable mosaic and mixup in final 20 epochs:

```python
if epoch >= epochs - 20:
    # Disable strong augmentation
    mosaic_enabled = False
    mixup_enabled = False
```

This allows model to adapt to real image distribution.

## Code Walkthrough

Implementation in `/Users/kevinyu/Projects/Nexus/nexus/models/cv/yolov10.py`.

### Decoupled Downsampling

```python
class DecoupledDownsample(nn.Module):
    """
    Spatial-channel decoupled downsampling.
    
    Reduces parameters while maintaining accuracy.
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Spatial downsampling (depth-wise)
        self.spatial_down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )
        
        # Channel transformation (point-wise)
        self.channel_trans = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.spatial_down(x)
        x = self.channel_trans(x)
        return x
```

### Large-Kernel Convolution

```python
class LargeKernelConv(nn.Module):
    """
    Large-kernel depth-wise convolution.
    
    Expands receptive field efficiently.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        
        # Large-kernel depth-wise
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size, padding=kernel_size//2,
            groups=in_channels
        )
        
        # Point-wise
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
```

### Partial Self-Attention

```python
class PartialSelfAttention(nn.Module):
    """
    Partial self-attention for efficient global modeling.
    
    Applies attention to half the channels, passes through the rest.
    """
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        
        assert dim % 2 == 0, "dim must be divisible by 2"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = (dim // 2) // num_heads
        
        # Attention for first half
        self.qkv = nn.Linear(dim // 2, (dim // 2) * 3)
        self.proj = nn.Linear(dim // 2, dim // 2)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Split channels
        x1, x2 = x.split(C // 2, dim=1)
        
        # Apply attention to x1
        x1 = x1.flatten(2).transpose(1, 2)  # [B, HW, C//2]
        
        qkv = self.qkv(x1).reshape(B, H*W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x1 = (attn @ v).transpose(1, 2).reshape(B, H*W, C//2)
        x1 = self.proj(x1)
        
        x1 = x1.transpose(1, 2).reshape(B, C//2, H, W)
        
        # Concatenate with x2 (pass-through)
        out = torch.cat([x1, x2], dim=1)
        
        return out
```

### Dual Head Design

```python
class DualDetectionHead(nn.Module):
    """
    Dual detection heads for YOLOv10.
    
    - One-to-Many head (training only)
    - One-to-One head (training + inference)
    """
    
    def __init__(self, num_classes=80, anchors_per_location=1):
        super().__init__()
        
        # One-to-Many head (traditional YOLO)
        self.o2m_head = nn.ModuleList([
            YOLOHead(num_classes, anchors=3) for _ in range(3)  # 3 scales
        ])
        
        # One-to-One head (NMS-free)
        self.o2o_head = nn.ModuleList([
            YOLOHead(num_classes, anchors=1) for _ in range(3)  # 3 scales, 1 anchor
        ])
    
    def forward(self, features):
        """
        Args:
            features: List of 3 feature maps from neck
            
        Returns:
            If training: (o2m_outputs, o2o_outputs)
            If inference: o2o_outputs only
        """
        if self.training:
            # Both heads during training
            o2m_outputs = [head(feat) for head, feat in zip(self.o2m_head, features)]
            o2o_outputs = [head(feat) for head, feat in zip(self.o2o_head, features)]
            return o2m_outputs, o2o_outputs
        else:
            # Only one-to-one head during inference
            o2o_outputs = [head(feat) for head, feat in zip(self.o2o_head, features)]
            return o2o_outputs
```

### One-to-One Assignment

```python
def one_to_one_assignment(predictions, targets):
    """
    Hungarian matching for one-to-one assignment.
    
    Args:
        predictions: List of (boxes, classes) for each image
        targets: List of ground truth boxes and classes
        
    Returns:
        matched_indices: List of (pred_idx, gt_idx) pairs for each image
    """
    from scipy.optimize import linear_sum_assignment
    
    all_indices = []
    
    for pred, tgt in zip(predictions, targets):
        pred_boxes, pred_classes = pred
        gt_boxes, gt_classes = tgt
        
        num_preds = len(pred_boxes)
        num_gts = len(gt_boxes)
        
        # Compute cost matrix
        # Classification cost
        pred_probs = pred_classes.softmax(-1)
        cost_class = -pred_probs[:, gt_classes]  # [num_preds, num_gts]
        
        # Box cost (CIoU)
        cost_box = 1 - compute_ciou(pred_boxes, gt_boxes)  # [num_preds, num_gts]
        
        # Total cost
        C = cost_class + 5.0 * cost_box
        
        # Hungarian algorithm
        pred_idx, gt_idx = linear_sum_assignment(C.cpu())
        
        all_indices.append((pred_idx, gt_idx))
    
    return all_indices
```

### Training Loop

```python
def train_epoch(model, dataloader, optimizer, epoch, cfg):
    """
    Train YOLOv10 for one epoch with dual assignments.
    """
    model.train()
    
    # Disable mosaic/mixup in final epochs
    if epoch >= cfg.epochs - 20:
        dataloader.dataset.mosaic = False
        dataloader.dataset.mixup = False
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.cuda()
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        
        # Forward
        o2m_outputs, o2o_outputs = model(images)
        
        # One-to-many loss
        o2m_loss = compute_yolo_loss(
            o2m_outputs, targets,
            assignment='one-to-many'
        )
        
        # One-to-one loss
        o2o_loss = compute_yolo_loss(
            o2o_outputs, targets,
            assignment='one-to-one'
        )
        
        # Total loss
        loss = o2m_loss + o2o_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: "
                  f"O2M Loss: {o2m_loss.item():.4f}, "
                  f"O2O Loss: {o2o_loss.item():.4f}")
```

### NMS-Free Inference

```python
@torch.no_grad()
def detect_objects(model, image, conf_threshold=0.25):
    """
    NMS-free inference with YOLOv10.
    
    No post-processing needed!
    """
    model.eval()
    
    # Preprocess
    image = preprocess(image)
    image = image.unsqueeze(0).cuda()
    
    # Forward (only one-to-one head)
    outputs = model(image)
    
    # Parse outputs
    boxes, classes, scores = parse_outputs(outputs)
    
    # Filter by confidence (no NMS needed!)
    keep = scores > conf_threshold
    boxes = boxes[keep]
    classes = classes[keep]
    scores = scores[keep]
    
    return boxes, classes, scores
```

## Optimization Tricks

### 1. Gradient Accumulation

Simulate larger batch size:

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. EMA (Exponential Moving Average)

Smooth model weights for better generalization:

```python
ema = ModelEMA(model, decay=0.9999)

# After each step
ema.update(model)

# Use EMA for evaluation
ema_model = ema.module
```

### 3. Multi-Scale Training

Train with various input sizes:

```python
scales = [480, 512, 544, 576, 608, 640, 672, 704, 736]
img_size = random.choice(scales)
images = F.interpolate(images, size=img_size)
```

### 4. Knowledge Distillation

Distill from larger YOLOv10 model:

```python
# Teacher predictions
with torch.no_grad():
    teacher_outputs = teacher_model(images)

# Student predictions
student_outputs = student_model(images)

# Distillation loss
distill_loss = kl_divergence(student_outputs, teacher_outputs)

total_loss = task_loss + 0.5 * distill_loss
```

### 5. Mixed Precision Training

Use FP16 for faster training:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 6. Model Pruning

Prune less important filters:

```python
# Compute importance scores
importance = compute_filter_importance(model)

# Prune low-importance filters
pruned_model = prune_filters(model, importance, prune_ratio=0.3)
```

### 7. Quantization

INT8 quantization for deployment:

```python
# Post-training quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Conv2d, nn.Linear},
    dtype=torch.qint8
)
```

## Experiments & Results

### COCO Detection Results

**YOLOv10-N** (Nano):
- mAP: 38.5
- AP50: 53.8
- AP75: 41.3
- Parameters: 2.3M
- FLOPs: 6.7G
- Latency: 1.79ms
- FPS: 559

**YOLOv10-S** (Small):
- mAP: 46.3
- AP50: 62.9
- AP75: 50.1
- Parameters: 7.2M
- FLOPs: 21.6G
- Latency: 2.49ms
- FPS: 402

**YOLOv10-M** (Medium):
- mAP: 51.1
- AP50: 68.1
- AP75: 55.4
- Parameters: 15.4M
- FLOPs: 59.1G
- Latency: 4.74ms
- FPS: 211

**YOLOv10-L** (Large):
- mAP: 54.4
- AP50: 71.5
- AP75: 58.8
- Parameters: 24.4M
- FLOPs: 120.3G
- Latency: 7.28ms
- FPS: 137

**YOLOv10-X** (Extra Large):
- mAP: 56.2
- AP50: 73.1
- AP75: 60.7
- Parameters: 29.5M
- FLOPs: 160.4G
- Latency: 10.70ms
- FPS: 93

### Comparison with YOLOv8

| Model | mAP | Params | FLOPs | Latency | FPS |
|-------|-----|--------|-------|---------|-----|
| YOLOv8-N | 37.3 | 3.2M | 8.7G | 1.93ms | 518 |
| YOLOv10-N | 38.5 | 2.3M | 6.7G | 1.79ms | 559 |
| YOLOv8-S | 44.9 | 11.2M | 28.6G | 2.84ms | 352 |
| YOLOv10-S | 46.3 | 7.2M | 21.6G | 2.49ms | 402 |
| YOLOv8-M | 50.2 | 25.9M | 78.9G | 5.67ms | 176 |
| YOLOv10-M | 51.1 | 15.4M | 59.1G | 4.74ms | 211 |
| YOLOv8-L | 52.9 | 43.7M | 165.2G | 8.41ms | 119 |
| YOLOv10-L | 54.4 | 24.4M | 120.3G | 7.28ms | 137 |

YOLOv10 achieves higher accuracy with fewer parameters and lower latency!

### Comparison with RT-DETR

| Model | mAP | Params | FLOPs | Latency |
|-------|-----|--------|-------|---------|
| RT-DETR-R18 | 46.5 | 20M | 60G | 4.58ms |
| YOLOv10-S | 46.3 | 7.2M | 21.6G | 2.49ms |
| RT-DETR-R50 | 53.1 | 42M | 136G | 9.20ms |
| YOLOv10-L | 54.4 | 24.4M | 120.3G | 7.28ms |

YOLOv10 is more efficient than RT-DETR.

### Ablation Studies

**Effect of Dual Assignments**:
- One-to-many only (with NMS): 52.8 mAP, 8.1ms
- One-to-one only: 50.2 mAP, 7.0ms
- Dual (our approach): 54.4 mAP, 7.3ms

**Effect of NMS-Free Design**:
- With NMS: 54.1 mAP, 8.7ms latency
- Without NMS (YOLOv10): 54.4 mAP, 7.3ms latency

**Effect of Architectural Improvements**:
- Baseline: 52.1 mAP
- + Decoupled downsampling: 52.9 mAP
- + Large-kernel conv: 53.6 mAP
- + Partial self-attention: 54.4 mAP

**Effect of Model Size**:
- Depth 0.33, Width 0.25 (N): 38.5 mAP
- Depth 0.33, Width 0.50 (S): 46.3 mAP
- Depth 0.67, Width 0.75 (M): 51.1 mAP
- Depth 1.00, Width 1.00 (L): 54.4 mAP

### Latency Breakdown

On T4 GPU, YOLOv10-L (640×640):

| Component | Time (ms) | % Total |
|-----------|-----------|---------|
| Backbone | 2.8 | 38% |
| Neck | 1.9 | 26% |
| One-to-one head | 2.6 | 36% |
| Post-processing | 0.0 | 0% |
| **Total** | **7.3** | **100%** |

Note: No NMS latency!

### Comparison with One-Stage Detectors

| Model | Type | mAP | FPS | NMS Required |
|-------|------|-----|-----|--------------|
| YOLOv5-L | Anchor-based | 49.0 | 108 | Yes |
| YOLOv6-L | Anchor-free | 52.8 | 116 | Yes |
| YOLOv7-L | Anchor-based | 51.4 | 94 | Yes |
| YOLOv8-L | Anchor-free | 52.9 | 119 | Yes |
| YOLOv9-E | Anchor-free | 55.6 | 82 | Yes |
| **YOLOv10-L** | **NMS-free** | **54.4** | **137** | **No** |

YOLOv10 eliminates NMS bottleneck!

## Common Pitfalls

### 1. Not Disabling Mosaic in Final Epochs

Problem: Strong augmentation hurts final convergence.

Solution: Disable mosaic/mixup in last 20 epochs.

```python
if epoch >= total_epochs - 20:
    train_loader.dataset.mosaic = False
    train_loader.dataset.mixup = False
```

### 2. Using NMS on YOLOv10

Problem: YOLOv10 is designed to be NMS-free.

Solution: Don't apply NMS! Just filter by confidence.

```python
# Wrong
boxes = nms(predictions, iou_threshold=0.45)

# Correct
boxes = predictions[predictions[:, 4] > conf_threshold]  # No NMS!
```

### 3. Only Training One-to-One Head

Problem: Missing rich supervision from one-to-many branch.

Solution: Always train both heads.

```python
# Wrong
loss = one_to_one_loss(o2o_outputs, targets)

# Correct
loss = one_to_many_loss(o2m_outputs, targets) + one_to_one_loss(o2o_outputs, targets)
```

### 4. Using One-to-Many Head at Inference

Problem: One-to-many produces duplicates.

Solution: Only use one-to-one head at inference.

```python
def forward(self, x):
    if self.training:
        return self.o2m_head(x), self.o2o_head(x)
    else:
        return self.o2o_head(x)  # Only one-to-one!
```

### 5. Small Batch Size

Problem: Batch size < 16 hurts batch norm statistics.

Solution: Use larger batch size or gradient accumulation.

```python
# Gradient accumulation for effective batch size 64
effective_batch_size = 64
accumulation_steps = effective_batch_size // actual_batch_size
```

### 6. Not Using EMA

Problem: Model weights are noisy without EMA.

Solution: Always use EMA.

```python
ema = ModelEMA(model, decay=0.9999)

# Update after each batch
ema.update(model)

# Use EMA for evaluation
evaluate(ema.module)
```

### 7. Wrong Input Normalization

Problem: Using ImageNet normalization for YOLO.

Solution: Normalize to [0, 1].

```python
# Wrong
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

# Correct
images = images.float() / 255.0  # Just rescale to [0, 1]
```

### 8. Not Warming Up Learning Rate

Problem: Large learning rate at start causes instability.

Solution: Use learning rate warmup.

```python
if epoch < warmup_epochs:
    lr = base_lr * (epoch / warmup_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

## References

### Original Paper

- **YOLOv10: Real-Time End-to-End Object Detection**
  - Authors: Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding
  - Conference: Preprint 2024
  - Paper: https://arxiv.org/abs/2405.14458
  - Code: https://github.com/THU-MIG/yolov10

### Related YOLO Papers

- **YOLOv1: You Only Look Once: Unified, Real-Time Object Detection**
  - CVPR 2016
  - https://arxiv.org/abs/1506.02640

- **YOLOv3: An Incremental Improvement**
  - 2018
  - https://arxiv.org/abs/1804.02767

- **YOLOv4: Optimal Speed and Accuracy of Object Detection**
  - 2020
  - https://arxiv.org/abs/2004.10934

- **YOLOv5**
  - Ultralytics 2020
  - https://github.com/ultralytics/yolov5

- **YOLOv6: A Single-Stage Object Detection Framework**
  - 2022
  - https://arxiv.org/abs/2209.02976

- **YOLOv7: Trainable Bag-of-Freebies**
  - CVPR 2023
  - https://arxiv.org/abs/2207.02696

- **YOLOv8**
  - Ultralytics 2023
  - https://github.com/ultralytics/ultralytics

- **YOLOv9: Learning What You Want to Learn**
  - 2024
  - https://arxiv.org/abs/2402.13616

### Related Work

- **DETR: End-to-End Object Detection with Transformers**
  - ECCV 2020
  - https://arxiv.org/abs/2005.12872
  - Inspiration for one-to-one assignment

- **RT-DETR: Real-Time DEtection TRansformer**
  - CVPR 2024
  - https://arxiv.org/abs/2304.08069
  - Comparison baseline

### Implementation Resources

- Official PyTorch implementation: https://github.com/THU-MIG/yolov10
- Ultralytics integration: https://github.com/ultralytics/ultralytics
- ONNX export: https://github.com/THU-MIG/yolov10/blob/main/docs/export.md
- Local implementation: `/Users/kevinyu/Projects/Nexus/nexus/models/cv/yolov10.py`

### Deployment Resources

- **TensorRT**: https://github.com/triple-Mu/YOLOv10-TensorRT
- **OpenVINO**: https://docs.openvino.ai/latest/notebooks/yolov10-optimization-with-output.html
- **NCNN**: https://github.com/Daming-TF/YOLOv10-ncnn
- **CoreML**: Export via Ultralytics

### Tutorials and Blogs

- "YOLOv10 Explained": https://blog.roboflow.com/yolov10/
- "NMS-Free Object Detection": https://medium.com/@yolov10/nms-free-detection
- Ultralytics YOLOv10 guide: https://docs.ultralytics.com/models/yolov10/

### Applications

- **Autonomous Driving**: Real-time vehicle/pedestrian detection
- **Robotics**: Object detection for manipulation
- **Surveillance**: Real-time security monitoring
- **Retail**: Inventory tracking, checkout automation
- **Agriculture**: Crop/pest detection
- **Manufacturing**: Quality inspection

### Benchmarks

- **COCO**: Primary benchmark (test-dev)
- **Objects365**: Large-scale pre-training
- **Roboflow100**: Domain-specific evaluation
- **BDD100K**: Autonomous driving
