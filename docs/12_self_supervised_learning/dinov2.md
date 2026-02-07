# DINOv2: Learning Robust Visual Features without Supervision

## Overview & Motivation

DINOv2 (Distillation with NO labels version 2) is a state-of-the-art self-distillation method that learns robust visual features through self-supervised learning at scale. Building upon the original DINO framework, DINOv2 trains on a carefully curated dataset of 142M images and achieves state-of-the-art performance on many downstream tasks with zero-shot transfer or simple linear probing, without ever seeing task-specific labels during pretraining.

### Key Innovation

**Self-distillation with image-level and patch-level objectives at unprecedented scale**:
- **Student-Teacher Framework**: Student predicts teacher's output (softmax distribution)
- **Teacher Architecture**: Exponential Moving Average (EMA) of student, with centering and sharpening mechanisms
- **Multi-crop Training**: Global views (224×224) + local views (96×96) for multi-scale learning
- **Curated Pre-training Data**: 142M carefully filtered images from diverse sources
- **Patch-level Learning**: Unlike classification models, learns features for every image patch
- **No Labels Required**: Completely self-supervised, no human annotations needed

### Why DINOv2 Matters

Traditional supervised learning requires millions of labeled images. DINOv2 shows that:
1. **Self-supervision at scale** can match or exceed supervised learning
2. **Data curation** matters as much as model architecture
3. **Universal features** can be learned without task-specific labels
4. **Zero-shot transfer** is possible for many computer vision tasks

## Theoretical Background

### Self-Supervised Learning Paradigm

DINOv2 belongs to the family of **knowledge distillation** methods where a student network learns to match the output of a teacher network. The key insight is that the teacher is not a separate pre-trained model, but an exponential moving average of the student itself.

### The Self-Distillation Framework

The core principle is **self-distillation without labels**:

```
Goal: Learn features that are invariant to image transformations
Method: Different views of same image should have similar representations
Challenge: Prevent collapse (all images mapping to same representation)
Solution: Centering + sharpening + high-dimensional output space
```

### Why This Works

**Intuition**: If the model can predict how one view of an image relates to another view, it must understand the semantic content of the image, not just low-level statistics.

**Mathematical Justification**: The combination of:
1. **Cross-entropy loss**: Encourages prediction accuracy
2. **Centering**: Prevents mode collapse
3. **Sharpening**: Encourages confident, peaked distributions
4. **High-dimensional output**: Allows rich feature representations

Together, these components create a training signal that encourages learning semantic features.

## Mathematical Formulation

### Loss Function

The primary loss is the cross-entropy between student and teacher distributions:

$$\mathcal{L} = -\sum_{x \in \{x_1^g, x_2^g\}} \sum_{x' \in V, x' \neq x} P_t(x) \log P_s(x')$$

Where:
- $V = \{x_1^g, x_2^g, x_1^l, ..., x_8^l\}$ is the set of all crops
- $x_1^g, x_2^g$ are the two global crops
- $x_1^l, ..., x_8^l$ are the local crops
- $P_t$ is the teacher's output distribution
- $P_s$ is the student's output distribution

**Key Property**: Student learns from all crops, teacher only processes global crops.

### Student Output Processing

The student processes all views through a standard softmax with temperature:

$$P_s(x) = \frac{\exp(g_s(x) / \tau_s)}{\sum_{k=1}^K \exp(g_{s,k}(x) / \tau_s)}$$

Where:
- $g_s(x)$ is the student's output logits
- $\tau_s = 0.1$ is the student temperature (fixed)
- $K$ is the output dimension (typically 65,536)

### Teacher Output Processing with Centering and Sharpening

The teacher processing involves three critical steps:

```python
# 1. Centering: subtract running mean
g_t_centered = g_t - c

# 2. Temperature scaling (sharpening)
g_t_sharp = g_t_centered / τ_t

# 3. Softmax
P_t = softmax(g_t_sharp)
```

Mathematically:

$$P_t(x) = \frac{\exp((g_t(x) - c) / \tau_t)}{\sum_{k=1}^K \exp((g_{t,k}(x) - c_k) / \tau_t)}$$

Where:
- $c$ is the center (running mean of teacher outputs)
- $\tau_t \in [0.04, 0.07]$ is the teacher temperature (scheduled)

### Centering Mechanism

Centering prevents collapse by ensuring the model cannot trivially output the same value for all images:

$$c \leftarrow m \cdot c + (1-m) \cdot \frac{1}{B} \sum_{i=1}^B g_t(x_i)$$

Where:
- $m = 0.9$ is the centering momentum
- $B$ is the batch size
- The center $c$ is updated using exponential moving average

**Why This Works**: By subtracting the mean, the model is forced to output diverse representations. If all outputs were identical, subtracting the center would zero them out, leading to uniform (uninformative) softmax distributions and high loss.

### Temperature Scheduling

Teacher temperature increases during training to gradually sharpen the distribution:

$$\tau_t(k) = \tau_{t,\text{start}} + (\tau_{t,\text{end}} - \tau_{t,\text{start}}) \cdot \frac{k}{K}$$

Where:
- $\tau_{t,\text{start}} = 0.04$
- $\tau_{t,\text{end}} = 0.07$
- $k$ is the current epoch
- $K$ is total epochs

**Intuition**: Start with very sharp distributions (low temperature) to establish strong gradients, then gradually increase temperature to allow more nuanced predictions.

### Multi-Crop Training Strategy

Multi-crop training exposes the model to both global context and local details:

**Global Crops** (2 crops, 224×224):
- Capture overall scene structure
- Provide global context
- Only processed by teacher

**Local Crops** (8 crops, 96×96):
- Capture fine-grained details
- Force learning of local features
- Only processed by student

$$\mathcal{L}_{\text{total}} = \sum_{i=1}^{2} \sum_{j \neq i}^{10} \mathcal{L}(P_t(x_i^g), P_s(x_j))$$

This creates $2 \times 9 = 18$ cross-entropy terms per image.

### EMA Teacher Update

The teacher is updated via exponential moving average:

$$\theta_t \leftarrow \lambda \theta_t + (1-\lambda) \theta_s$$

Where:
- $\theta_s$ are student parameters
- $\theta_t$ are teacher parameters
- $\lambda$ is the EMA coefficient

**EMA Schedule** (cosine schedule from 0.996 to 1.0):

$$\lambda(k) = 1 - (1 - \lambda_{\text{base}}) \cdot \frac{1 + \cos(\pi k / K)}{2}$$

Starting at $\lambda_{\text{base}} = 0.996$ and approaching 1.0 as training progresses.

## Architecture Details

### Vision Transformer Backbone

DINOv2 uses standard Vision Transformer (ViT) architectures:

**ViT-S/14** (Small):
- Parameters: 21M
- Embedding dimension: 384
- Layers: 12
- Attention heads: 6
- Patch size: 14×14

**ViT-B/14** (Base):
- Parameters: 86M
- Embedding dimension: 768
- Layers: 12
- Attention heads: 12
- Patch size: 14×14

**ViT-L/14** (Large):
- Parameters: 304M
- Embedding dimension: 1024
- Layers: 24
- Attention heads: 16
- Patch size: 14×14

**ViT-g/14** (Giant):
- Parameters: 1.1B
- Embedding dimension: 1536
- Layers: 40
- Attention heads: 24
- Patch size: 14×14

### Projection Head

After the ViT backbone, DINOv2 uses a 3-layer projection head:

```
ViT Output (d dimensions)
    ↓
Linear(d → 2048) + GELU
    ↓
Linear(2048 → 2048) + GELU
    ↓
Linear(2048 → 65536)
    ↓
L2 Normalization
```

The final dimension (65,536) is much larger than typical classification heads, allowing for rich, high-dimensional representations.

### Patch-Level vs Image-Level Features

DINOv2 produces features at two granularities:

**Image-level** (CLS token):
```python
features_image = model(image)[:, 0]  # CLS token, shape (B, D)
```

**Patch-level** (all tokens):
```python
features_patches = model(image)[:, 1:]  # All patches, shape (B, N, D)
```

This enables both:
- **Image classification** (use CLS token)
- **Dense prediction tasks** (use patch tokens for segmentation, detection)

## Implementation Details

### Complete Nexus Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from nexus.models.vision import VisionTransformer
from nexus.models.ssl.base import BaseSSL

class DINOv2Head(nn.Module):
    """3-layer projection head for DINOv2."""

    def __init__(self, in_dim, hidden_dim=2048, out_dim=65536):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layers(x)
        return F.normalize(x, dim=-1, p=2)  # L2 normalize


class DINOv2(BaseSSL):
    """DINOv2 self-supervised learning model."""

    def __init__(
        self,
        backbone="vit_base_patch14",
        img_size=224,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        out_dim=65536,
        student_temp=0.1,
        teacher_temp_schedule=(0.04, 0.07),
        warmup_teacher_temp_epochs=30,
        center_momentum=0.9,
        ema_momentum_schedule=(0.996, 1.0),
        local_crops_number=8,
        local_crops_scale=(0.05, 0.4),
        global_crops_scale=(0.4, 1.0),
    ):
        super().__init__()

        # Student network
        self.student_backbone = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.student_head = DINOv2Head(embed_dim, out_dim=out_dim)

        # Teacher network (same architecture, no gradients)
        self.teacher_backbone = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.teacher_head = DINOv2Head(embed_dim, out_dim=out_dim)

        # Initialize teacher with student weights
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        # Teacher has no gradients
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        # Training parameters
        self.student_temp = student_temp
        self.teacher_temp_start, self.teacher_temp_end = teacher_temp_schedule
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.center_momentum = center_momentum
        self.ema_start, self.ema_end = ema_momentum_schedule

        # Center (running mean of teacher outputs)
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Multi-crop parameters
        self.local_crops_number = local_crops_number
        self.local_crops_scale = local_crops_scale
        self.global_crops_scale = global_crops_scale

        self.epoch = 0

    def get_teacher_temp(self, epoch):
        """Get teacher temperature with warmup."""
        if epoch < self.warmup_teacher_temp_epochs:
            return self.teacher_temp_start
        else:
            progress = (epoch - self.warmup_teacher_temp_epochs) / \
                      (self.total_epochs - self.warmup_teacher_temp_epochs)
            return self.teacher_temp_start + \
                   (self.teacher_temp_end - self.teacher_temp_start) * progress

    def get_ema_momentum(self, epoch):
        """Get EMA momentum with cosine schedule."""
        progress = epoch / self.total_epochs
        return self.ema_end - (self.ema_end - self.ema_start) * \
               0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    @torch.no_grad()
    def update_teacher(self):
        """Update teacher with EMA of student."""
        m = self.get_ema_momentum(self.epoch)

        for param_s, param_t in zip(
            self.student_backbone.parameters(),
            self.teacher_backbone.parameters()
        ):
            param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)

        for param_s, param_t in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters()
        ):
            param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center with batch mean of teacher outputs."""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + \
                      batch_center * (1 - self.center_momentum)

    def forward(self, images):
        """
        Args:
            images: List of crops [global_1, global_2, local_1, ..., local_8]
                   Each crop shape: (B, 3, H, W)

        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics for logging
        """
        # Split crops
        global_crops = images[:2]
        local_crops = images[2:]

        # Student forward on all crops
        student_outputs = []
        for crop in global_crops + local_crops:
            features = self.student_backbone(crop)[:, 0]  # CLS token
            output = self.student_head(features)
            student_outputs.append(output)

        # Teacher forward on global crops only
        with torch.no_grad():
            teacher_outputs = []
            teacher_temp = self.get_teacher_temp(self.epoch)

            for crop in global_crops:
                features = self.teacher_backbone(crop)[:, 0]
                output = self.teacher_head(features)

                # Center and sharpen
                output = (output - self.center) / teacher_temp
                teacher_outputs.append(output)

            # Update center with mean of teacher outputs
            all_teacher = torch.cat(teacher_outputs)
            self.update_center(all_teacher)

        # Compute loss: student predicts teacher
        loss = 0
        n_loss_terms = 0

        for i, teacher_out in enumerate(teacher_outputs):
            # Teacher distribution
            teacher_probs = F.softmax(teacher_out, dim=-1)

            # Student predicts teacher from all other crops
            for j, student_out in enumerate(student_outputs):
                if i == j:  # Skip same crop
                    continue

                student_out = student_out / self.student_temp
                student_log_probs = F.log_softmax(student_out, dim=-1)

                loss += -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()
                n_loss_terms += 1

        loss /= n_loss_terms

        # Metrics
        metrics = {
            "loss": loss.item(),
            "teacher_temp": self.get_teacher_temp(self.epoch),
            "ema_momentum": self.get_ema_momentum(self.epoch).item(),
            "center_norm": self.center.norm().item(),
        }

        return loss, metrics


class MultiCropAugmentation:
    """Multi-crop data augmentation for DINOv2."""

    def __init__(
        self,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=8,
        global_size=224,
        local_size=96,
    ):
        from torchvision import transforms

        # Global crop augmentation
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                global_size,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

        # Local crop augmentation
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                local_size,
                scale=local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

        self.local_crops_number = local_crops_number

    def __call__(self, image):
        """
        Args:
            image: PIL Image

        Returns:
            List of crops [global_1, global_2, local_1, ..., local_N]
        """
        crops = []

        # Two global crops
        crops.append(self.global_transform(image))
        crops.append(self.global_transform(image))

        # N local crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))

        return crops
```

### Training Loop Example

```python
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from nexus.models.ssl import DINOv2, MultiCropAugmentation

# Configuration
config = {
    "backbone": "vit_base_patch14",
    "img_size": 224,
    "patch_size": 14,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "out_dim": 65536,
    "batch_size": 1024,  # Effective batch size across GPUs
    "epochs": 100,
    "lr": 0.0005,
    "weight_decay": 0.04,
    "warmup_epochs": 10,
}

# Model
model = DINOv2(
    backbone=config["backbone"],
    img_size=config["img_size"],
    patch_size=config["patch_size"],
    embed_dim=config["embed_dim"],
    depth=config["depth"],
    num_heads=config["num_heads"],
    out_dim=config["out_dim"],
)
model.total_epochs = config["epochs"]
model = model.cuda()

# Data
transform = MultiCropAugmentation(
    global_crops_scale=(0.4, 1.0),
    local_crops_scale=(0.05, 0.4),
    local_crops_number=8,
    global_size=224,
    local_size=96,
)
dataset = ImageFolder("/path/to/imagenet", transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    drop_last=True,
)

# Optimizer with layer-wise learning rate decay
param_groups = []
for name, param in model.student_backbone.named_parameters():
    # Get layer index
    if "blocks" in name:
        layer_id = int(name.split(".")[1])
    else:
        layer_id = -1

    # Layer-wise LR decay
    lr_scale = 0.75 ** (12 - layer_id) if layer_id >= 0 else 1.0
    param_groups.append({
        "params": param,
        "lr": config["lr"] * lr_scale,
        "weight_decay": config["weight_decay"],
    })

optimizer = torch.optim.AdamW(param_groups, lr=config["lr"])

# Learning rate schedule
def get_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        return base_lr * epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

# Training loop
for epoch in range(config["epochs"]):
    model.epoch = epoch
    model.train()

    # Update learning rate
    lr = get_lr(epoch, config["warmup_epochs"], config["epochs"], config["lr"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * (param_group.get("lr_scale", 1.0))

    for batch_idx, (images, _) in enumerate(dataloader):
        # images is a list of 10 crops per image
        images = [crop.cuda() for crop in images]

        # Forward pass
        loss, metrics = model(images)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.student_backbone.parameters(), 3.0)
        torch.nn.utils.clip_grad_norm_(model.student_head.parameters(), 3.0)

        optimizer.step()

        # Update teacher with EMA
        model.update_teacher()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {metrics['loss']:.4f}, "
                  f"Teacher Temp: {metrics['teacher_temp']:.4f}, "
                  f"EMA: {metrics['ema_momentum']:.4f}")

    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save({
            "epoch": epoch,
            "student_backbone": model.student_backbone.state_dict(),
            "student_head": model.student_head.state_dict(),
            "teacher_backbone": model.teacher_backbone.state_dict(),
            "teacher_head": model.teacher_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "center": model.center,
        }, f"dinov2_checkpoint_epoch_{epoch}.pth")
```

### Distributed Training

DINOv2 requires distributed training for large batch sizes:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

local_rank = setup_distributed()

# Model
model = DINOv2(...).cuda()
model = DDP(model, device_ids=[local_rank])

# Data with distributed sampler
sampler = torch.utils.data.DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
)
dataloader = DataLoader(dataset, sampler=sampler, ...)

# Training loop
for epoch in range(epochs):
    sampler.set_epoch(epoch)
    for images, _ in dataloader:
        # Training step
        ...
```

## Optimization Tricks

### 1. Temperature Scheduling

Teacher temperature critically affects learning dynamics:

```python
def get_teacher_temperature(epoch, total_epochs, warmup_epochs=30):
    """Warm up teacher temperature from 0.04 to 0.07."""
    if epoch < warmup_epochs:
        return 0.04

    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return 0.04 + (0.07 - 0.04) * progress
```

**Why this works**:
- Low temperature (0.04) creates sharp, confident distributions
- Provides strong gradients early in training
- Gradual increase allows more nuanced predictions later

### 2. Centering Momentum

```python
# High momentum for stability
center_momentum = 0.9

# Update rule
self.center = self.center * center_momentum + \
              batch_center * (1 - center_momentum)
```

**Tuning advice**:
- Higher momentum (0.95-0.99): More stable, slower adaptation
- Lower momentum (0.8-0.9): Faster adaptation, potentially less stable
- DINOv2 uses 0.9 as a good balance

### 3. EMA Momentum Schedule

```python
def cosine_ema_schedule(epoch, total_epochs, start=0.996, end=1.0):
    """Cosine schedule from start to end."""
    progress = epoch / total_epochs
    return end - (end - start) * 0.5 * (1 + math.cos(math.pi * progress))
```

**Why cosine schedule**:
- Start at 0.996: Teacher tracks student relatively closely
- End at 1.0: Teacher becomes nearly frozen, providing stable targets
- Smooth transition prevents disruption

### 4. Layer-wise Learning Rate Decay

```python
def get_layer_lr_scale(layer_id, num_layers, decay_rate=0.75):
    """Earlier layers get smaller learning rates."""
    return decay_rate ** (num_layers - layer_id - 1)

# Example for 12-layer ViT
for layer_id in range(12):
    lr_scale = get_layer_lr_scale(layer_id, 12, decay_rate=0.75)
    print(f"Layer {layer_id}: {lr_scale:.4f}x base LR")
```

**Rationale**:
- Earlier layers learn more general features (should change slowly)
- Later layers learn more task-specific features (can change faster)
- Improves training stability and final performance

### 5. Gradient Clipping

```python
# Clip gradients to prevent instability
max_norm = 3.0
torch.nn.utils.clip_grad_norm_(
    model.student_backbone.parameters(),
    max_norm=max_norm
)
```

### 6. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, _ in dataloader:
    with autocast():
        loss, metrics = model(images)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)

    scaler.step(optimizer)
    scaler.update()
```

**Benefits**:
- 2-3× faster training
- Enables larger batch sizes
- Minimal accuracy loss with proper clipping

### 7. Multi-Crop Strategy Details

```python
# Global crops: capture overall structure
global_crops_scale = (0.4, 1.0)  # 40% to 100% of image
global_size = 224

# Local crops: capture fine details
local_crops_scale = (0.05, 0.4)  # 5% to 40% of image
local_size = 96
local_crops_number = 8
```

**Design rationale**:
- Global crops: Provide context, processed by teacher
- Local crops: Force learning of local features, processed by student
- Asymmetry: Student sees more views, encourages generalization

### 8. Data Augmentation Details

```python
from torchvision import transforms

# Strong augmentation for self-supervised learning
augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                          saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
```

## Experiments & Results

### ImageNet-1K Linear Probing

Training a linear classifier on frozen features:

| Model | Params | Resolution | Top-1 Acc | Training Data |
|-------|--------|-----------|-----------|---------------|
| DINOv2-S/14 | 21M | 518 | 79.0% | 142M images |
| DINOv2-B/14 | 86M | 518 | 82.1% | 142M images |
| DINOv2-L/14 | 304M | 518 | 82.8% | 142M images |
| **DINOv2-g/14** | **1.1B** | **518** | **83.5%** | **142M images** |

**Comparison to supervised**:
- Supervised ViT-H/14: 88.5% (trained on labeled ImageNet-22K + ImageNet-1K)
- DINOv2-g/14: 83.5% (no labels whatsoever!)

**Key insights**:
- Self-supervised learning at scale approaches supervised performance
- Larger models benefit more from self-supervised learning
- Higher resolution (518 vs 224) significantly improves performance

### Zero-Shot Transfer (k-NN Classification)

Using DINOv2 features directly without any fine-tuning:

| Task | Dataset | DINOv2-g/14 | Supervised ViT-H/14 |
|------|---------|-------------|---------------------|
| Image Classification | ImageNet-1K | 81.2% | 78.3% |
| Fine-grained | iNaturalist | 75.9% | 71.2% |
| Scene Recognition | Places205 | 63.8% | 61.5% |

**Remarkable finding**: Zero-shot DINOv2 features outperform supervised features for k-NN!

### Dense Prediction Tasks

Using patch-level features for pixel-wise prediction:

**Semantic Segmentation (ADE20K)**:
| Method | Backbone | mIoU | Training |
|--------|----------|------|----------|
| Supervised | ViT-L/14 | 45.8 | ImageNet labels |
| MAE | ViT-L/14 | 46.4 | Self-supervised |
| **DINOv2** | **ViT-L/14** | **47.2** | **Self-supervised** |

**Depth Estimation (NYUv2)**:
| Method | Backbone | RMSE | δ₁ |
|--------|----------|------|-----|
| Supervised | ViT-L/14 | 0.089 | 88.2% |
| **DINOv2** | **ViT-L/14** | **0.077** | **91.3%** |

### Instance Segmentation (COCO)

Fine-tuning Mask R-CNN with DINOv2 backbone:

| Backbone | AP (box) | AP (mask) |
|----------|----------|-----------|
| ResNet-50 (supervised) | 38.2 | 34.5 |
| ViT-B/14 (MAE) | 42.1 | 38.6 |
| **ViT-B/14 (DINOv2)** | **43.7** | **39.8** |

### Ablation Studies

**Effect of Dataset Size**:
| Training Images | Top-1 (Linear) |
|----------------|----------------|
| 1M | 75.3% |
| 10M | 78.9% |
| 50M | 81.2% |
| **142M** | **82.1%** |

**Scaling matters**: More diverse, curated data significantly improves performance.

**Effect of Multi-Crop**:
| Configuration | Top-1 (Linear) |
|---------------|----------------|
| 2 global only | 79.1% |
| 2 global + 4 local | 80.8% |
| **2 global + 8 local** | **82.1%** |
| 2 global + 12 local | 82.0% |

**Sweet spot**: 8 local crops balances performance and compute.

**Effect of Teacher Temperature**:
| Schedule | Top-1 (Linear) |
|----------|----------------|
| Fixed 0.04 | 80.3% |
| Fixed 0.07 | 79.8% |
| **0.04 → 0.07** | **82.1%** |

**Temperature scheduling is crucial** for optimal performance.

**Effect of Output Dimension**:
| Output Dim | Top-1 (Linear) |
|------------|----------------|
| 256 | 77.2% |
| 2048 | 79.8% |
| 8192 | 81.3% |
| **65536** | **82.1%** |

**High-dimensional output space** allows richer representations.

### Comparison to Other Methods

| Method | Type | ImageNet Top-1 | Dense Tasks |
|--------|------|----------------|-------------|
| SimCLR | Contrastive | 69.3% | Moderate |
| MoCo v3 | Contrastive | 72.8% | Good |
| MAE | Reconstruction | 67.8% | Good |
| iBOT | Masked + Distill | 78.5% | Very Good |
| data2vec | Masked + Distill | 74.2% | Very Good |
| **DINOv2** | **Distillation** | **82.1%** | **Excellent** |

**DINOv2 advantages**:
- Best zero-shot transfer
- Excellent dense prediction features
- No negative pairs needed
- Scales well with data and model size

## Common Pitfalls

### 1. Model Collapse

**Symptoms**:
- Loss drops to near zero very quickly
- All images produce identical embeddings
- Teacher outputs become uniform

**Root cause**: The model finds a trivial solution where all inputs map to the same output.

**Solutions**:

```python
# 1. Ensure centering is enabled
self.center_momentum = 0.9  # Not too high, not too low

# 2. Use appropriate teacher temperature
teacher_temp_start = 0.04  # Low = sharp = strong signal
teacher_temp_end = 0.07

# 3. High-dimensional output space
out_dim = 65536  # Larger space = harder to collapse

# 4. Proper initialization
self.teacher.load_state_dict(self.student.state_dict())

# 5. Monitor metrics
if loss < 0.1 and epoch < 5:
    print("Warning: Potential collapse!")
```

### 2. Insufficient Batch Size

**Symptoms**:
- Training is unstable
- Poor downstream performance
- High variance in loss

**Root cause**: DINOv2 relies on batch statistics for centering. Small batches give noisy estimates.

**Solutions**:

```python
# 1. Use distributed training
# Effective batch size should be ≥ 1024
world_size = 8  # Number of GPUs
local_batch_size = 128
effective_batch_size = local_batch_size * world_size  # 1024

# 2. Gradient accumulation (if limited GPUs)
accumulation_steps = 8
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Adjust learning rate for batch size
# Linear scaling rule
base_lr = 0.0005
batch_size = 1024
lr = base_lr * batch_size / 256
```

### 3. Poor Data Quality

**Symptoms**:
- Model learns specific artifacts
- Poor generalization
- Downstream performance plateaus early

**Root cause**: Training data contains duplicates, watermarks, or insufficient diversity.

**Solutions**:

```python
# 1. Deduplicate dataset
# Use perceptual hashing or SSCD (Self-supervised copy detection)
from nexus.data.deduplication import deduplicate_images
dataset = deduplicate_images(dataset, threshold=0.95)

# 2. Filter low-quality images
from nexus.data.quality import ImageQualityFilter
quality_filter = ImageQualityFilter(
    min_resolution=224,
    max_aspect_ratio=3.0,
    check_corrupted=True,
)
dataset = quality_filter.filter(dataset)

# 3. Ensure diversity
# Balance datasets from multiple sources
# DINOv2 uses: ImageNet, Web data, Curated data
```

### 4. Incorrect EMA Update

**Symptoms**:
- Teacher and student diverge
- Training becomes unstable after some epochs
- Loss stops decreasing

**Root cause**: EMA update implemented incorrectly or with wrong momentum.

**Solutions**:

```python
# Correct EMA update
@torch.no_grad()
def update_teacher(self, momentum):
    for param_s, param_t in zip(
        self.student.parameters(),
        self.teacher.parameters()
    ):
        # Correct: teacher = m * teacher + (1-m) * student
        param_t.data.mul_(momentum).add_(
            param_s.detach().data,
            alpha=1 - momentum
        )

# Common mistake: forgetting detach()
# This creates a gradient graph through teacher!
# param_t.data = momentum * param_t.data + (1-momentum) * param_s.data  # WRONG

# Monitor EMA momentum
momentum = self.get_ema_momentum(epoch)
print(f"Epoch {epoch}, EMA momentum: {momentum:.6f}")
```

### 5. Wrong Encoder at Inference

**Symptoms**:
- Good training metrics but poor evaluation
- Features don't work for downstream tasks

**Root cause**: Using teacher encoder instead of student for inference.

**Solution**:

```python
# For inference, use STUDENT encoder (the one trained with gradients)
@torch.no_grad()
def extract_features(self, images):
    self.student.eval()  # Use student!
    features = self.student_backbone(images)
    return features

# Common mistake: using teacher
# features = self.teacher_backbone(images)  # WRONG

# Note: For DINOv2 specifically, both work well, but student is standard
```

### 6. Improper Learning Rate

**Symptoms**:
- Loss explodes or doesn't decrease
- Training is very slow
- Features don't transfer well

**Solutions**:

```python
# 1. Scale LR with batch size (linear scaling rule)
base_lr = 0.0005  # For batch size 256
actual_batch_size = 1024
lr = base_lr * actual_batch_size / 256  # 0.002

# 2. Use warmup
warmup_epochs = 10
if epoch < warmup_epochs:
    lr = base_lr * epoch / warmup_epochs

# 3. Cosine decay after warmup
else:
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))

# 4. Layer-wise LR decay
for layer_id in range(num_layers):
    layer_lr = lr * (0.75 ** (num_layers - layer_id - 1))
```

### 7. Augmentation Imbalance

**Symptoms**:
- Model learns augmentation artifacts
- Poor zero-shot transfer
- Features too sensitive to transformations

**Solutions**:

```python
# Ensure augmentations are balanced and appropriate
augmentation = transforms.Compose([
    # Not too extreme crops
    transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),  # Not (0.08, 1.0)

    # Reasonable color jitter
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # Not too strong

    # Moderate grayscale
    transforms.RandomGrayscale(p=0.2),  # Not 0.5

    # Careful with blur
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
    ], p=0.5),  # Not always applied
])

# Test augmentations visually
import matplotlib.pyplot as plt
image = Image.open("test.jpg")
augmented = [augmentation(image) for _ in range(16)]
# Visualize to ensure they look reasonable
```

### 8. Memory Issues

**Symptoms**:
- Out of memory errors
- Very slow training
- Can't use large models or batch sizes

**Solutions**:

```python
# 1. Gradient checkpointing
from torch.utils.checkpoint import checkpoint

class ViTWithCheckpointing(nn.Module):
    def forward(self, x):
        for block in self.blocks:
            x = checkpoint(block, x)  # Trades compute for memory
        return x

# 2. Mixed precision
from torch.cuda.amp import autocast
with autocast():
    loss = model(images)

# 3. Efficient attention (if using very large images)
# Use Flash Attention or other efficient implementations

# 4. Smaller local crops
local_crops_size = 96  # Instead of 128 or 196

# 5. Reduce number of local crops
local_crops_number = 6  # Instead of 8 or 10
```

## References

```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Vo, Huy V and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}

@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J{\'e}gou, Herv{\'e} and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9650--9660},
  year={2021}
}

@article{darcet2023vision,
  title={Vision Transformers Need Registers},
  author={Darcet, Timoth{\'e}e and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  journal={arXiv preprint arXiv:2309.16588},
  year={2023}
}

@inproceedings{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  booktitle={ICLR},
  year={2021}
}

@article{chen2020simple,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={International Conference on Machine Learning},
  pages={1597--1607},
  year={2020}
}

@inproceedings{he2020momentum,
  title={Momentum Contrast for Unsupervised Visual Representation Learning},
  author={He, Kaiming and Fan, Haoqi and Wu, Yuxin and Xie, Saining and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9729--9738},
  year={2020}
}

@article{he2022masked,
  title={Masked Autoencoders Are Scalable Vision Learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16000--16009},
  year={2022}
}

@article{zhou2021ibot,
  title={iBOT: Image BERT Pre-Training with Online Tokenizer},
  author={Zhou, Jinghao and Wei, Chen and Wang, Huiyu and Shen, Wei and Xie, Cihang and Yuille, Alan and Kong, Tao},
  journal={arXiv preprint arXiv:2111.07832},
  year={2021}
}

@article{baevski2022data2vec,
  title={data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language},
  author={Baevski, Alexei and Hsu, Wei-Ning and Xu, Qiantong and Babu, Arun and Gu, Jiatao and Auli, Michael},
  booktitle={International Conference on Machine Learning},
  pages={1298--1312},
  year={2022}
}
```

## Summary

DINOv2 represents a major milestone in self-supervised learning:

**Key Strengths**:
- State-of-the-art zero-shot transfer performance
- Excellent features for both image-level and pixel-level tasks
- Scales effectively with data and model size
- Simple and elegant training procedure
- No negative pairs or complex augmentations required

**Key Hyperparameters**:
- Batch size: ≥ 1024 (distributed training essential)
- Output dimension: 65,536 (high-dimensional space)
- Student temperature: 0.1 (fixed)
- Teacher temperature: 0.04 → 0.07 (scheduled)
- EMA momentum: 0.996 → 1.0 (cosine schedule)
- Centering momentum: 0.9
- Learning rate: 0.0005 × (batch_size / 256)
- Weight decay: 0.04
- Multi-crop: 2 global (224×224) + 8 local (96×96)

**When to use DINOv2**:
- Need strong visual representations without labels
- Want zero-shot transfer capabilities
- Working on dense prediction tasks (segmentation, detection)
- Have access to large-scale diverse image data
- Can train with large batch sizes (distributed setup)

**Comparison to alternatives**:
- vs MAE: Better downstream transfer, but needs larger batches
- vs SimCLR/MoCo: Better at dense tasks, no negative pairs needed
- vs Supervised: Approaches supervised performance without labels

**Official Resources**:
- **Paper**: https://arxiv.org/abs/2304.07193
- **Code**: https://github.com/facebookresearch/dinov2
- **Models**: https://github.com/facebookresearch/dinov2#pretrained-models
- **Nexus Implementation**: `nexus/models/ssl/dinov2.py`

DINOv2 demonstrates that with careful design of the learning objective, data curation, and training at scale, self-supervised learning can produce visual features that rival or exceed those learned with full supervision.
