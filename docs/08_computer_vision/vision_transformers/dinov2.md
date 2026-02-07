# DINOv2 (Self-Distillation with No Labels v2)

## Overview & Motivation

DINOv2 is a self-supervised learning method that trains Vision Transformers without any labels by distilling knowledge from a teacher network to a student network. The teacher is an exponential moving average (EMA) of the student, creating a self-distillation loop that learns rich visual representations from unlabeled images.

**Key Innovation**: Self-supervised student-teacher framework with multi-crop augmentation, centering to prevent mode collapse, and massive-scale pre-training (142M images) producing universal visual features.

**Why It Matters**:
- Matches or exceeds supervised pre-training on downstream tasks
- Learns semantic features without labels (enabling zero-shot transfer)
- Provides dense features useful for segmentation, depth, etc.
- Serves as foundation for CLIP, Segment Anything, and other models
- Scales to giant models (1B+ parameters) with improved quality

## Theoretical Background

### Problem Setting
Learn visual representations from unlabeled images that transfer well to downstream tasks. Avoid mode collapse (all images mapped to same representation) and dimensional collapse (representations lie in low-dimensional subspace).

### Core Approach
1. Create two views of same image: global crops (large) and local crops (small)
2. Pass global crops through teacher network (EMA of student)
3. Pass all crops through student network
4. Teacher produces "soft targets" with low temperature (sharp distribution)
5. Student tries to match teacher's output with higher temperature
6. Use centering to prevent mode collapse
7. Update teacher weights via EMA

### Key Insight
The teacher provides slowly evolving targets that are more stable than the student's predictions. Multi-crop augmentation forces the network to learn scale-invariant features. Centering prevents trivial solutions where all images collapse to the same output.

## Mathematical Formulation

### 1. Multi-Crop Augmentation
Generate multiple views of image x:
```
Global crops: x_g1, x_g2  (224×224, large scale)
Local crops: x_l1, ..., x_lK  (96×96, small scale, K=6-10)

All crops: {x_g1, x_g2, x_l1, ..., x_lK}
```

### 2. Student and Teacher Networks
```
Student: f_θ (parameters θ, trained via backprop)
Teacher: f_θ' (parameters θ', updated via EMA)

θ'_t = λ·θ'_{t-1} + (1-λ)·θ_t

where λ starts at 0.996 and increases to 1.0 using cosine schedule
```

### 3. Projection Head
Map backbone features to normalized probability distribution:
```
# Backbone output
z = ViT(x) ∈ R^D  (CLS token)

# MLP projection head
h = MLP(z) ∈ R^{D_h}
h = LayerNorm(h)
h = L2Normalize(h)

# Prototype layer (learned codebook)
logits = h^T · W ∈ R^K  (K prototypes)

# Center and sharpen with temperature τ
P = Softmax((logits - c) / τ) ∈ R^K

where c is exponentially updated center:
c_t = m·c_{t-1} + (1-m)·Mean(logits_teacher)
```

### 4. DINO Loss
Cross-entropy between teacher and student distributions:
```
Teacher output (global crops only):
P_t^{g1} = g_θ'(x_g1, τ_t)  # τ_t = 0.04 (sharp)
P_t^{g2} = g_θ'(x_g2, τ_t)

Student output (all crops):
P_s^i = g_θ(x_i, τ_s)  # τ_s = 0.1 (smooth)
for i ∈ {g1, g2, l1, ..., lK}

Loss (exclude same-view pairs):
L = - Σ_{i,j≠i} P_t^i · log(P_s^j)

Total: 2×(K+1) cross-entropy terms per image
```

### 5. Centering Mechanism
Prevent mode collapse by subtracting running mean:
```
c_t = 0.9·c_{t-1} + 0.1·Mean_batch(logits_teacher)

logits_centered = logits - c

This prevents one prototype from dominating
```

### 6. Complete Training Algorithm
```
for batch (x₁, ..., x_B):
    # Generate multi-crop views
    global_crops, local_crops = augment(x)

    # Teacher forward (no grad)
    with torch.no_grad():
        teacher_out = [teacher(g, τ=0.04) for g in global_crops]
        update_center(teacher_out)

    # Student forward
    all_crops = global_crops + local_crops
    student_out = [student(c, τ=0.1) for c in all_crops]

    # Compute loss
    loss = cross_entropy(student_out, teacher_out)

    # Update student
    loss.backward()
    optimizer.step()

    # Update teacher (EMA)
    update_teacher_ema(student, teacher, momentum=λ)
```

### 7. Koleo Regularization (DINOv2 Enhancement)
Encourage uniform distribution of features:
```
# Kolmogorov-Smirnov regularization
For batch features F = [f₁, ..., f_B]:

d_ij = ||f_i - f_j||²  # Pairwise distances
d_nn_i = min_{j≠i} d_ij  # Nearest neighbor distance

L_koleo = -Mean(log(d_nn_i))

Total loss: L = L_dino + λ_koleo·L_koleo
```

## High-Level Intuition

```
Input Image
    ↓
[Multi-Crop Augmentation]
    ├─────────────┬─────────────┬──────────────┐
Global Crop 1  Global Crop 2  Local Crops (6×)
  (224×224)      (224×224)      (96×96)
    │                │               │
    ↓                ↓               ↓
┌─────────────────────────────────────────────┐
│           STUDENT (Gradient Flow)           │
│                                             │
│  ┌────────────────┐                        │
│  │ ViT Backbone   │                        │
│  │ (12-40 layers) │                        │
│  └────────────────┘                        │
│          ↓                                  │
│  ┌────────────────┐                        │
│  │ DINO Head      │                        │
│  │ • MLP (3 layer)│                        │
│  │ • Bottleneck   │                        │
│  │ • L2 Normalize │                        │
│  │ • Prototypes   │                        │
│  └────────────────┘                        │
│          ↓                                  │
│    [τ_s = 0.1]  ← Higher temperature       │
│          ↓                                  │
│   Student Probs P_s                         │
└─────────────────────────────────────────────┘
                    ↓
              [Cross-Entropy]
                    ↑
┌─────────────────────────────────────────────┐
│          TEACHER (No Gradient)              │
│                                             │
│  ┌────────────────┐                        │
│  │ ViT Backbone   │                        │
│  │ (EMA of student)│                       │
│  └────────────────┘                        │
│          ↓                                  │
│  ┌────────────────┐                        │
│  │ DINO Head      │                        │
│  │ (EMA of student)│                       │
│  └────────────────┘                        │
│          ↓                                  │
│  [Centering: - c]                          │
│          ↓                                  │
│    [τ_t = 0.04]  ← Lower temperature       │
│          ↓                                  │
│   Teacher Probs P_t (sharp targets)        │
└─────────────────────────────────────────────┘
    │
    ↓
[EMA Update]
θ' ← 0.996·θ' + 0.004·θ
```

**Why It Works**:
- Teacher provides stable, slowly changing targets
- Multi-crop forces learning scale-invariant features
- Centering prevents collapse to uniform distribution
- Temperature difference: teacher sharp → student smooth

## Implementation Details

### Model Variants

| Variant | Params | Embed Dim | Depth | Heads | Patch | Dataset |
|---------|--------|-----------|-------|-------|-------|---------|
| ViT-S/14 | 21M | 384 | 12 | 6 | 14 | LVD-142M |
| ViT-B/14 | 86M | 768 | 12 | 12 | 14 | LVD-142M |
| ViT-L/14 | 304M | 1024 | 24 | 16 | 14 | LVD-142M |
| ViT-g/14 | 1.1B | 1536 | 40 | 24 | 14 | LVD-142M |

### Configuration Example

```python
config = {
    # Backbone (ViT)
    "img_size": 224,
    "patch_size": 14,
    "in_channels": 3,
    "embed_dim": 768,           # ViT-B
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "drop_path_rate": 0.1,

    # DINO Head
    "out_dim": 65536,           # Number of prototypes
    "head_hidden_dim": 2048,
    "head_bottleneck_dim": 256,
    "head_nlayers": 3,

    # Training
    "student_temp": 0.1,
    "teacher_temp": 0.04,
    "ema_momentum": 0.996,      # Increases to 1.0
    "center_momentum": 0.9,
    "koleo_weight": 0.1,        # DINOv2 enhancement

    # Multi-crop
    "global_crops_scale": (0.4, 1.0),
    "local_crops_scale": (0.05, 0.4),
    "local_crops_number": 8,
}
```

## Code Walkthrough

Reference: `Nexus/nexus/models/cv/dinov2.py`

### 1. DINO Projection Head

```python
class DINOHead(NexusModule):
    """Projection head with centering and sharpening."""

    def __init__(self, in_dim, hidden_dim=2048, bottleneck_dim=256,
                 out_dim=65536, nlayers=3):
        super().__init__()

        # MLP layers
        layers = []
        for i in range(nlayers - 1):
            dim_in = in_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

        # Bottleneck projection
        self.last_layer = nn.Linear(hidden_dim, bottleneck_dim)
        self.last_layer_norm = nn.LayerNorm(bottleneck_dim)

        # Prototype layer (no bias)
        self.prototypes = nn.Linear(bottleneck_dim, out_dim, bias=False)

        # Center buffer (prevents collapse)
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, x, temperature=0.1):
        """
        Args:
            x: (B, in_dim) features from backbone
            temperature: Temperature for softmax sharpening

        Returns:
            log_probs: (B, out_dim)
        """
        x = self.mlp(x)
        x = self.last_layer(x)
        x = self.last_layer_norm(x)

        # L2 normalization
        x = F.normalize(x, dim=-1)

        # Compute logits
        logits = self.prototypes(x)

        # Apply centering and temperature
        logits = (logits - self.center) / temperature

        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_output, momentum=0.9):
        """Update center with teacher's mean output."""
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * momentum + batch_center * (1 - momentum)
```

### 2. Student-Teacher Framework

```python
class StudentTeacher(NexusModule):
    """EMA-based student-teacher framework."""

    def __init__(self, student, student_head, teacher_head, ema_momentum=0.996):
        super().__init__()
        self.student = student
        self.student_head = student_head

        # Teacher is a copy of student (no gradient)
        self.teacher = copy.deepcopy(student)
        self.teacher.requires_grad_(False)
        self.teacher_head = teacher_head
        self.teacher_head.requires_grad_(False)

        self.ema_momentum = ema_momentum

    @torch.no_grad()
    def update_teacher(self, momentum=None):
        """Update teacher via EMA."""
        m = momentum if momentum is not None else self.ema_momentum

        # Update backbone
        for param_s, param_t in zip(
            self.student.parameters(),
            self.teacher.parameters()
        ):
            param_t.data = param_t.data * m + param_s.data * (1 - m)

        # Update head
        for param_s, param_t in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters()
        ):
            param_t.data = param_t.data * m + param_s.data * (1 - m)
```

### 3. Multi-Crop Augmentation

```python
class MultiCropAugmentation:
    """Generate global and local crops."""

    def __init__(self, global_crops_scale=(0.4, 1.0),
                 local_crops_scale=(0.05, 0.4),
                 local_crops_number=8,
                 size=224, local_size=96):
        # Global crops (2)
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size, scale=global_crops_scale, interpolation=Image.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Local crops (6-10)
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                local_size, scale=local_crops_scale, interpolation=Image.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.local_crops_number = local_crops_number

    def __call__(self, image):
        crops = []
        # Global crops
        crops.append(self.global_transform(image))
        crops.append(self.global_transform(image))
        # Local crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops
```

### 4. DINO Loss

```python
class DINOv2Loss(NexusModule):
    """Cross-entropy loss for self-distillation."""

    def __init__(self, student_temp=0.1, teacher_temp=0.04):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def forward(self, student_outputs, teacher_outputs):
        """
        Args:
            student_outputs: List of log-probs for all crops
            teacher_outputs: List of log-probs for global crops only

        Returns:
            loss: Scalar
        """
        total_loss = 0.0
        n_loss_terms = 0

        for t_idx, t_out in enumerate(teacher_outputs):
            # Teacher targets (detach)
            teacher_probs = torch.exp(t_out).detach()

            for s_idx, s_out in enumerate(student_outputs):
                # Skip same-view pairs
                if s_idx == t_idx:
                    continue

                # Cross-entropy: -sum(p_teacher * log(p_student))
                loss = -torch.sum(teacher_probs * s_out, dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        return total_loss / n_loss_terms
```

### 5. Complete Training Loop

```python
def train_dinov2(model, dataloader, optimizer, config):
    """DINOv2 training loop."""

    # Multi-crop augmentation
    augmentation = MultiCropAugmentation(
        global_crops_scale=config["global_crops_scale"],
        local_crops_scale=config["local_crops_scale"],
        local_crops_number=config["local_crops_number"]
    )

    for epoch in range(config["epochs"]):
        for batch_idx, images in enumerate(dataloader):
            # Generate multi-crop views
            all_crops = []
            for img in images:
                crops = augmentation(img)
                all_crops.append(crops)

            # Separate global and local
            global_crops = [crop[0:2] for crop in all_crops]
            local_crops = [crop[2:] for crop in all_crops]

            # Student forward (all crops)
            student_outputs = []
            for crop_set in all_crops:
                for crop in crop_set:
                    feat = model.forward_backbone(crop)
                    out = model.student_head(
                        feat["embeddings"],
                        temperature=config["student_temp"]
                    )
                    student_outputs.append(out)

            # Teacher forward (global only, no grad)
            teacher_outputs = []
            with torch.no_grad():
                for crop_set in global_crops:
                    for crop in crop_set:
                        feat = model.teacher(crop)
                        out = model.teacher_head(
                            feat["embeddings"],
                            temperature=config["teacher_temp"]
                        )
                        teacher_outputs.append(out)

                # Update center
                all_teacher_logits = torch.cat(teacher_outputs, dim=0)
                model.teacher_head.update_center(all_teacher_logits)

            # Compute loss
            loss = model.loss_fn(student_outputs, teacher_outputs)

            # Backward and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update teacher via EMA
            m = cosine_scheduler(
                base_value=config["ema_momentum"],
                final_value=1.0,
                epochs=config["epochs"],
                niter_per_ep=len(dataloader),
                warmup_epochs=10,
                start_warmup_value=0.996
            )
            model.update_teacher(momentum=m[epoch * len(dataloader) + batch_idx])
```

## Optimization Tricks

### 1. Cosine EMA Schedule

```python
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0):
    """Cosine schedule for EMA momentum."""
    warmup_schedule = np.linspace(start_warmup_value, base_value,
                                  warmup_epochs * niter_per_ep)

    iters = np.arange(epochs * niter_per_ep - warmup_epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * \
               (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule
```

### 2. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    student_out = model.student(crops)
    teacher_out = model.teacher(crops)
    loss = dino_loss(student_out, teacher_out)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Gradient Clipping

```python
# Clip gradients for stability
torch.nn.utils.clip_grad_norm_(
    model.student.parameters(),
    max_norm=3.0
)
```

### 4. LARS Optimizer

```python
from torch_optimizer import LARS

optimizer = LARS(
    model.student.parameters(),
    lr=0.3,  # Base LR scaled with batch size
    weight_decay=1e-6,
    momentum=0.9
)
```

### 5. Efficient Multi-Crop Processing

```python
# Batch all crops together for efficient forward pass
def batch_crops(crops_list):
    """
    Args:
        crops_list: List of [global1, global2, local1, ..., localK]
    Returns:
        Batched tensor (B*(2+K), C, H, W)
    """
    all_crops = []
    for crops in crops_list:
        all_crops.extend(crops)
    return torch.stack(all_crops)

# Single forward pass
batched = batch_crops(all_crops)
features = model(batched)

# Split back
B = len(all_crops)
n_crops = len(all_crops[0])
features = features.view(B, n_crops, -1)
```

## Experiments & Results

### ImageNet-1K Linear Probing

**Setup**:
- Pre-training: LVD-142M (142M curated images)
- Evaluation: Freeze backbone, train linear classifier
- Epochs: 100 (linear probing)

**Results** (Top-1 Accuracy):

| Method | Backbone | Pre-train | Linear Probe |
|--------|----------|-----------|--------------|
| Supervised | ViT-B/16 | ImageNet-1K | 81.8% |
| SimCLR | ViT-B/16 | ImageNet-1K | 74.2% |
| MoCo v3 | ViT-B/16 | ImageNet-1K | 76.7% |
| DINO | ViT-B/16 | ImageNet-1K | 78.2% |
| **DINOv2** | **ViT-B/14** | **LVD-142M** | **84.5%** |
| **DINOv2** | **ViT-L/14** | **LVD-142M** | **86.3%** |
| **DINOv2** | **ViT-g/14** | **LVD-142M** | **86.8%** |

DINOv2 **surpasses supervised pre-training** with self-supervision!

### K-NN Classification (No Training)

Nearest neighbor classification in feature space:

| Model | k=1 | k=10 | k=20 |
|-------|-----|------|------|
| Supervised ViT-B | 68.2% | 75.3% | 76.8% |
| DINO ViT-B | 74.5% | 78.8% | 79.5% |
| DINOv2 ViT-B | **80.6%** | **82.1%** | **82.5%** |
| DINOv2 ViT-g | **83.2%** | **84.5%** | **84.8%** |

Strong semantic features enable zero-shot transfer.

### Dense Prediction Tasks

**Semantic Segmentation (ADE20K)**:

| Backbone | Method | mIoU |
|----------|--------|------|
| ViT-B (sup) | Linear | 42.3% |
| DINOv2 ViT-B | Linear | **47.2%** |
| DINOv2 ViT-L | Linear | **51.6%** |

**Depth Estimation (NYUv2)**:

| Backbone | RMSE ↓ | δ₁ ↑ |
|----------|--------|------|
| ViT-B (sup) | 0.512 | 85.2% |
| DINOv2 ViT-B | **0.439** | **90.1%** |

### Qualitative Analysis

**PCA Visualization**:
First 3 principal components of DINOv2 features naturally correspond to semantic regions (e.g., sky, grass, objects) without any supervision.

**Attention Maps**:
Self-attention heads learn to focus on object boundaries, parts, and semantic regions automatically.

### Ablation Studies

**Effect of Components**:

| Configuration | Accuracy |
|---------------|----------|
| Baseline (no DINO) | 76.8% |
| + Multi-crop | 80.2% |
| + Centering | 82.1% |
| + Koleo reg | 83.5% |
| + Large data | **84.5%** |

**Number of Local Crops**:

| Local Crops | Accuracy | Training Time |
|-------------|----------|---------------|
| 0 | 81.2% | 1.0× |
| 4 | 83.1% | 1.3× |
| 8 | **84.5%** | 1.6× |
| 12 | 84.6% | 2.0× |

8 crops provide best accuracy/speed trade-off.

## Common Pitfalls

### 1. Forgetting to Detach Teacher
Wrong: Teacher gradients flow back
```python
teacher_out = self.teacher(x)
loss = F.cross_entropy(student_out, teacher_out)  # BUG
```

Correct: Always detach or use `@torch.no_grad()`
```python
with torch.no_grad():
    teacher_out = self.teacher(x)
loss = F.cross_entropy(student_out, teacher_out)
```

### 2. Not Updating Center
Wrong: Forgetting center update
```python
loss = dino_loss(student_out, teacher_out)  # Mode collapse!
```

Correct: Update center every iteration
```python
with torch.no_grad():
    teacher_head.update_center(teacher_out)
loss = dino_loss(student_out, teacher_out)
```

### 3. Wrong EMA Momentum Schedule
Wrong: Fixed momentum
```python
self.update_teacher(momentum=0.996)  # Suboptimal
```

Correct: Cosine schedule 0.996 → 1.0
```python
m = cosine_schedule(base=0.996, final=1.0, epoch=epoch)
self.update_teacher(momentum=m)
```

### 4. Temperature Too High
Wrong: Equal temperatures
```python
student_temp = 0.1
teacher_temp = 0.1  # BUG: Too high for teacher
```

Correct: Teacher should be sharper
```python
student_temp = 0.1
teacher_temp = 0.04  # Sharp targets
```

### 5. Insufficient Augmentation
Wrong: Weak augmentation
```python
transform = Compose([Resize(224), ToTensor()])  # Too weak
```

Correct: Strong multi-crop augmentation
```python
transform = Compose([
    RandomResizedCrop(...),
    ColorJitter(...),
    GaussianBlur(...),
    Solarization(...),
    ...
])
```

## References

### Original Papers
1. **Emerging Properties in Self-Supervised Vision Transformers (DINO)**
   Caron, M., Touvron, H., Misra, I., et al., ICCV 2021
   https://arxiv.org/abs/2104.14294

2. **DINOv2: Learning Robust Visual Features without Supervision**
   Oquab, M., Darcet, T., Moutakanni, T., et al., 2023
   https://arxiv.org/abs/2304.07193

### Related Work
3. **Self-Distillation with No Labels**
   Grill, J., et al., NeurIPS 2020 (BYOL)

4. **An Empirical Study of Training Self-Supervised Vision Transformers**
   Chen, X., et al., ICCV 2021 (MoCo v3)

### Applications
5. **Segment Anything (SAM)**
   Uses DINOv2 as backbone
   https://arxiv.org/abs/2304.02643

6. **Depth Anything**
   Depth estimation with DINOv2 features

### Implementation
- Official: https://github.com/facebookresearch/dinov2
- Nexus: `Nexus/nexus/models/cv/dinov2.py`
- Hugging Face: https://huggingface.co/facebook/dinov2-base

### Datasets
7. **LVD-142M Dataset**
   Large-scale curated dataset for DINOv2
   142M images from diverse sources
