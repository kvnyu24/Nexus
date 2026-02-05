# Vision Transformer (ViT)

## Overview & Motivation

The Vision Transformer (ViT) revolutionized computer vision by applying pure transformer architectures to images, eliminating the need for convolutions. By treating images as sequences of patches, ViT demonstrates that transformers can match or exceed CNN performance when trained on sufficient data.

**Key Innovation**: Split images into patches, linearly embed them, add positional encodings, and process with standard Transformer encoders.

**Why It Matters**:
- Unified architecture across vision and language
- Scales better with data than CNNs
- Foundation for CLIP, DINO, MAE, etc.
- Excellent transfer learning capabilities

## Theoretical Background

### Problem Setting
Map image x ∈ R^(H×W×C) to class probabilities over K classes.

### Core Approach
1. Split image into N fixed-size patches (P×P)
2. Linearly embed each patch to dimension D
3. Add learnable positional embeddings
4. Prepend learnable [CLS] token
5. Process with L Transformer encoder layers
6. Classify using [CLS] token representation

## Mathematical Formulation

### 1. Patch Embedding
```
N = HW / P²  (number of patches)
z₀ = [x_class; x_p¹E; x_p²E; ...; x_pᴺE] + E_pos
```
where:
- x_pⁱ ∈ R^(P²·C): flattened i-th patch
- E ∈ R^((P²·C)×D): patch embedding matrix
- x_class: learnable class token
- E_pos ∈ R^((N+1)×D): positional embedding

### 2. Transformer Encoder
For layers ℓ = 1...L:
```
z'_ℓ = MSA(LN(z_{ℓ-1})) + z_{ℓ-1}
z_ℓ = MLP(LN(z'_ℓ)) + z'_ℓ
```

### 3. Multi-Head Self-Attention
```
MSA(Z) = Concat(head₁, ..., headₕ)W^O
headᵢ = Attention(ZWᵢ^Q, ZWᵢ^K, ZWᵢ^V)
Attention(Q,K,V) = softmax(QKᵀ/√d_k)V
```

### 4. MLP Block
```
MLP(x) = W₂·GELU(W₁·x)
```
where W₁ ∈ R^(D×4D) and W₂ ∈ R^(4D×D)

### 5. Classification
```
y = LN(z_L⁰)
p(class) = softmax(W_head·y + b)
```

### 6. Layer Scale (Modern Enhancement)
```
z'_ℓ = γ₁ ⊙ MSA(LN(z_{ℓ-1})) + z_{ℓ-1}
z_ℓ = γ₂ ⊙ MLP(LN(z'_ℓ)) + z'_ℓ
```
where γ₁, γ₂ are learnable, initialized to ~10⁻⁶

## High-Level Intuition

```
Input Image (224×224×3)
    ↓
[Patch Embedding]
16×16 patches → 196 tokens
    ↓
[Add CLS token + Position Embeddings]
[CLS] + 196 patches → 197 tokens (dim D)
    ↓
┌─────────────────────┐
│ Transformer Block 1 │
│ • LayerNorm         │
│ • Self-Attention    │  ← All patches attend to each other
│ • Residual          │
│ • LayerNorm         │
│ • MLP (4×D)         │
│ • Residual          │
└─────────────────────┘
    ↓
(Repeat L times)
    ↓
[LayerNorm]
    ↓
[Linear Classifier]
Use [CLS] token → K classes
```

**Attention Patterns**:
- **Early layers**: Local patterns (edges, textures)
- **Middle layers**: Object parts (faces, wheels)
- **Late layers**: Semantic concepts (animals, vehicles)

## Implementation Details

### Model Variants

| Variant | Layers | Hidden | MLP | Heads | Params | Patch |
|---------|--------|--------|-----|-------|--------|-------|
| ViT-Ti  | 12     | 192    | 768 | 3     | 5.7M   | 16    |
| ViT-S   | 12     | 384    | 1536| 6     | 22M    | 16    |
| ViT-B   | 12     | 768    | 3072| 12    | 86M    | 16    |
| ViT-L   | 24     | 1024   | 4096| 16    | 307M   | 16    |
| ViT-H   | 32     | 1280   | 5120| 16    | 632M   | 14    |

### Configuration Example

```python
config = {
    "image_size": 224,
    "patch_size": 16,
    "in_channels": 3,
    "num_classes": 1000,
    "embed_dim": 768,            # ViT-B
    "num_layers": 12,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "dropout": 0.1,
    "layer_scale_init_value": 1e-6,
    "use_flash_attention": True,
    "distillation": False,       # DeiT-style
}
```

## Code Walkthrough

Reference: `/Users/kevinyu/Projects/Nexus/nexus/models/cv/vit.py`

### 1. Patch Embedding

```python
class PatchEmbedding(NexusModule):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        # Conv2d acts as patch extraction + linear projection
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.projection(x)     # (B, E, H', W')
        x = x.flatten(2)           # (B, E, N)
        x = x.transpose(1, 2)      # (B, N, E)
        x = self.norm(x)
        return x
```

**Why Conv2d?** More efficient than explicit patch extraction + linear layer.

### 2. Transformer Block with Layer Scale

```python
class TransformerBlock(NexusModule):
    def __init__(self, dim, num_heads, mlp_ratio=4.0,
                 layer_scale_init_value=1e-6):
        super().__init__()

        self.attention = UnifiedAttention(
            hidden_size=dim,
            num_heads=num_heads,
            use_flash_attention=True
        )
        self.norm1 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

        # Layer scale for training stability
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x):
        x = x + self.gamma1 * self.attention(self.norm1(x))
        x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x
```

**Key Features**:
- Pre-normalization (more stable than post-norm)
- Layer scale allows fine-grained residual control
- Flash attention for memory efficiency

### 3. Positional Embedding Interpolation

```python
def interpolate_pos_encoding(self, x, h, w):
    """Support variable input resolutions."""
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1

    if npatch == N and w == h:
        return self.pos_embed

    # Separate CLS token embedding
    class_pos_embed = self.pos_embed[:, :1]
    patch_pos_embed = self.pos_embed[:, 1:]

    # Reshape to 2D and interpolate
    dim = x.shape[-1]
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(N**0.5), int(N**0.5), dim).permute(0, 3, 1, 2),
        size=(h, w),
        mode='bicubic',
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    return torch.cat((class_pos_embed, patch_pos_embed), dim=1)
```

**Benefits**:
- Train on 224×224, test on 384×384 or higher
- Smooth adaptation of learned positional patterns

### 4. Forward Pass

```python
def forward(self, image):
    B = image.shape[0]

    # Patch embedding
    x = self.patch_embed(image)

    # Add CLS token
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # Add position embeddings
    x = x + self.interpolate_pos_encoding(x,
                                          self.patch_embed.grid_size,
                                          self.patch_embed.grid_size)
    x = self.pos_drop(x)

    # Apply transformer layers
    features = []
    for layer in self.transformer_layers:
        x = layer(x)
        features.append(x)

    # Classification
    x = self.norm(x)
    logits = self.head(x[:, 0])

    return {
        "logits": logits,
        "embeddings": x[:, 0],
        "features": features
    }
```

## Optimization Tricks

### 1. Data Augmentation

ViT requires **stronger augmentation** than CNNs:

```python
from timm.data import create_transform

# Training augmentation
train_transform = create_transform(
    input_size=224,
    is_training=True,
    auto_augment='rand-m9-mstd0.5-inc1',  # RandAugment
    re_prob=0.25,                          # Random Erasing
    re_mode='pixel',
    interpolation='bicubic',
)

# Additional augmentation
from timm.data.mixup import Mixup
mixup_fn = Mixup(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,
    label_smoothing=0.1,
)
```

### 2. Learning Rate Schedule

```python
# Warmup crucial for stability
base_lr = 0.001
batch_size = 512
warmup_epochs = 5

# Linear scaling rule
lr = base_lr * (effective_batch_size / 512)

# Cosine decay with warmup
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_epochs * steps_per_epoch,
    num_training_steps=total_epochs * steps_per_epoch
)
```

### 3. Stochastic Depth (DropPath)

```python
# Progressive drop path rates
drop_path_rate = 0.1
dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

for i, block in enumerate(transformer_blocks):
    block.drop_path = DropPath(dpr[i])
```

### 4. Weight Initialization

```python
def init_weights_vit(self):
    # Positional embeddings
    nn.init.trunc_normal_(self.pos_embed, std=0.02)
    nn.init.trunc_normal_(self.cls_token, std=0.02)

    # Linear layers
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
```

### 5. Memory Optimizations

```python
# Flash Attention (3-5x speedup)
config["use_flash_attention"] = True

# Gradient Checkpointing (trade compute for memory)
from torch.utils.checkpoint import checkpoint
for block in model.transformer_layers:
    block.forward = lambda x: checkpoint(block.forward, x)

# Mixed Precision Training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs["logits"], labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Experiments & Results

### ImageNet-1K Classification

**Setup**:
- Dataset: ImageNet-1K (1.28M images, 1K classes)
- Pre-training: JFT-300M (300M images)
- Resolution: 224×224 → 384×384 fine-tuning
- Batch size: 4096
- Optimizer: AdamW (lr=0.001, wd=0.1)
- Epochs: 300

**Results** (Top-1 Accuracy):

| Model | Params | Pre-train | 224px | 384px |
|-------|--------|-----------|-------|-------|
| ResNet-50 | 25M | ImageNet-1K | 76.5% | 78.0% |
| ResNet-152 | 60M | ImageNet-1K | 78.3% | 79.8% |
| EfficientNet-B7 | 66M | ImageNet-1K | 84.3% | 84.5% |
| ViT-B/16 | 86M | JFT-300M | **77.9%** | **84.1%** |
| ViT-L/16 | 307M | JFT-300M | **76.5%** | **85.3%** |
| ViT-H/14 | 632M | JFT-300M | **78.0%** | **88.5%** |

**Key Observations**:
1. ViT needs more data than CNNs (poor when trained only on ImageNet-1K)
2. With large-scale pre-training, ViT outperforms CNNs
3. Larger models scale better
4. Higher resolution fine-tuning gives significant gains

### Transfer Learning

**Few-Shot Learning** (average across 19 datasets):

| Model | 1-shot | 5-shot | 10-shot |
|-------|--------|--------|---------|
| ResNet-152 | 42.3% | 62.1% | 69.8% |
| ViT-B/16 | **45.7%** | **68.4%** | **75.2%** |
| ViT-L/16 | **49.1%** | **73.2%** | **79.8%** |

**Fine-tuning Efficiency**:
- Converges 2-3× faster than ResNets
- Lower learning rates optimal (10× smaller)
- Linear probing achieves 90% of full fine-tuning

### Computational Cost

**Training** (ViT-B/16, ImageNet-1K, 300 epochs):
- Hardware: 8× V100 32GB
- Time: ~225 hours
- Throughput: ~1800 images/sec

**Inference** (batch_size=1):
- CPU (Xeon): 85 ms/image
- GPU (V100): 12 ms/image
- GPU + Flash Attention: 7 ms/image

## Common Pitfalls

### 1. Small Dataset Training
Wrong: Training ViT-B on small datasets (e.g., CIFAR-10) from scratch
```python
model = VisionTransformer(config)  # Likely to underfit
```

Correct: Use pre-trained models or smaller architectures
```python
model = VisionTransformer.from_pretrained("vit_base_patch16_224")
model.head = nn.Linear(768, 10)  # Replace head
```

### 2. Weak Augmentation
Wrong: Basic augmentation
```python
transforms = Compose([Resize(224), RandomHorizontalFlip()])
```

Correct: Strong augmentation
```python
transforms = create_transform(
    auto_augment='rand-m9-mstd0.5-inc1',
    re_prob=0.25,
)
```

### 3. Incorrect LR Scaling
Wrong: Fixed learning rate
```python
optimizer = AdamW(model.parameters(), lr=0.001)
```

Correct: Scale with batch size
```python
lr = base_lr * (batch_size / 512)
optimizer = AdamW(model.parameters(), lr=lr)
```

### 4. Missing Warmup
Wrong: No learning rate warmup
```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

Correct: Warmup then cosine decay
```python
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=5 * steps_per_epoch,
    num_training_steps=total_steps
)
```

### 5. Deep Models Without Layer Scale
Wrong: Deep model without layer scale
```python
config = {"num_layers": 24}  # Training instability
```

Correct: Enable layer scale
```python
config = {"num_layers": 24, "layer_scale_init_value": 1e-6}
```

## References

### Original Papers
1. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**
   Dosovitskiy et al., ICLR 2021
   https://arxiv.org/abs/2010.11929

2. **Training data-efficient image transformers & distillation through attention (DeiT)**
   Touvron et al., ICML 2021
   https://arxiv.org/abs/2012.12877

3. **Going deeper with Image Transformers (CaiT)**
   Touvron et al., ICCV 2021
   https://arxiv.org/abs/2103.17239

### Training Insights
4. **How to train your ViT? Data, Augmentation, and Regularization**
   Steiner et al., 2021
   https://arxiv.org/abs/2106.10270

5. **Three things everyone should know about Vision Transformers**
   Dosovitskiy et al., 2022
   https://arxiv.org/abs/2203.09795

### Implementation
- Official: https://github.com/google-research/vision_transformer
- timm: https://github.com/huggingface/pytorch-image-models
- Nexus: `/Users/kevinyu/Projects/Nexus/nexus/models/cv/vit.py`
