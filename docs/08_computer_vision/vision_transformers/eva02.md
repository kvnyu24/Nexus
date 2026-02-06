# EVA-02 (Enhanced Vision-language pre-training with Augmented supervision)

## Overview & Motivation

EVA-02 is a Vision Transformer that combines masked image modeling (MIM) pre-training with architectural innovations: 2D Rotary Position Embeddings (RoPE) and SwiGLU activations. It achieves state-of-the-art performance by using a CLIP teacher to generate reconstruction targets for MIM, avoiding the need for hand-crafted tokenizers.

**Key Innovation**: Replaces absolute positional embeddings with 2D RoPE for better resolution extrapolation, and uses SwiGLU activations for improved expressiveness, while training via MIM with CLIP-generated targets.

**Why It Matters**:
- Better length extrapolation than standard ViT (train 224px, test 1024px+)
- No need for discrete tokenizers (VQ-VAE, DALL-E) for MIM
- State-of-the-art on ImageNet, COCO, ADE20K
- Stronger foundation for vision-language models
- Scales efficiently to giant models (7B parameters)

## Theoretical Background

### Problem Setting
Standard ViT uses absolute positional embeddings that don't extrapolate well to different resolutions. MIM methods typically require discrete visual tokenizers, adding complexity. We need a unified approach that scales well and transfers to downstream tasks.

### Core Approach
1. Use Masked Image Modeling (MIM) as pre-training objective
2. Generate reconstruction targets using pre-trained CLIP image encoder
3. Replace absolute positional embeddings with 2D RoPE
4. Replace GELU MLPs with SwiGLU for better capacity
5. Train on large-scale data with strong augmentation

### Key Insight
RoPE encodes relative positions via rotations in complex space, naturally extending to 2D grids. This provides better inductive bias for vision. CLIP features as targets provide semantic supervision without discrete tokenization. SwiGLU increases model capacity without massive parameter growth.

## Mathematical Formulation

### 1. Masked Image Modeling Setup
```
Input image: x ∈ R^(H×W×3)
Patch tokenization: {p₁, p₂, ..., p_N} where N = HW/P²

Random masking: Keep ratio r (e.g., 40%)
Visible patches: V = {p_i | i ∈ visible}
Masked patches: M = {p_i | i ∈ masked}

Goal: Predict CLIP features for masked patches
```

### 2. 2D Rotary Position Embeddings (RoPE)
For position (x, y) in 2D grid:
```
# Standard 1D RoPE for position m:
Θ = {θ₁, θ₂, ..., θ_{d/2}} where θ_i = 10000^(-2i/d)

RoPE(q, m) = [
    q₁cos(mθ₁) - q₂sin(mθ₁),
    q₂cos(mθ₁) + q₁sin(mθ₁),
    q₃cos(mθ₂) - q₄sin(mθ₂),
    q₄cos(mθ₂) + q₃sin(mθ₂),
    ...
]

# 2D Extension for grid position (x, y):
Split dimension: d → d/2 + d/2 (for x and y)

RoPE_x(q, x) = apply RoPE to first d/2 dimensions with position x
RoPE_y(q, y) = apply RoPE to last d/2 dimensions with position y

RoPE_2D(q, (x,y)) = Concat(RoPE_x(q[:d/2], x), RoPE_y(q[d/2:], y))
```

### 3. SwiGLU Activation
Swish-Gated Linear Unit:
```
Standard MLP:
MLP(x) = W₂·GELU(W₁·x)
Hidden dim: 4D

SwiGLU:
SwiGLU(x) = (Swish(W₁·x) ⊙ W₂·x) · W₃
where Swish(x) = x·σ(x) = x·sigmoid(x)

Hidden dim: (8/3)·D (compensates for gating)

Benefits:
- Gating provides adaptive feature selection
- Swish is smooth, avoids dead neurons
- Better capacity-to-parameter ratio
```

### 4. EVA-02 Transformer Block
```
Input: z ∈ R^(N×D)

# Apply 2D RoPE to Q and K (skip CLS token)
Generate (cos_emb, sin_emb) for grid positions
Q_rope = RoPE_2D(Q, positions)
K_rope = RoPE_2D(K, positions)

# Multi-head self-attention
Attn(Q_rope, K_rope, V) = Softmax(Q_rope·K_rope^T / √d_k) · V

# Residual connection
z' = z + Attn(LN(z))

# SwiGLU MLP
z'' = z' + SwiGLU(LN(z'))

Output: z''
```

### 5. MIM Pre-training with CLIP Teacher
```
# CLIP teacher (frozen)
Teacher: f_CLIP (pre-trained)

# Student encoder (trainable)
Student: f_EVA

For image x:
    # Generate CLIP targets
    with torch.no_grad():
        x_patches = extract_all_patches(x)
        targets = f_CLIP(x_patches)  # (N, D_clip)
        targets = L2Normalize(targets)

    # Mask patches
    visible, masked = random_mask(x_patches, mask_ratio=0.6)

    # Encode visible patches
    predictions = f_EVA(visible)  # (N_visible, D)

    # Predict masked patch features
    masked_pred = predictions[masked_indices]

    # L2 loss
    loss = MSE(masked_pred, targets[masked_indices])
```

### 6. Complete EVA-02 Architecture
```
Input: x ∈ R^(B×3×H×W)

# Patch embedding
Patches = Conv2d(x, kernel=P, stride=P) ∈ R^(B×D×H'×W')
Patches = Reshape(Patches) → R^(B×N×D) where N=H'×W'

# Add CLS token
z₀ = [CLS; Patches] ∈ R^(B×(N+1)×D)

# Generate 2D RoPE embeddings
(cos_emb, sin_emb) = RoPE_2D(H', W', device)

# Transformer blocks with RoPE
For l = 1 to L:
    z_l = EVA02Block(z_{l-1}, rope_emb=(cos_emb, sin_emb))

# Layer norm
z_L = LayerNorm(z_L)

# Extract CLS token
output = z_L[:, 0] ∈ R^(B×D)
```

### 7. Resolution Extrapolation
Training: 224×224 → 14×14 patches
```
RoPE frequencies: computed for 14×14 grid
```

Testing: 1024×1024 → 64×64 patches
```
RoPE automatically extends to 64×64 grid
No interpolation artifacts like absolute PE
```

## High-Level Intuition

```
Input Image (224×224)
        ↓
  [Random Masking]
     40% visible, 60% masked
        ↓
   ┌────┴────┐
   │         │
Visible   Masked
Patches   Patches
   │         │
   ↓         ↓
┌──────────────────────────┐
│    EVA-02 Encoder        │
│  (Student, Trainable)    │
│                          │
│  ┌──────────────────┐   │
│  │ Patch Embedding  │   │
│  │ + CLS Token      │   │
│  └──────────────────┘   │
│          ↓               │
│  [No absolute PE!]       │
│          ↓               │
│  ┌──────────────────┐   │
│  │ Block 1          │   │
│  │ • 2D RoPE        │   │ ← Relative position
│  │ • Self-Attention │   │    via rotation
│  │ • SwiGLU MLP     │   │
│  └──────────────────┘   │
│          ↓               │
│  (Repeat 12-40 layers)  │
│          ↓               │
│  ┌──────────────────┐   │
│  │ Decoder Head     │   │
│  │ Predict features │   │
│  └──────────────────┘   │
└──────────────────────────┘
          ↓
  Predicted Features
   for masked patches
          ↓
       [L2 Loss]
          ↑
┌──────────────────────────┐
│   CLIP Teacher           │
│   (Frozen)               │
│                          │
│  Extract patch features  │
│  for ALL patches         │
│  → Ground truth targets  │
└──────────────────────────┘

After Pre-training:
┌──────────────────────────┐
│  EVA-02 Backbone         │
│  • Transfer to tasks     │
│  • Linear probe          │
│  • Fine-tune             │
│  • Extract features      │
└──────────────────────────┘
```

**RoPE Visualization (1D)**:
```
Position 0: q_vector → [q₁, q₂, q₃, q₄]
Position 1: q_vector → rotate by θ
Position 2: q_vector → rotate by 2θ
...

Relative position m-n encoded in rotation difference!
```

**SwiGLU vs GELU**:
```
GELU MLP:              SwiGLU:
x → [Linear]           x → [Linear₁]
    ↓                      ↓
  [GELU]              [Swish]  [Linear₂]
    ↓                    ↓         ↓
  [Linear]              [Multiply] ← Gating!
    ↓                      ↓
  output               [Linear₃]
                          ↓
                       output

More expressive!
```

## Implementation Details

### Model Variants

| Variant | Embed Dim | Depth | Heads | Params | Patch | Data |
|---------|-----------|-------|-------|--------|-------|------|
| EVA02-S | 384 | 12 | 6 | 22M | 14 | Merged-30M |
| EVA02-B | 768 | 12 | 12 | 86M | 14 | Merged-30M |
| EVA02-L | 1024 | 24 | 16 | 304M | 14 | Merged-30M |
| EVA02-E | 1408 | 56 | 16 | 5.0B | 14 | Merged-2B |

### Configuration Example

```python
config = {
    # Architecture
    "img_size": 224,
    "patch_size": 14,
    "in_channels": 3,
    "embed_dim": 1024,         # EVA02-L
    "depth": 24,
    "num_heads": 16,
    "mlp_ratio": 4.0,          # Not used with SwiGLU
    "use_swiglu": True,        # Use SwiGLU instead of GELU
    "use_rope": True,          # Use 2D RoPE instead of abs PE
    "rope_temperature": 10000.0,

    # Training
    "dropout": 0.0,
    "drop_path_rate": 0.1,
    "mask_ratio": 0.6,         # MIM masking ratio

    # Output
    "num_classes": 0,          # 0 for feature extraction
    "global_pool": "token",    # "token" or "avg"
}
```

## Code Walkthrough

Reference: `/Users/kevinyu/Projects/Nexus/nexus/models/cv/eva02.py`

### 1. 2D Rotary Position Embeddings

```python
class RoPE2D(nn.Module):
    """2D RoPE for vision."""

    def __init__(self, dim, max_resolution=224, temperature=10000.0):
        super().__init__()
        assert dim % 4 == 0, "Dim must be divisible by 4"

        self.dim = dim
        self.temperature = temperature

        # Frequency bands
        dim_per_axis = dim // 4
        freqs = 1.0 / (temperature ** (
            torch.arange(0, dim_per_axis, 2).float() / dim_per_axis
        ))
        self.register_buffer("freqs", freqs)

    def forward(self, h, w, device):
        """Generate 2D RoPE embeddings.

        Args:
            h, w: Grid height and width
            device: Device to create tensors on

        Returns:
            (cos_emb, sin_emb): Each (h*w, dim)
        """
        # Coordinate grids
        h_coords = torch.arange(h, device=device, dtype=torch.float32)
        w_coords = torch.arange(w, device=device, dtype=torch.float32)

        # Compute frequencies for each axis
        h_freqs = torch.outer(h_coords, self.freqs)  # (h, dim//4)
        w_freqs = torch.outer(w_coords, self.freqs)  # (w, dim//4)

        # Expand to grid
        h_grid = h_freqs.unsqueeze(1).expand(h, w, -1)  # (h, w, dim//4)
        w_grid = w_freqs.unsqueeze(0).expand(h, w, -1)  # (h, w, dim//4)

        # Duplicate for cos/sin pairs
        freqs_h = torch.stack([h_grid, h_grid], dim=-1).flatten(-2)
        freqs_w = torch.stack([w_grid, w_grid], dim=-1).flatten(-2)

        # Combine both axes: (h, w, dim)
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)

        # Flatten spatial: (h*w, dim)
        freqs = freqs.reshape(-1, self.dim)

        return freqs.cos(), freqs.sin()

def apply_rope_2d(x, cos_emb, sin_emb):
    """Apply 2D RoPE.

    Args:
        x: (B, N, dim)
        cos_emb, sin_emb: (N, dim)

    Returns:
        Rotated x: (B, N, dim)
    """
    # Split into pairs
    x1, x2 = x.chunk(2, dim=-1)

    # Split embeddings
    cos_half = cos_emb.chunk(2, dim=-1)
    sin_half = sin_emb.chunk(2, dim=-1)

    # Rotate
    out1 = x1 * cos_half[0] - x2 * sin_half[0]
    out2 = x1 * sin_half[1] + x2 * cos_half[1]

    return torch.cat([out1, out2], dim=-1)
```

### 2. SwiGLU Activation

```python
class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (..., dim)
        Returns:
            (..., dim)
        """
        # Swish activation: x * sigmoid(x)
        swish_out = F.silu(self.w1(x))

        # Gate
        gated = swish_out * self.w2(x)

        # Project back
        out = self.w3(gated)
        return self.dropout(out)
```

### 3. EVA-02 Transformer Block

```python
class EVA02Block(NexusModule):
    """Transformer block with RoPE and SwiGLU."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0,
                 dropout=0.0, drop_path=0.0,
                 use_swiglu=True, rope_2d=None):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        # SwiGLU or standard MLP
        if use_swiglu:
            hidden_dim = int(dim * 8 / 3)  # Adjusted ratio
            self.mlp = SwiGLU(dim, hidden_dim, dropout)
        else:
            hidden_dim = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.rope_2d = rope_2d

    def forward(self, x, rope_emb=None):
        """
        Args:
            x: (B, N, dim) where N = h*w + 1 (CLS token)
            rope_emb: Optional (cos_emb, sin_emb) for RoPE

        Returns:
            (B, N, dim)
        """
        shortcut = x
        x = self.norm1(x)

        # Apply RoPE to Q and K (skip CLS token)
        if rope_emb is not None and self.rope_2d is not None:
            cos_emb, sin_emb = rope_emb

            # Split CLS and patches
            cls_token = x[:, :1]
            patches = x[:, 1:]

            # Apply RoPE to patches
            patches = apply_rope_2d(patches, cos_emb, sin_emb)

            # Recombine
            x = torch.cat([cls_token, patches], dim=1)

        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = shortcut + self.drop_path(attn_out)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
```

### 4. Complete EVA-02 Model

```python
class EVA02(NexusModule):
    def __init__(self, config):
        super().__init__(config)

        self.img_size = config.get("img_size", 224)
        self.patch_size = config.get("patch_size", 14)
        self.embed_dim = config.get("embed_dim", 1024)
        self.depth = config.get("depth", 24)
        self.num_heads = config.get("num_heads", 16)
        self.use_swiglu = config.get("use_swiglu", True)
        self.use_rope = config.get("use_rope", True)

        self.grid_size = self.img_size // self.patch_size
        self.num_patches = self.grid_size ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.get("in_channels", 3),
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # 2D RoPE or standard PE
        if self.use_rope:
            self.rope_2d = RoPE2D(
                dim=self.embed_dim,
                max_resolution=self.grid_size,
                temperature=config.get("rope_temperature", 10000.0)
            )
        else:
            self.rope_2d = None
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + 1, self.embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(
            0, config.get("drop_path_rate", 0.1), self.depth
        )]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            EVA02Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=config.get("dropout", 0.0),
                drop_path=dpr[i],
                use_swiglu=self.use_swiglu,
                rope_2d=self.rope_2d
            ) for i in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        # Classification head (optional)
        num_classes = config.get("num_classes", 0)
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.init_weights_vit()

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            dict with embeddings, patch_tokens, logits
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Positional encoding
        if self.use_rope:
            # Generate RoPE for current resolution
            rope_emb = self.rope_2d(self.grid_size, self.grid_size, x.device)
        else:
            x = x + self.pos_embed
            rope_emb = None

        # Transformer blocks
        for block in self.blocks:
            x = block(x, rope_emb=rope_emb)

        x = self.norm(x)

        # Global pooling
        if config.get("global_pool", "token") == "avg":
            embeddings = x[:, 1:].mean(dim=1)
        else:
            embeddings = x[:, 0]

        output = {
            "embeddings": embeddings,
            "patch_tokens": x[:, 1:]
        }

        # Classification
        if config.get("num_classes", 0) > 0:
            logits = self.head(embeddings)
            output["logits"] = logits

        return output
```

## Optimization Tricks

### 1. Resolution Extrapolation

```python
# Train at 224px
model = EVA02(config={"img_size": 224, "patch_size": 14})
train(model)

# Test at higher resolution (no retraining!)
test_images = load_images(size=448)  # 2× resolution
with torch.no_grad():
    features = model(test_images)  # RoPE handles it automatically
```

### 2. Efficient MIM Training

```python
# Block-wise masking (faster than random)
def blockwise_mask(x, mask_ratio=0.6, block_size=4):
    """Mask in blocks for efficiency."""
    B, N, D = x.shape
    num_blocks = int(N / block_size)
    num_mask = int(num_blocks * mask_ratio)

    # Random block indices
    mask_idx = torch.randperm(num_blocks)[:num_mask]

    # Expand to patches
    mask = torch.zeros(num_blocks, dtype=torch.bool)
    mask[mask_idx] = True
    mask = mask.repeat_interleave(block_size)[:N]

    return mask
```

### 3. Mixed Precision with SwiGLU

```python
# SwiGLU benefits from FP32 in gating
class SwiGLU(nn.Module):
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            # Compute gating in FP32 for stability
            swish = F.silu(self.w1(x.float()))
            gate = self.w2(x.float())
            gated = (swish * gate).to(x.dtype)

        return self.w3(gated)
```

### 4. CLIP Target Caching

```python
# Pre-compute CLIP targets to save memory
@torch.no_grad()
def precompute_clip_targets(dataset, clip_model):
    """Cache CLIP features for entire dataset."""
    targets = []
    for images in dataset:
        feats = clip_model(images)
        targets.append(feats.cpu())
    return targets

# Use cached targets during training
clip_targets = precompute_clip_targets(train_dataset, clip_teacher)
```

### 5. Gradient Checkpointing

```python
# Save memory on deep models
from torch.utils.checkpoint import checkpoint

for i, block in enumerate(self.blocks):
    if self.training and i % 3 == 0:  # Every 3rd block
        x = checkpoint(block, x, rope_emb)
    else:
        x = block(x, rope_emb)
```

## Experiments & Results

### ImageNet-1K Classification

**Setup**:
- Pre-training: MIM on Merged-30M
- Fine-tuning: ImageNet-1K, 224×224
- Evaluation: 224×224 and 448×448

**Results** (Top-1 Accuracy):

| Model | Params | 224px | 448px |
|-------|--------|-------|-------|
| ViT-L/14 | 304M | 85.2% | 86.2% |
| BEiT-L/14 | 304M | 85.7% | 86.8% |
| **EVA02-L/14** | **304M** | **88.3%** | **89.2%** |
| **EVA02-E/14** | **5.0B** | **89.6%** | **90.1%** |

EVA-02 achieves new SOTA!

### Zero-Shot Transfer (Linear Probe)

| Model | IN-1K | IN-V2 | IN-Real | Avg |
|-------|-------|-------|---------|-----|
| CLIP-L | 75.3% | 69.8% | 84.2% | 76.4% |
| DINOv2-L | 86.3% | 77.1% | 91.2% | 84.9% |
| **EVA02-L** | **87.1%** | **78.5%** | **92.1%** | **85.9%** |

### Dense Prediction (COCO Object Detection)

**Cascade Mask R-CNN backbone**:

| Backbone | box AP | mask AP |
|----------|--------|---------|
| Swin-L | 53.9 | 46.7 |
| ViT-L | 54.1 | 47.0 |
| **EVA02-L** | **55.8** | **48.5** |

### Resolution Extrapolation

**Training: 224×224, Testing: Variable**:

| Resolution | ViT-L (Interp) | EVA02-L (RoPE) |
|-----------|---------------|---------------|
| 224 | 88.3% | 88.3% |
| 336 | 88.7% (+0.4) | 89.1% (+0.8) |
| 448 | 88.9% (+0.6) | 89.5% (+1.2) |
| 560 | 88.6% (+0.3) | 89.7% (+1.4) |
| 1024 | 87.2% (-1.1) | 89.3% (+1.0) |

RoPE extrapolates much better than interpolated absolute PE!

### Ablation Studies

**Architectural Choices**:

| Config | ImageNet Top-1 |
|--------|---------------|
| Baseline (abs PE + GELU) | 87.1% |
| + RoPE | 87.8% (+0.7) |
| + SwiGLU | 87.9% (+0.8) |
| + Both (EVA02) | **88.3%** (+1.2) |

**MIM Target Choice**:

| Target | Top-1 |
|--------|-------|
| Pixels | 85.2% |
| DALL-E tokens | 86.5% |
| CLIP features | **88.3%** |

CLIP features provide best semantic supervision.

## Common Pitfalls

### 1. RoPE Dimension Mismatch
Wrong: Not divisible by 4
```python
config = {"embed_dim": 770}  # BUG: 770 % 4 != 0
```

Correct: Ensure divisibility
```python
config = {"embed_dim": 768}  # 768 % 4 == 0
```

### 2. Applying RoPE to CLS Token
Wrong: RoPE on all tokens
```python
x_rope = apply_rope_2d(x, cos, sin)  # BUG: CLS has no position
```

Correct: Skip CLS token
```python
cls = x[:, :1]
patches = apply_rope_2d(x[:, 1:], cos, sin)
x = torch.cat([cls, patches], dim=1)
```

### 3. Wrong SwiGLU Hidden Dimension
Wrong: Using standard 4× ratio
```python
hidden_dim = dim * 4  # Too large for SwiGLU
```

Correct: Use 8/3× ratio
```python
hidden_dim = int(dim * 8 / 3)  # Compensates for gating
```

### 4. Not Extrapolating Resolution
Wrong: Retraining for each resolution
```python
model_224 = EVA02(img_size=224)
model_448 = EVA02(img_size=448)  # Unnecessary!
```

Correct: One model, multiple resolutions
```python
model = EVA02(img_size=224)
# Use for any resolution ≥ 224
```

### 5. Forgetting to Freeze CLIP Teacher
Wrong: CLIP gradients flow
```python
clip_targets = clip_model(images)  # BUG: Gradients!
```

Correct: Always no_grad
```python
with torch.no_grad():
    clip_targets = clip_model(images)
```

## References

### Original Papers
1. **EVA-02: A Visual Representation for Neon Genesis**
   Fang, Y., Sun, Q., Wang, X., et al., 2023
   https://arxiv.org/abs/2303.11331

2. **EVA: Exploring the Limits of Masked Visual Representation Learning at Scale**
   Fang, Y., et al., CVPR 2023
   https://arxiv.org/abs/2211.07636

### Architectural Components
3. **RoFormer: Enhanced Transformer with Rotary Position Embedding**
   Su, J., et al., 2021
   https://arxiv.org/abs/2104.09864

4. **GLU Variants Improve Transformer**
   Shazeer, N., 2020
   https://arxiv.org/abs/2002.05202

### Pre-training Methods
5. **Masked Autoencoders Are Scalable Vision Learners (MAE)**
   He, K., et al., CVPR 2022
   https://arxiv.org/abs/2111.06377

6. **BEiT: BERT Pre-Training of Image Transformers**
   Bao, H., et al., ICLR 2022
   https://arxiv.org/abs/2106.08254

### Implementation
- Official: https://github.com/baaivision/EVA
- Nexus: `/Users/kevinyu/Projects/Nexus/nexus/models/cv/eva02.py`
- Hugging Face: https://huggingface.co/BAAI/EVA02

### Downstream Applications
7. **CLIP: Learning Transferable Visual Models**
   Used as teacher for EVA-02
   https://arxiv.org/abs/2103.00020
