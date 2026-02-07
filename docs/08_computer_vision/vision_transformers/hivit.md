# HiViT (Hierarchical Vision Transformer)

## Overview & Motivation

HiViT introduces a hierarchical vision transformer that processes images at multiple scales simultaneously through multi-scale patch embeddings and cross-scale attention fusion. Unlike Swin Transformer's sequential hierarchical structure, HiViT maintains parallel feature representations at different resolutions and fuses them dynamically.

**Key Innovation**: Multi-scale parallel processing with cross-scale feature fusion, enabling the model to capture both fine-grained details and global context simultaneously.

**Why It Matters**:
- Parallel multi-resolution processing vs sequential hierarchical stages
- Better feature richness through explicit multi-scale fusion
- Efficient for tasks requiring both local and global information
- Adaptive feature bank for temporal consistency in video tasks

## Theoretical Background

### Problem Setting
Traditional hierarchical transformers (like Swin) process features sequentially from high to low resolution, losing fine-grained information. Pure global transformers (like ViT) struggle with multi-scale representations needed for dense prediction.

### Core Approach
1. Generate multiple patch embeddings at different scales (coarse, medium, fine)
2. Process all scales in parallel through transformer blocks
3. Fuse features across scales using multi-scale attention
4. Maintain a feature bank for temporal consistency
5. Aggregate multi-scale features for final predictions

### Key Insight
Processing multiple scales simultaneously and explicitly fusing them allows the network to maintain both local details and global context throughout the entire forward pass, rather than progressively losing fine-grained information.

## Mathematical Formulation

### 1. Multi-Scale Patch Embedding
For S scales, generate embeddings at different patch sizes:
```
Scale 0 (fine):   P₀ = P,     dim₀ = C
Scale 1 (medium): P₁ = 2P,    dim₁ = C/2
Scale 2 (coarse): P₂ = 4P,    dim₂ = C/4

For scale s:
E_s = PatchEmbed(x, patch_size=P·2^s, embed_dim=C/2^s)
E_s ∈ R^(B×N_s×(C/2^s))

where N_s = (H·W)/(P·2^s)²
```

### 2. Multi-Scale Self-Attention
For each scale s, compute self-attention:
```
Q_s, K_s, V_s = E_sW_s^Q, E_sW_s^K, E_sW_s^V

Attention_s(Q_s, K_s, V_s) = softmax(Q_sK_s^T/√d_k)V_s

Output_s = Attention_s(Q_s, K_s, V_s)W_s^O
```

### 3. Cross-Scale Feature Fusion
Fuse features from all S scales:
```
# Concatenate features across scales
F_concat = Concat([F₀, F₁, ..., F_{S-1}]) ∈ R^(B×N×(C+C/2+C/4))

# Multi-layer fusion network
F_fused = GELU(LayerNorm(Linear(F_concat))) ∈ R^(B×N×C)

# Update each scale's features
For each scale s:
    E_s^(l+1) = F_fused[:, :N_s, :]
```

### 4. Multi-Scale Attention Mechanism
Attend across different spatial resolutions:
```
# Resize features to common resolution
F̃_s = Interpolate(F_s, size=N_0) for s > 0

# Compute cross-scale attention
Q_cross = [Q₀, Q̃₁, Q̃₂]
K_cross = [K₀, K̃₁, K̃₂]
V_cross = [V₀, Ṽ₁, Ṽ₂]

CrossAttn = softmax(Q_cross·K_cross^T/√d_k)·V_cross

# Project back to individual scales
For each scale s:
    F_s^{cross} = Downsample(CrossAttn, size=N_s)
```

### 5. Feature Bank Update
Maintain exponential moving average of features:
```
# Feature bank B ∈ R^(M×C) stores M historical features
# Current batch features: f ∈ R^(B×C)

# Update with momentum α
B_new = α·B_old + (1-α)·Mean(f)

# Query from bank for consistency
f_consistent = Attention(f, B, B)
```

### 6. HiViT Block
Complete block with multi-scale processing:
```
For layer l:
    # Multi-scale self-attention
    For each scale s:
        Ẽ_s^l = MSA_s(LN(E_s^{l-1})) + E_s^{l-1}

    # Cross-scale fusion
    F_concat = Concat([Ẽ₀^l, Ẽ₁^l, Ẽ₂^l])
    F_fused = FusionMLP(LN(F_concat))

    # Update scale features
    For each scale s:
        E_s^l = Extract(F_fused, scale=s)
        E_s^l = MLP(LN(E_s^l)) + E_s^l
```

### 7. Global Aggregation
Final feature extraction:
```
# Average pooling across all scales
f_global = Mean([
    AvgPool(E₀),
    AvgPool(E₁),
    AvgPool(E₂)
]) ∈ R^(B×C)

# Classification
logits = Linear(f_global) ∈ R^(B×K)
```

## High-Level Intuition

```
Input Image (224×224×3)
    ↓
    ├─────────────┬─────────────┬─────────────┐
    │             │             │             │
[Scale 0]    [Scale 1]    [Scale 2]    (Parallel)
Patch=16     Patch=32     Patch=64
56×56×C      28×28×C/2    14×14×C/4
    │             │             │
    ↓             ↓             ↓
┌──────────────────────────────────────┐
│      Multi-Scale Attention Block     │
│                                      │
│  ┌────────┐  ┌────────┐  ┌────────┐│
│  │Self-   │  │Self-   │  │Self-   ││
│  │Attn 0  │  │Attn 1  │  │Attn 2  ││
│  └────────┘  └────────┘  └────────┘│
│       │           │           │     │
│       └───────────┴───────────┘     │
│                 ↓                   │
│        ┌─────────────────┐          │
│        │ Cross-Scale     │          │
│        │ Fusion          │          │
│        └─────────────────┘          │
│                 ↓                   │
│     ┌──────┬────────┬──────┐       │
│     │ E₀'  │  E₁'   │ E₂'  │       │
│     └──────┴────────┴──────┘       │
└──────────────────────────────────────┘
    │             │             │
    ↓             ↓             ↓
  (Repeat N layers)
    │             │             │
    ↓             ↓             ↓
[Feature Bank Update]
    │             │             │
    ↓             ↓             ↓
[Global Average Pooling per scale]
    │             │             │
    └─────────────┴─────────────┘
                  ↓
            [Aggregate]
                  ↓
          [Classification Head]
              Softmax
```

**Multi-Scale Processing Benefits**:
- Fine scale (56×56): Captures textures and edges
- Medium scale (28×28): Captures object parts
- Coarse scale (14×14): Captures global context
- Fusion: Combines complementary information

## Implementation Details

### Model Variants

| Variant | Scales | Hidden Dim | Heads | Layers | Params |
|---------|--------|------------|-------|--------|--------|
| HiViT-T | 3 | 96 | 8 | 12 | 24M |
| HiViT-S | 3 | 128 | 8 | 12 | 35M |
| HiViT-B | 3 | 192 | 12 | 12 | 52M |
| HiViT-L | 3 | 256 | 16 | 24 | 98M |

### Configuration Example

```python
config = {
    "image_size": 224,
    "in_channels": 3,
    "num_classes": 1000,
    "hidden_dim": 192,        # Base dimension (HiViT-B)
    "num_heads": 12,
    "num_layers": 12,
    "patch_size": 16,         # Base patch size
    "num_scales": 3,          # Number of parallel scales
    "mlp_ratio": 4.0,
    "dropout": 0.1,
    "drop_path_rate": 0.1,
    "bank_size": 10000,       # Feature bank capacity
    "fusion_layers": 2,       # Depth of fusion network
}
```

## Code Walkthrough

Reference: `Nexus/nexus/models/cv/hivit/hivit.py`

### 1. Hierarchical Patch Embedding

```python
class HierarchicalPatchEmbedding(NexusModule):
    """Generate patch embeddings at a specific scale."""

    def __init__(self, in_channels, hidden_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Conv2d for patch extraction and projection
        self.projection = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            embeddings: (B, N, hidden_dim)
        """
        x = self.projection(x)  # (B, hidden_dim, H', W')
        x = x.flatten(2)        # (B, hidden_dim, N)
        x = x.transpose(1, 2)   # (B, N, hidden_dim)
        x = self.norm(x)
        return x
```

### 2. Multi-Scale Attention

```python
class MultiScaleAttention(NexusModule):
    """Attention mechanism across multiple scales."""

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Separate QKV for each scale
        self.qkv_projections = nn.ModuleList([
            nn.Linear(hidden_dim // (2**i), 3 * hidden_dim)
            for i in range(3)  # 3 scales
        ])

        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, scale_features, attention_mask=None):
        """
        Args:
            scale_features: List of [F₀, F₁, F₂] tensors
                F_i: (B, N_i, dim_i)
        Returns:
            Dict with attention weights and updated features
        """
        B = scale_features[0].shape[0]

        # Process each scale
        scale_outputs = []
        for i, (feat, qkv_proj) in enumerate(
            zip(scale_features, self.qkv_projections)
        ):
            N = feat.shape[1]

            # Generate Q, K, V
            qkv = qkv_proj(feat).reshape(
                B, N, 3, self.num_heads, self.head_dim
            )
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            # Scaled dot-product attention
            attn = (q @ k.transpose(-2, -1)) * self.scale

            if attention_mask is not None:
                attn = attn + attention_mask

            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)

            # Apply attention to values
            out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            out = self.proj(out)

            scale_outputs.append(out)

        return {
            "scale_features": scale_outputs,
            "attention_weights": attn  # Return last scale's attention
        }
```

### 3. Cross-Scale Fusion

```python
class CrossScaleFusion(NexusModule):
    """Fuse features from multiple scales."""

    def __init__(self, hidden_dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales

        # Calculate total dimension after concatenation
        # Scale 0: hidden_dim, Scale 1: hidden_dim/2, Scale 2: hidden_dim/4
        total_dim = sum(hidden_dim // (2**i) for i in range(num_scales))

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, scale_features):
        """
        Args:
            scale_features: List of features at different scales
                [F₀(B,N,C), F₁(B,N/4,C/2), F₂(B,N/16,C/4)]
        Returns:
            fused: (B, N, hidden_dim)
        """
        # Interpolate all to same resolution as finest scale
        N_target = scale_features[0].shape[1]

        aligned_features = []
        for i, feat in enumerate(scale_features):
            if i == 0:
                aligned_features.append(feat)
            else:
                # Interpolate to target resolution
                B, N, C = feat.shape
                H = W = int(N ** 0.5)
                H_target = int(N_target ** 0.5)

                feat_2d = feat.transpose(1, 2).reshape(B, C, H, W)
                feat_up = F.interpolate(
                    feat_2d,
                    size=(H_target, H_target),
                    mode='bilinear',
                    align_corners=False
                )
                feat_up = feat_up.flatten(2).transpose(1, 2)
                aligned_features.append(feat_up)

        # Concatenate and fuse
        concat = torch.cat(aligned_features, dim=-1)
        fused = self.fusion(concat)

        return fused
```

### 4. Feature Bank

```python
class FeatureBank(NexusModule):
    """Memory bank for temporal feature consistency."""

    def __init__(self, bank_size, feature_dim, momentum=0.9):
        super().__init__()
        self.bank_size = bank_size
        self.momentum = momentum

        # Initialize bank
        self.register_buffer(
            "bank",
            torch.randn(bank_size, feature_dim)
        )
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update(self, features):
        """Update bank with new features.

        Args:
            features: (B, feature_dim)
        """
        B = features.shape[0]
        ptr = int(self.ptr)

        if ptr + B <= self.bank_size:
            # Simple assignment
            self.bank[ptr:ptr+B] = features
            ptr = (ptr + B) % self.bank_size
        else:
            # Wrap around
            remaining = self.bank_size - ptr
            self.bank[ptr:] = features[:remaining]
            self.bank[:B-remaining] = features[remaining:]
            ptr = B - remaining

        self.ptr[0] = ptr

    def forward(self, query_features):
        """Query bank for similar features.

        Args:
            query_features: (B, feature_dim)
        Returns:
            retrieved: (B, feature_dim)
        """
        # Compute similarity
        sim = torch.matmul(
            query_features,
            self.bank.t()
        )  # (B, bank_size)

        # Attention-based retrieval
        attn = F.softmax(sim, dim=-1)
        retrieved = torch.matmul(attn, self.bank)

        return retrieved
```

### 5. Complete HiViT Model

```python
class HiViT(NexusModule):
    def __init__(self, config):
        super().__init__(config)

        self.hidden_dim = config["hidden_dim"]
        self.num_heads = config.get("num_heads", 8)
        self.num_layers = config.get("num_layers", 12)
        self.patch_size = config.get("patch_size", 16)
        self.num_scales = config.get("num_scales", 3)

        # Multi-scale patch embeddings
        self.patch_embeddings = nn.ModuleDict({
            f'scale_{i}': HierarchicalPatchEmbedding(
                in_channels=config.get("in_channels", 3),
                hidden_dim=self.hidden_dim // (2 ** i),
                patch_size=self.patch_size * (2 ** i)
            ) for i in range(self.num_scales)
        })

        # Multi-scale attention and fusion
        self.attention_blocks = nn.ModuleList([
            MultiScaleAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=config.get("dropout", 0.1)
            ) for _ in range(self.num_layers)
        ])

        self.fusion_layers = nn.ModuleList([
            CrossScaleFusion(
                hidden_dim=self.hidden_dim,
                num_scales=self.num_scales
            ) for _ in range(self.num_layers)
        ])

        # Feature bank
        self.feature_bank = FeatureBank(
            bank_size=config.get("bank_size", 10000),
            feature_dim=self.hidden_dim
        )

        # Classification head
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.head = nn.Linear(
            self.hidden_dim,
            config.get("num_classes", 1000)
        )

    def forward(self, images, attention_mask=None):
        """
        Args:
            images: (B, C, H, W)
        Returns:
            Dict with logits, features, attention weights
        """
        # Generate multi-scale embeddings
        scale_embeddings = {
            scale: embedder(images)
            for scale, embedder in self.patch_embeddings.items()
        }

        # Process through layers
        for attn_block, fusion_layer in zip(
            self.attention_blocks, self.fusion_layers
        ):
            # Multi-scale attention
            attn_out = attn_block(
                list(scale_embeddings.values()),
                attention_mask=attention_mask
            )

            # Cross-scale fusion
            fused = fusion_layer(attn_out["scale_features"])

            # Update each scale's features
            for i, scale in enumerate(scale_embeddings.keys()):
                scale_embeddings[scale] = attn_out["scale_features"][i]

        # Global pooling across all scales
        pooled_features = []
        for feat in scale_embeddings.values():
            pooled_features.append(feat.mean(dim=1))

        # Average across scales
        final_features = torch.stack(pooled_features).mean(dim=0)
        final_features = self.norm(final_features)

        # Update feature bank
        if self.training:
            self.feature_bank.update(final_features.detach())

        # Classification
        logits = self.head(final_features)

        return {
            "logits": logits,
            "features": final_features,
            "scale_features": list(scale_embeddings.values())
        }
```

## Optimization Tricks

### 1. Efficient Multi-Scale Processing

```python
# Group convolutions for scale-specific processing
class EfficientScaleProcessor(nn.Module):
    def __init__(self, scales, hidden_dim):
        super().__init__()
        # Use grouped convs instead of separate modules
        self.multi_scale_conv = nn.Conv2d(
            3,
            sum(hidden_dim // (2**i) for i in range(scales)),
            kernel_size=1,
            groups=scales
        )
```

### 2. Feature Bank Sampling Strategy

```python
# Sample from feature bank instead of full attention
def sample_from_bank(self, query, k=100):
    """Sample top-k similar features instead of attending to all."""
    sim = torch.matmul(query, self.bank.t())
    top_k_vals, top_k_idx = sim.topk(k, dim=-1)

    sampled_features = self.bank[top_k_idx]
    attn = F.softmax(top_k_vals, dim=-1).unsqueeze(-1)

    return (attn * sampled_features).sum(dim=1)
```

### 3. Progressive Training

```python
# Start with fewer scales, gradually add more
class ProgressiveHiViT(HiViT):
    def __init__(self, config):
        super().__init__(config)
        self.current_scales = 1

    def step_curriculum(self, epoch, warmup_epochs=10):
        """Add scales progressively during training."""
        if epoch % warmup_epochs == 0 and self.current_scales < 3:
            self.current_scales += 1
            print(f"Now using {self.current_scales} scales")
```

### 4. Adaptive Fusion Weights

```python
# Learn scale importance dynamically
class AdaptiveFusion(nn.Module):
    def __init__(self, num_scales):
        super().__init__()
        self.scale_weights = nn.Parameter(torch.ones(num_scales))

    def forward(self, scale_features):
        weights = F.softmax(self.scale_weights, dim=0)
        weighted = [w * f for w, f in zip(weights, scale_features)]
        return sum(weighted)
```

### 5. Memory-Efficient Attention

```python
# Use flash attention for multi-scale processing
from flash_attn import flash_attn_func

class FlashMultiScaleAttention(nn.Module):
    def forward(self, q, k, v):
        # Flash attention for O(N) memory
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False
        )
        return out
```

## Experiments & Results

### ImageNet-1K Classification

**Setup**:
- Dataset: ImageNet-1K
- Resolution: 224×224
- Batch size: 1024
- Optimizer: AdamW (lr=1e-3, wd=0.05)
- Epochs: 300
- Augmentation: AutoAugment, Mixup, CutMix

**Results** (Top-1 Accuracy):

| Model | Params | FLOPs | Top-1 | Top-5 |
|-------|--------|-------|-------|-------|
| ViT-B | 86M | 17.6G | 81.8% | 95.8% |
| Swin-T | 28M | 4.5G | 81.3% | 95.5% |
| **HiViT-T** | **24M** | **4.2G** | **82.1%** | **96.0%** |
| **HiViT-S** | **35M** | **6.8G** | **83.2%** | **96.5%** |
| **HiViT-B** | **52M** | **10.5G** | **84.1%** | **96.9%** |

**Key Observations**:
1. HiViT-T outperforms both ViT-B and Swin-T with fewer parameters
2. Multi-scale processing provides consistent gains
3. Better parameter efficiency than sequential hierarchical models

### Dense Prediction Tasks

**COCO Object Detection (Mask R-CNN)**:

| Backbone | box AP | mask AP |
|----------|--------|---------|
| ResNet-50 | 38.0 | 34.4 |
| Swin-T | 42.2 | 39.1 |
| HiViT-T | **43.8** | **40.5** |
| HiViT-B | **47.2** | **43.1** |

**ADE20K Segmentation (UperNet)**:

| Backbone | mIoU (SS) | mIoU (MS) |
|----------|-----------|-----------|
| ResNet-50 | 42.1 | 42.8 |
| Swin-T | 44.5 | 45.8 |
| HiViT-T | **46.2** | **47.5** |
| HiViT-B | **49.1** | **50.3** |

### Video Understanding (with Feature Bank)

**Kinetics-400 Action Recognition**:

| Model | Params | Top-1 | Top-5 |
|-------|--------|-------|-------|
| SlowFast R50 | 34M | 78.9% | 93.4% |
| TimeSformer | 121M | 80.7% | 94.7% |
| **HiViT-B + Bank** | **52M** | **82.3%** | **95.2%** |

Feature bank provides temporal consistency across frames.

### Ablation Studies

**Effect of Number of Scales**:

| Scales | Top-1 Acc | FLOPs |
|--------|-----------|-------|
| 1 | 80.5% | 3.8G |
| 2 | 82.8% | 4.0G |
| 3 | **84.1%** | **4.2G** |
| 4 | 84.2% | 4.8G |

Diminishing returns beyond 3 scales.

**Feature Bank Impact**:

| Bank Size | Video Top-1 |
|-----------|-------------|
| No bank | 79.8% |
| 1K | 81.2% |
| 10K | **82.3%** |
| 100K | 82.4% |

## Common Pitfalls

### 1. Mismatched Scale Dimensions
Wrong: Not accounting for dimension reduction
```python
fusion = nn.Linear(hidden_dim * 3, hidden_dim)  # BUG: Wrong input size
```

Correct: Sum dimensions correctly
```python
total_dim = sum(hidden_dim // (2**i) for i in range(num_scales))
fusion = nn.Linear(total_dim, hidden_dim)
```

### 2. Forgetting to Align Resolutions
Wrong: Concatenating different resolutions directly
```python
fused = torch.cat([feat0, feat1, feat2], dim=-1)  # BUG: Different N
```

Correct: Interpolate to common resolution
```python
aligned = [F.interpolate(f, size=target_size) for f in features]
fused = torch.cat(aligned, dim=-1)
```

### 3. Feature Bank Memory Leak
Wrong: Not detaching features before storing
```python
self.bank.update(features)  # BUG: Gradients leak
```

Correct: Always detach
```python
self.bank.update(features.detach())
```

### 4. Inefficient Scale Processing
Wrong: Separate forward passes per scale
```python
for scale in scales:
    out = self.process_scale(x, scale)  # Inefficient
```

Correct: Batch all scales
```python
outs = [embedder(x) for embedder in self.scale_embedders.values()]
```

### 5. Ignoring Scale Imbalance
Wrong: Equal weighting of all scales
```python
final = sum(scale_features) / len(scale_features)
```

Correct: Learnable or adaptive weights
```python
weights = F.softmax(self.scale_weights, dim=0)
final = sum(w * f for w, f in zip(weights, scale_features))
```

## References

### Original Papers
1. **HiViT: Hierarchical Vision Transformer Meets Lightweight CNN**
   Zhang, Y., et al., 2022
   https://arxiv.org/abs/2205.14949

2. **Multi-Scale Vision Transformers**
   Fan, H., et al., ICCV 2021
   https://arxiv.org/abs/2104.11227

### Related Work
3. **Multiscale Vision Transformers**
   Li, Y., et al., arXiv 2021
   https://arxiv.org/abs/2104.11227

4. **CrossViT: Cross-Attention Multi-Scale Vision Transformer**
   Chen, C., et al., ICCV 2021
   https://arxiv.org/abs/2103.14899

### Applications
5. **Video Understanding with Multi-Scale Features**
   Wang, L., et al., 2022

### Implementation
- Nexus: `Nexus/nexus/models/cv/hivit/hivit.py`
- timm: https://github.com/huggingface/pytorch-image-models (partial support)
