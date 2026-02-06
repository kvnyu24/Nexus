# Swin Transformer

## Overview & Motivation

The Swin Transformer introduces a hierarchical vision transformer architecture that builds representations at multiple scales using shifted windows. Unlike ViT's global self-attention, Swin computes attention within local non-overlapping windows that shift between layers, enabling efficient computation while capturing both local and global dependencies.

**Key Innovation**: Shifted window-based multi-head self-attention (SW-MSA) that maintains linear computational complexity relative to image size while enabling cross-window connections.

**Why It Matters**:
- Hierarchical feature maps ideal for dense prediction tasks (detection, segmentation)
- Linear complexity O(n) vs quadratic O(n²) of standard ViT
- State-of-the-art on COCO object detection and ADE20K segmentation
- Serves as general-purpose backbone for both vision and vision-language tasks

## Theoretical Background

### Problem Setting
Standard ViT computes global attention over all patches, resulting in quadratic complexity unsuitable for high-resolution images. Dense prediction tasks require multi-scale hierarchical features like CNNs provide.

### Core Approach
1. Partition image into non-overlapping windows
2. Compute self-attention within each window locally
3. Shift window positions between layers for cross-window connections
4. Progressively merge patches to build hierarchical representations
5. Use relative position bias for better positional awareness

### Key Insight
Local attention in non-overlapping windows is efficient (linear complexity), but shifting windows between consecutive layers enables information flow across windows, achieving the modeling power of global attention with local efficiency.

## Mathematical Formulation

### 1. Patch Partitioning and Embedding
```
Input: x ∈ R^(H×W×3)
Patches: Split into M×M non-overlapping patches of size P×P
Embedding: z₀ = Linear(Flatten(patches)) ∈ R^((HW/P²)×C)
where C = 96 (base dimension)
```

### 2. Window-Based Multi-Head Self-Attention (W-MSA)
Partition feature map into M×M windows of size W×W:
```
Number of windows: n_w = ⌈H/W⌉ × ⌈W/W⌉
Tokens per window: W²

For window i:
Q, K, V = z_iW^Q, z_iW^K, z_iW^V
Attention(Q,K,V) = SoftMax(QK^T/√d + B)V
```

where B ∈ R^(W²×W²) is the relative position bias.

### 3. Shifted Window Multi-Head Self-Attention (SW-MSA)
Shift windows by (⌊W/2⌋, ⌊W/2⌋) pixels:
```
Layer l (even): Regular windows starting at (0,0)
Layer l+1 (odd): Shifted windows starting at (⌊W/2⌋, ⌊W/2⌋)
```

To maintain efficiency, use cyclic shifting and masking:
```
Shifted-z = CyclicShift(z, shift_size=⌊W/2⌋)
Attention with mask M to prevent invalid cross-window connections
Output = ReverseCyclicShift(Attention(Shifted-z, M))
```

### 4. Relative Position Bias
For each head, learnable relative position bias:
```
B ∈ R^((2W-1)×(2W-1))  # All possible relative positions
B̂ ∈ R^(W²×W²)          # Indexed bias for actual pairs

Attention = SoftMax(QK^T/√d + B̂)V
```

### 5. Swin Transformer Block
```
ẑ^l = W-MSA(LN(z^(l-1))) + z^(l-1)
z^l = MLP(LN(ẑ^l)) + ẑ^l

ẑ^(l+1) = SW-MSA(LN(z^l)) + z^l
z^(l+1) = MLP(LN(ẑ^(l+1))) + ẑ^(l+1)
```

### 6. Patch Merging
Down-sample feature maps between stages:
```
Concatenate 2×2 neighboring patches: R^(H×W×C) → R^(H/2×W/2×4C)
Linear projection: R^(H/2×W/2×4C) → R^(H/2×W/2×2C)
```

### 7. Complexity Analysis
```
W-MSA complexity: Ω(W-MSA) = 4hwC² + 2(W²)hwC
Global MSA complexity: Ω(MSA) = 4hwC² + 2h²w²C

For h=w=56, W=7, C=128:
W-MSA: 6.5 GFLOPs
Global MSA: 71.4 GFLOPs (11× more)
```

## High-Level Intuition

```
Input Image (224×224×3)
    ↓
[Patch Partition] 4×4 patches
56×56×96 feature map
    ↓
┌─────────────────────────┐
│ Stage 1: 56×56×96       │
│ ┌─────────────────────┐ │
│ │ Swin Block (W-MSA)  │ │  Regular 7×7 windows
│ │ Swin Block (SW-MSA) │ │  Shifted windows
│ └─────────────────────┘ │
│ Repeated N times        │
└─────────────────────────┘
    ↓
[Patch Merging] 2×2→1
28×28×192 feature map
    ↓
┌─────────────────────────┐
│ Stage 2: 28×28×192      │
│ W-MSA / SW-MSA blocks   │
└─────────────────────────┘
    ↓
[Patch Merging]
14×14×384 feature map
    ↓
┌─────────────────────────┐
│ Stage 3: 14×14×384      │
│ W-MSA / SW-MSA blocks   │
└─────────────────────────┘
    ↓
[Patch Merging]
7×7×768 feature map
    ↓
┌─────────────────────────┐
│ Stage 4: 7×7×768        │
│ W-MSA / SW-MSA blocks   │
└─────────────────────────┘
    ↓
[Classification Head]
Global Average Pool → Linear → Softmax
```

**Window Shifting Visualization**:
```
Layer l (W-MSA):        Layer l+1 (SW-MSA):
┌───┬───┬───┬───┐      ┌─┬─────┬─────┬───┐
│ 0 │ 1 │ 2 │ 3 │      │ │     │     │   │
├───┼───┼───┼───┤      ├─┼─────┼─────┼───┤
│ 4 │ 5 │ 6 │ 7 │  →   │ │  0  │  1  │ 2 │
├───┼───┼───┼───┤      ├─┼─────┼─────┼───┤
│ 8 │ 9 │10 │11 │      │ │  3  │  4  │ 5 │
├───┼───┼───┼───┤      ├─┼─────┼─────┼───┤
│12 │13 │14 │15 │      │ │     │     │   │
└───┴───┴───┴───┘      └─┴─────┴─────┴───┘
```

## Implementation Details

### Model Variants

| Variant | Layers | Hidden Dim | Heads | Params | Window | FLOPs |
|---------|--------|------------|-------|--------|--------|-------|
| Swin-T  | [2,2,6,2] | 96 | [3,6,12,24] | 28M | 7 | 4.5G |
| Swin-S  | [2,2,18,2] | 96 | [3,6,12,24] | 50M | 7 | 8.7G |
| Swin-B  | [2,2,18,2] | 128 | [4,8,16,32] | 88M | 7 | 15.4G |
| Swin-L  | [2,2,18,2] | 192 | [6,12,24,48] | 197M | 7 | 34.5G |

### Configuration Example

```python
config = {
    "image_size": 224,
    "patch_size": 4,
    "in_channels": 3,
    "num_classes": 1000,
    "embed_dim": 96,           # C (Swin-T)
    "depths": [2, 2, 6, 2],    # Layers per stage
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "drop_path_rate": 0.2,     # Stochastic depth
    "use_checkpoint": False,   # Gradient checkpointing
}
```

## Code Walkthrough

Reference: `/Users/kevinyu/Projects/Nexus/nexus/models/cv/swin_transformer.py`

### 1. Window Partition and Reverse

```python
def window_partition(x, window_size):
    """Partition feature map into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size: Window size

    Returns:
        windows: (B*num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """Reverse window partition back to feature map.

    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size: Window size
        H, W: Height and width of feature map

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x
```

### 2. Window Attention with Relative Position Bias

```python
class WindowAttention(NexusModule):
    def __init__(self, dim, window_size, num_heads, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )

        # Pre-compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = coords.flatten(1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B*num_windows, N, C)
            mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).contiguous()

        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply attention mask for shifted windows
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

### 3. Swin Transformer Block

```python
class SwinTransformerBlock(NexusModule):
    def __init__(self, dim, num_heads, window_size, shift_size=0,
                 mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size, num_heads, dropout
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask_matrix=None):
        """
        Args:
            x: (B, H*W, C)
            mask_matrix: Attention mask for shifted windows
        """
        H, W = self.H, self.W
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # Partition into windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size ** 2, C)

        # W-MSA or SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
```

### 4. Patch Merging

```python
class PatchMerging(NexusModule):
    """Merge 2×2 neighboring patches and reduce spatial resolution."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
            H, W: Spatial dimensions
        Returns:
            Merged features (B, H/2*W/2, 2*C)
        """
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # Concatenate 2×2 patches
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right

        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x
```

## Optimization Tricks

### 1. Efficient Attention Mask Generation

```python
def create_mask(H, W, window_size, shift_size, device):
    """Generate attention mask for SW-MSA."""
    img_mask = torch.zeros((1, H, W, 1), device=device)

    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    # Partition into windows
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)

    # Create attention mask: same region → 0, different → -100
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

    return attn_mask
```

### 2. Gradient Checkpointing for Memory

```python
from torch.utils.checkpoint import checkpoint

class SwinTransformer(NexusModule):
    def __init__(self, config):
        super().__init__(config)
        self.use_checkpoint = config.get("use_checkpoint", False)
        # ... rest of init

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        return x
```

### 3. Stochastic Depth Schedule

```python
# Linear decay schedule for drop path
drop_path_rate = 0.2
depths = [2, 2, 6, 2]
total_depth = sum(depths)

dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

# Assign to blocks
block_idx = 0
for stage_idx, num_blocks in enumerate(depths):
    for i in range(num_blocks):
        block = SwinTransformerBlock(
            drop_path=dpr[block_idx],
            ...
        )
        block_idx += 1
```

### 4. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in dataloader:
    with autocast():
        outputs = model(images)
        loss = criterion(outputs["logits"], labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 5. Learning Rate Scaling

```python
# Layer-wise LR decay (better for Swin)
from timm.optim import create_optimizer_v2

optimizer = create_optimizer_v2(
    model,
    opt='adamw',
    lr=1e-3,
    weight_decay=0.05,
    layer_decay=0.9,  # Decay earlier layers
)
```

## Experiments & Results

### ImageNet-1K Classification

**Setup**:
- Dataset: ImageNet-1K (1.28M images)
- Resolution: 224×224 → 384×384 fine-tuning
- Batch size: 1024
- Optimizer: AdamW (lr=1e-3, wd=0.05)
- Epochs: 300
- Augmentation: RandAugment, Mixup, CutMix, Random Erasing

**Results** (Top-1 Accuracy):

| Model | Params | FLOPs | 224px | 384px |
|-------|--------|-------|-------|-------|
| ResNet-50 | 25M | 4.1G | 76.2% | 78.3% |
| DeiT-B | 86M | 17.5G | 81.8% | 83.1% |
| ViT-B/16 | 86M | 17.6G | 77.9% | 84.1% |
| **Swin-T** | **28M** | **4.5G** | **81.3%** | **83.2%** |
| **Swin-S** | **50M** | **8.7G** | **83.0%** | **84.5%** |
| **Swin-B** | **88M** | **15.4G** | **83.5%** | **85.2%** |
| **Swin-L** | **197M** | **34.5G** | **86.3%** | **87.3%** |

**Key Observations**:
1. Swin-T achieves 81.3% with only 28M parameters (vs 86M for ViT-B)
2. Better parameter efficiency than ViT at all scales
3. Hierarchical features enable strong performance across resolutions

### COCO Object Detection (Cascade Mask R-CNN)

| Backbone | Params | FLOPs | box AP | mask AP |
|----------|--------|-------|--------|---------|
| ResNet-50 | 82M | 739G | 46.3 | 40.0 |
| ResNeXt-101 | 140M | 972G | 48.1 | 41.4 |
| Swin-T | 86M | 745G | **50.5** | **43.7** |
| Swin-S | 107M | 838G | **51.9** | **45.0** |
| Swin-B | 145M | 982G | **51.9** | **45.0** |
| Swin-L | 284M | 1382G | **53.9** | **46.7** |

**Gains**: +4.2 box AP over ResNet-50 with similar compute

### ADE20K Semantic Segmentation (UperNet)

| Backbone | Params | FLOPs | mIoU (SS) | mIoU (MS) |
|----------|--------|-------|-----------|-----------|
| ResNet-101 | 86M | 1029G | 44.9 | 45.9 |
| DeiT-B | 144M | 1185G | 45.9 | 47.3 |
| Swin-T | 60M | 945G | **44.5** | **45.8** |
| Swin-S | 81M | 1038G | **47.6** | **49.5** |
| Swin-B | 121M | 1188G | **48.1** | **49.7** |
| Swin-L | 234M | 1612G | **52.1** | **53.5** |

**Gains**: +7.2 mIoU over ResNet-101 (Swin-L)

### Throughput Comparison (V100)

| Model | Throughput (img/s) | Memory (GB) |
|-------|-------------------|-------------|
| ResNet-50 | 1390 | 6.3 |
| ViT-B/16 | 292 | 11.2 |
| Swin-T | 755 | 7.8 |
| Swin-B | 278 | 12.1 |

Swin-T is 2.6× faster than ViT-B with better accuracy.

## Common Pitfalls

### 1. Incorrect Window Size for Resolution
Wrong: Fixed window size for variable resolutions
```python
config = {"window_size": 7, "image_size": 256}  # 256 not divisible by 7
```

Correct: Ensure H and W are divisible by window size
```python
# Option 1: Adjust image size
config = {"window_size": 7, "image_size": 224}  # 224 = 7×32

# Option 2: Adjust window size
config = {"window_size": 8, "image_size": 256}  # 256 = 8×32
```

### 2. Missing Attention Mask for Shifted Windows
Wrong: Not applying mask to shifted windows
```python
if self.shift_size > 0:
    shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size))
    attn_windows = self.attn(shifted_x, mask=None)  # BUG: Missing mask
```

Correct: Always use attention mask for shifted windows
```python
if self.shift_size > 0:
    shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size))
    attn_mask = self.create_mask()
    attn_windows = self.attn(shifted_x, mask=attn_mask)
```

### 3. Forgetting to Update H, W After Patch Merging
Wrong: Not tracking spatial dimensions
```python
for stage in self.stages:
    x = stage(x)  # BUG: Lost track of H, W
```

Correct: Maintain H and W throughout
```python
H, W = self.patch_embed.grid_size, self.patch_embed.grid_size
for stage in self.stages:
    x = stage(x, H, W)
    if hasattr(stage, 'downsample'):
        H, W = H // 2, W // 2
```

### 4. Inefficient Relative Position Bias
Wrong: Computing on every forward pass
```python
def forward(self, x):
    # Compute relative coordinates every time
    coords = compute_relative_coords(self.window_size)
    bias = self.bias_table[coords]
    ...
```

Correct: Pre-compute and register as buffer
```python
def __init__(self):
    ...
    # Compute once during initialization
    relative_position_index = self._compute_indices()
    self.register_buffer("relative_position_index", relative_position_index)

def forward(self, x):
    # Just index
    bias = self.bias_table[self.relative_position_index]
    ...
```

### 5. Wrong Patch Merging Implementation
Wrong: Simple strided convolution loses information
```python
self.downsample = nn.Conv2d(dim, 2*dim, kernel_size=2, stride=2)
```

Correct: Concatenate then project
```python
class PatchMerging(nn.Module):
    def forward(self, x):
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        return self.reduction(x)
```

## References

### Original Papers
1. **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**
   Liu, Z., Lin, Y., Cao, Y., et al., ICCV 2021
   https://arxiv.org/abs/2103.14030

2. **Swin Transformer V2: Scaling Up Capacity and Resolution**
   Liu, Z., Hu, H., Lin, Y., et al., CVPR 2022
   https://arxiv.org/abs/2111.09883

### Applications
3. **Video Swin Transformer**
   Liu, Z., Ning, J., Cao, Y., et al., CVPR 2022
   https://arxiv.org/abs/2106.13230

4. **SimMIM: A Simple Framework for Masked Image Modeling**
   Xie, Z., Zhang, Z., Cao, Y., et al., CVPR 2022
   https://arxiv.org/abs/2111.09886

### Implementation
- Official: https://github.com/microsoft/Swin-Transformer
- timm: https://github.com/huggingface/pytorch-image-models
- Nexus: `/Users/kevinyu/Projects/Nexus/nexus/models/cv/swin_transformer.py`

### Analysis
5. **Understanding the Role of Shifted Windows**
   Liu et al., Analysis Paper, 2021

6. **On the Relationship Between Self-Attention and Convolutional Layers**
   Cordonnier et al., ICLR 2020
   https://arxiv.org/abs/1911.03584
