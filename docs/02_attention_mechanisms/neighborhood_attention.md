# Neighborhood Attention (NATTEN)

## Overview & Motivation

Neighborhood Attention (NA) is a locality-aware attention mechanism that restricts each token to attend only to its local spatial or temporal neighborhood, defined by a sliding window kernel. Introduced by Hassani et al. in 2022, it brings the efficiency of local receptive fields from CNNs to transformers while maintaining the flexibility of learned, data-dependent attention weights.

**Key Innovation**: Instead of computing attention over all N tokens (O(N²)) or using fixed local windows, NA applies a sliding window of size k×k that moves across spatial dimensions, giving each position a local neighborhood of k² tokens to attend to. This achieves O(N·k²) complexity while preserving spatial inductive biases crucial for vision tasks.

**Why Neighborhood Attention?**
- **Spatial Locality**: Explicitly models the fact that nearby pixels are more related than distant ones
- **Efficient**: O(N·k²) complexity instead of O(N²), enabling high-resolution processing
- **Flexible**: Data-dependent attention weights (unlike CNNs' fixed kernels)
- **Dilatable**: Dilated neighborhoods expand receptive fields without increasing computation
- **Performance**: Matches or exceeds Swin Transformer on ImageNet, COCO, and ADE20K
- **Vision-Optimized**: Designed specifically for 2D/3D spatial data (images, video, point clouds)

**When to Use Neighborhood Attention:**
- High-resolution images (1024×1024+) where global attention is prohibitive
- Video processing (spatial + temporal neighborhoods)
- Dense prediction tasks (segmentation, detection) where local detail matters
- Hierarchical vision architectures (like Swin, but simpler)
- When you want CNN-like inductive biases with transformer flexibility

## Theoretical Background

### The Locality Principle in Vision

Natural images exhibit strong **spatial locality**:
- Nearby pixels are highly correlated (smooth regions, object parts)
- Semantic relationships decay with distance
- Local features (edges, textures) compose into global structures

CNNs exploit this via fixed local filters. Transformers use global attention, losing locality bias. Neighborhood Attention bridges this gap: **local receptive fields with learned attention**.

### Attention Patterns: Global vs. Local vs. Neighborhood

**Global Attention (Standard ViT)**:
```
Each token attends to ALL N tokens
Complexity: O(N²)
Pattern: Fully connected graph
```

**Shifted Window Attention (Swin)**:
```
Partition image into windows, attend within windows
Shift windows between layers for cross-window interaction
Complexity: O(N) but requires window partitioning logic
Pattern: Non-overlapping patches, then shifted patches
```

**Neighborhood Attention (NAT)**:
```
Each token attends to k×k neighborhood (centered on itself)
No partitioning, shifts naturally like a convolution
Complexity: O(N·k²)
Pattern: Sliding window, overlapping neighborhoods
```

### Comparison Table

| Method | Complexity | Receptive Field | Implementation | Flexibility |
|--------|-----------|-----------------|----------------|-------------|
| Global Attention | O(N²) | All tokens | Simple | Full |
| Swin (Shifted Windows) | O(N) | Limited by window | Complex partitioning | Medium |
| Neighborhood Attention | O(N·k²) | Local neighborhood | Sliding window | High |
| Convolution | O(N·k²) | Local kernel | Simple | Fixed weights |

### Why Neighborhood Attention Beats Shifted Windows

1. **Simplicity**: No window partitioning, shifting logic, or cyclic shifts
2. **Uniformity**: Every token has the same neighborhood size (except boundaries)
3. **Translation Equivariance**: Like convolutions, sliding window is equivariant
4. **Dilated Neighborhoods**: Easy to expand receptive field without complexity increase
5. **Performance**: 1.6% higher ImageNet top-1 accuracy than Swin at same FLOPs

### Dilated Neighborhoods

Like dilated convolutions, dilated NA expands the receptive field without increasing kernel size:

```
Standard NA (kernel=3, dilation=1):
  X X X
  X O X
  X X X
  Receptive field: 3×3 = 9 pixels

Dilated NA (kernel=3, dilation=2):
  X . . X . . X
  . . . . . . .
  . . . . . . .
  X . . O . . X
  . . . . . . .
  . . . . . . .
  X . . X . . X
  Receptive field: 7×7 grid, 9 pixels sampled
```

**Effective receptive field** = kernel_size + (kernel_size - 1) × (dilation - 1)

For kernel=7, dilation=3: Effective field = 7 + 6×2 = 19 pixels

### Computational Complexity Analysis

**Time Complexity**:
```
Standard Attention: O(N² · d)
  - For each of N tokens, compute attention over N tokens

Neighborhood Attention: O(N · k² · d)
  - For each of N tokens, compute attention over k² neighbors
  - For images: N = H·W, so O(H·W·k²·d)
```

**Space Complexity**:
```
Standard Attention: O(N²)  [attention matrix]
Neighborhood Attention: O(N·k²) = O(H·W·k²)
```

**Example**: For 224×224 image with kernel_size=7:
- N = 224² = 50,176 tokens
- Global attention: 50,176² = 2.5 billion pairs
- Neighborhood attention: 50,176 × 49 = 2.5 million pairs
- **1000x reduction** in attention pairs!

**Scaling**:
```
As image size doubles (H, W → 2H, 2W):
- Global attention: 4x slower (quadratic)
- Neighborhood attention: 4x slower (linear in N)
- But global becomes 16N² while NA stays 4N·k²
```

## Mathematical Formulation

### 2D Neighborhood Attention (Images)

Given an input feature map X ∈ ℝ^(H×W×C):

**Step 1: Spatial Embeddings**
```
For each spatial position (i, j):
  Q_{i,j}, K_{i,j}, V_{i,j} = X_{i,j} W_Q, X_{i,j} W_K, X_{i,j} W_V
  where W_Q, W_K, W_V ∈ ℝ^(C×d_h)
```

**Step 2: Neighborhood Definition**
```
For position (i, j) with kernel size k (k is odd):
  N(i,j) = {(i+δy, j+δx) : δy, δx ∈ [-k/2, k/2]} ∩ [0,H)×[0,W)

  Neighborhood size = k² (except at boundaries)
```

**Step 3: Attention Computation**
```
Attention scores for position (i,j):
  S_{i,j}[p,q] = (Q_{i,j} · K_{p,q}^T) / √d_h    ∀(p,q) ∈ N(i,j)

Add relative position bias:
  S'_{i,j}[p,q] = S_{i,j}[p,q] + B[p-i, q-j]
  where B ∈ ℝ^(k×k) is a learnable bias table

Attention weights:
  A_{i,j}[p,q] = softmax(S'_{i,j}[p,q])    over all (p,q) ∈ N(i,j)

Output:
  O_{i,j} = Σ_{(p,q)∈N(i,j)} A_{i,j}[p,q] · V_{p,q}
```

**Full Formulation**:
```
NA2D(X)_{i,j} = softmax((Q_{i,j} K_{N(i,j)}^T) / √d_h + B) V_{N(i,j)}

where:
  Q_{i,j} ∈ ℝ^(1×d_h)           [query at position i,j]
  K_{N(i,j)} ∈ ℝ^(k²×d_h)       [keys in neighborhood]
  V_{N(i,j)} ∈ ℝ^(k²×d_h)       [values in neighborhood]
  B ∈ ℝ^(k×k)                   [relative position bias]
```

### 1D Neighborhood Attention (Sequences)

For 1D sequences (e.g., time series, text):

```
Given X ∈ ℝ^(N×d):

For position i:
  N(i) = {i+δ : δ ∈ [-k/2, k/2]} ∩ [0, N)

NA1D(X)_i = softmax((Q_i K_{N(i)}^T) / √d_h + b) V_{N(i)}

where b ∈ ℝ^k is 1D relative position bias
```

### Dilated Neighborhood Attention

With dilation factor d:

```
N^d(i,j) = {(i+d·δy, j+d·δx) : δy, δx ∈ [-k/2, k/2]} ∩ [0,H)×[0,W)

Effective receptive field: k_eff = k + (k-1)·(d-1)
Attention pairs: Still k² (sparse sampling over larger area)
```

**Multi-Head Extension**:
```
For H heads with head dimension d_h = d/H:

MultiHead-NA(X) = Concat(head₁, ..., head_H) W_O

where head_h = NA(XW_Q^h, XW_K^h, XW_V^h)
```

### Relative Position Bias

Unlike absolute position encodings, NA uses **learnable relative position bias**:

```
For 2D, bias table B ∈ ℝ^((2k-1)×(2k-1))
  Relative positions range from -(k-1) to +(k-1) in each dimension

For heads h:
  B^h[Δy, Δx] = learnable parameter for offset (Δy, Δx)

Added to attention scores:
  S_{i,j}[p,q] += B^h[p-i, q-j]
```

This allows the model to learn position-dependent attention patterns (e.g., stronger attention to immediate neighbors).

### Boundary Handling

At image boundaries, neighborhoods are truncated:

```
For position (i,j) near edge:
  N(i,j) = {(p,q) ∈ N_full(i,j) : 0 ≤ p < H, 0 ≤ q < W}

Softmax normalizes over available neighbors only:
  A_{i,j}[p,q] = exp(S_{i,j}[p,q]) / Σ_{(p',q')∈N(i,j)} exp(S_{i,j}[p',q'])
```

Options:
1. **Truncated** (default): Smaller neighborhoods at edges
2. **Padded**: Pad with zeros, attend to padding (increases k²)
3. **Reflected**: Reflect boundary pixels

## Implementation Details

### Core Algorithm (2D Case)

**Pseudocode**:
```python
def neighborhood_attention_2d(X, kernel_size=7, dilation=1):
    """
    X: (B, H, W, C) input features
    Returns: (B, H, W, C) output features
    """
    B, H, W, C = X.shape
    half_k = kernel_size // 2

    # 1. Project to Q, K, V
    Q = linear(X, W_q)  # (B, H, W, d)
    K = linear(X, W_k)
    V = linear(X, W_v)

    # 2. Reshape for multi-head
    Q = rearrange(Q, 'b h w (heads d) -> b heads h w d')
    K = rearrange(K, 'b h w (heads d) -> b heads h w d')
    V = rearrange(V, 'b h w (heads d) -> b heads h w d')

    # 3. Gather neighborhoods for each position
    K_neighbors = gather_2d_neighbors(K, kernel_size, dilation)
    # Shape: (B, heads, H, W, k², d)
    V_neighbors = gather_2d_neighbors(V, kernel_size, dilation)

    # 4. Compute attention scores
    # Q: (B, heads, H, W, d)
    # K_neighbors: (B, heads, H, W, k², d)
    scores = einsum('bhwid,bhwikd->bhwik', Q, K_neighbors) / sqrt(d)

    # 5. Add relative position bias
    scores = scores + relative_position_bias  # (heads, k, k)

    # 6. Mask invalid neighbors (at boundaries)
    mask = compute_boundary_mask(H, W, kernel_size, dilation)
    scores = scores.masked_fill(~mask, -inf)

    # 7. Softmax
    attn = softmax(scores, dim=-1)  # (B, heads, H, W, k²)
    attn = dropout(attn)

    # 8. Weighted sum of values
    out = einsum('bhwik,bhwikd->bhwid', attn, V_neighbors)

    # 9. Merge heads and project
    out = rearrange(out, 'b heads h w d -> b h w (heads d)')
    out = linear(out, W_o)

    return out
```

### Efficient Neighborhood Gathering

The key operation is gathering k² neighbors for each of H×W positions:

**Method 1: Unfold (PyTorch)**
```python
def gather_neighbors_unfold(x, kernel_size, dilation=1):
    """
    x: (B, H, W, C)
    Returns: (B, H, W, k², C)
    """
    x = x.permute(0, 3, 1, 2)  # (B, C, H, W) for unfold

    # Pad for boundaries
    pad = (kernel_size // 2) * dilation
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0)

    # Unfold creates sliding windows
    # unfold(dim, size, step)
    x_unfold = x_pad.unfold(2, kernel_size, 1)  # Unfold height
    x_unfold = x_unfold.unfold(3, kernel_size, 1)  # Unfold width
    # Shape: (B, C, H, W, k, k)

    # Reshape to (B, H, W, k², C)
    x_unfold = x_unfold.permute(0, 2, 3, 4, 5, 1)
    x_unfold = x_unfold.reshape(B, H, W, kernel_size**2, C)

    return x_unfold
```

**Method 2: Explicit Indexing (Flexible)**
```python
def gather_neighbors_index(x, kernel_size, dilation=1):
    """
    More flexible, supports arbitrary dilation
    """
    B, H, W, C = x.shape
    half_k = kernel_size // 2

    # Pad
    pad = half_k * dilation
    x_pad = F.pad(x, (0, 0, pad, pad, pad, pad))

    neighborhoods = []
    for dy in range(-half_k, half_k + 1):
        for dx in range(-half_k, half_k + 1):
            # Compute indices for this offset
            y_offset = pad + dy * dilation
            x_offset = pad + dx * dilation

            # Gather all positions with this offset
            neighbor = x_pad[:, y_offset:y_offset+H, x_offset:x_offset+W, :]
            neighborhoods.append(neighbor)

    # Stack: (B, H, W, k², C)
    return torch.stack(neighborhoods, dim=3)
```

**Method 3: CUDA Kernel (NATTEN Library)**
```cpp
// Optimized CUDA implementation
// Fuses gathering + attention computation
// Avoids materializing (H×W×k²×C) tensor

template <typename scalar_t>
__global__ void natten_forward_kernel(
    scalar_t* output,
    const scalar_t* query,
    const scalar_t* key,
    const scalar_t* value,
    int batch, int heads, int height, int width,
    int kernel_size, int dilation
) {
    // Each thread computes attention for one query position
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * heads * height * width) return;

    // Decode position
    int b = idx / (heads * height * width);
    int h = (idx / (height * width)) % heads;
    int i = (idx / width) % height;
    int j = idx % width;

    // Compute attention only over neighborhood (in registers)
    float sum = 0;
    float max_score = -INFINITY;

    // First pass: compute max for numerical stability
    for (int dy = -kernel_size/2; dy <= kernel_size/2; dy++) {
        for (int dx = -kernel_size/2; dx <= kernel_size/2; dx++) {
            int ni = i + dy * dilation;
            int nj = j + dx * dilation;
            if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                float score = dot_product(query[...], key[...]);
                max_score = max(max_score, score);
            }
        }
    }

    // Second pass: compute softmax and output
    // ... (similar structure)
}
```

### Multi-Resolution Support

For hierarchical architectures (like Swin), downsample via patch merging:

```python
class PatchMerging(nn.Module):
    """Downsample 2x2 patches to 1 token, 4x channels"""
    def forward(self, x):
        # x: (B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Top-right
        x2 = x[:, 0::2, 1::2, :]  # Bottom-left
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = self.linear(x)  # Project to 2C
        return x
```

### Relative Position Bias Initialization

```python
# Initialize with truncated normal (small values)
self.rpb = nn.Parameter(torch.zeros(num_heads, kernel_size, kernel_size))
nn.init.trunc_normal_(self.rpb, std=0.02)

# Or use distance-based initialization
def init_rpb_distance(rpb, kernel_size):
    """Initialize based on spatial distance"""
    half_k = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            dy = i - half_k
            dx = j - half_k
            dist = math.sqrt(dy**2 + dx**2)
            # Closer neighbors get higher initial bias
            rpb[:, i, j] = -0.1 * dist
```

### Memory Optimization

**Checkpointing for Deep Networks**:
```python
from torch.utils.checkpoint import checkpoint

class NABlock(nn.Module):
    def forward(self, x):
        # Use gradient checkpointing to save memory
        if self.training:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x):
        x = x + self.na(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

**Mixed Precision**:
```python
with torch.cuda.amp.autocast():
    output = na_layer(input)  # Runs in fp16
    # Attention scores computed in fp32 for stability
```

## Code Examples

### Example 1: Basic 2D Neighborhood Attention

```python
import torch
import torch.nn as nn
from nexus.components.attention import NeighborhoodAttention2D

# Create NA2D layer
na2d = NeighborhoodAttention2D(
    d_model=256,
    num_heads=8,
    kernel_size=7,
    dilation=1,
    dropout=0.1
)

# Example: 224x224 image with 256 channels
batch_size = 4
height, width = 224, 224
x = torch.randn(batch_size, height, width, 256)

# Forward pass
output, attn_weights = na2d(x, output_attentions=True)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")
# Output:
# Input shape: torch.Size([4, 224, 224, 256])
# Output shape: torch.Size([4, 224, 224, 256])
# Attention weights shape: torch.Size([4, 8, 50176, 49])
# 50176 = 224*224 positions, 49 = 7*7 neighborhood
```

### Example 2: 1D Neighborhood Attention (Sequences)

```python
from nexus.components.attention import NeighborhoodAttention1D

# For time-series or 1D sequences
na1d = NeighborhoodAttention1D(
    d_model=512,
    num_heads=8,
    kernel_size=9,  # Each token attends to 9 neighbors
    dilation=1,
    dropout=0.0
)

# Example: Audio features (B, T, C)
batch_size = 2
seq_len = 1000
x = torch.randn(batch_size, seq_len, 512)

output, _ = na1d(x)
print(f"Output shape: {output.shape}")  # (2, 1000, 512)
```

### Example 3: Dilated Neighborhood Attention

```python
# Dilated NA expands receptive field without increasing computation
na_dilated = NeighborhoodAttention2D(
    d_model=384,
    num_heads=6,
    kernel_size=7,   # Still 7x7 = 49 attention pairs
    dilation=3,      # But spanning 19x19 effective area
    dropout=0.1
)

x = torch.randn(2, 56, 56, 384)
output, _ = na_dilated(x)

# Effective receptive field: 7 + (7-1)*2 = 19 pixels
# But still only 49 attention computations per position
```

### Example 4: Neighborhood Attention Transformer Block

```python
class NATBlock(nn.Module):
    """Full transformer block with NA"""
    def __init__(self, dim, num_heads, kernel_size=7, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.na = NeighborhoodAttention2D(
            d_model=dim,
            num_heads=num_heads,
            kernel_size=kernel_size
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # x: (B, H, W, C)
        # NA with residual
        x = x + self.na(self.norm1(x))[0]
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x

# Stack multiple blocks
nat_stage = nn.Sequential(*[
    NATBlock(dim=192, num_heads=6, kernel_size=7)
    for _ in range(2)
])

x = torch.randn(1, 56, 56, 192)
out = nat_stage(x)  # (1, 56, 56, 192)
```

### Example 5: Hierarchical NAT (Like Swin)

```python
class HierarchicalNAT(nn.Module):
    """Multi-stage NAT with downsampling"""
    def __init__(self):
        super().__init__()

        # Stage 1: 56x56, C=96
        self.stage1 = nn.Sequential(*[
            NATBlock(dim=96, num_heads=3, kernel_size=7)
            for _ in range(2)
        ])
        self.downsample1 = PatchMerging(96)

        # Stage 2: 28x28, C=192
        self.stage2 = nn.Sequential(*[
            NATBlock(dim=192, num_heads=6, kernel_size=7)
            for _ in range(2)
        ])
        self.downsample2 = PatchMerging(192)

        # Stage 3: 14x14, C=384
        self.stage3 = nn.Sequential(*[
            NATBlock(dim=384, num_heads=12, kernel_size=7)
            for _ in range(6)
        ])
        self.downsample3 = PatchMerging(384)

        # Stage 4: 7x7, C=768
        self.stage4 = nn.Sequential(*[
            NATBlock(dim=768, num_heads=24, kernel_size=7)
            for _ in range(2)
        ])

    def forward(self, x):
        # x: (B, 224, 224, 3) - raw image
        x = self.patch_embed(x)  # -> (B, 56, 56, 96)

        x = self.stage1(x)           # (B, 56, 56, 96)
        x = self.downsample1(x)      # (B, 28, 28, 192)

        x = self.stage2(x)           # (B, 28, 28, 192)
        x = self.downsample2(x)      # (B, 14, 14, 384)

        x = self.stage3(x)           # (B, 14, 14, 384)
        x = self.downsample3(x)      # (B, 7, 7, 768)

        x = self.stage4(x)           # (B, 7, 7, 768)

        # Global average pooling
        x = x.mean(dim=[1, 2])       # (B, 768)
        return x
```

### Example 6: Using Official NATTEN Library

```python
# Install: pip install natten
from natten import NeighborhoodAttention2D as NATTEN2D
import torch.nn as nn

class OptimizedNATBlock(nn.Module):
    """Using official CUDA-optimized NATTEN"""
    def __init__(self, dim, num_heads, kernel_size=7):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # Official NATTEN - optimized CUDA kernels
        self.na = NATTEN2D(
            dim=dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=1,
            qkv_bias=True
        )

    def forward(self, x):
        # NATTEN expects (B, C, H, W) format
        B, H, W, C = x.shape
        x_norm = self.norm(x)
        x_norm = x_norm.permute(0, 3, 1, 2)  # -> (B, C, H, W)

        attn_out = self.na(x_norm)  # CUDA-optimized

        attn_out = attn_out.permute(0, 2, 3, 1)  # -> (B, H, W, C)
        return x + attn_out

# Example
block = OptimizedNATBlock(dim=256, num_heads=8, kernel_size=7)
x = torch.randn(2, 56, 56, 256).cuda()
out = block(x)  # Fast CUDA execution
```

### Example 7: Video Processing (3D NA)

```python
class NeighborhoodAttention3D(nn.Module):
    """3D NA for video: spatial + temporal neighborhoods"""
    def __init__(self, dim, num_heads, kernel_size=(3, 7, 7)):
        super().__init__()
        # kernel_size = (T, H, W)
        self.t_kernel = kernel_size[0]  # Temporal window
        self.h_kernel = kernel_size[1]  # Spatial height
        self.w_kernel = kernel_size[2]  # Spatial width

        # Similar to 2D, but extend to 3D
        # ... (implementation details)

    def forward(self, x):
        # x: (B, T, H, W, C) - video frames
        # Each position (t, h, w) attends to 3x7x7 neighborhood
        pass

# Example usage
na3d = NeighborhoodAttention3D(
    dim=384,
    num_heads=6,
    kernel_size=(3, 7, 7)  # 3 frames, 7x7 spatial
)

# Video input: 16 frames of 224x224
video = torch.randn(1, 16, 224, 224, 384)
output = na3d(video)  # (1, 16, 224, 224, 384)
```

### Example 8: Combining NA with Other Mechanisms

```python
class HybridAttentionBlock(nn.Module):
    """Combine NA (local) with sparse global attention"""
    def __init__(self, dim, num_heads):
        super().__init__()
        # Local attention via NA
        self.local_attn = NeighborhoodAttention2D(
            d_model=dim,
            num_heads=num_heads // 2,
            kernel_size=7
        )
        # Global attention on downsampled features
        self.global_attn = MultiHeadAttention(
            d_model=dim,
            num_heads=num_heads // 2
        )
        self.downsample = nn.AvgPool2d(4)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        # x: (B, H, W, C)

        # Local branch
        local = self.local_attn(x)[0]

        # Global branch (on downsampled)
        B, H, W, C = x.shape
        x_down = self.downsample(x.permute(0,3,1,2)).permute(0,2,3,1)
        x_down_flat = x_down.reshape(B, -1, C)
        global_flat, _ = self.global_attn(x_down_flat)
        global_out = global_flat.reshape(B, H//4, W//4, C)
        global_out = self.upsample(global_out.permute(0,3,1,2)).permute(0,2,3,1)

        # Combine
        return (local + global_out) / 2
```

## Comparative Analysis

### NAT vs. Swin Transformer

| Aspect | Swin Transformer | NAT (Neighborhood Attention) |
|--------|-----------------|------------------------------|
| **Attention Pattern** | Non-overlapping windows, shifted | Sliding overlapping neighborhoods |
| **Implementation** | Window partitioning + cyclic shift | Direct neighborhood gathering |
| **Complexity** | O(M²·N) where M is window size | O(k²·N) where k is kernel size |
| **Uniformity** | Window edges have discontinuities | Every token treated uniformly |
| **Receptive Field Growth** | Requires window shifting | Natural sliding overlap |
| **Code Complexity** | High (partition, shift, merge logic) | Low (like convolution) |
| **ImageNet Top-1** | 83.3% (Swin-B) | 84.3% (NAT-B) at same FLOPs |
| **Training Speed** | Baseline | 1.2x faster (less overhead) |

**Key Difference**: Swin partitions the image into non-overlapping windows (e.g., 7×7), computes attention within each window, then shifts windows by half the window size in the next layer to enable cross-window communication. NAT simply slides a kernel (like convolution) over all positions.

### NAT vs. ViT (Global Attention)

| Aspect | ViT | NAT |
|--------|-----|-----|
| **Complexity** | O(N²) | O(N·k²) |
| **Max Resolution (A100 40GB)** | ~384×384 | ~1024×1024+ |
| **Inductive Bias** | None (learns from data) | Spatial locality |
| **Data Efficiency** | Needs 300M+ images | Works with ImageNet-1K |
| **Global Context** | Immediate (1 layer) | Grows with depth |

### NAT vs. Convolution

| Aspect | Convolution | NAT |
|--------|-------------|-----|
| **Weights** | Fixed per kernel position | Data-dependent (attention) |
| **Flexibility** | Low (same filter everywhere) | High (adapts per position) |
| **Context Modeling** | Local only | Local with attention |
| **Performance** | Good baseline | +2-3% on ImageNet |

### Benchmarks: ImageNet-1K Classification

| Model | Params | FLOPs | Top-1 Acc | Throughput (img/s) |
|-------|--------|-------|-----------|-------------------|
| Swin-T | 28M | 4.5G | 81.3% | 755 |
| **NAT-T** | **28M** | **4.3G** | **83.2%** | **738** |
| Swin-S | 50M | 8.7G | 83.0% | 436 |
| **NAT-S** | **51M** | **7.8G** | **83.7%** | **438** |
| Swin-B | 88M | 15.4G | 83.3% | 278 |
| **NAT-B** | **90M** | **13.7G** | **84.3%** | **281** |

**Key Takeaways**:
- NAT consistently outperforms Swin at similar FLOPs
- 1-1.5% higher accuracy across all model sizes
- Slightly better throughput (simpler implementation)

### COCO Object Detection (Mask R-CNN)

| Backbone | Params | APbox | APmask |
|----------|--------|-------|--------|
| Swin-T | 48M | 46.0 | 41.6 |
| **NAT-T** | **48M** | **47.7** | **42.6** |
| Swin-S | 69M | 48.5 | 43.3 |
| **NAT-S** | **70M** | **48.4** | **43.2** |
| Swin-B | 107M | 48.5 | 43.4 |
| **NAT-B** | **108M** | **49.5** | **44.0** |

NAT-T shows the biggest gains (+1.7 AP), suggesting NA is particularly effective for dense prediction.

### ADE20K Semantic Segmentation (UperNet)

| Backbone | Params | mIoU | mIoU (MS) |
|----------|--------|------|-----------|
| Swin-T | 60M | 44.5 | 45.8 |
| **NAT-T** | **58M** | **45.1** | **46.4** |
| Swin-S | 81M | 47.6 | 49.5 |
| **NAT-S** | **82M** | **48.4** | **50.3** |
| Swin-B | 121M | 48.1 | 49.7 |
| **NAT-B** | **123M** | **49.0** | **50.6** |

NAT's local attention preserves fine-grained spatial details crucial for segmentation.

### Dilated NAT Results

Using mixed kernel sizes and dilations:

| Model | Config | ImageNet Top-1 | COCO APbox |
|-------|--------|----------------|-----------|
| NAT-T | kernel=7, dilation=1 | 83.2% | 47.7 |
| **DiNAT-T** | **Mixed (3,5,7), dilation (1,2,3)** | **83.8%** | **48.5** |
| NAT-S | kernel=7, dilation=1 | 83.7% | 48.4 |
| **DiNAT-S** | **Mixed** | **84.6%** | **49.8** |

Dilated NA expands receptive fields, improving global context modeling.

## Practical Considerations

### When to Use Neighborhood Attention

**Best Use Cases:**
1. **High-Resolution Images** (>512×512)
   - Global attention becomes prohibitive
   - NA scales linearly with image size
   - Example: Medical imaging (1024×1024+), satellite imagery

2. **Dense Prediction Tasks**
   - Segmentation, detection, depth estimation
   - Local detail preservation is critical
   - Example: Semantic segmentation with 512×512 inputs

3. **Video Understanding**
   - 3D NA over spatial + temporal dimensions
   - Efficient for long videos (100+ frames)
   - Example: Action recognition, video object segmentation

4. **Limited Compute Budget**
   - NAT trains 20-30% faster than ViT
   - Lower memory footprint enables larger batches
   - Example: Training on single GPU

5. **Data-Constrained Scenarios**
   - Spatial locality bias helps with small datasets
   - ImageNet-1K sufficient (vs. ViT needing ImageNet-21K)
   - Example: Fine-tuning on domain-specific datasets

**When NOT to Use:**
1. **Small Images** (<224×224): Global attention is fast enough
2. **Tasks Requiring Long-Range Dependencies**: NLP, where tokens 1000 positions apart interact
3. **Pre-training on Massive Data**: ViT's lack of bias can be advantageous with 100M+ images

### Hyperparameter Selection

**Kernel Size**:
```
Small (3-5): Fast, local features, good for early layers
Medium (7): Standard choice, balanced
Large (11-13): Slower, larger receptive field, good for late layers

Rule of thumb: kernel_size ≈ feature_map_size / 20
  For 56×56 feature maps: kernel=7 is good
  For 14×14 feature maps: kernel=3-5 sufficient
```

**Dilation**:
```
1 (no dilation): Default, dense neighborhoods
2-3: Moderate expansion, good for middle layers
4+: Large receptive field, use sparingly (can miss local details)

Strategy: Increase dilation in deeper layers
  Layers 1-4:   dilation=1
  Layers 5-8:   dilation=2
  Layers 9-12:  dilation=3
```

**Number of Heads**:
```
Follows standard transformer rules:
  dim=96   → heads=3
  dim=192  → heads=6
  dim=384  → heads=12
  dim=768  → heads=24

Constraint: dim must be divisible by num_heads
```

**MLP Ratio**:
```
Typical: 4.0 (MLP hidden dim = 4 × embedding dim)
Smaller models: 3.0 (reduce parameters)
Larger models: 4.0-4.5 (more capacity)
```

### Training Tips

**Optimizer Settings**:
```python
# AdamW with layer-wise learning rate decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.05
)

# Cosine LR schedule
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=300,  # epochs
    eta_min=1e-6
)
```

**Layer-wise LR Decay**:
```python
# Deeper layers use smaller learning rates
def get_layer_lr_decay(model, lr, decay_rate=0.95):
    param_groups = []
    num_layers = len(model.layers)
    for i, layer in enumerate(model.layers):
        lr_scale = decay_rate ** (num_layers - i)
        param_groups.append({
            'params': layer.parameters(),
            'lr': lr * lr_scale
        })
    return param_groups
```

**Regularization**:
```python
# Drop path (stochastic depth)
drop_path_rate = 0.1  # For NAT-T
drop_path_rate = 0.3  # For NAT-B
drop_path_rate = 0.5  # For NAT-L

# Dropout in attention and MLP
attn_dropout = 0.0   # Usually no dropout in attention
mlp_dropout = 0.1    # Dropout in MLP

# Label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Data Augmentation**:
```python
# Standard for vision transformers
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25)
])
```

**Mixed Precision Training**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in dataloader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Common Pitfalls

**1. Incorrect Input Format**:
```python
# WRONG: (B, C, H, W) - channel-first
x = torch.randn(2, 256, 56, 56)
output = na2d(x)  # ERROR

# CORRECT: (B, H, W, C) - channel-last
x = torch.randn(2, 56, 56, 256)
output = na2d(x)  # OK
```

**2. Forgetting Boundary Padding**:
```python
# Without proper padding, boundary tokens have smaller neighborhoods
# Make sure your implementation pads correctly or uses boundary masks
```

**3. Mismatched Kernel Size**:
```python
# Kernel size must be odd for symmetric neighborhoods
kernel_size = 8  # WRONG - even kernel
kernel_size = 7  # CORRECT - odd kernel
```

**4. Excessive Dilation**:
```python
# Too much dilation loses local detail
# BAD: dilation=10 with kernel=7 -> 67×67 effective area, very sparse
# GOOD: dilation=2-3 for balanced local-global trade-off
```

**5. Ignoring Relative Position Bias**:
```python
# Position bias is crucial for NA performance
# Don't initialize to zeros - use truncated normal
nn.init.trunc_normal_(self.rpb, std=0.02)
```

### Performance Optimization

**CUDA Kernel Usage**:
```bash
# Install official NATTEN library for 3-5x speedup
pip install natten

# Supports CUDA 11.x, 12.x
# Provides optimized kernels for 1D, 2D, 3D NA
```

**Batch Size Tuning**:
```python
# NA uses less memory than global attention
# You can use larger batch sizes

# ViT-B: batch_size=64 on A100 (40GB)
# NAT-B: batch_size=96 on A100 (40GB)
# 50% larger batches → faster training
```

**Gradient Checkpointing**:
```python
# For very deep networks (>30 layers)
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    for layer in self.layers:
        x = checkpoint(layer, x)  # Trade compute for memory
    return x
```

**Multi-GPU Training**:
```python
# NAT works well with DDP
from torch.nn.parallel import DistributedDataParallel as DDP

model = NAT(...)
model = DDP(model, device_ids=[local_rank])

# Larger effective batch size = better convergence
# 8 GPUs × 96 batch/GPU = 768 total batch size
```

## Visualizations

### Attention Pattern Visualization

**Local Neighborhood Structure**:
```
For kernel_size=7, dilation=1:

Image Grid (each cell is a pixel):
┌─────────────────────────────────┐
│ . . . . . . . . . . . . . . . . │
│ . . . . . . . . . . . . . . . . │
│ . . . ■ ■ ■ ■ ■ ■ ■ . . . . . . │
│ . . . ■ ■ ■ ■ ■ ■ ■ . . . . . . │
│ . . . ■ ■ ■ ■ ■ ■ ■ . . . . . . │
│ . . . ■ ■ ■ █ ■ ■ ■ . . . . . . │  █ = query token
│ . . . ■ ■ ■ ■ ■ ■ ■ . . . . . . │  ■ = attending neighbors (49 tokens)
│ . . . ■ ■ ■ ■ ■ ■ ■ . . . . . . │  . = not in neighborhood
│ . . . ■ ■ ■ ■ ■ ■ ■ . . . . . . │
│ . . . . . . . . . . . . . . . . │
│ . . . . . . . . . . . . . . . . │
└─────────────────────────────────┘

For kernel_size=7, dilation=2:

┌─────────────────────────────────┐
│ ■ . ■ . ■ . ■ . ■ . ■ . ■ . . . │
│ . . . . . . . . . . . . . . . . │
│ ■ . ■ . ■ . ■ . ■ . ■ . ■ . . . │
│ . . . . . . . . . . . . . . . . │
│ ■ . ■ . ■ . ■ . ■ . ■ . ■ . . . │
│ . . . . . . . . . . . . . . . . │
│ ■ . ■ . ■ . █ . ■ . ■ . ■ . . . │  Effective field: 13×13
│ . . . . . . . . . . . . . . . . │  Attending tokens: still 49
│ ■ . ■ . ■ . ■ . ■ . ■ . ■ . . . │  Sparsely sampled
│ . . . . . . . . . . . . . . . . │
└─────────────────────────────────┘
```

**Sliding Window vs. Shifted Window**:
```
Neighborhood Attention (Sliding):
┌─────┬─────┬─────┬─────┐    Every token has own 3×3 neighborhood
│ ■■■ │ ■■■ │ ■■■ │ ■■  │    Overlapping windows
│ ■█■ │ █■■ │ █■■ │ █■  │    Continuous coverage
│ ■■■ │ ■■■ │ ■■■ │ ■■  │
├─────┼─────┼─────┼─────┤
│ ■■■ │ ■■■ │ ■■■ │ ■■  │
│ ■█■ │ █■■ │ █■■ │ █■  │
└─────┴─────┴─────┴─────┘

Swin (Shifted Windows):
Layer 1:                        Layer 2 (shifted):
┌─────────┬─────────┐          ┌──┬─────────┬──────┐
│ ■ ■ ■   │   ■ ■ ■ │          │■ │ ■ ■ ■   │  ■ ■ │
│ ■ ■ ■   │   ■ ■ ■ │          │■ │ ■ ■ ■   │  ■ ■ │
│ ■ ■ ■   │   ■ ■ ■ │          ├──┼─────────┼──────┤
├─────────┼─────────┤          │■ │ ■ ■ ■   │  ■ ■ │
│ ■ ■ ■   │   ■ ■ ■ │          │■ │ ■ ■ ■   │  ■ ■ │
│ ■ ■ ■   │   ■ ■ ■ │          └──┴─────────┴──────┘
│ ■ ■ ■   │   ■ ■ ■ │
└─────────┴─────────┘
Non-overlapping windows         Shifted to enable cross-window
Tokens at edges can't attend    interaction (complex logic)
```

### Complexity Visualization

**Memory Usage vs. Sequence Length**:
```
Memory (GB)
    │
 40 │                                 Global Attn ───────────
    │                              ╱
 30 │                          ╱
    │                      ╱
 20 │                  ╱
    │              ╱                 NA (k=7) ─────────────
 10 │          ╱               ─────
    │      ╱           ─────
  0 │──────────────────────────────────────────> Seq Length
    0    512   1024   2048   4096   8192

Global: O(N²) - quadratic growth
NA: O(N·k²) - linear growth
```

**Receptive Field Growth Over Layers**:
```
Layer 1 (k=7):      Layer 2:           Layer 3:
  ■ ■ ■               ■ ■ ■ ■ ■           ■ ■ ■ ■ ■ ■ ■
  ■ █ ■               ■ ■ ■ ■ ■           ■ ■ ■ ■ ■ ■ ■
  ■ ■ ■               ■ ■ █ ■ ■           ■ ■ ■ ■ ■ ■ ■
  (7×7=49)            ■ ■ ■ ■ ■           ■ ■ ■ █ ■ ■ ■
                      ■ ■ ■ ■ ■           ■ ■ ■ ■ ■ ■ ■
                      (13×13=169)         ■ ■ ■ ■ ■ ■ ■
                                          ■ ■ ■ ■ ■ ■ ■
                                          (19×19=361)

Effective RF = 7 + 6×(L-1) for kernel=7
After L layers: RF ≈ 6L+1 pixels
```

### Hierarchical Architecture

```
Input: 224×224×3 RGB Image
           │
           ▼
    ┌─────────────┐
    │ Patch Embed │  4×4 patches → 56×56×96
    └─────────────┘
           │
           ▼
    ╔═════════════╗
    ║  Stage 1    ║  56×56, C=96, 2× NA blocks (k=7)
    ╚═════════════╝
           │ Downsample 2×
           ▼
    ╔═════════════╗
    ║  Stage 2    ║  28×28, C=192, 2× NA blocks (k=7)
    ╚═════════════╝
           │ Downsample 2×
           ▼
    ╔═════════════╗
    ║  Stage 3    ║  14×14, C=384, 6× NA blocks (k=7)
    ╚═════════════╝
           │ Downsample 2×
           ▼
    ╔═════════════╗
    ║  Stage 4    ║  7×7, C=768, 2× NA blocks (k=7)
    ╚═════════════╝
           │ Global Pool
           ▼
    ┌─────────────┐
    │  FC Layer   │  → 1000 classes
    └─────────────┘
```

## References & Resources

### Key Papers

1. **Neighborhood Attention Transformer (NAT)** - CVPR 2023
   - Hassani, A., Walton, S., Li, J., Li, S., & Shi, H.
   - "Neighborhood Attention Transformer"
   - Paper: https://arxiv.org/abs/2204.07143
   - Introduces NA, shows superiority over Swin

2. **Dilated Neighborhood Attention Transformer (DiNAT)** - 2022
   - Hassani, A., & Shi, H.
   - "Dilated Neighborhood Attention Transformer"
   - Paper: https://arxiv.org/abs/2209.15001
   - Extends NA with dilated neighborhoods

3. **NATTEN: Neighborhood Attention Extension** - Software
   - Hassani, A.
   - CUDA-optimized implementation
   - Code: https://github.com/SHI-Labs/NATTEN

### Related Work

4. **Swin Transformer** - ICCV 2021
   - Liu, Z., et al.
   - "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
   - Paper: https://arxiv.org/abs/2103.14030
   - NAT's main comparison baseline

5. **Vision Transformer (ViT)** - ICLR 2021
   - Dosovitskiy, A., et al.
   - "An Image is Worth 16x16 Words"
   - Paper: https://arxiv.org/abs/2010.11929
   - Foundational global attention for vision

6. **Local Attention** - Various
   - Parmar, N., et al. "Image Transformer" (2018)
   - Ramachandran, P., et al. "Stand-Alone Self-Attention" (2019)
   - Early work on local attention for images

### Code & Implementations

**Official Implementations**:
- NATTEN Library (CUDA): https://github.com/SHI-Labs/NATTEN
- NAT Models: https://github.com/SHI-Labs/Neighborhood-Attention-Transformer
- Pre-trained weights: Available on GitHub releases

**Nexus Implementation**:
```python
from nexus.components.attention import NeighborhoodAttention2D, NeighborhoodAttention1D
```

**Example Notebooks**:
- NAT for ImageNet classification
- DiNAT for object detection (COCO)
- 3D NA for video understanding

### Benchmarks & Datasets

**ImageNet-1K**:
- 1.28M training images, 50K validation
- 1000 classes
- Used for all NAT classification benchmarks

**COCO**:
- Object detection and instance segmentation
- 118K training, 5K validation images
- NAT used as backbone for Mask R-CNN

**ADE20K**:
- Semantic segmentation
- 150 categories, 20K training, 2K validation
- NAT as backbone for UperNet

### Tutorials & Guides

1. **Using NATTEN**:
   ```bash
   pip install natten
   ```
   Documentation: https://www.shi-labs.com/natten/

2. **Training NAT from Scratch**:
   - Requires ImageNet-1K dataset
   - 300 epochs, AdamW optimizer
   - 8 GPUs, batch size 128 per GPU
   - Full recipe in official repo

3. **Fine-tuning Pre-trained NAT**:
   ```python
   from natten import NATClassifier

   model = NATClassifier.from_pretrained('NAT-Tiny')
   # Fine-tune on your dataset
   ```

### Community Resources

- **Discussion**: GitHub Issues on SHI-Labs/NATTEN
- **Papers with Code**: https://paperswithcode.com/method/neighborhood-attention
- **Reddit**: r/MachineLearning discussions on local attention

### Citation

If you use Neighborhood Attention in your work, please cite:

```bibtex
@inproceedings{hassani2023neighborhood,
  title={Neighborhood Attention Transformer},
  author={Hassani, Ali and Walton, Steven and Li, Jiachen and Li, Shen and Shi, Humphrey},
  booktitle={CVPR},
  year={2023}
}

@article{hassani2022dilated,
  title={Dilated Neighborhood Attention Transformer},
  author={Hassani, Ali and Shi, Humphrey},
  journal={arXiv preprint arXiv:2209.15001},
  year={2022}
}

@software{hassani2022natten,
  title={NATTEN: Neighborhood Attention Extension},
  author={Hassani, Ali},
  year={2022},
  url={https://github.com/SHI-Labs/NATTEN}
}
```

---

**Document Version**: 1.0
**Last Updated**: 2025-02-06
**Nexus Component**: `nexus.components.attention.NeighborhoodAttention2D`
**Status**: Production-ready, CUDA-optimized version available
