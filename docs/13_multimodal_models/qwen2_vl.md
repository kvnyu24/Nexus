# Qwen2-VL: Dynamic Resolution Vision-Language Model with M-RoPE

## 1. Overview & Motivation

Qwen2-VL is an advanced multimodal language model from Alibaba that introduces Multimodal Rotary Position Embedding (M-RoPE) to handle images and videos at arbitrary resolutions without interpolation. This enables more natural processing of visual data while maintaining spatial and temporal coherence.

### Key Innovations
- **M-RoPE (Multimodal RoPE)**: Extends RoPE to 2D/3D for images/videos with native spatial encoding
- **Dynamic Resolution**: Process images at native resolution without resizing or interpolation
- **Naive Dynamic Resolution**: Simple yet effective approach to variable image sizes
- **Unified Video Support**: Seamlessly handles both images and video with temporal M-RoPE

### Why Qwen2-VL?
Traditional vision-language models face challenges with:
- Fixed input resolutions requiring interpolation
- Loss of fine-grained spatial information through resizing
- Inefficient processing of high-resolution images
- Difficulty modeling temporal relationships in videos

Qwen2-VL addresses these through its novel M-RoPE mechanism and dynamic resolution support.

## 2. Theoretical Background

### 2.1 Multimodal Rotary Position Embedding

**Standard RoPE** (for 1D sequences):
$$
\begin{pmatrix} q_0 \\ q_1 \end{pmatrix} \mapsto \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}
$$

**M-RoPE** extends this to multiple dimensions (2D for images, 3D for videos):

For an image position $(h, w)$:
$$
\text{RoPE}_{2D}(x, h, w) = \text{RoPE}_{1D}(x[:d/2], h) \oplus \text{RoPE}_{1D}(x[d/2:], w)
$$

where $\oplus$ denotes concatenation, and each dimension gets half the embedding dimensions.

### 2.2 Dynamic Resolution Without Interpolation

**Key Idea**: Instead of interpolating positional embeddings, directly encode actual positions:

```python
# Traditional approach (requires interpolation)
pos_embed_fixed = pos_embed[:, :H*W, :]  # Fixed size
if H*W != pos_embed_fixed.size(1):
    pos_embed = interpolate(pos_embed_fixed, H*W)  # Interpolate

# Qwen2-VL approach (no interpolation)
h_positions = torch.arange(H)  # Actual row positions
w_positions = torch.arange(W)  # Actual column positions
pos_embed = M_RoPE(h_positions, w_positions)  # Direct encoding
```

### 2.3 Spatiotemporal Encoding for Video

For video with frames at positions $(t, h, w)$:

$$
\text{M-RoPE}_{3D}(x, t, h, w) = \text{RoPE}_{1D}(x[:d/3], t) \oplus \text{RoPE}_{1D}(x[d/3:2d/3], h) \oplus \text{RoPE}_{1D}(x[2d/3:], w)
$$

## 3. Mathematical Formulation

### 3.1 M-RoPE Computation

**Inverse Frequencies** for each axis:
$$
\theta_i^{(axis)} = 10000^{-2i/d_{axis}}, \quad i = 0, 1, ..., d_{axis}/2 - 1
$$

where $d_{axis} = d / \text{num\_axes}$ (2 for images, 3 for videos).

**Positional Encoding**:
$$
\begin{align}
\text{freq}_{i,j}^{(axis)} &= pos_j \cdot \theta_i^{(axis)} \\
\text{emb}_{i,j}^{(axis)} &= [\cos(\text{freq}_{i,j}^{(axis)}), \sin(\text{freq}_{i,j}^{(axis)})] \\
\text{M-RoPE}(pos) &= \text{concat}[\text{emb}^{(h)}, \text{emb}^{(w)}]
\end{align}
$$

### 3.2 Rotation Application

For query/key vectors $q, k \in \mathbb{R}^{d}$:

$$
\begin{align}
q_{rotated} &= \text{apply\_rotation}(q, \cos_{\text{M-RoPE}}, \sin_{\text{M-RoPE}}) \\
&= [q_{:d/2} \odot \cos_{:d/2} - q_{d/2:} \odot \sin_{:d/2}, \, q_{d/2:} \odot \cos_{d/2:} + q_{:d/2} \odot \sin_{d/2:}]
\end{align}
$$

where $\odot$ is element-wise multiplication.

### 3.3 Attention with M-RoPE

$$
\begin{align}
Q, K, V &= W_Q X, W_K X, W_V X \\
Q_{rot}, K_{rot} &= \text{M-RoPE}(Q, pos), \text{M-RoPE}(K, pos) \\
\text{Attn}(Q, K, V) &= \text{softmax}\left(\frac{Q_{rot} K_{rot}^T}{\sqrt{d_k}}\right) V
\end{align}
$$

### 3.4 Naive Dynamic Resolution

**Pixel Shuffle-like Encoding**:
$$
\begin{align}
P &= \text{Conv2D}_{p \times p}(I) \in \mathbb{R}^{d \times H' \times W'} \\
H' &= H / p, \quad W' = W / p \\
\text{Features} &= \text{Flatten}(P) \in \mathbb{R}^{(H' \cdot W') \times d}
\end{align}
$$

where $p$ is patch size, and no interpolation is needed for different $H, W$.

## 4. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Qwen2-VL Architecture                    │
└──────────────────────────────────────────────────────────────┘

Input: Image I ∈ R^(3×H×W) or Video V ∈ R^(T×3×H×W)

                        ┌────────────────┐
                        │ Dynamic Vision │
                        │    Encoder     │
                        └────────┬───────┘
                                 │
                    V ∈ R^(N×d_v), Pos ∈ R^(N×2/3)
                                 │
                        ┌────────▼───────┐
                        │  V-L Projector │
                        └────────┬───────┘
                                 │
                          V_proj ∈ R^(N×d)
                                 │
            ┌────────────────────┴────────────────────┐
            │                                         │
            ▼                                         ▼
    ┌───────────────┐                        ┌──────────────┐
    │   M-RoPE      │                        │  Text Embed  │
    │  Encoding     │                        │              │
    └───────┬───────┘                        └──────┬───────┘
            │                                        │
      cos, sin ∈ R^(N×d)                      T ∈ R^(L×d)
            │                                        │
            └────────────────┬───────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Transformer    │
                    │  with M-RoPE    │
                    └────────┬────────┘
                             │
                    Fused ∈ R^((N+L)×d)
                             │
                    ┌────────▼────────┐
                    │  Output Head    │
                    └─────────────────┘

M-RoPE Position Encoding:
    2D (Image):  pos = (h, w)     → d/2 for h, d/2 for w
    3D (Video):  pos = (t, h, w)  → d/3 for each dimension
```

## 5. Implementation Details

### 5.1 M-RoPE Implementation

Located in `/Users/kevinyu/Projects/Nexus/nexus/models/multimodal/qwen2_vl.py`:

```python
class MultimodalRotaryEmbedding(NexusModule):
    def __init__(
        self,
        dim: int,
        max_position: int = 2048,
        base: float = 10000.0,
        position_axes: int = 2  # 2 for images, 3 for videos
    ):
        super().__init__()
        self.dim = dim
        self.position_axes = position_axes
        self.dim_per_axis = dim // position_axes

        # Compute inverse frequencies for each axis
        inv_freq_per_axis = []
        for _ in range(position_axes):
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim_per_axis, 2).float()
                        / self.dim_per_axis)
            )
            inv_freq_per_axis.append(inv_freq)

        self.register_buffer("inv_freq", torch.cat(inv_freq_per_axis))

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: [B, N, position_axes] where
                      - 2D: positions are (h, w)
                      - 3D: positions are (t, h, w)
        Returns:
            cos, sin: [B, N, dim]
        """
        batch_size, seq_len, _ = positions.shape

        # Split positions by axis
        pos_axes = torch.split(positions, 1, dim=-1)

        # Compute embeddings for each axis
        cos_parts, sin_parts = [], []

        for axis_idx, pos in enumerate(pos_axes):
            pos = pos.squeeze(-1)  # [B, N]

            # Get inverse frequencies for this axis
            start_idx = axis_idx * (self.dim_per_axis // 2)
            end_idx = (axis_idx + 1) * (self.dim_per_axis // 2)
            inv_freq = self.inv_freq[start_idx:end_idx]

            # Compute frequencies
            freqs = torch.einsum('bn,d->bnd', pos.float(), inv_freq)

            # Duplicate for cos/sin pairs
            emb = torch.cat([freqs, freqs], dim=-1)  # [B, N, dim_per_axis]

            cos_parts.append(torch.cos(emb))
            sin_parts.append(torch.sin(emb))

        # Concatenate all axes
        cos = torch.cat(cos_parts, dim=-1)
        sin = torch.cat(sin_parts, dim=-1)

        return cos, sin
```

### 5.2 Dynamic Vision Encoder

```python
class DynamicVisionEncoder(NexusModule):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 1024,
        patch_size: int = 14,
        merge_size: int = 2
    ):
        super().__init__()
        self.patch_size = patch_size
        self.merge_size = merge_size

        # Patch embedding (supports any resolution)
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Optional spatial merge for efficiency
        self.merge = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=merge_size,
            stride=merge_size
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: [B, 3, H, W] - any resolution!
        Returns:
            features: [B, H'*W', hidden_dim]
            positions: [B, H'*W', 2] - (h, w) positions
        """
        B, C, H, W = images.shape

        # Extract patches (no resizing!)
        x = self.patch_embed(images)  # [B, hidden_dim, H_patch, W_patch]

        # Optional merge for efficiency
        if self.merge_size > 1:
            x = self.merge(x)

        H_patch, W_patch = x.shape[2], x.shape[3]

        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # [B, H_patch*W_patch, hidden_dim]

        # Generate 2D positions for M-RoPE
        h_pos = torch.arange(H_patch, device=images.device).unsqueeze(1).expand(-1, W_patch)
        w_pos = torch.arange(W_patch, device=images.device).unsqueeze(0).expand(H_patch, -1)

        positions = torch.stack([h_pos, w_pos], dim=-1)  # [H_patch, W_patch, 2]
        positions = positions.reshape(-1, 2)  # [H_patch*W_patch, 2]
        positions = positions.unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]

        return x, positions
```

### 5.3 Applying M-RoPE to Attention

```python
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        x: [B, N, num_heads, head_dim]
        cos, sin: [B, N, head_dim]
    Returns:
        rotated: [B, N, num_heads, head_dim]
    """
    # Split into first half and second half
    x1, x2 = x.chunk(2, dim=-1)

    # Expand cos/sin for multi-head
    cos = cos.unsqueeze(2)  # [B, N, 1, head_dim]
    sin = sin.unsqueeze(2)

    # Apply rotation
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)

    return rotated
```

## 6. Code Walkthrough

### Step 1: Model Initialization

```python
from nexus.models.multimodal import Qwen2VL

model = Qwen2VL(
    visual_hidden_dim=1024,
    text_hidden_dim=4096,
    num_visual_layers=12,
    patch_size=14,
    merge_size=2,
    max_spatial_position=128,
    use_temporal=False  # True for video
)
```

### Step 2: Process Images at Any Resolution

```python
import torch

# Different resolutions - no problem!
image1 = torch.randn(1, 3, 336, 336)  # Square
image2 = torch.randn(1, 3, 448, 672)  # Rectangular
image3 = torch.randn(1, 3, 1024, 768)  # High-res

# All work without interpolation
outputs1 = model(images=image1)
outputs2 = model(images=image2)
outputs3 = model(images=image3)

# Visual features have different lengths based on resolution
print(outputs1['visual_embeds'].shape)  # [1, 576, 4096]
print(outputs2['visual_embeds'].shape)  # [1, 1536, 4096]
print(outputs3['visual_embeds'].shape)  # [1, 4096, 4096]
```

### Step 3: Video Processing with Temporal M-RoPE

```python
# Enable temporal M-RoPE
model = Qwen2VL(
    visual_hidden_dim=1024,
    text_hidden_dim=4096,
    use_temporal=True  # 3D M-RoPE
)

# Process video
video_frames = torch.randn(1, 8, 3, 336, 336)  # 8 frames
outputs = model(video_frames=video_frames)

# Positions are now 3D: (t, h, w)
print(outputs['spatial_positions'].shape)  # [1, 8*576, 3]
```

### Step 4: Combining with Text

```python
# Prepare text
text_embeds = torch.randn(1, 100, 4096)

# Multimodal forward
outputs = model(
    images=images,
    text_embeds=text_embeds
)

# Fused embeddings with M-RoPE applied
multimodal_embeds = outputs['multimodal_embeds']  # [1, 576+100, 4096]
```

## 7. Optimization Tricks

### 7.1 Efficient High-Resolution Processing

**Adaptive Merge Size**: Adjust merge_size based on input resolution
```python
def get_optimal_merge_size(image_size):
    if image_size <= 512:
        return 1  # No merge for small images
    elif image_size <= 1024:
        return 2  # 2x merge
    else:
        return 4  # 4x merge for very high-res
```

**Patch Size Selection**:
```python
# Smaller patch size for high-resolution images
if H > 1024 or W > 1024:
    patch_size = 7  # Smaller patches
else:
    patch_size = 14  # Standard patches
```

### 7.2 M-RoPE Caching

Cache computed frequencies for efficiency:
```python
@torch.no_grad()
def precompute_freqs(max_h, max_w, dim, device):
    """Precompute and cache M-RoPE frequencies"""
    h_positions = torch.arange(max_h, device=device)
    w_positions = torch.arange(max_w, device=device)

    # Compute all combinations
    h_grid, w_grid = torch.meshgrid(h_positions, w_positions, indexing='ij')
    positions = torch.stack([h_grid.flatten(), w_grid.flatten()], dim=-1)

    cos, sin = mrope.forward(positions.unsqueeze(0))
    return cos, sin

# Use cached frequencies
freqs_cache = precompute_freqs(128, 128, 4096, 'cuda')
```

### 7.3 Memory-Efficient Video Processing

**Frame Batching**: Process video frames in batches
```python
def process_video_efficiently(video_frames, batch_size=4):
    T = video_frames.shape[1]
    all_features = []

    for i in range(0, T, batch_size):
        batch = video_frames[:, i:i+batch_size]
        batch_features = model.encode_images(
            batch.reshape(-1, 3, H, W)
        )
        all_features.append(batch_features)

    return torch.cat(all_features, dim=1)
```

## 8. Experiments & Results

### 8.1 Benchmark Performance

| Benchmark | Qwen-VL | Qwen2-VL | Improvement |
|-----------|---------|----------|-------------|
| MMBench | 61.8 | 73.4 | +11.6 |
| SeedBench | 62.3 | 72.7 | +10.4 |
| MME | 1487 | 1872 | +385 |
| TextVQA | 63.8 | 84.5 | +20.7 |
| DocVQA | 65.1 | 94.5 | +29.4 |

### 8.2 Resolution Scaling

**Performance vs. Resolution**:
| Resolution | Parameters | Latency (ms) | VQA Accuracy |
|------------|-----------|--------------|--------------|
| 336×336 | 7B | 120 | 78.5 |
| 448×448 | 7B | 180 | 81.2 |
| 672×672 | 7B | 320 | 83.7 |
| 1024×1024 | 7B | 720 | 85.1 |

### 8.3 M-RoPE Ablation

| Position Encoding | MMBench | TextVQA | Inference Speed |
|-------------------|---------|---------|-----------------|
| Learned 2D PE | 68.2 | 76.3 | 1.0x |
| Interpolated RoPE | 70.1 | 79.8 | 1.1x |
| M-RoPE (ours) | 73.4 | 84.5 | 1.2x |

### 8.4 Video Understanding

**Video QA Benchmarks**:
- NExT-QA: 68.2% accuracy
- MSVD-QA: 72.5% accuracy
- ActivityNet-QA: 45.3% accuracy

## 9. Common Pitfalls

### 9.1 Position Encoding Issues

**Pitfall**: Exceeding maximum position embeddings
```python
# Wrong: No check for position range
positions = torch.arange(H * W)  # May exceed max_position

# Correct: Clip or adjust positions
max_pos = model.mrope.max_position
if H * W > max_pos:
    # Either reduce resolution or use adaptive encoding
    scale = max_pos / (H * W)
    positions = positions * scale
```

### 9.2 Dimension Mismatch

**Pitfall**: Dimension not divisible by position_axes
```python
# Wrong: hidden_dim not divisible by 2 (for images)
model = Qwen2VL(text_hidden_dim=4095, position_axes=2)  # Error!

# Correct: Ensure divisibility
model = Qwen2VL(text_hidden_dim=4096, position_axes=2)  # 4096 % 2 == 0
```

### 9.3 Video Processing Memory

**Pitfall**: Loading entire video into memory
```python
# Wrong: Process all frames at once
video = torch.randn(1, 1000, 3, 1024, 1024)  # OOM!
output = model(video_frames=video)

# Correct: Process in chunks
chunk_size = 8
outputs = []
for i in range(0, 1000, chunk_size):
    chunk = video[:, i:i+chunk_size]
    output_chunk = model(video_frames=chunk)
    outputs.append(output_chunk)
```

## 10. References

### Papers
1. **Qwen2-VL**: "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution"
   - https://arxiv.org/abs/2409.12191 (Alibaba Cloud, 2024)

2. **RoFormer (RoPE)**: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
   - https://arxiv.org/abs/2104.09864

3. **ViT**: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
   - https://arxiv.org/abs/2010.11929

4. **Pixel Shuffle**: "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"
   - https://arxiv.org/abs/1609.05158

### Resources
- Qwen Blog: https://qwenlm.github.io/blog/qwen2-vl/
- Implementation: `/Users/kevinyu/Projects/Nexus/nexus/models/multimodal/qwen2_vl.py`
- Hugging Face: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

### Related Models
- Qwen-VL (predecessor)
- LLaVA-NeXT (also supports dynamic resolution)
- InternVL (high-resolution vision-language model)
