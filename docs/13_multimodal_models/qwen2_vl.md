# Qwen2-VL: Dynamic Resolution Vision-Language Model with M-RoPE

## 1. Overview & Motivation

Qwen2-VL is an advanced multimodal language model from Alibaba that introduces Multimodal Rotary Position Embedding (M-RoPE) to handle images and videos at arbitrary resolutions without interpolation. This enables more natural processing of visual data while maintaining spatial and temporal coherence.

### Key Innovations
- **M-RoPE (Multimodal RoPE)**: Extends RoPE to 2D/3D for images/videos with native spatial encoding
- **Dynamic Resolution**: Process images at native resolution without resizing or interpolation
- **Naive Dynamic Resolution**: Simple yet effective approach to variable image sizes
- **Unified Video Support**: Seamlessly handles both images and video with temporal M-RoPE
- **Enhanced Visual Understanding**: Superior performance on high-resolution visual tasks
- **Efficient Training**: No complex interpolation schemes needed

### Why Qwen2-VL?
Traditional vision-language models face challenges with:
- Fixed input resolutions requiring interpolation
- Loss of fine-grained spatial information through resizing
- Inefficient processing of high-resolution images
- Difficulty modeling temporal relationships in videos
- Complex position encoding schemes for different resolutions
- Poor generalization to unseen aspect ratios

Qwen2-VL addresses these through its novel M-RoPE mechanism and dynamic resolution support.

### Applications
- **Visual Question Answering**: Complex reasoning over high-resolution images
- **Document Understanding**: OCR and layout analysis without preprocessing
- **Video Understanding**: Temporal reasoning with 3D positional encodings
- **Medical Imaging**: High-resolution image analysis preserving spatial detail
- **Satellite Imagery**: Processing large-scale images at native resolution
- **Fine-grained Recognition**: Detailed visual understanding tasks

## 2. Theoretical Background

### 2.1 Multimodal Rotary Position Embedding

**Standard RoPE** (for 1D sequences):

RoPE applies rotation matrices to query and key vectors in attention:

$$
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix} \mapsto \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
$$

where $m$ is the position index and $\theta_i = 10000^{-2i/d}$.

**M-RoPE** extends this to multiple dimensions (2D for images, 3D for videos):

For an image position $(h, w)$:
$$
\text{RoPE}_{2D}(x, h, w) = \text{RoPE}_{1D}(x[:d/2], h) \oplus \text{RoPE}_{1D}(x[d/2:], w)
$$

where $\oplus$ denotes concatenation, and each dimension gets half the embedding dimensions.

**Key Properties**:
1. **Relative Position Encoding**: Attention score depends only on relative positions
2. **Extrapolation**: Can handle positions beyond training range
3. **Efficiency**: Applied via rotation, no learned parameters
4. **Composability**: Each axis is independent

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

**Advantages**:
- No interpolation artifacts
- Perfect preservation of spatial relationships
- Works with arbitrary aspect ratios
- Efficient computation

### 2.3 Spatiotemporal Encoding for Video

For video with frames at positions $(t, h, w)$:

$$
\text{M-RoPE}_{3D}(x, t, h, w) = \text{RoPE}_{1D}(x[:d/3], t) \oplus \text{RoPE}_{1D}(x[d/3:2d/3], h) \oplus \text{RoPE}_{1D}(x[2d/3:], w)
$$

Each dimension (time, height, width) gets $d/3$ of the total dimension.

**Temporal Modeling**:
- Frame index $t$ encoded in first $d/3$ dimensions
- Enables understanding of motion and temporal dynamics
- Supports variable-length videos naturally

### 2.4 Attention with Relative 2D/3D Positions

The attention computation becomes position-aware:

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q_{rot}(pos_q) \cdot K_{rot}^T(pos_k)}{\sqrt{d_k}}\right) V
$$

The attention score between positions $(h_1, w_1)$ and $(h_2, w_2)$ depends on:
$$
\Delta h = h_2 - h_1, \quad \Delta w = w_2 - w_1
$$

This provides natural translation equivariance.

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
\text{M-RoPE}(pos) &= \text{concat}[\text{emb}^{(t)}, \text{emb}^{(h)}, \text{emb}^{(w)}]
\end{align}
$$

### 3.2 Rotation Application

For query/key vectors $q, k \in \mathbb{R}^{d}$:

$$
\begin{align}
q_{rotated} &= \text{apply\_rotation}(q, \cos_{\text{M-RoPE}}, \sin_{\text{M-RoPE}}) \\
&= [q_{even} \odot \cos - q_{odd} \odot \sin, \, q_{odd} \odot \cos + q_{even} \odot \sin]
\end{align}
$$

where $\odot$ is element-wise multiplication, and:
- $q_{even} = [q_0, q_2, q_4, ...]$
- $q_{odd} = [q_1, q_3, q_5, ...]$

### 3.3 Multi-Head Attention with M-RoPE

$$
\begin{align}
Q, K, V &= W_Q X, W_K X, W_V X \\
Q_{rot}, K_{rot} &= \text{M-RoPE}(Q, pos), \text{M-RoPE}(K, pos) \\
\text{head}_i &= \text{softmax}\left(\frac{Q_{rot}^{(i)} {K_{rot}^{(i)}}^T}{\sqrt{d_k}}\right) V^{(i)} \\
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, ..., \text{head}_h) W_O
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

**Position Grid Construction**:
$$
\begin{align}
\text{pos}_h &= [0, 1, 2, ..., H'-1] \\
\text{pos}_w &= [0, 1, 2, ..., W'-1] \\
\text{PosGrid} &= \text{meshgrid}(\text{pos}_h, \text{pos}_w) \in \mathbb{R}^{H' \times W' \times 2}
\end{align}
$$

### 3.5 Training Objective

**Vision-Language Modeling Loss**:
$$
\mathcal{L}_{VLM} = -\sum_{t=1}^{T} \log P(x_t | V, x_{<t})
$$

where $V$ is the visual features and $x_t$ are text tokens.

**Contrastive Vision-Text Alignment** (pre-training):
$$
\mathcal{L}_{contrast} = -\log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(v_i, t_j)/\tau)}
$$

**Total Loss**:
$$
\mathcal{L} = \mathcal{L}_{VLM} + \lambda_{contrast} \mathcal{L}_{contrast}
$$

## 4. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Qwen2-VL Architecture                    │
└──────────────────────────────────────────────────────────────┘

Input: Image I ∈ R^(3×H×W) or Video V ∈ R^(T×3×H×W)

                        ┌────────────────┐
                        │ Dynamic Vision │
                        │    Encoder     │
                        │   (Any Res!)   │
                        └────────┬───────┘
                                 │
                    V ∈ R^(N×d_v), Pos ∈ R^(N×2/3)
                                 │
                        ┌────────▼───────┐
                        │  V-L Projector │
                        │  (Multi-layer) │
                        └────────┬───────┘
                                 │
                          V_proj ∈ R^(N×d)
                                 │
            ┌────────────────────┴────────────────────┐
            │                                         │
            ▼                                         ▼
    ┌───────────────┐                        ┌──────────────┐
    │   M-RoPE      │                        │  Text Embed  │
    │  Encoding     │                        │  + RoPE 1D   │
    │  (2D or 3D)   │                        │              │
    └───────┬───────┘                        └──────┬───────┘
            │                                        │
      cos, sin ∈ R^(N×d)                      T ∈ R^(L×d)
            │                                        │
            └────────────────┬───────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Transformer    │
                    │  with M-RoPE    │
                    │  (32 layers)    │
                    └────────┬────────┘
                             │
                    Fused ∈ R^((N+L)×d)
                             │
                    ┌────────▼────────┐
                    │  Output Head    │
                    │  (Next Token)   │
                    └─────────────────┘

M-RoPE Position Encoding:
    2D (Image):  pos = (h, w)     → d/2 for h, d/2 for w
    3D (Video):  pos = (t, h, w)  → d/3 for each dimension

Vision Encoder Details:
┌─────────────────────────────────┐
│ Image → Conv(patch_size) → Patches
│         ↓
│    Flatten (H/p × W/p patches)
│         ↓
│    Linear Projection → d_v
│         ↓
│    ViT Layers (optional)
└─────────────────────────────────┘
```

### Detailed Component Flow

**Vision Path**:
1. Image/Video → Patch Embedding (Conv2D)
2. Flatten spatial dims → (B, N, d_v) where N = H' × W' (× T for video)
3. Generate 2D/3D position grid
4. Project to text dimension → (B, N, d)
5. Compute M-RoPE cos/sin for positions

**Text Path**:
1. Tokens → Embedding lookup → (B, L, d)
2. Add 1D RoPE for sequential positions

**Fusion**:
1. Concatenate visual and text tokens
2. Apply unified transformer with M-RoPE
3. Visual tokens use 2D/3D RoPE, text tokens use 1D RoPE
4. Cross-modal attention naturally emerges

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

### 5.4 Complete Qwen2-VL Model

```python
class Qwen2VL(NexusModule):
    def __init__(
        self,
        visual_hidden_dim: int = 1024,
        text_hidden_dim: int = 4096,
        num_visual_layers: int = 12,
        num_text_layers: int = 32,
        num_heads: int = 32,
        patch_size: int = 14,
        merge_size: int = 2,
        max_spatial_position: int = 128,
        use_temporal: bool = False,
        vocab_size: int = 151936
    ):
        super().__init__()

        # Vision encoder
        self.vision_encoder = DynamicVisionEncoder(
            hidden_dim=visual_hidden_dim,
            patch_size=patch_size,
            merge_size=merge_size
        )

        # Vision-to-text projection
        self.vision_projection = nn.Sequential(
            nn.Linear(visual_hidden_dim, text_hidden_dim),
            nn.GELU(),
            nn.Linear(text_hidden_dim, text_hidden_dim)
        )

        # M-RoPE for vision
        position_axes = 3 if use_temporal else 2
        self.mrope = MultimodalRotaryEmbedding(
            dim=text_hidden_dim,
            max_position=max_spatial_position,
            position_axes=position_axes
        )

        # Text embedding
        self.text_embed = nn.Embedding(vocab_size, text_hidden_dim)

        # 1D RoPE for text
        self.text_rope = RotaryEmbedding(dim=text_hidden_dim // num_heads)

        # Unified transformer
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=text_hidden_dim,
                num_heads=num_heads
            )
            for _ in range(num_text_layers)
        ])

        self.norm = nn.LayerNorm(text_hidden_dim)
        self.lm_head = nn.Linear(text_hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None
    ):
        # Process vision
        if images is not None:
            vision_features, positions = self.vision_encoder(images)
            vision_embeds = self.vision_projection(vision_features)

            # Compute M-RoPE
            cos, sin = self.mrope(positions)

        elif video_frames is not None:
            # Process video with temporal dimension
            B, T, C, H, W = video_frames.shape
            frames = video_frames.reshape(B * T, C, H, W)

            vision_features, spatial_pos = self.vision_encoder(frames)

            # Add temporal positions
            temporal_pos = torch.arange(T, device=frames.device)
            temporal_pos = temporal_pos.repeat_interleave(spatial_pos.shape[1])
            temporal_pos = temporal_pos.unsqueeze(0).expand(B, -1).unsqueeze(-1)

            positions = torch.cat([temporal_pos, spatial_pos], dim=-1)

            vision_embeds = self.vision_projection(vision_features)
            cos, sin = self.mrope(positions)
        else:
            vision_embeds = None
            cos, sin = None, None

        # Process text
        if input_ids is not None:
            text_embeds = self.text_embed(input_ids)
            text_rope_cos, text_rope_sin = self.text_rope(
                torch.arange(input_ids.shape[1], device=input_ids.device)
            )

        # Combine modalities
        if vision_embeds is not None and text_embeds is not None:
            combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
            # Vision uses M-RoPE, text uses 1D RoPE
            # This is handled in attention layers
        elif vision_embeds is not None:
            combined_embeds = vision_embeds
        else:
            combined_embeds = text_embeds

        # Apply transformer
        hidden_states = combined_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos=cos, sin=sin)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'visual_embeds': vision_embeds,
            'spatial_positions': positions if vision_embeds is not None else None
        }
```

## 6. Code Walkthrough

### Step 1: Model Initialization

```python
from nexus.models.multimodal import Qwen2VL

model = Qwen2VL(
    visual_hidden_dim=1024,
    text_hidden_dim=4096,
    num_visual_layers=12,
    num_text_layers=32,
    num_heads=32,
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

### Step 5: Vision-Language Task

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Prepare input
image = load_image("example.jpg")
question = "What objects are in this image?"

# Tokenize
input_ids = tokenizer.encode(question, return_tensors='pt')

# Forward pass
outputs = model(images=image, input_ids=input_ids)

# Generate response
generated = model.generate(
    images=image,
    input_ids=input_ids,
    max_length=100,
    temperature=0.7
)

response = tokenizer.decode(generated[0])
print(response)
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

# Usage
H, W = image.shape[-2:]
merge_size = get_optimal_merge_size(max(H, W))
vision_encoder.merge_size = merge_size
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

### 7.4 Flash Attention Integration

```python
import torch.nn.functional as F

class FlashMRoPEAttention(nn.Module):
    def forward(self, q, k, v, cos, sin):
        # Apply M-RoPE
        q_rot = apply_rotary_emb(q, cos, sin)
        k_rot = apply_rotary_emb(k, cos, sin)

        # Use flash attention
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q_rot, k_rot, v,
                dropout_p=0.0,
                is_causal=False
            )
        else:
            # Fallback
            attn = torch.matmul(q_rot, k_rot.transpose(-2, -1))
            attn = attn / math.sqrt(q.shape[-1])
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)

        return out
```

### 7.5 Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class Qwen2VL(NexusModule):
    def __init__(self, *args, use_gradient_checkpointing=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def forward(self, *args, **kwargs):
        # ... encode vision and text ...

        hidden_states = combined_embeds
        for layer in self.layers:
            if self.training and self.use_gradient_checkpointing:
                hidden_states = checkpoint(layer, hidden_states, cos, sin)
            else:
                hidden_states = layer(hidden_states, cos, sin)

        # ... rest of forward ...
```

### 7.6 Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(
            images=batch['images'],
            input_ids=batch['input_ids']
        )
        loss = compute_loss(outputs, batch['labels'])

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

### 7.7 Multi-Resolution Training

```python
# Train with varying resolutions
resolutions = [(336, 336), (448, 448), (672, 672), (384, 512), (512, 384)]

for batch in dataloader:
    # Randomly sample resolution
    H, W = random.choice(resolutions)

    # Resize images
    images = F.interpolate(batch['images'], size=(H, W))

    # Forward pass (M-RoPE handles different resolutions)
    outputs = model(images=images, input_ids=batch['input_ids'])
    loss = compute_loss(outputs, batch['labels'])
    loss.backward()
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
| ChartQA | 58.3 | 79.7 | +21.4 |
| AI2D | 62.1 | 75.9 | +13.8 |

### 8.2 Resolution Scaling

**Performance vs. Resolution**:
| Resolution | Parameters | Latency (ms) | VQA Accuracy | Memory (GB) |
|------------|-----------|--------------|--------------|-------------|
| 336×336 | 7B | 120 | 78.5 | 12 |
| 448×448 | 7B | 180 | 81.2 | 16 |
| 672×672 | 7B | 320 | 83.7 | 24 |
| 1024×1024 | 7B | 720 | 85.1 | 40 |

**Key Insight**: M-RoPE enables seamless scaling to higher resolutions with consistent performance gains.

### 8.3 M-RoPE Ablation

| Position Encoding | MMBench | TextVQA | Inference Speed | Parameter Count |
|-------------------|---------|---------|-----------------|-----------------|
| Learned 2D PE | 68.2 | 76.3 | 1.0x | +50M |
| Interpolated RoPE | 70.1 | 79.8 | 1.1x | +0M |
| M-RoPE (ours) | 73.4 | 84.5 | 1.2x | +0M |

**Analysis**:
- M-RoPE is parameter-free (unlike learned PE)
- No interpolation means better extrapolation
- Faster inference due to efficient rotation operations

### 8.4 Video Understanding

**Video QA Benchmarks**:
| Dataset | Qwen2-VL | VideoChat | VideoLLaMA | GPT-4V |
|---------|----------|-----------|------------|--------|
| NExT-QA | 68.2% | 61.5% | 59.8% | 72.1% |
| MSVD-QA | 72.5% | 68.3% | 65.7% | 75.3% |
| ActivityNet-QA | 45.3% | 42.1% | 40.8% | 48.9% |
| STAR | 58.7% | 54.2% | 52.6% | 61.4% |

**Temporal M-RoPE Benefits**:
- Natural encoding of frame positions
- Better temporal reasoning
- Consistent performance across video lengths

### 8.5 Multi-Modal Reasoning

**MathVista Benchmark**:
- Overall: 58.3%
- Geometry: 62.7%
- Algebra: 56.1%
- Statistics: 59.8%

**ScienceQA**:
- Overall: 83.2%
- Physics: 85.1%
- Chemistry: 81.7%
- Biology: 82.9%

### 8.6 Document Understanding

**OCR-Free Document Tasks**:
| Task | Qwen2-VL | Donut | LayoutLM | Pix2Struct |
|------|----------|-------|----------|------------|
| DocVQA | 94.5 | 85.1 | 89.7 | 88.3 |
| InfographicVQA | 78.3 | 69.2 | 72.8 | 74.1 |
| TableVQA | 82.7 | 75.8 | 79.3 | 80.2 |
| VisualMRC | 71.4 | 64.3 | 68.7 | 69.5 |

### 8.7 Computational Efficiency

**Training Efficiency**:
| Model Size | Training Time (hours) | GPU Hours | Memory/GPU |
|------------|----------------------|-----------|------------|
| 2B | 120 | 960 (8×A100) | 32GB |
| 7B | 240 | 1920 (8×A100) | 64GB |
| 72B | 720 | 5760 (8×A100) | 80GB |

**Inference Throughput**:
| Batch Size | Resolution | Throughput (samples/s) | Latency (ms) |
|------------|-----------|------------------------|--------------|
| 1 | 336×336 | 8.3 | 120 |
| 4 | 336×336 | 24.1 | 166 |
| 1 | 672×672 | 3.1 | 322 |
| 4 | 672×672 | 8.7 | 459 |

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

### 9.4 Interpolation Confusion

**Pitfall**: Trying to interpolate M-RoPE embeddings
```python
# Wrong: Interpolating rotary embeddings
cos_small, sin_small = mrope(positions_small)
cos_large = F.interpolate(cos_small, size=large_size)  # Don't do this!

# Correct: Compute M-RoPE directly for target size
cos_large, sin_large = mrope(positions_large)  # Direct computation
```

### 9.5 Mixed Position Encoding

**Pitfall**: Inconsistent position encoding for vision and text
```python
# Wrong: Using 1D RoPE for everything
text_rope = RotaryEmbedding(dim)
cos, sin = text_rope(all_positions)  # Ignores 2D structure!

# Correct: Use M-RoPE for vision, 1D RoPE for text
vision_cos, vision_sin = mrope(vision_positions)  # 2D/3D
text_cos, text_sin = text_rope(text_positions)  # 1D
# Apply appropriate encoding to each modality
```

### 9.6 Aspect Ratio Handling

**Pitfall**: Not preserving aspect ratios
```python
# Wrong: Squashing images to fixed size
images = F.interpolate(images, size=(336, 336))  # Distorts aspect ratio

# Correct: Pad to preserve aspect ratio or use dynamic resolution
H, W = images.shape[-2:]
if H > W:
    new_H = 672
    new_W = int(W * (672 / H))
else:
    new_W = 672
    new_H = int(H * (672 / W))
images = F.interpolate(images, size=(new_H, new_W))
# M-RoPE handles any resolution!
```

### 9.7 Training Stability

**Pitfall**: Unstable training with high-resolution images
```python
# Wrong: No gradient clipping
loss.backward()
optimizer.step()

# Correct: Clip gradients and use warmup
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# Also use learning rate warmup
lr = base_lr * min(step / warmup_steps, 1.0)
```

## 10. References

### Papers

1. **Qwen2-VL**: "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution"
   - https://arxiv.org/abs/2409.12191 (Alibaba Cloud, 2024)
   - Introduces M-RoPE and dynamic resolution handling

2. **RoFormer (RoPE)**: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
   - https://arxiv.org/abs/2104.09864
   - Foundation for rotary position embeddings

3. **ViT**: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
   - https://arxiv.org/abs/2010.11929
   - Vision transformer architecture

4. **Pixel Shuffle**: "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"
   - https://arxiv.org/abs/1609.05158
   - Sub-pixel convolution for upsampling

5. **LLaVA**: "Visual Instruction Tuning"
   - https://arxiv.org/abs/2304.08485
   - Vision-language instruction tuning

6. **Flamingo**: "Flamingo: a Visual Language Model for Few-Shot Learning"
   - https://arxiv.org/abs/2204.14198
   - Cross-attention for vision-language fusion

7. **BLIP-2**: "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"
   - https://arxiv.org/abs/2301.12597
   - Efficient vision-language alignment

8. **InternVL**: "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks"
   - https://arxiv.org/abs/2312.14238
   - Large-scale vision-language model

### Resources
- Qwen Blog: https://qwenlm.github.io/blog/qwen2-vl/
- Implementation: `/Users/kevinyu/Projects/Nexus/nexus/models/multimodal/qwen2_vl.py`
- Hugging Face: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
- Official Code: https://github.com/QwenLM/Qwen2-VL

### Related Models in Nexus
- **Qwen-VL** (predecessor): Basic vision-language model
- **LLaVA-NeXT**: `/Users/kevinyu/Projects/Nexus/nexus/models/multimodal/llava_next.py`
- **InternVL**: High-resolution vision-language model
- **Phi-3-Vision**: `/Users/kevinyu/Projects/Nexus/nexus/models/multimodal/phi3_vision.py`
- **NVLM**: `/Users/kevinyu/Projects/Nexus/docs/13_multimodal_models/nvlm.md`
- **Molmo**: `/Users/kevinyu/Projects/Nexus/nexus/models/multimodal/molmo.py`

### Benchmarks & Datasets
- **MMBench**: https://github.com/open-compass/MMBench
- **SeedBench**: https://github.com/AILab-CVC/SEED-Bench
- **TextVQA**: https://textvqa.org/
- **DocVQA**: https://www.docvqa.org/
- **ChartQA**: https://github.com/vis-nlp/ChartQA
- **NExT-QA**: https://doc-doc.github.io/docs/nextqa.html

## Summary

Qwen2-VL represents a significant advancement in multimodal vision-language models through its innovative M-RoPE mechanism and dynamic resolution support.

**Key Contributions**:
1. **M-RoPE**: Extends rotary position embeddings to 2D/3D for images/videos
2. **Dynamic Resolution**: Processes images at any resolution without interpolation
3. **Unified Architecture**: Seamlessly handles images, videos, and text
4. **State-of-the-art Performance**: Excellent results across vision-language benchmarks
5. **Efficient Design**: Parameter-free position encoding with fast inference

**When to Use Qwen2-VL**:
- High-resolution image understanding is critical
- Document OCR and layout analysis tasks
- Video understanding with temporal reasoning
- Variable aspect ratio inputs
- Multi-modal reasoning over vision and text

**Implementation Highlights**:
- M-RoPE computation for 2D/3D positions
- Dynamic vision encoder supporting any resolution
- Efficient caching and gradient checkpointing
- Flash attention integration for speed
- Mixed precision training support

**Performance Summary**:
- MMBench: 73.4 (SOTA for 7B models)
- TextVQA: 84.5 (+20.7 over Qwen-VL)
- DocVQA: 94.5 (near-perfect OCR-free understanding)
- Video QA: Competitive with specialized video models
- Scales efficiently from 336×336 to 1024×1024 resolution
