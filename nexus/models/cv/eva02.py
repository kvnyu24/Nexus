"""
EVA-02: A Visual Representation for Neon Genesis

Implementation of EVA-02, a masked image modeling (MIM) pre-trained Vision Transformer
with Rotary Position Embeddings (RoPE) and SwiGLU activations. EVA-02 achieves strong
performance on various vision tasks through improved architectural choices.

Reference:
    Fang, Y., Sun, Q., Wang, X., et al. (2023).
    "EVA-02: A Visual Representation for Neon Genesis."
    arXiv:2303.11331

Key Components:
    - EVA02Block: Transformer block with RoPE and SwiGLU MLP
    - RoPE2D: 2D Rotary Position Embeddings for vision
    - SwiGLU: Swish-gated linear unit activation
    - EVA02: Full ViT model with MIM pre-training support

Architecture Details:
    - Uses RoPE instead of absolute positional embeddings
    - SwiGLU activations in MLP blocks for better expressiveness
    - Supports various sizes (Small, Base, Large, Giant)
    - MIM pre-training with CLIP teacher for target generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple

from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin


class RoPE2D(nn.Module):
    """2D Rotary Position Embeddings for Vision Transformers.

    Extends the RoPE mechanism from language models to 2D spatial grids by applying
    separate rotations for x and y coordinates. This allows for better length
    extrapolation and positional awareness.

    Args:
        dim: Feature dimension (must be divisible by 4).
        max_resolution: Maximum spatial resolution (height/width). Default: 224.
        temperature: Temperature parameter for frequency scaling. Default: 10000.0.
    """

    def __init__(
        self,
        dim: int,
        max_resolution: int = 224,
        temperature: float = 10000.0,
    ):
        super().__init__()
        assert dim % 4 == 0, "Dimension must be divisible by 4 for 2D RoPE"

        self.dim = dim
        self.max_resolution = max_resolution
        self.temperature = temperature

        # Pre-compute frequency bands
        dim_per_axis = dim // 4
        freqs = 1.0 / (temperature ** (torch.arange(0, dim_per_axis, 2).float() / dim_per_axis))
        self.register_buffer("freqs", freqs)

    def forward(self, h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 2D RoPE embeddings for a given spatial resolution.

        Args:
            h: Height of the feature map.
            w: Width of the feature map.
            device: Device to create tensors on.

        Returns:
            Tuple of (cos_emb, sin_emb) each of shape (h*w, dim).
        """
        # Generate coordinate grids
        h_coords = torch.arange(h, device=device, dtype=torch.float32)
        w_coords = torch.arange(w, device=device, dtype=torch.float32)

        # Compute frequencies for each axis
        h_freqs = torch.outer(h_coords, self.freqs)  # (h, dim//4)
        w_freqs = torch.outer(w_coords, self.freqs)  # (w, dim//4)

        # Expand to grid: (h, w, dim//4)
        h_grid = h_freqs.unsqueeze(1).expand(h, w, -1)
        w_grid = w_freqs.unsqueeze(0).expand(h, w, -1)

        # Concatenate and duplicate for cos/sin pairs
        # Each axis contributes dim//2 dimensions
        freqs_h = torch.stack([h_grid, h_grid], dim=-1).flatten(-2)  # (h, w, dim//2)
        freqs_w = torch.stack([w_grid, w_grid], dim=-1).flatten(-2)  # (h, w, dim//2)

        # Combine both axes: (h, w, dim)
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)

        # Flatten spatial dimensions: (h*w, dim)
        freqs = freqs.reshape(-1, self.dim)

        # Compute cos and sin
        cos_emb = freqs.cos()
        sin_emb = freqs.sin()

        return cos_emb, sin_emb


def apply_rope_2d(x: torch.Tensor, cos_emb: torch.Tensor, sin_emb: torch.Tensor) -> torch.Tensor:
    """Apply 2D RoPE to input features.

    Args:
        x: Input tensor of shape (B, N, dim) where N = h*w.
        cos_emb: Cosine embeddings of shape (N, dim).
        sin_emb: Sine embeddings of shape (N, dim).

    Returns:
        Rotated features of shape (B, N, dim).
    """
    # Reshape for rotation: split into pairs
    x1, x2 = x.chunk(2, dim=-1)  # Each (B, N, dim//2)

    # Rotate using RoPE formula
    # [x1*cos - x2*sin, x1*sin + x2*cos]
    cos_half = cos_emb.chunk(2, dim=-1)
    sin_half = sin_emb.chunk(2, dim=-1)

    out1 = x1 * cos_half[0] - x2 * sin_half[0]
    out2 = x1 * sin_half[1] + x2 * cos_half[1]

    return torch.cat([out1, out2], dim=-1)


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (SwiGLU) activation.

    SwiGLU is a gated activation function that uses Swish (SiLU) as the activation:
        SwiGLU(x) = Swish(W1*x) ⊙ (W2*x)

    This provides better expressiveness than standard GELU or ReLU activations.

    Args:
        dim: Input dimension.
        hidden_dim: Hidden dimension (typically ~2.67x input dim for SwiGLU).
        dropout: Dropout rate. Default: 0.0.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Output tensor of shape (..., dim).
        """
        # Swish activation: x * sigmoid(x)
        swish_out = F.silu(self.w1(x))
        gated = swish_out * self.w2(x)
        out = self.w3(gated)
        out = self.dropout(out)
        return out


class EVA02Block(NexusModule):
    """Transformer block for EVA-02 with RoPE and SwiGLU.

    Combines multi-head self-attention with 2D RoPE positional encoding and
    SwiGLU activation in the MLP block.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio (default 4.0, but SwiGLU uses 8/3).
        dropout: Dropout rate.
        drop_path: Stochastic depth rate.
        use_swiglu: Whether to use SwiGLU (True) or standard GELU (False).
        rope_2d: Optional pre-initialized RoPE2D module.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_swiglu: bool = True,
        rope_2d: Optional[RoPE2D] = None,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.use_swiglu = use_swiglu
        self.rope_2d = rope_2d

        self.norm1 = nn.LayerNorm(dim)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)

        # MLP with SwiGLU or standard GELU
        if use_swiglu:
            # SwiGLU typically uses smaller hidden dim: 8/3 * dim
            hidden_dim = int(dim * 8 / 3)
            self.mlp = SwiGLU(dim, hidden_dim, dropout)
        else:
            hidden_dim = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
            )

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        rope_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, dim) where N = h*w + 1 (with CLS token).
            rope_emb: Optional tuple of (cos_emb, sin_emb) for RoPE.

        Returns:
            Output tensor of shape (B, N, dim).
        """
        # Self-attention with RoPE
        shortcut = x
        x = self.norm1(x)

        # Apply RoPE to query and key (skip CLS token at position 0)
        if rope_emb is not None and self.rope_2d is not None:
            cos_emb, sin_emb = rope_emb
            # Split CLS token and patch tokens
            cls_token = x[:, :1]
            patch_tokens = x[:, 1:]

            # Apply RoPE to patches only
            patch_tokens = apply_rope_2d(patch_tokens, cos_emb, sin_emb)

            # Recombine
            x = torch.cat([cls_token, patch_tokens], dim=1)

        attn_out, _ = self.attn(x, x, x)
        x = shortcut + self.drop_path(attn_out)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class DropPath(nn.Module):
    """Stochastic Depth / Drop Path regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x * mask / keep_prob


class EVA02(WeightInitMixin, NexusModule):
    """EVA-02: A Visual Representation with RoPE and SwiGLU.

    Vision Transformer with 2D Rotary Position Embeddings and SwiGLU activations.
    Designed for masked image modeling pre-training and various downstream tasks.

    Config:
        img_size (int): Input image resolution. Default: 224.
        patch_size (int): Patch size. Default: 16.
        in_channels (int): Number of input channels. Default: 3.
        embed_dim (int): Embedding dimension. Default: 768.
        depth (int): Number of transformer blocks. Default: 12.
        num_heads (int): Number of attention heads. Default: 12.
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1.
        use_swiglu (bool): Use SwiGLU activation. Default: True.
        use_rope (bool): Use 2D RoPE. Default: True.
        rope_temperature (float): RoPE temperature. Default: 10000.0.
        num_classes (int): Number of output classes (0 for feature extraction). Default: 0.
        global_pool (str): Global pooling type ("token" or "avg"). Default: "token".

    Example:
        >>> config = {
        ...     "img_size": 224,
        ...     "embed_dim": 768,
        ...     "depth": 12,
        ...     "num_heads": 12,
        ...     "use_rope": True,
        ...     "use_swiglu": True,
        ... }
        >>> model = EVA02(config)
        >>> images = torch.randn(2, 3, 224, 224)
        >>> output = model(images)
        >>> output["embeddings"].shape
        torch.Size([2, 768])
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.img_size = config.get("img_size", 224)
        self.patch_size = config.get("patch_size", 16)
        self.in_channels = config.get("in_channels", 3)
        self.embed_dim = config.get("embed_dim", 768)
        self.depth = config.get("depth", 12)
        self.num_heads = config.get("num_heads", 12)
        self.mlp_ratio = config.get("mlp_ratio", 4.0)
        self.dropout = config.get("dropout", 0.0)
        self.drop_path_rate = config.get("drop_path_rate", 0.1)
        self.use_swiglu = config.get("use_swiglu", True)
        self.use_rope = config.get("use_rope", True)
        self.num_classes = config.get("num_classes", 0)
        self.global_pool = config.get("global_pool", "token")

        self.grid_size = self.img_size // self.patch_size
        self.num_patches = self.grid_size ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            self.in_channels, self.embed_dim,
            kernel_size=self.patch_size, stride=self.patch_size,
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # 2D RoPE (no learnable positional embeddings when using RoPE)
        if self.use_rope:
            self.rope_2d = RoPE2D(
                dim=self.embed_dim,
                max_resolution=self.grid_size,
                temperature=config.get("rope_temperature", 10000.0),
            )
        else:
            self.rope_2d = None
            # Standard learnable positional embeddings
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + 1, self.embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=self.dropout)

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            EVA02Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                drop_path=dpr[i],
                use_swiglu=self.use_swiglu,
                rope_2d=self.rope_2d,
            )
            for i in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        # Classification head (optional)
        if self.num_classes > 0:
            self.head = nn.Linear(self.embed_dim, self.num_classes)
        else:
            self.head = nn.Identity()

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.init_weights_vit()

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            Dictionary containing:
                embeddings: Global features (B, embed_dim).
                patch_tokens: Patch-level features (B, num_patches, embed_dim).
                logits: Classification logits (if num_classes > 0).
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding or prepare RoPE
        if self.use_rope:
            # Generate RoPE embeddings for current resolution
            rope_emb = self.rope_2d(self.grid_size, self.grid_size, x.device)
        else:
            x = x + self.pos_embed
            rope_emb = None

        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, rope_emb=rope_emb)

        x = self.norm(x)

        # Global pooling
        if self.global_pool == "avg":
            # Average pooling over patch tokens (exclude CLS)
            embeddings = x[:, 1:].mean(dim=1)
        else:
            # CLS token
            embeddings = x[:, 0]

        output = {
            "embeddings": embeddings,
            "patch_tokens": x[:, 1:],
        }

        # Classification head
        if self.num_classes > 0:
            logits = self.head(embeddings)
            output["logits"] = logits

        return output


# Model variants
EVA02_VARIANTS = {
    "eva02_small": {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
    },
    "eva02_base": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
    },
    "eva02_large": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
    },
    "eva02_giant": {
        "embed_dim": 1408,
        "depth": 40,
        "num_heads": 16,
    },
}


__all__ = [
    "EVA02",
    "EVA02Block",
    "RoPE2D",
    "SwiGLU",
    "apply_rope_2d",
    "EVA02_VARIANTS",
]
