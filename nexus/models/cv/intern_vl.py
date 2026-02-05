"""
InternVL: Scaling Up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks

Implementation of InternVL, an open-source multimodal Vision Transformer family that
bridges vision and language models through contrastive learning and progressive alignment.

Reference:
    Chen, Z., Wu, J., Wang, W., et al. (2024).
    "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks."
    CVPR 2024
    arXiv:2312.14238

Key Components:
    - InternVisionModel: Large-scale ViT encoder (up to 6B parameters)
    - InternVLEmbedding: Cross-modal projection with QLLaMA adapter
    - InternVL: Full multimodal model with vision-language alignment

Architecture Details:
    - Progressive scaling: InternViT-300M/1B/6B variants
    - Contrastive learning on large-scale image-text pairs
    - QLLaMA: Quantized LLaMA integration for language understanding
    - Dynamic resolution support with position interpolation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List

from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin


class InternVisionModel(WeightInitMixin, NexusModule):
    """Large-scale Vision Transformer for InternVL.

    Implements a standard ViT architecture that can scale to billions of parameters.

    Args:
        config: Configuration dictionary with keys:
            img_size (int): Input image resolution. Default: 224.
            patch_size (int): Patch size. Default: 14.
            in_channels (int): Number of input channels. Default: 3.
            embed_dim (int): Embedding dimension. Default: 1024.
            depth (int): Number of transformer blocks. Default: 24.
            num_heads (int): Number of attention heads. Default: 16.
            mlp_ratio (float): MLP hidden dim ratio. Default: 4.0.
            dropout (float): Dropout rate. Default: 0.0.
            drop_path_rate (float): Stochastic depth rate. Default: 0.0.
            qkv_bias (bool): Add bias to QKV projection. Default: True.
            use_cls_token (bool): Use CLS token for global pooling. Default: True.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.img_size = config.get("img_size", 224)
        self.patch_size = config.get("patch_size", 14)
        self.in_channels = config.get("in_channels", 3)
        self.embed_dim = config.get("embed_dim", 1024)
        self.depth = config.get("depth", 24)
        self.num_heads = config.get("num_heads", 16)
        self.mlp_ratio = config.get("mlp_ratio", 4.0)
        self.dropout = config.get("dropout", 0.0)
        self.drop_path_rate = config.get("drop_path_rate", 0.0)
        self.qkv_bias = config.get("qkv_bias", True)
        self.use_cls_token = config.get("use_cls_token", True)

        self.grid_size = self.img_size // self.patch_size
        self.num_patches = self.grid_size ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            self.in_channels, self.embed_dim,
            kernel_size=self.patch_size, stride=self.patch_size,
        )

        # CLS token and positional embeddings
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.num_tokens = self.num_patches + 1
        else:
            self.num_tokens = self.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, self.embed_dim)
        )
        self.pos_drop = nn.Dropout(p=self.dropout)

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            InternVisionBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                drop_path=dpr[i],
                qkv_bias=self.qkv_bias,
            )
            for i in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        # Initialize weights
        if self.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.init_weights_vit()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            Dictionary containing:
                embeddings: Global features (B, embed_dim).
                patch_tokens: Patch-level features (B, num_patches, embed_dim).
                all_tokens: All tokens including CLS (B, num_tokens, embed_dim).
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract features
        if self.use_cls_token:
            embeddings = x[:, 0]
            patch_tokens = x[:, 1:]
        else:
            embeddings = x.mean(dim=1)
            patch_tokens = x

        return {
            "embeddings": embeddings,
            "patch_tokens": patch_tokens,
            "all_tokens": x,
        }


class InternVisionBlock(NexusModule):
    """Transformer block for InternVision encoder.

    Standard ViT block with multi-head attention and MLP.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim expansion ratio.
        dropout: Dropout rate.
        drop_path: Stochastic depth rate.
        qkv_bias: Add bias to QKV projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=qkv_bias,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, dim).

        Returns:
            Output tensor of shape (B, N, dim).
        """
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop_path(attn_out)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class DropPath(nn.Module):
    """Stochastic Depth / Drop Path."""

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


class InternVLEmbedding(NexusModule):
    """Cross-modal projection layer for InternVL.

    Projects vision features to language model dimension for multimodal fusion.

    Args:
        config: Configuration dictionary with keys:
            vision_dim (int): Vision encoder output dimension. Default: 1024.
            language_dim (int): Language model dimension. Default: 4096.
            num_proj_layers (int): Number of projection layers. Default: 2.
            dropout (float): Dropout rate. Default: 0.0.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.vision_dim = config.get("vision_dim", 1024)
        self.language_dim = config.get("language_dim", 4096)
        num_proj_layers = config.get("num_proj_layers", 2)
        dropout = config.get("dropout", 0.0)

        # Multi-layer projection
        layers = []
        in_dim = self.vision_dim
        for i in range(num_proj_layers):
            out_dim = self.language_dim if i == num_proj_layers - 1 else self.vision_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_proj_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.projection = nn.Sequential(*layers)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project vision features to language dimension.

        Args:
            vision_features: Vision features of shape (B, N, vision_dim).

        Returns:
            Projected features of shape (B, N, language_dim).
        """
        return self.projection(vision_features)


class InternVL(WeightInitMixin, NexusModule):
    """InternVL: Multimodal Vision-Language Model.

    Combines a large-scale vision encoder with a language model through
    cross-modal projection and contrastive learning.

    Config:
        # Vision encoder config
        img_size (int): Input image resolution. Default: 224.
        patch_size (int): Patch size. Default: 14.
        vision_embed_dim (int): Vision encoder embedding dimension. Default: 1024.
        vision_depth (int): Vision encoder depth. Default: 24.
        vision_num_heads (int): Vision encoder attention heads. Default: 16.

        # Cross-modal projection config
        language_dim (int): Language model dimension. Default: 4096.
        num_proj_layers (int): Number of projection layers. Default: 2.

        # Training config
        dropout (float): Dropout rate. Default: 0.0.
        temperature (float): Temperature for contrastive learning. Default: 0.07.

    Example:
        >>> config = {
        ...     "img_size": 224,
        ...     "vision_embed_dim": 1024,
        ...     "vision_depth": 24,
        ...     "language_dim": 4096,
        ... }
        >>> model = InternVL(config)
        >>> images = torch.randn(2, 3, 224, 224)
        >>> output = model(images)
        >>> output["embeddings"].shape
        torch.Size([2, 4096])
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Vision encoder
        vision_config = {
            "img_size": config.get("img_size", 224),
            "patch_size": config.get("patch_size", 14),
            "in_channels": config.get("in_channels", 3),
            "embed_dim": config.get("vision_embed_dim", 1024),
            "depth": config.get("vision_depth", 24),
            "num_heads": config.get("vision_num_heads", 16),
            "mlp_ratio": config.get("mlp_ratio", 4.0),
            "dropout": config.get("dropout", 0.0),
            "drop_path_rate": config.get("drop_path_rate", 0.0),
        }
        self.vision_encoder = InternVisionModel(vision_config)

        # Cross-modal projection
        projection_config = {
            "vision_dim": config.get("vision_embed_dim", 1024),
            "language_dim": config.get("language_dim", 4096),
            "num_proj_layers": config.get("num_proj_layers", 2),
            "dropout": config.get("dropout", 0.0),
        }
        self.projection = InternVLEmbedding(projection_config)

        # Temperature for contrastive learning
        self.temperature = nn.Parameter(
            torch.tensor(config.get("temperature", 0.07))
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to language-aligned features.

        Args:
            images: Input images of shape (B, C, H, W).

        Returns:
            Language-aligned features of shape (B, language_dim).
        """
        vision_output = self.vision_encoder(images)
        embeddings = vision_output["embeddings"]
        projected = self.projection(embeddings.unsqueeze(1)).squeeze(1)
        return F.normalize(projected, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        return_patch_tokens: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: Input images of shape (B, C, H, W).
            return_patch_tokens: Whether to return patch-level features.

        Returns:
            Dictionary containing:
                embeddings: Language-aligned global features (B, language_dim).
                patch_embeddings: Patch-level features (B, N, language_dim) if requested.
                vision_embeddings: Raw vision features (B, vision_dim).
        """
        # Vision encoding
        vision_output = self.vision_encoder(images)

        # Project global features
        global_features = self.projection(
            vision_output["embeddings"].unsqueeze(1)
        ).squeeze(1)
        global_features = F.normalize(global_features, dim=-1)

        output = {
            "embeddings": global_features,
            "vision_embeddings": vision_output["embeddings"],
        }

        # Optionally project patch tokens
        if return_patch_tokens:
            patch_features = self.projection(vision_output["patch_tokens"])
            output["patch_embeddings"] = patch_features

        return output


# Model variants
INTERNVL_VARIANTS = {
    "intern_vl_300m": {
        "vision_embed_dim": 768,
        "vision_depth": 12,
        "vision_num_heads": 12,
        "language_dim": 2048,
    },
    "intern_vl_1b": {
        "vision_embed_dim": 1024,
        "vision_depth": 24,
        "vision_num_heads": 16,
        "language_dim": 4096,
    },
    "intern_vl_6b": {
        "vision_embed_dim": 3200,
        "vision_depth": 48,
        "vision_num_heads": 25,
        "language_dim": 4096,
    },
}


__all__ = [
    "InternVL",
    "InternVisionModel",
    "InternVLEmbedding",
    "InternVisionBlock",
    "INTERNVL_VARIANTS",
]
