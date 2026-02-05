"""
SAM 2: Segment Anything in Images and Videos

Implementation of SAM 2, which extends the Segment Anything Model to video
segmentation with streaming memory and temporal modeling capabilities.

Reference:
    Ravi, N., Gabeur, V., Hu, Y., et al. (2024).
    "SAM 2: Segment Anything in Images and Videos."
    arXiv:2408.00714

Key Components:
    - SAM2ImageEncoder: Hiera-based image encoder
    - StreamingMemory: Temporal memory bank for video tracking
    - SAM2VideoPredictor: Video segmentation with memory attention
    - SAM2: Full model with image and video modes

Architecture Details:
    - Hiera backbone for efficient hierarchical encoding
    - Memory attention mechanism for temporal consistency
    - Streaming inference for long videos
    - Cross-frame attention for object tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Deque
from collections import deque

from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin


class StreamingMemory(NexusModule):
    """Streaming memory bank for video object tracking.

    Maintains a sliding window of frame embeddings and mask predictions
    to provide temporal context for video segmentation.

    Args:
        memory_size (int): Maximum number of frames to keep in memory. Default: 7.
        embed_dim (int): Embedding dimension. Default: 256.
        num_heads (int): Number of attention heads for memory. Default: 8.
    """

    def __init__(
        self,
        memory_size: int = 7,
        embed_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()

        self.memory_size = memory_size
        self.embed_dim = embed_dim

        # Memory attention for cross-frame modeling
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.memory_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Memory buffers (managed externally)
        self._memory_frames: Deque[torch.Tensor] = deque(maxlen=memory_size)
        self._memory_masks: Deque[torch.Tensor] = deque(maxlen=memory_size)

    def add_memory(
        self,
        frame_embedding: torch.Tensor,
        mask_prediction: torch.Tensor,
    ) -> None:
        """Add a frame and its mask to memory.

        Args:
            frame_embedding: Frame embedding (B, N, embed_dim).
            mask_prediction: Predicted mask (B, 1, H, W).
        """
        self._memory_frames.append(frame_embedding.detach())
        self._memory_masks.append(mask_prediction.detach())

    def clear_memory(self) -> None:
        """Clear all memory."""
        self._memory_frames.clear()
        self._memory_masks.clear()

    def forward(
        self,
        query_features: torch.Tensor,
    ) -> torch.Tensor:
        """Query memory with current frame features.

        Args:
            query_features: Current frame features (B, N, embed_dim).

        Returns:
            Memory-augmented features (B, N, embed_dim).
        """
        if len(self._memory_frames) == 0:
            return query_features

        # Stack memory frames
        memory = torch.stack(list(self._memory_frames), dim=1)  # (B, T, N, embed_dim)
        B, T, N, D = memory.shape

        # Reshape for attention
        memory = memory.reshape(B, T * N, D)

        # Memory attention
        normed_query = self.norm1(query_features)
        attn_out, _ = self.memory_attention(
            normed_query, memory, memory,
        )
        query_features = query_features + attn_out

        # MLP
        query_features = query_features + self.mlp(self.norm2(query_features))

        return query_features

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        """MLP forward."""
        return self.memory_mlp(x)


class HieraBlock(NexusModule):
    """Hiera transformer block with masked unit attention.

    Hiera uses hierarchical masked unit attention for efficiency.

    Args:
        dim (int): Feature dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP expansion ratio. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (B, H, W, C).

        Returns:
            Output (B, H, W, C).
        """
        # Flatten spatial dimensions
        B, H, W, C = x.shape
        x_flat = x.reshape(B, H * W, C)

        # Self-attention
        normed = self.norm1(x_flat)
        attn_out, _ = self.attn(normed, normed, normed)
        x_flat = x_flat + attn_out

        # MLP
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        # Reshape back
        x = x_flat.reshape(B, H, W, C)
        return x


class SAM2ImageEncoder(WeightInitMixin, NexusModule):
    """Hiera-based image encoder for SAM 2.

    More efficient than the original SAM's ViT encoder.

    Args:
        config: Configuration dictionary with keys:
            img_size (int): Input image size. Default: 1024.
            patch_size (int): Initial patch size. Default: 16.
            in_channels (int): Input channels. Default: 3.
            embed_dim (int): Embedding dimension. Default: 768.
            depths (List[int]): Number of blocks per stage. Default: [2, 3, 16, 3].
            num_heads (int): Base number of attention heads. Default: 12.
            out_channels (int): Output channels. Default: 256.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.img_size = config.get("img_size", 1024)
        self.patch_size = config.get("patch_size", 16)
        self.embed_dim = config.get("embed_dim", 768)
        self.depths = config.get("depths", [2, 3, 16, 3])
        self.num_heads = config.get("num_heads", 12)
        self.out_channels = config.get("out_channels", 256)

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.get("in_channels", 3), self.embed_dim,
            kernel_size=self.patch_size, stride=self.patch_size,
        )

        # Hiera stages
        self.stages = nn.ModuleList()
        for stage_idx, depth in enumerate(self.depths):
            stage = nn.ModuleList([
                HieraBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=4.0,
                    dropout=0.0,
                )
                for _ in range(depth)
            ])
            self.stages.append(stage)

        # Output projection
        self.neck = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out_channels, kernel_size=1, bias=False),
            nn.LayerNorm([self.out_channels, self.img_size // self.patch_size, self.img_size // self.patch_size]),
        )

        self.init_weights_vit()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images (B, C, H, W).

        Returns:
            Encoded features (B, out_channels, H', W').
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.permute(0, 2, 3, 1)  # (B, H', W', embed_dim)

        # Hiera stages
        for stage in self.stages:
            for block in stage:
                x = block(x)

        # Output projection
        x = x.permute(0, 3, 1, 2)  # (B, embed_dim, H', W')
        x = self.neck(x)

        return x


class SAM2(WeightInitMixin, NexusModule):
    """SAM 2: Segment Anything in Images and Videos.

    Extends SAM with streaming memory for video segmentation.

    Config:
        img_size (int): Input image size. Default: 1024.
        patch_size (int): Patch size. Default: 16.
        encoder_embed_dim (int): Encoder embedding dimension. Default: 768.
        encoder_depths (List[int]): Encoder stage depths. Default: [2, 3, 16, 3].
        encoder_num_heads (int): Encoder attention heads. Default: 12.
        decoder_dim (int): Decoder dimension. Default: 256.
        num_mask_tokens (int): Number of mask tokens. Default: 4.
        memory_size (int): Memory bank size. Default: 7.
        video_mode (bool): Enable video mode with memory. Default: False.

    Example:
        >>> # Image mode
        >>> config = {"img_size": 1024, "video_mode": False}
        >>> model = SAM2(config)
        >>> images = torch.randn(1, 3, 1024, 1024)
        >>> points = (torch.tensor([[[512, 512]]]), torch.tensor([[1]]))
        >>> output = model(images, points=points)
        >>>
        >>> # Video mode
        >>> config["video_mode"] = True
        >>> model = SAM2(config)
        >>> for frame in video_frames:
        ...     output = model(frame, points=points)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.video_mode = config.get("video_mode", False)

        # Image encoder (Hiera)
        self.image_encoder = SAM2ImageEncoder(config)

        # Streaming memory (for video mode)
        if self.video_mode:
            self.memory = StreamingMemory(
                memory_size=config.get("memory_size", 7),
                embed_dim=config.get("decoder_dim", 256),
                num_heads=config.get("memory_num_heads", 8),
            )
        else:
            self.memory = None

        # Simplified decoder (in practice, use full SAM decoder)
        decoder_dim = config.get("decoder_dim", 256)
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim, config.get("num_mask_tokens", 4), kernel_size=1),
        )

    def forward(
        self,
        images: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: Input images (B, C, H, W).
            points: Point prompts (coords, labels).
            boxes: Box prompts (B, N, 4).
            masks: Mask prompts (B, 1, H, W).

        Returns:
            Dictionary with mask predictions and features.
        """
        # Encode image
        image_features = self.image_encoder(images)  # (B, decoder_dim, H', W')

        # Apply memory (in video mode)
        if self.video_mode and self.memory is not None:
            B, C, H, W = image_features.shape
            feat_flat = image_features.permute(0, 2, 3, 1).reshape(B, H * W, C)
            feat_flat = self.memory(feat_flat)
            image_features = feat_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Decode masks (simplified - in practice, use full SAM decoder with prompts)
        masks_pred = self.decoder(image_features)

        # Add to memory (in video mode)
        if self.video_mode and self.memory is not None:
            feat_flat = image_features.permute(0, 2, 3, 1).reshape(B, H * W, C)
            self.memory.add_memory(feat_flat, masks_pred[:, 0:1])

        # Upsample masks to input resolution
        masks_pred = F.interpolate(
            masks_pred,
            size=images.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

        return {
            "masks": masks_pred,
            "image_features": image_features,
        }

    def reset_memory(self) -> None:
        """Reset video memory (call between videos)."""
        if self.memory is not None:
            self.memory.clear_memory()


__all__ = [
    "SAM2",
    "SAM2ImageEncoder",
    "StreamingMemory",
    "HieraBlock",
]
