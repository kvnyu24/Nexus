"""Phi-3-Vision: Lightweight multimodal model with 128K context.

Phi-3-Vision is Microsoft's lightweight yet powerful multimodal model that extends
the Phi-3 language model with vision capabilities while maintaining efficiency.

Key features:
- Lightweight architecture (3.8B-4.2B parameters)
- 128K token context length support
- Strong performance despite small size
- Efficient for edge deployment
- Multi-image understanding
- High-resolution image support

References:
    - Phi-3 Technical Report: https://arxiv.org/abs/2404.14219 (Microsoft, 2024)
    - Blog: https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/

Authors: Microsoft Research (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
from nexus.core.base import NexusModule


class EfficientImageEncoder(NexusModule):
    """Efficient vision encoder optimized for lightweight deployment.

    Uses a streamlined ViT architecture with efficiency optimizations
    for edge deployment scenarios.

    Args:
        in_channels: Input channels
        hidden_dim: Hidden dimension
        num_layers: Number of encoder layers
        num_heads: Number of attention heads
        patch_size: Size of image patches
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        patch_size: int = 16,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # Efficient patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, hidden_dim) * 0.02)

        # Lightweight transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images efficiently.

        Args:
            images: [batch_size, 3, H, W]

        Returns:
            Image features [batch_size, num_patches, hidden_dim]
        """
        B, C, H, W = images.shape

        # Extract patches
        x = self.patch_embed(images)  # [B, hidden_dim, H', W']
        H_patch, W_patch = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]

        # Add positional embeddings
        num_patches = x.shape[1]
        if num_patches <= self.pos_embed.shape[1]:
            pos_embed = self.pos_embed[:, :num_patches, :]
        else:
            # Interpolate if needed
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=num_patches,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        x = x + pos_embed

        # Apply transformer
        x = self.encoder(x)
        x = self.norm(x)

        return x


class LongContextProjector(NexusModule):
    """Projector optimized for long context (128K tokens).

    Args:
        visual_dim: Visual encoder dimension
        text_dim: Language model dimension
        compression_ratio: Compression ratio for visual tokens
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 3072,
        compression_ratio: int = 2,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.compression_ratio = compression_ratio

        # Compression layer (reduce token count for long context)
        if compression_ratio > 1:
            self.compress = nn.Conv1d(
                visual_dim,
                visual_dim,
                kernel_size=compression_ratio,
                stride=compression_ratio
            )
        else:
            self.compress = nn.Identity()

        # Projection layers
        self.projector = nn.Sequential(
            nn.Linear(visual_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Project and optionally compress visual features.

        Args:
            visual_features: [batch_size, num_patches, visual_dim]

        Returns:
            Projected features [batch_size, compressed_patches, text_dim]
        """
        # Compress if needed
        if self.compression_ratio > 1:
            # Transpose for conv1d
            x = visual_features.transpose(1, 2)  # [B, visual_dim, num_patches]
            x = self.compress(x)
            x = x.transpose(1, 2)  # [B, compressed_patches, visual_dim]
        else:
            x = visual_features

        # Project to text space
        x = self.projector(x)

        return x


class MultiImageFusion(NexusModule):
    """Module for fusing multiple images in a single context.

    Args:
        hidden_dim: Hidden dimension
        max_images: Maximum number of images
    """

    def __init__(
        self,
        hidden_dim: int = 3072,
        max_images: int = 16,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.max_images = max_images

        # Image separator tokens
        self.image_sep_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Optional cross-image attention
        self.cross_image_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=hidden_dim // 256,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        image_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Fuse multiple images with separators.

        Args:
            image_features: List of image features, each [batch_size, num_patches, hidden_dim]

        Returns:
            Fused features [batch_size, total_patches, hidden_dim]
        """
        B = image_features[0].shape[0]
        sep_token = self.image_sep_token.expand(B, -1, -1)

        # Interleave images with separators
        fused_features = []
        for i, img_feat in enumerate(image_features):
            fused_features.append(img_feat)
            if i < len(image_features) - 1:  # Don't add separator after last image
                fused_features.append(sep_token)

        fused = torch.cat(fused_features, dim=1)

        # Apply cross-image attention for multi-image reasoning
        attn_out, _ = self.cross_image_attn(fused, fused, fused)
        fused = self.norm(fused + attn_out)

        return fused


class Phi3Vision(NexusModule):
    """Phi-3-Vision: Lightweight multimodal model with 128K context.

    Microsoft's efficient vision-language model that combines the small
    but powerful Phi-3 language model with vision capabilities, supporting
    extremely long contexts (128K tokens).

    Key advantages:
    - Compact size (3.8B-4.2B parameters)
    - Long context support (128K tokens)
    - Multi-image understanding
    - Efficient inference
    - Strong performance despite small size

    Args:
        visual_encoder_dim: Dimension of vision encoder
        language_model_dim: Dimension of language model
        num_visual_layers: Number of vision encoder layers
        patch_size: Vision patch size
        compression_ratio: Visual token compression ratio
        max_images: Maximum images per context
        max_context_length: Maximum context length (default: 128K)

    Example:
        >>> model = Phi3Vision(
        ...     visual_encoder_dim=768,
        ...     language_model_dim=3072,
        ...     num_visual_layers=12,
        ...     compression_ratio=2,
        ...     max_images=8
        ... )
        >>> # Single image
        >>> images = torch.randn(2, 3, 336, 336)
        >>> text_embeds = torch.randn(2, 100, 3072)
        >>> output = model(images=[images], text_embeds=text_embeds)
        >>> # Multiple images
        >>> images1 = torch.randn(2, 3, 336, 336)
        >>> images2 = torch.randn(2, 3, 336, 336)
        >>> output = model(images=[images1, images2], text_embeds=text_embeds)
    """

    def __init__(
        self,
        visual_encoder_dim: int = 768,
        language_model_dim: int = 3072,
        num_visual_layers: int = 12,
        patch_size: int = 16,
        compression_ratio: int = 2,
        max_images: int = 16,
        max_context_length: int = 131072,  # 128K
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.visual_encoder_dim = visual_encoder_dim
        self.language_model_dim = language_model_dim
        self.max_images = max_images
        self.max_context_length = max_context_length
        self.compression_ratio = compression_ratio

        # Efficient vision encoder
        self.vision_encoder = EfficientImageEncoder(
            in_channels=3,
            hidden_dim=visual_encoder_dim,
            num_layers=num_visual_layers,
            num_heads=visual_encoder_dim // 64,
            patch_size=patch_size
        )

        # Long-context projector with compression
        self.projector = LongContextProjector(
            visual_dim=visual_encoder_dim,
            text_dim=language_model_dim,
            compression_ratio=compression_ratio
        )

        # Multi-image fusion
        self.multi_image_fusion = MultiImageFusion(
            hidden_dim=language_model_dim,
            max_images=max_images
        )

        # Special tokens
        self.image_start_token = nn.Parameter(
            torch.randn(1, 1, language_model_dim) * 0.02
        )
        self.image_end_token = nn.Parameter(
            torch.randn(1, 1, language_model_dim) * 0.02
        )

    def encode_single_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a single image.

        Args:
            image: [batch_size, 3, H, W]

        Returns:
            Image features [batch_size, compressed_patches, language_model_dim]
        """
        # Extract visual features
        visual_features = self.vision_encoder(image)

        # Project and compress
        projected_features = self.projector(visual_features)

        return projected_features

    def encode_images(
        self,
        images: List[torch.Tensor]
    ) -> torch.Tensor:
        """Encode multiple images with fusion.

        Args:
            images: List of image tensors, each [batch_size, 3, H, W]

        Returns:
            Fused image features [batch_size, total_patches, language_model_dim]
        """
        if len(images) > self.max_images:
            raise ValueError(
                f"Number of images ({len(images)}) exceeds max_images ({self.max_images})"
            )

        # Encode each image
        encoded_images = []
        for image in images:
            encoded = self.encode_single_image(image)
            encoded_images.append(encoded)

        # Fuse multiple images if needed
        if len(encoded_images) > 1:
            fused_features = self.multi_image_fusion(encoded_images)
        else:
            fused_features = encoded_images[0]

        return fused_features

    def forward(
        self,
        images: List[torch.Tensor],
        text_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through Phi-3-Vision.

        Args:
            images: List of image tensors, each [batch_size, 3, H, W]
            text_embeds: Text embeddings [batch_size, text_seq_len, language_model_dim]
            attention_mask: Attention mask for text

        Returns:
            Dictionary containing:
                - visual_features: Encoded and fused visual features
                - multimodal_embeds: Combined vision-language embeddings
                - attention_mask: Extended attention mask
        """
        outputs = {}
        B = images[0].shape[0]

        # Encode images (single or multiple)
        visual_features = self.encode_images(images)
        outputs['visual_features'] = visual_features

        # Add boundary tokens
        image_start = self.image_start_token.expand(B, -1, -1)
        image_end = self.image_end_token.expand(B, -1, -1)

        # Combine with text if provided
        if text_embeds is not None:
            # Check total context length
            total_length = visual_features.shape[1] + 2 + text_embeds.shape[1]
            if total_length > self.max_context_length:
                import warnings
                warnings.warn(
                    f"Total context length ({total_length}) exceeds max_context_length "
                    f"({self.max_context_length}). Consider increasing compression_ratio."
                )

            # Concatenate: [image_start, visual_features, image_end, text]
            multimodal_embeds = torch.cat([
                image_start,
                visual_features,
                image_end,
                text_embeds
            ], dim=1)

            outputs['multimodal_embeds'] = multimodal_embeds

            # Extend attention mask
            if attention_mask is not None:
                visual_mask = torch.ones(
                    B,
                    visual_features.shape[1] + 2,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                extended_mask = torch.cat([visual_mask, attention_mask], dim=1)
                outputs['attention_mask'] = extended_mask
        else:
            # Vision-only mode
            multimodal_embeds = torch.cat([
                image_start,
                visual_features,
                image_end
            ], dim=1)
            outputs['multimodal_embeds'] = multimodal_embeds

        return outputs


class Phi3VisionConfig:
    """Configuration presets for Phi-3-Vision models."""

    @staticmethod
    def phi3_vision_128k():
        """Standard Phi-3-Vision with 128K context."""
        return {
            'visual_encoder_dim': 768,
            'language_model_dim': 3072,
            'num_visual_layers': 12,
            'patch_size': 16,
            'compression_ratio': 2,
            'max_images': 16,
            'max_context_length': 131072
        }

    @staticmethod
    def phi3_vision_instruct():
        """Instruction-tuned variant."""
        return {
            'visual_encoder_dim': 768,
            'language_model_dim': 3072,
            'num_visual_layers': 12,
            'patch_size': 16,
            'compression_ratio': 2,
            'max_images': 8,
            'max_context_length': 131072
        }


# Export
__all__ = [
    'Phi3Vision',
    'Phi3VisionConfig',
    'EfficientImageEncoder',
    'LongContextProjector',
    'MultiImageFusion'
]
