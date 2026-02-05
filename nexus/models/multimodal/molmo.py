"""Molmo: Fully open vision-language model.

Molmo is a family of fully open vision-language models from AI2 (Allen Institute for AI),
notable for being completely open-source including training data, code, and weights.

Key features:
- Fully open ecosystem (data, code, weights, training recipes)
- Efficient multimodal architecture
- Strong performance on visual reasoning tasks
- Designed for accessibility and reproducibility
- Pointing capabilities for fine-grained spatial understanding

References:
    - Molmo: https://molmo.allenai.org/ (AI2, 2024)
    - Paper: "Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models"
    - Dataset: PixMo (high-quality multimodal instruction data)

Authors: Deitke et al., Allen Institute for AI (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from nexus.core.base import NexusModule


class PixelShuffleVisionEncoder(NexusModule):
    """Vision encoder using pixel shuffle for efficient processing.

    Args:
        in_channels: Input channels (default: 3 for RGB)
        hidden_dim: Hidden dimension
        num_layers: Number of encoder layers
        patch_size: Initial patch size
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 768,
        num_layers: int = 12,
        patch_size: int = 16,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Vision transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=12,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images.

        Args:
            images: [batch_size, 3, H, W]

        Returns:
            Visual features [batch_size, num_patches, hidden_dim]
        """
        # Extract patches
        x = self.patch_embed(images)  # [B, hidden_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x


class PointingModule(NexusModule):
    """Module for generating spatial pointing outputs.

    Enables the model to point to specific regions in images,
    useful for visual grounding and spatial reasoning tasks.

    Args:
        hidden_dim: Hidden dimension
        num_points: Number of points to predict
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_points: int = 10,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.num_points = num_points

        # Point prediction head
        self.point_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_points * 2)  # (x, y) coordinates per point
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict pointing coordinates.

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]

        Returns:
            Point coordinates [batch_size, num_points, 2] normalized to [0, 1]
        """
        # Use last token for pointing prediction (or pooled representation)
        pooled = hidden_states[:, -1, :]  # [B, hidden_dim]

        # Predict points
        points = self.point_predictor(pooled)  # [B, num_points * 2]
        points = points.view(-1, self.num_points, 2)  # [B, num_points, 2]

        # Apply sigmoid to normalize to [0, 1]
        points = torch.sigmoid(points)

        return points


class MultimodalProjector(NexusModule):
    """Projects visual features to language model space.

    Args:
        visual_dim: Dimension of visual encoder
        text_dim: Dimension of language model
        num_layers: Number of projection layers
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 2048,
        num_layers: int = 2,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)

        layers = []
        for i in range(num_layers):
            in_dim = visual_dim if i == 0 else text_dim
            layers.extend([
                nn.Linear(in_dim, text_dim),
                nn.GELU() if i < num_layers - 1 else nn.Identity()
            ])

        self.projector = nn.Sequential(*layers)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Project visual features.

        Args:
            visual_features: [batch_size, num_patches, visual_dim]

        Returns:
            Projected features [batch_size, num_patches, text_dim]
        """
        return self.projector(visual_features)


class Molmo(NexusModule):
    """Molmo: Fully open vision-language model from AI2.

    A completely open multimodal model including training data (PixMo),
    model weights, and training code. Designed for accessibility and
    reproducibility in multimodal AI research.

    Key capabilities:
    - Image understanding and visual reasoning
    - Visual question answering
    - Spatial pointing for fine-grained grounding
    - Document understanding
    - Chart and diagram interpretation

    Args:
        visual_encoder_dim: Dimension of vision encoder
        language_model_dim: Dimension of language model
        num_visual_layers: Number of vision encoder layers
        patch_size: Vision patch size
        enable_pointing: Whether to enable spatial pointing
        num_points: Number of pointing locations to predict

    Example:
        >>> model = Molmo(
        ...     visual_encoder_dim=768,
        ...     language_model_dim=2048,
        ...     num_visual_layers=12,
        ...     enable_pointing=True
        ... )
        >>> images = torch.randn(2, 3, 336, 336)
        >>> text_embeds = torch.randn(2, 50, 2048)
        >>> output = model(images, text_embeds)
        >>> # Access pointing predictions if enabled
        >>> points = output['pointing_coords']  # [2, 10, 2]
    """

    def __init__(
        self,
        visual_encoder_dim: int = 768,
        language_model_dim: int = 2048,
        num_visual_layers: int = 12,
        patch_size: int = 16,
        enable_pointing: bool = True,
        num_points: int = 10,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.visual_encoder_dim = visual_encoder_dim
        self.language_model_dim = language_model_dim
        self.enable_pointing = enable_pointing

        # Vision encoder
        self.vision_encoder = PixelShuffleVisionEncoder(
            in_channels=3,
            hidden_dim=visual_encoder_dim,
            num_layers=num_visual_layers,
            patch_size=patch_size
        )

        # Multimodal projector
        self.projector = MultimodalProjector(
            visual_dim=visual_encoder_dim,
            text_dim=language_model_dim,
            num_layers=2
        )

        # Optional pointing module
        if enable_pointing:
            self.pointing_module = PointingModule(
                hidden_dim=language_model_dim,
                num_points=num_points
            )

        # Special tokens
        self.vision_start_token = nn.Parameter(
            torch.randn(1, 1, language_model_dim)
        )
        self.vision_end_token = nn.Parameter(
            torch.randn(1, 1, language_model_dim)
        )

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to visual features.

        Args:
            images: Input images [batch_size, 3, H, W]

        Returns:
            Visual features [batch_size, num_patches, language_model_dim]
        """
        # Extract visual features
        visual_features = self.vision_encoder(images)

        # Project to language space
        projected_features = self.projector(visual_features)

        return projected_features

    def forward(
        self,
        images: torch.Tensor,
        text_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_pointing: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through Molmo.

        Args:
            images: Input images [batch_size, 3, H, W]
            text_embeds: Text embeddings [batch_size, text_seq_len, language_model_dim]
            attention_mask: Attention mask for text
            return_pointing: Whether to compute pointing predictions

        Returns:
            Dictionary containing:
                - visual_features: Encoded visual features
                - multimodal_embeds: Fused vision-language embeddings
                - pointing_coords: Pointing coordinates (if enabled and requested)
                - attention_mask: Extended attention mask
        """
        outputs = {}
        B = images.shape[0]

        # Encode images
        visual_features = self.encode_images(images)
        outputs['visual_features'] = visual_features

        # Add vision boundary tokens
        vision_start = self.vision_start_token.expand(B, -1, -1)
        vision_end = self.vision_end_token.expand(B, -1, -1)

        # Combine with text if provided
        if text_embeds is not None:
            # Concatenate: [vision_start, visual_features, vision_end, text]
            multimodal_embeds = torch.cat([
                vision_start,
                visual_features,
                vision_end,
                text_embeds
            ], dim=1)

            outputs['multimodal_embeds'] = multimodal_embeds

            # Extend attention mask
            if attention_mask is not None:
                visual_mask = torch.ones(
                    B,
                    visual_features.shape[1] + 2,  # +2 for start/end tokens
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                extended_mask = torch.cat([visual_mask, attention_mask], dim=1)
                outputs['attention_mask'] = extended_mask

            # Compute pointing predictions if requested
            if return_pointing and self.enable_pointing:
                pointing_coords = self.pointing_module(multimodal_embeds)
                outputs['pointing_coords'] = pointing_coords

        else:
            # Vision-only mode
            multimodal_embeds = torch.cat([
                vision_start,
                visual_features,
                vision_end
            ], dim=1)
            outputs['multimodal_embeds'] = multimodal_embeds

        return outputs


class MolmoConfig:
    """Configuration class for Molmo models.

    Provides preset configurations for different Molmo model sizes.
    """

    @staticmethod
    def molmo_7b():
        """Configuration for Molmo-7B model."""
        return {
            'visual_encoder_dim': 1024,
            'language_model_dim': 4096,
            'num_visual_layers': 12,
            'patch_size': 16,
            'enable_pointing': True,
            'num_points': 10
        }

    @staticmethod
    def molmo_72b():
        """Configuration for Molmo-72B model."""
        return {
            'visual_encoder_dim': 1280,
            'language_model_dim': 8192,
            'num_visual_layers': 18,
            'patch_size': 14,
            'enable_pointing': True,
            'num_points': 20
        }

    @staticmethod
    def molmo_base():
        """Configuration for Molmo-Base model."""
        return {
            'visual_encoder_dim': 768,
            'language_model_dim': 2048,
            'num_visual_layers': 12,
            'patch_size': 16,
            'enable_pointing': True,
            'num_points': 10
        }


# Export
__all__ = [
    'Molmo',
    'MolmoConfig',
    'PixelShuffleVisionEncoder',
    'PointingModule',
    'MultimodalProjector'
]
