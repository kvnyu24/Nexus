"""LLaVA-NeXT / LLaVA-OneVision implementation.

LLaVA-NeXT (and LLaVA-OneVision) are advanced open-source multimodal LLM architectures
that extend the original LLaVA with improved visual understanding capabilities.

Key features:
- Supports multiple image resolutions and aspect ratios
- Enhanced visual-language alignment
- Improved spatial reasoning and detail perception
- Unified architecture for images and videos (OneVision)

References:
    - LLaVA-NeXT: https://llava-vl.github.io/blog/2024-01-30-llava-next/
    - LLaVA-OneVision: https://llava-vl.github.io/blog/2024-08-05-llava-onevision/ (2024)
    - Original LLaVA: https://arxiv.org/abs/2304.08485

Authors: Haotian Liu et al. (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from nexus.core.base import NexusModule


class DynamicImageProcessor(NexusModule):
    """Dynamic image processor that handles variable resolutions and aspect ratios.

    Args:
        base_size: Base resolution for image processing
        max_patches: Maximum number of image patches
        patch_size: Size of each image patch
        hidden_dim: Hidden dimension for embeddings
    """

    def __init__(
        self,
        base_size: int = 336,
        max_patches: int = 5,
        patch_size: int = 14,
        hidden_dim: int = 1024,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.base_size = base_size
        self.max_patches = max_patches
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Image encoder (simplified ViT-like encoder)
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, (base_size // patch_size) ** 2, hidden_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Process images with dynamic resolution.

        Args:
            images: Input images [batch_size, 3, height, width]

        Returns:
            Image embeddings [batch_size, num_patches, hidden_dim]
        """
        B = images.shape[0]

        # Extract patches
        x = self.patch_embed(images)  # [B, hidden_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]

        # Add positional embeddings (with interpolation if needed)
        if x.shape[1] != self.pos_embed.shape[1]:
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2).unsqueeze(0),
                size=x.shape[1],
                mode='linear',
                align_corners=False
            ).squeeze(0).transpose(1, 2)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        return x


class AnyResolutionProjector(NexusModule):
    """Projects variable-resolution visual features to language model space.

    Handles any-resolution visual encoding by processing image patches
    at multiple scales and aspect ratios.

    Args:
        visual_dim: Dimension of visual features
        text_dim: Dimension of text/language model
        num_layers: Number of projection layers
    """

    def __init__(
        self,
        visual_dim: int = 1024,
        text_dim: int = 4096,
        num_layers: int = 2,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.visual_dim = visual_dim
        self.text_dim = text_dim

        # Multi-layer projector with GELU activations
        layers = []
        for i in range(num_layers):
            in_dim = visual_dim if i == 0 else text_dim
            layers.extend([
                nn.Linear(in_dim, text_dim),
                nn.GELU()
            ])

        # Remove last activation
        layers = layers[:-1]
        self.projector = nn.Sequential(*layers)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Project visual features to language space.

        Args:
            visual_features: Visual embeddings [batch_size, num_patches, visual_dim]

        Returns:
            Projected features [batch_size, num_patches, text_dim]
        """
        return self.projector(visual_features)


class LLaVANeXT(NexusModule):
    """LLaVA-NeXT: Advanced open-source multimodal LLM.

    Extends the original LLaVA architecture with:
    - Dynamic resolution processing (up to 4x higher resolution)
    - Improved visual-language alignment
    - Better spatial reasoning and OCR capabilities
    - Support for multiple images per input

    Args:
        visual_encoder_dim: Dimension of visual encoder output
        language_model_dim: Dimension of language model
        num_visual_tokens: Number of visual tokens per image
        max_images: Maximum number of images per input
        projector_layers: Number of projection layers
        use_video: Whether to support video input (OneVision)

    Example:
        >>> model = LLaVANeXT(
        ...     visual_encoder_dim=1024,
        ...     language_model_dim=4096,
        ...     num_visual_tokens=576,
        ... )
        >>> images = torch.randn(2, 3, 672, 672)  # Higher resolution
        >>> text_embeds = torch.randn(2, 100, 4096)
        >>> output = model(images, text_embeds)
    """

    def __init__(
        self,
        visual_encoder_dim: int = 1024,
        language_model_dim: int = 4096,
        num_visual_tokens: int = 576,
        max_images: int = 8,
        projector_layers: int = 2,
        use_video: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.visual_encoder_dim = visual_encoder_dim
        self.language_model_dim = language_model_dim
        self.num_visual_tokens = num_visual_tokens
        self.max_images = max_images
        self.use_video = use_video

        # Dynamic image processor
        self.image_processor = DynamicImageProcessor(
            base_size=336,
            hidden_dim=visual_encoder_dim
        )

        # Any-resolution projector
        self.projector = AnyResolutionProjector(
            visual_dim=visual_encoder_dim,
            text_dim=language_model_dim,
            num_layers=projector_layers
        )

        # Video temporal encoder (for OneVision variant)
        if use_video:
            self.temporal_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=language_model_dim,
                    nhead=8,
                    dim_feedforward=language_model_dim * 4,
                    batch_first=True
                ),
                num_layers=2
            )

        # Special tokens
        self.image_start_token = nn.Parameter(torch.randn(1, 1, language_model_dim))
        self.image_end_token = nn.Parameter(torch.randn(1, 1, language_model_dim))

    def encode_images(
        self,
        images: torch.Tensor,
        num_images_per_sample: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Encode images with dynamic resolution support.

        Args:
            images: Input images [total_images, 3, H, W]
            num_images_per_sample: Number of images per sample in batch

        Returns:
            Encoded image features [batch_size, num_visual_tokens, language_model_dim]
        """
        # Process images
        visual_features = self.image_processor(images)

        # Project to language space
        projected_features = self.projector(visual_features)

        # Reshape if multiple images per sample
        if num_images_per_sample is not None:
            batch_outputs = []
            offset = 0
            for num_imgs in num_images_per_sample:
                # Concatenate features for all images in this sample
                sample_features = projected_features[offset:offset + num_imgs]
                batch_outputs.append(sample_features.flatten(0, 1))
                offset += num_imgs
            projected_features = torch.stack(batch_outputs, dim=0)

        return projected_features

    def encode_video(self, video_frames: torch.Tensor) -> torch.Tensor:
        """Encode video frames with temporal modeling (OneVision).

        Args:
            video_frames: Video frames [batch_size, num_frames, 3, H, W]

        Returns:
            Encoded video features [batch_size, num_visual_tokens, language_model_dim]
        """
        if not self.use_video:
            raise ValueError("Video encoding not enabled. Set use_video=True.")

        B, T = video_frames.shape[:2]

        # Encode each frame
        frames = video_frames.reshape(B * T, *video_frames.shape[2:])
        frame_features = self.encode_images(frames)

        # Reshape and apply temporal encoder
        frame_features = frame_features.reshape(B, T, -1, self.language_model_dim)
        frame_features = frame_features.flatten(1, 2)  # [B, T*num_patches, dim]

        video_features = self.temporal_encoder(frame_features)
        return video_features

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        num_images_per_sample: Optional[List[int]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass combining visual and text inputs.

        Args:
            images: Input images [total_images, 3, H, W]
            text_embeds: Text embeddings [batch_size, seq_len, language_model_dim]
            video_frames: Video frames [batch_size, num_frames, 3, H, W]
            num_images_per_sample: Number of images per sample
            attention_mask: Attention mask for text

        Returns:
            Dictionary containing:
                - multimodal_embeds: Fused embeddings
                - visual_features: Encoded visual features
        """
        outputs = {}

        # Encode visual inputs
        if video_frames is not None:
            visual_features = self.encode_video(video_frames)
        elif images is not None:
            visual_features = self.encode_images(images, num_images_per_sample)
        else:
            raise ValueError("Either images or video_frames must be provided")

        outputs['visual_features'] = visual_features

        # Combine with text if provided
        if text_embeds is not None:
            B = text_embeds.shape[0]

            # Add special tokens
            image_start = self.image_start_token.expand(B, -1, -1)
            image_end = self.image_end_token.expand(B, -1, -1)

            # Concatenate: [image_start, visual_features, image_end, text]
            multimodal_embeds = torch.cat([
                image_start,
                visual_features,
                image_end,
                text_embeds
            ], dim=1)

            outputs['multimodal_embeds'] = multimodal_embeds

            # Extend attention mask if provided
            if attention_mask is not None:
                visual_mask = torch.ones(
                    B, visual_features.shape[1] + 2,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                extended_mask = torch.cat([visual_mask, attention_mask], dim=1)
                outputs['attention_mask'] = extended_mask
        else:
            outputs['multimodal_embeds'] = visual_features

        return outputs


class LLaVAOneVision(LLaVANeXT):
    """LLaVA-OneVision: Unified image and video understanding.

    Extends LLaVA-NeXT with native video understanding capabilities,
    unifying image and video processing in a single architecture.

    Reference:
        LLaVA-OneVision: https://llava-vl.github.io/blog/2024-08-05-llava-onevision/
    """

    def __init__(
        self,
        visual_encoder_dim: int = 1024,
        language_model_dim: int = 4096,
        num_visual_tokens: int = 576,
        max_images: int = 8,
        max_frames: int = 32,
        projector_layers: int = 2,
        config: Optional[Dict[str, Any]] = None
    ):
        # Always enable video for OneVision
        super().__init__(
            visual_encoder_dim=visual_encoder_dim,
            language_model_dim=language_model_dim,
            num_visual_tokens=num_visual_tokens,
            max_images=max_images,
            projector_layers=projector_layers,
            use_video=True,
            config=config
        )
        self.max_frames = max_frames


# Export
__all__ = ['LLaVANeXT', 'LLaVAOneVision', 'DynamicImageProcessor', 'AnyResolutionProjector']
