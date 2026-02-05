"""
MedSAM: Segment Anything in Medical Images

Implementation of MedSAM, a universal medical image segmentation model based on
SAM but fine-tuned on a large-scale medical imaging dataset covering diverse
anatomies, modalities, and organs.

Reference:
    Ma, J., He, Y., Li, F., et al. (2024).
    "Segment Anything in Medical Images."
    Nature Communications
    arXiv:2304.12306

Key Components:
    - MedSAM: SAM architecture fine-tuned for medical imaging
    - Handles diverse medical imaging modalities (CT, MRI, X-ray, ultrasound, etc.)
    - Trained on 1M+ medical image-mask pairs
    - Domain-specific prompt strategies for clinical use

Architecture Details:
    - Same architecture as SAM (ViT-based)
    - Fine-tuned on medical domain data
    - Supports 11 medical imaging modalities
    - Covers 30+ anatomical structures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin


class MedSAMEncoder(WeightInitMixin, NexusModule):
    """Medical image encoder adapted from SAM's ViT encoder.

    Args:
        config: Configuration dictionary with keys:
            img_size (int): Input image size. Default: 1024.
            patch_size (int): Patch size. Default: 16.
            in_channels (int): Input channels (1 for grayscale, 3 for RGB). Default: 3.
            embed_dim (int): Embedding dimension. Default: 768.
            depth (int): Number of transformer layers. Default: 12.
            num_heads (int): Number of attention heads. Default: 12.
            out_channels (int): Output channels. Default: 256.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.img_size = config.get("img_size", 1024)
        self.patch_size = config.get("patch_size", 16)
        self.in_channels = config.get("in_channels", 3)
        self.embed_dim = config.get("embed_dim", 768)
        self.depth = config.get("depth", 12)
        self.num_heads = config.get("num_heads", 12)
        self.out_channels = config.get("out_channels", 256)

        self.grid_size = self.img_size // self.patch_size

        # Patch embedding - adapted for medical images (may have 1 channel)
        self.patch_embed = nn.Conv2d(
            self.in_channels, self.embed_dim,
            kernel_size=self.patch_size, stride=self.patch_size,
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.grid_size, self.grid_size, self.embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            MedSAMBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=4.0,
            )
            for _ in range(self.depth)
        ])

        # Neck for dimension reduction
        self.neck = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out_channels, kernel_size=1, bias=False),
            nn.LayerNorm([self.out_channels, self.grid_size, self.grid_size]),
        )

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.init_weights_vit()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode medical images.

        Args:
            x: Input images (B, C, H, W).

        Returns:
            Image embeddings (B, out_channels, H', W').
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.permute(0, 2, 3, 1)  # (B, H', W', embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Neck
        x = x.permute(0, 3, 1, 2)  # (B, embed_dim, H', W')
        x = self.neck(x)

        return x


class MedSAMBlock(NexusModule):
    """Transformer block for MedSAM encoder.

    Args:
        dim (int): Feature dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP expansion ratio. Default: 4.0.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (B, H, W, C).

        Returns:
            Output (B, H, W, C).
        """
        # Flatten spatial dims
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


class MedSAMDecoder(NexusModule):
    """Lightweight decoder for medical image segmentation.

    Args:
        decoder_dim (int): Decoder dimension. Default: 256.
        num_mask_tokens (int): Number of mask output tokens. Default: 1.
    """

    def __init__(
        self,
        decoder_dim: int = 256,
        num_mask_tokens: int = 1,
    ):
        super().__init__()

        self.decoder_dim = decoder_dim
        self.num_mask_tokens = num_mask_tokens

        # Upsampling path
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(decoder_dim, decoder_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(decoder_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(decoder_dim // 2, decoder_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(decoder_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(decoder_dim // 4, decoder_dim // 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(decoder_dim // 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(decoder_dim // 8, decoder_dim // 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        # Mask prediction
        self.mask_head = nn.Conv2d(decoder_dim // 16, num_mask_tokens, kernel_size=1)

    def forward(
        self,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Predict segmentation masks.

        Args:
            image_embeddings: Encoder output (B, decoder_dim, H', W').

        Returns:
            Mask predictions (B, num_mask_tokens, H, W).
        """
        x = self.upsample(image_embeddings)
        masks = self.mask_head(x)
        return masks


class MedSAM(WeightInitMixin, NexusModule):
    """MedSAM: Universal Medical Image Segmentation Model.

    Adapts SAM for medical imaging with domain-specific fine-tuning.
    Supports diverse modalities and anatomies.

    Config:
        img_size (int): Input image size. Default: 1024.
        patch_size (int): Patch size. Default: 16.
        in_channels (int): Input channels (1 for grayscale, 3 for RGB). Default: 3.
        encoder_embed_dim (int): Encoder embedding dimension. Default: 768.
        encoder_depth (int): Encoder depth. Default: 12.
        encoder_num_heads (int): Encoder attention heads. Default: 12.
        decoder_dim (int): Decoder dimension. Default: 256.
        num_mask_tokens (int): Number of output masks. Default: 1.
        modality (str): Medical imaging modality for preprocessing. Default: "ct".

    Example:
        >>> # CT scan segmentation
        >>> config = {"img_size": 1024, "in_channels": 1, "modality": "ct"}
        >>> model = MedSAM(config)
        >>> images = torch.randn(1, 1, 1024, 1024)  # Grayscale CT
        >>> boxes = torch.tensor([[[100, 100, 500, 500]]])  # Bounding box prompt
        >>> output = model(images, boxes=boxes)
        >>> output["masks"].shape
        torch.Size([1, 1, 1024, 1024])
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.img_size = config.get("img_size", 1024)
        self.modality = config.get("modality", "ct")

        # Medical image encoder
        encoder_config = {
            "img_size": self.img_size,
            "patch_size": config.get("patch_size", 16),
            "in_channels": config.get("in_channels", 3),
            "embed_dim": config.get("encoder_embed_dim", 768),
            "depth": config.get("encoder_depth", 12),
            "num_heads": config.get("encoder_num_heads", 12),
            "out_channels": config.get("decoder_dim", 256),
        }
        self.encoder = MedSAMEncoder(encoder_config)

        # Decoder
        self.decoder = MedSAMDecoder(
            decoder_dim=config.get("decoder_dim", 256),
            num_mask_tokens=config.get("num_mask_tokens", 1),
        )

        # Modality-specific preprocessing parameters (learnable)
        self.register_buffer(
            "window_center",
            torch.tensor(config.get("window_center", 40.0)),
        )
        self.register_buffer(
            "window_width",
            torch.tensor(config.get("window_width", 400.0)),
        )

    def preprocess_medical_image(self, x: torch.Tensor) -> torch.Tensor:
        """Apply modality-specific preprocessing.

        Args:
            x: Raw medical images (B, C, H, W).

        Returns:
            Preprocessed images (B, C, H, W).
        """
        if self.modality in ["ct", "mri"]:
            # Windowing for CT/MRI
            lower = self.window_center - self.window_width / 2
            upper = self.window_center + self.window_width / 2
            x = torch.clamp(x, lower, upper)
            x = (x - lower) / (upper - lower)
        else:
            # Standard normalization for other modalities
            x = (x - x.mean()) / (x.std() + 1e-6)

        return x

    def encode_box_prompt(self, boxes: torch.Tensor) -> torch.Tensor:
        """Encode box prompts for medical ROI specification.

        Args:
            boxes: Box coordinates (B, num_boxes, 4) as [x1, y1, x2, y2].

        Returns:
            Box embeddings.
        """
        # Simplified - in practice, use full prompt encoder
        return boxes

    def forward(
        self,
        images: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: Input medical images (B, C, H, W).
            boxes: Box prompts (B, num_boxes, 4) - commonly used in medical imaging.
            points: Point prompts (coords, labels).
            masks: Mask prompts (B, 1, H, W).

        Returns:
            Dictionary with mask predictions and features.
        """
        # Preprocess medical images
        images = self.preprocess_medical_image(images)

        # Encode image
        image_embeddings = self.encoder(images)

        # Decode masks (simplified - in practice, incorporate prompts)
        masks_pred = self.decoder(image_embeddings)

        return {
            "masks": masks_pred,
            "image_embeddings": image_embeddings,
            "boxes": boxes if boxes is not None else None,
        }


__all__ = [
    "MedSAM",
    "MedSAMEncoder",
    "MedSAMDecoder",
    "MedSAMBlock",
]
