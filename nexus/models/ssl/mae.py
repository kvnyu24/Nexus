"""MAE: Masked Autoencoder for Self-Supervised Learning.

Reference: "Masked Autoencoders Are Scalable Vision Learners"
(He et al., 2022)

MAE learns visual representations by masking a large portion of image
patches and training an autoencoder to reconstruct the missing pixels.
The key insight is the asymmetric encoder-decoder design: a large
encoder processes only visible patches, while a lightweight decoder
reconstructs the full image from encoded visible patches plus mask tokens.

Architecture:
    - MAEEncoder: ViT encoder that processes only visible (unmasked) patches
    - MAEDecoder: Lightweight transformer decoder that reconstructs pixel values
    - MAE: Full model combining encoder and decoder

Key properties:
    - Masks a high ratio (75%) of patches for efficiency
    - Encoder processes only visible patches (computational savings)
    - Lightweight decoder for reconstruction
    - Reconstruction target is normalized pixel values
    - Learns strong representations without contrastive pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple
from ...core.base import NexusModule


class MAEEncoder(NexusModule):
    """Vision Transformer encoder for MAE.

    Processes only visible (unmasked) patches, making it computationally
    efficient even with large ViT architectures. The masking is performed
    by simply removing masked tokens before the transformer.

    Args:
        config: Configuration dictionary with:
            - img_size: Input image size. Default: 224.
            - patch_size: Patch size. Default: 16.
            - in_channels: Input channels. Default: 3.
            - encoder_dim: Encoder embedding dimension. Default: 768.
            - encoder_heads: Number of attention heads. Default: 12.
            - encoder_layers: Number of transformer layers. Default: 12.
            - mlp_ratio: MLP hidden dimension ratio. Default: 4.0.
            - dropout: Dropout rate. Default: 0.0.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.img_size = config.get("img_size", 224)
        self.patch_size = config.get("patch_size", 16)
        self.in_channels = config.get("in_channels", 3)
        self.encoder_dim = config.get("encoder_dim", 768)
        self.num_heads = config.get("encoder_heads", 12)
        self.num_layers = config.get("encoder_layers", 12)
        self.mlp_ratio = config.get("mlp_ratio", 4.0)
        self.dropout = config.get("dropout", 0.0)

        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.patch_dim = self.in_channels * self.patch_size * self.patch_size

        # Patch embedding via linear projection
        self.patch_embed = nn.Conv2d(
            self.in_channels,
            self.encoder_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embedding (includes CLS token position)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.encoder_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_dim,
            nhead=self.num_heads,
            dim_feedforward=int(self.encoder_dim * self.mlp_ratio),
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        self.norm = nn.LayerNorm(self.encoder_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode visible patches.

        Args:
            x: Input images (B, C, H, W).
            mask: Boolean mask (B, N). True = keep (visible), False = mask out.

        Returns:
            Tuple of (encoded_tokens, visible_indices).
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # Add positional embedding (excluding CLS position)
        x = x + self.pos_embed[:, 1:, :]

        if mask is not None:
            # Keep only visible patches
            visible_list = []
            idx_list = []
            for i in range(batch_size):
                vis_idx = mask[i].nonzero(as_tuple=False).squeeze(-1)
                visible_list.append(x[i, vis_idx])
                idx_list.append(vis_idx)

            # Pad to same length
            max_vis = max(v.shape[0] for v in visible_list)
            padded = torch.zeros(
                batch_size, max_vis, self.encoder_dim, device=x.device
            )
            for i, v in enumerate(visible_list):
                padded[i, : v.shape[0]] = v
            x = padded
            indices = torch.stack(
                [
                    F.pad(idx, (0, max_vis - idx.shape[0]), value=0)
                    for idx in idx_list
                ]
            )
        else:
            indices = (
                torch.arange(self.num_patches, device=x.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        return x, indices


class MAEDecoder(NexusModule):
    """Lightweight decoder for MAE pixel reconstruction.

    Receives encoded visible patches and mask tokens, and reconstructs
    the pixel values for masked patches. The decoder is intentionally
    lightweight (smaller dim, fewer layers) as it only needs to
    reconstruct local pixel patterns.

    Args:
        config: Configuration dictionary with:
            - encoder_dim: Encoder embedding dimension. Default: 768.
            - decoder_dim: Decoder embedding dimension. Default: 512.
            - decoder_heads: Number of attention heads. Default: 16.
            - decoder_layers: Number of transformer layers. Default: 8.
            - patch_size: Patch size for computing output dim. Default: 16.
            - in_channels: Number of input channels. Default: 3.
            - img_size: Image size. Default: 224.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.encoder_dim = config.get("encoder_dim", 768)
        self.decoder_dim = config.get("decoder_dim", 512)
        self.num_heads = config.get("decoder_heads", 16)
        self.num_layers = config.get("decoder_layers", 8)
        self.patch_size = config.get("patch_size", 16)
        self.in_channels = config.get("in_channels", 3)
        self.img_size = config.get("img_size", 224)

        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.patch_dim = self.in_channels * self.patch_size * self.patch_size

        # Project from encoder dim to decoder dim
        self.embed_proj = nn.Linear(self.encoder_dim, self.decoder_dim)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Decoder positional embedding (includes CLS position)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.decoder_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.decoder_dim,
            nhead=self.num_heads,
            dim_feedforward=self.decoder_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=self.num_layers
        )

        self.norm = nn.LayerNorm(self.decoder_dim)

        # Prediction head: project to pixel values
        self.pred_head = nn.Linear(self.decoder_dim, self.patch_dim)

    def forward(
        self,
        encoded: torch.Tensor,
        visible_indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode and reconstruct masked patches.

        Args:
            encoded: Encoded visible tokens from encoder (B, N_vis+1, D_enc).
                Includes CLS token at position 0.
            visible_indices: Indices of visible patches (B, N_vis).
            mask: Boolean mask (B, N). True = visible, False = masked.

        Returns:
            Reconstructed patches for all positions (B, N, patch_dim).
        """
        batch_size = encoded.shape[0]

        # Project encoder output to decoder dimension
        x = self.embed_proj(encoded)

        # Separate CLS and patch tokens
        cls_token = x[:, :1, :]
        visible_tokens = x[:, 1:, :]

        # Create full sequence with mask tokens
        full_tokens = self.mask_token.expand(
            batch_size, self.num_patches, -1
        ).clone()

        # Place visible tokens at their positions
        for i in range(batch_size):
            num_vis = mask[i].sum().item()
            vis_idx = visible_indices[i, :int(num_vis)]
            full_tokens[i, vis_idx] = visible_tokens[i, :int(num_vis)]

        # Add positional embeddings
        full_tokens = full_tokens + self.pos_embed[:, 1:, :]

        # Prepend CLS token
        cls_token = cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls_token, full_tokens], dim=1)

        # Decoder transformer
        x = self.transformer(x)
        x = self.norm(x)

        # Remove CLS token and project to pixel space
        x = x[:, 1:, :]
        reconstructed = self.pred_head(x)

        return reconstructed


class MAE(NexusModule):
    """Masked Autoencoder (MAE) for self-supervised visual learning.

    Complete MAE model with asymmetric encoder-decoder architecture.
    During training, randomly masks patches and reconstructs pixel
    values for the masked patches. During inference, the encoder
    alone is used as a feature extractor.

    Args:
        config: Configuration dictionary with:
            - img_size: Input image size. Default: 224.
            - patch_size: Patch size. Default: 16.
            - in_channels: Input channels. Default: 3.
            - encoder_dim: Encoder dim. Default: 768.
            - decoder_dim: Decoder dim. Default: 512.
            - mask_ratio: Fraction of patches to mask. Default: 0.75.
            - norm_pix_loss: Normalize pixel targets. Default: True.
            See MAEEncoder and MAEDecoder for additional config options.

    Example:
        >>> config = {"img_size": 224, "patch_size": 16, "mask_ratio": 0.75}
        >>> model = MAE(config)
        >>> images = torch.randn(8, 3, 224, 224)
        >>> loss, reconstructed, mask = model(images)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.mask_ratio = config.get("mask_ratio", 0.75)
        self.norm_pix_loss = config.get("norm_pix_loss", True)
        self.patch_size = config.get("patch_size", 16)
        self.img_size = config.get("img_size", 224)
        self.in_channels = config.get("in_channels", 3)

        self.num_patches = (self.img_size // self.patch_size) ** 2

        # Encoder and decoder
        self.encoder = MAEEncoder(config)
        self.decoder = MAEDecoder(config)

    def _patchify(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to patch sequences.

        Args:
            images: (B, C, H, W)

        Returns:
            Patches: (B, N, patch_dim)
        """
        p = self.patch_size
        h = w = self.img_size // p

        x = images.reshape(
            images.shape[0], self.in_channels, h, p, w, p
        )
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, h, w, C, p, p)
        x = x.reshape(images.shape[0], h * w, -1)  # (B, N, C*p*p)

        return x

    def _generate_mask(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Generate random mask for patches.

        Args:
            batch_size: Batch size.
            device: Device.

        Returns:
            Boolean mask (B, N). True = visible, False = masked.
        """
        num_visible = int(self.num_patches * (1 - self.mask_ratio))

        mask = torch.zeros(
            batch_size, self.num_patches, dtype=torch.bool, device=device
        )

        for i in range(batch_size):
            perm = torch.randperm(self.num_patches, device=device)
            mask[i, perm[:num_visible]] = True

        return mask

    def forward(
        self,
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for MAE training.

        Args:
            images: Input images (B, C, H, W).
            mask: Optional pre-computed mask (B, N). If None, random mask
                is generated.

        Returns:
            Tuple of (loss, reconstructed_patches, mask).
        """
        batch_size = images.shape[0]
        device = images.device

        # Generate mask if not provided
        if mask is None:
            mask = self._generate_mask(batch_size, device)

        # Encode visible patches
        encoded, visible_indices = self.encoder(images, mask)

        # Decode and reconstruct
        reconstructed = self.decoder(encoded, visible_indices, mask)

        # Compute reconstruction loss (only on masked patches)
        target = self._patchify(images)

        if self.norm_pix_loss:
            # Normalize target by patch statistics
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        # Loss only on masked patches
        loss_mask = ~mask  # True where masked
        loss = (reconstructed - target) ** 2
        loss = loss.mean(dim=-1)  # Per-patch loss

        # Average over masked patches only
        loss = (loss * loss_mask.float()).sum() / loss_mask.float().sum()

        return loss, reconstructed, mask

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images without masking (for downstream tasks).

        Args:
            images: Input images (B, C, H, W).

        Returns:
            Encoded representations (B, N+1, D). Includes CLS token.
        """
        encoded, _ = self.encoder(images, mask=None)
        return encoded
