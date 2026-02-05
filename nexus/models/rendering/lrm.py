"""
LRM: Large Reconstruction Model for Single Image to 3D

Implementation of LRM, which reconstructs high-quality 3D neural radiance fields
from a single image in just 5 seconds using a transformer-based architecture and
triplane representation.

Reference:
    Hong, Y., Zhang, K., Gu, J., et al. (2024).
    "LRM: Large Reconstruction Model for Single Image to 3D."
    ICLR 2024
    arXiv:2311.04400

Key Components:
    - ImageEncoder: DINO-based image feature extractor
    - TransformerDecoder: Triplane prediction from image features
    - TriplaneRenderer: NeRF rendering from triplane representation
    - LRM: Full single-image to 3D reconstruction model

Architecture Details:
    - Input: Single RGB image
    - Output: Triplane representation (3 planes of 32x32x40 features)
    - Transformer decoder with camera-aware attention
    - Fast rendering via triplane interpolation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin


class ImageEncoder(WeightInitMixin, NexusModule):
    """DINO-based image encoder for LRM.

    Args:
        config: Configuration dictionary with keys:
            img_size (int): Input image size. Default: 224.
            patch_size (int): Patch size. Default: 14.
            embed_dim (int): Embedding dimension. Default: 768.
            depth (int): Number of transformer layers. Default: 12.
            num_heads (int): Number of attention heads. Default: 12.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.img_size = config.get("img_size", 224)
        self.patch_size = config.get("patch_size", 14)
        self.embed_dim = config.get("embed_dim", 768)
        self.depth = config.get("depth", 12)
        self.num_heads = config.get("num_heads", 12)

        self.grid_size = self.img_size // self.patch_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.grid_size * self.grid_size, self.embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                dim_feedforward=self.embed_dim * 4,
                batch_first=True,
            )
            for _ in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.init_weights_vit()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to features.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Image features (B, N, embed_dim).
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x


class TriplaneDecoder(nn.Module):
    """Transformer decoder that predicts triplane representation.

    Args:
        embed_dim (int): Input embedding dimension. Default: 768.
        triplane_dim (int): Triplane feature dimension. Default: 40.
        triplane_resolution (int): Triplane spatial resolution. Default: 32.
        num_layers (int): Number of decoder layers. Default: 6.
        num_heads (int): Number of attention heads. Default: 8.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        triplane_dim: int = 40,
        triplane_resolution: int = 32,
        num_layers: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()

        self.triplane_dim = triplane_dim
        self.triplane_resolution = triplane_resolution

        # Learnable triplane queries (3 planes)
        num_triplane_tokens = 3 * triplane_resolution * triplane_resolution
        self.triplane_queries = nn.Parameter(
            torch.randn(1, num_triplane_tokens, embed_dim)
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Projection to triplane features
        self.triplane_proj = nn.Linear(embed_dim, triplane_dim)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Decode triplane from image features.

        Args:
            image_features: Encoded image features (B, N, embed_dim).

        Returns:
            Triplane features (B, 3, triplane_dim, H, W).
        """
        B = image_features.shape[0]

        # Expand queries for batch
        queries = self.triplane_queries.expand(B, -1, -1)

        # Decode with cross-attention to image features
        triplane_features = self.decoder(queries, image_features)

        # Project to triplane dimension
        triplane_features = self.triplane_proj(triplane_features)

        # Reshape to 3 planes
        # (B, 3*H*W, D) -> (B, 3, H, W, D)
        triplane_features = triplane_features.reshape(
            B, 3, self.triplane_resolution, self.triplane_resolution, self.triplane_dim
        )

        # Rearrange to (B, 3, D, H, W) for convolution operations
        triplane_features = triplane_features.permute(0, 1, 4, 2, 3)

        return triplane_features


class TriplaneRenderer(nn.Module):
    """Render NeRF from triplane representation.

    Args:
        triplane_dim (int): Triplane feature dimension. Default: 40.
        hidden_dim (int): MLP hidden dimension. Default: 128.
    """

    def __init__(
        self,
        triplane_dim: int = 40,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # MLP for density and color prediction
        self.density_net = nn.Sequential(
            nn.Linear(triplane_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.color_net = nn.Sequential(
            nn.Linear(triplane_dim * 3 + 3, hidden_dim),  # +3 for view direction
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def sample_triplane(
        self,
        triplane: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Sample features from triplane at given 3D coordinates.

        Args:
            triplane: Triplane features (B, 3, D, H, W).
            coords: 3D coordinates (B, N, 3) in [-1, 1].

        Returns:
            Sampled features (B, N, D*3).
        """
        B, N = coords.shape[:2]

        # Sample from each plane
        # XY plane (index 0): use x, y coordinates
        # XZ plane (index 1): use x, z coordinates
        # YZ plane (index 2): use y, z coordinates

        features = []

        # XY plane
        xy_coords = coords[:, :, [0, 1]].unsqueeze(1)  # (B, 1, N, 2)
        xy_features = F.grid_sample(
            triplane[:, 0],  # (B, D, H, W)
            xy_coords,
            align_corners=False,
            mode='bilinear',
        )
        features.append(xy_features.squeeze(2).transpose(1, 2))  # (B, N, D)

        # XZ plane
        xz_coords = coords[:, :, [0, 2]].unsqueeze(1)
        xz_features = F.grid_sample(
            triplane[:, 1],
            xz_coords,
            align_corners=False,
            mode='bilinear',
        )
        features.append(xz_features.squeeze(2).transpose(1, 2))

        # YZ plane
        yz_coords = coords[:, :, [1, 2]].unsqueeze(1)
        yz_features = F.grid_sample(
            triplane[:, 2],
            yz_coords,
            align_corners=False,
            mode='bilinear',
        )
        features.append(yz_features.squeeze(2).transpose(1, 2))

        # Concatenate features from all planes
        combined = torch.cat(features, dim=-1)  # (B, N, D*3)

        return combined

    def forward(
        self,
        triplane: torch.Tensor,
        coords: torch.Tensor,
        view_dirs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render density and color from triplane.

        Args:
            triplane: Triplane features (B, 3, D, H, W).
            coords: 3D coordinates (B, N, 3).
            view_dirs: View directions (B, N, 3).

        Returns:
            Tuple of (density, rgb).
        """
        # Sample triplane
        features = self.sample_triplane(triplane, coords)

        # Predict density
        density = self.density_net(features)

        # Predict color (conditioned on view direction)
        color_input = torch.cat([features, view_dirs], dim=-1)
        rgb = self.color_net(color_input)

        return density, rgb


class LRM(WeightInitMixin, NexusModule):
    """LRM: Large Reconstruction Model for Single Image to 3D.

    Fast single-image 3D reconstruction using triplane representation.

    Config:
        # Image encoder config
        img_size (int): Input image size. Default: 224.
        encoder_embed_dim (int): Encoder embedding dimension. Default: 768.
        encoder_depth (int): Encoder depth. Default: 12.
        encoder_num_heads (int): Encoder attention heads. Default: 12.

        # Triplane decoder config
        triplane_dim (int): Triplane feature dimension. Default: 40.
        triplane_resolution (int): Triplane spatial resolution. Default: 32.
        decoder_num_layers (int): Decoder layers. Default: 6.
        decoder_num_heads (int): Decoder attention heads. Default: 8.

        # Renderer config
        renderer_hidden_dim (int): Renderer MLP hidden dimension. Default: 128.

        # Rendering config
        num_samples (int): Samples per ray. Default: 64.

    Example:
        >>> config = {
        ...     "img_size": 224,
        ...     "encoder_embed_dim": 768,
        ...     "triplane_dim": 40,
        ... }
        >>> model = LRM(config)
        >>> image = torch.randn(1, 3, 224, 224)
        >>> output = model(image)
        >>> triplane = output["triplane"]
        >>> triplane.shape
        torch.Size([1, 3, 40, 32, 32])
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Image encoder
        encoder_config = {
            "img_size": config.get("img_size", 224),
            "patch_size": config.get("patch_size", 14),
            "embed_dim": config.get("encoder_embed_dim", 768),
            "depth": config.get("encoder_depth", 12),
            "num_heads": config.get("encoder_num_heads", 12),
        }
        self.encoder = ImageEncoder(encoder_config)

        # Triplane decoder
        self.decoder = TriplaneDecoder(
            embed_dim=config.get("encoder_embed_dim", 768),
            triplane_dim=config.get("triplane_dim", 40),
            triplane_resolution=config.get("triplane_resolution", 32),
            num_layers=config.get("decoder_num_layers", 6),
            num_heads=config.get("decoder_num_heads", 8),
        )

        # Renderer
        self.renderer = TriplaneRenderer(
            triplane_dim=config.get("triplane_dim", 40),
            hidden_dim=config.get("renderer_hidden_dim", 128),
        )

    def forward(
        self,
        images: torch.Tensor,
        rays_o: Optional[torch.Tensor] = None,
        rays_d: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct 3D from single image.

        Args:
            images: Input images (B, 3, H, W).
            rays_o: Optional ray origins for rendering (B, N, 3).
            rays_d: Optional ray directions for rendering (B, N, 3).

        Returns:
            Dictionary with triplane and optional rendered outputs.
        """
        # Encode image
        image_features = self.encoder(images)

        # Decode triplane
        triplane = self.decoder(image_features)

        output = {
            "triplane": triplane,
            "image_features": image_features,
        }

        # Optional rendering
        if rays_o is not None and rays_d is not None:
            # Sample points along rays (simplified)
            t_vals = torch.linspace(0.0, 1.0, 64, device=images.device)
            coords = rays_o.unsqueeze(2) + rays_d.unsqueeze(2) * t_vals.view(1, 1, -1, 1)

            # Normalize to [-1, 1]
            coords = coords * 2.0 - 1.0

            # Render
            B, N, S = coords.shape[:3]
            coords_flat = coords.reshape(B, N * S, 3)
            view_dirs = rays_d.unsqueeze(2).expand(-1, -1, S, -1).reshape(B, N * S, 3)

            density, rgb = self.renderer(triplane, coords_flat, view_dirs)

            output["density"] = density.reshape(B, N, S, 1)
            output["rgb"] = rgb.reshape(B, N, S, 3)

        return output


__all__ = [
    "LRM",
    "ImageEncoder",
    "TriplaneDecoder",
    "TriplaneRenderer",
]
