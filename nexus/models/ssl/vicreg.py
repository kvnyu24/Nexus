"""VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.

Reference: "VICReg: Variance-Invariance-Covariance Regularization for
Self-Supervised Learning" (Bardes et al., Meta AI, ICLR 2022)

VICReg is a non-contrastive self-supervised learning method that avoids
dimensional collapse through three simple regularization terms:
    1. Invariance: Embedding similarity between augmented views (MSE)
    2. Variance: Maintains variance of embeddings along each dimension (hinge loss)
    3. Covariance: Decorrelates different embedding dimensions

Unlike contrastive methods, VICReg doesn't require negative pairs, large batches,
or momentum encoders. It's simpler than other non-contrastive methods (BYOL, SimSiam)
while achieving competitive or better performance.

Architecture:
    - Encoder: Backbone network (e.g., ResNet, ViT)
    - Projector: MLP projector head (expander architecture)
    - VICRegModel: Complete SSL system with VICReg loss

Key properties:
    - No negative pairs required
    - No momentum encoder required
    - Simple and stable training
    - Explicit variance and covariance regularization
    - Works with smaller batch sizes than contrastive methods
    - Achieves strong performance on ImageNet and transfer tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from ...core.base import NexusModule


class VICRegProjector(NexusModule):
    """MLP projector head for VICReg.

    Uses an expander architecture: projects to a higher-dimensional space,
    applies non-linearity and normalization, then projects back.

    Args:
        config: Configuration dictionary with:
            - input_dim: Input dimension from encoder. Default: 2048.
            - hidden_dim: Hidden dimension. Default: 8192.
            - output_dim: Output embedding dimension. Default: 8192.
            - num_layers: Number of layers. Default: 3.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.input_dim = config.get("input_dim", 2048)
        self.hidden_dim = config.get("hidden_dim", 8192)
        self.output_dim = config.get("output_dim", 8192)
        self.num_layers = config.get("num_layers", 3)

        # Build MLP
        layers = []

        # First layer: expand dimension
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # Hidden layers
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # Last layer: no activation or normalization
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder features to embedding space.

        Args:
            x: Input features (B, D_in).

        Returns:
            Projected embeddings (B, D_out).
        """
        return self.projector(x)


class VICRegLoss(nn.Module):
    """VICReg loss function.

    Combines three terms:
        1. Invariance (sim_loss): MSE between embeddings of augmented views
        2. Variance (std_loss): Maintains std along each dimension above threshold
        3. Covariance (cov_loss): Decorrelates dimensions (off-diagonal covariance)

    Args:
        lambda_param: Weight for invariance loss. Default: 25.0.
        mu_param: Weight for variance loss. Default: 25.0.
        nu_param: Weight for covariance loss. Default: 1.0.
        variance_threshold: Target minimum std for each dimension. Default: 1.0.
        eps: Small constant for numerical stability. Default: 1e-4.
    """

    def __init__(
        self,
        lambda_param: float = 25.0,
        mu_param: float = 25.0,
        nu_param: float = 1.0,
        variance_threshold: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()

        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.variance_threshold = variance_threshold
        self.eps = eps

    def invariance_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """Invariance loss: MSE between embeddings.

        Args:
            z1: Embeddings from view 1 (B, D).
            z2: Embeddings from view 2 (B, D).

        Returns:
            Invariance loss (scalar).
        """
        return F.mse_loss(z1, z2)

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Variance loss: maintains variance above threshold.

        Uses hinge loss to penalize dimensions with low variance.

        Args:
            z: Embeddings (B, D).

        Returns:
            Variance loss (scalar).
        """
        # Compute std along batch dimension for each feature
        std = torch.sqrt(z.var(dim=0) + self.eps)

        # Hinge loss: penalize if std < threshold
        loss = F.relu(self.variance_threshold - std).mean()

        return loss

    def covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Covariance loss: decorrelates dimensions.

        Penalizes off-diagonal elements of the covariance matrix.

        Args:
            z: Embeddings (B, D).

        Returns:
            Covariance loss (scalar).
        """
        batch_size, dim = z.shape

        # Center the embeddings
        z = z - z.mean(dim=0, keepdim=True)

        # Compute covariance matrix: (D, D)
        cov = (z.T @ z) / (batch_size - 1)

        # Sum of squared off-diagonal elements
        # Diagonal elements should be large (high variance)
        # Off-diagonal should be small (uncorrelated)
        cov_diag = torch.diagonal(cov)
        off_diag = cov.pow(2).sum() - cov_diag.pow(2).sum()
        loss = off_diag / dim

        return loss

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute VICReg loss.

        Args:
            z1: Embeddings from view 1 (B, D).
            z2: Embeddings from view 2 (B, D).

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        # Invariance loss
        sim_loss = self.invariance_loss(z1, z2)

        # Variance loss (applied to both views)
        std_loss = (self.variance_loss(z1) + self.variance_loss(z2)) / 2

        # Covariance loss (applied to both views)
        cov_loss = (self.covariance_loss(z1) + self.covariance_loss(z2)) / 2

        # Total loss
        total_loss = (
            self.lambda_param * sim_loss
            + self.mu_param * std_loss
            + self.nu_param * cov_loss
        )

        loss_dict = {
            "total_loss": total_loss.item(),
            "sim_loss": sim_loss.item(),
            "std_loss": std_loss.item(),
            "cov_loss": cov_loss.item(),
        }

        return total_loss, loss_dict


class VICRegEncoder(NexusModule):
    """Simple encoder for VICReg.

    Can use any backbone (ResNet, ViT, etc.). This is a simple
    Vision Transformer encoder for demonstration.

    Args:
        config: Configuration dictionary with:
            - encoder_dim: Embedding dimension. Default: 768.
            - num_heads: Number of attention heads. Default: 12.
            - num_layers: Number of transformer layers. Default: 12.
            - patch_size: Size of image patches. Default: 16.
            - img_size: Input image size. Default: 224.
            - dropout: Dropout rate. Default: 0.0.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.encoder_dim = config.get("encoder_dim", 768)
        self.num_heads = config.get("num_heads", 12)
        self.num_layers = config.get("num_layers", 12)
        self.patch_size = config.get("patch_size", 16)
        self.img_size = config.get("img_size", 224)
        self.dropout = config.get("dropout", 0.0)

        self.num_patches = (self.img_size // self.patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, self.encoder_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.encoder_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Class token
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, self.encoder_dim)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_dim,
            nhead=self.num_heads,
            dim_feedforward=self.encoder_dim * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        self.norm = nn.LayerNorm(self.encoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to features.

        Args:
            x: Input images (B, C, H, W).

        Returns:
            Encoded features (B, D).
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # Return class token representation
        return x[:, 0]


class VICRegModel(NexusModule):
    """VICReg: Variance-Invariance-Covariance Regularization.

    Complete self-supervised learning system using VICReg loss.
    Trains by processing two augmented views of images and matching
    their embeddings while maintaining variance and decorrelation.

    Training procedure:
        1. Generate two augmented views of each image
        2. Encode both views through shared encoder
        3. Project to high-dimensional embedding space
        4. Compute VICReg loss (invariance + variance + covariance)
        5. Update encoder and projector via gradient descent

    Args:
        config: Configuration dictionary with:
            - encoder_dim: Encoder output dimension. Default: 768.
            - projector_hidden_dim: Projector hidden dim. Default: 8192.
            - projector_output_dim: Projector output dim. Default: 8192.
            - lambda_param: Invariance loss weight. Default: 25.0.
            - mu_param: Variance loss weight. Default: 25.0.
            - nu_param: Covariance loss weight. Default: 1.0.
            - variance_threshold: Minimum std threshold. Default: 1.0.
            - Additional encoder config options.

    Example:
        >>> config = {
        ...     "encoder_dim": 768,
        ...     "projector_output_dim": 8192,
        ...     "lambda_param": 25.0
        ... }
        >>> model = VICRegModel(config)
        >>> view1 = torch.randn(32, 3, 224, 224)
        >>> view2 = torch.randn(32, 3, 224, 224)  # Different augmentation
        >>> loss, metrics = model(view1, view2)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.encoder_dim = config.get("encoder_dim", 768)
        projector_hidden_dim = config.get("projector_hidden_dim", 8192)
        projector_output_dim = config.get("projector_output_dim", 8192)

        # Encoder
        self.encoder = VICRegEncoder(config)

        # Projector
        projector_config = {
            "input_dim": self.encoder_dim,
            "hidden_dim": projector_hidden_dim,
            "output_dim": projector_output_dim,
            "num_layers": config.get("projector_layers", 3),
        }
        self.projector = VICRegProjector(projector_config)

        # Loss function
        self.criterion = VICRegLoss(
            lambda_param=config.get("lambda_param", 25.0),
            mu_param=config.get("mu_param", 25.0),
            nu_param=config.get("nu_param", 1.0),
            variance_threshold=config.get("variance_threshold", 1.0),
        )

    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass for VICReg training.

        Args:
            view1: First augmented view (B, C, H, W).
            view2: Second augmented view (B, C, H, W).

        Returns:
            Tuple of (loss, metrics_dict).
        """
        # Encode both views
        h1 = self.encoder(view1)  # (B, D_enc)
        h2 = self.encoder(view2)  # (B, D_enc)

        # Project to embedding space
        z1 = self.projector(h1)  # (B, D_proj)
        z2 = self.projector(h2)  # (B, D_proj)

        # Compute VICReg loss
        loss, loss_dict = self.criterion(z1, z2)

        # Add embedding statistics
        metrics = {
            **loss_dict,
            "z1_mean": z1.mean().item(),
            "z1_std": z1.std().item(),
            "z2_mean": z2.mean().item(),
            "z2_std": z2.std().item(),
        }

        return loss, metrics

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to features (for evaluation/downstream tasks).

        Args:
            x: Input images (B, C, H, W).

        Returns:
            Encoded features (B, D).
        """
        with torch.no_grad():
            return self.encoder(x)
