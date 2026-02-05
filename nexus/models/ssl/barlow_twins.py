"""Barlow Twins: Self-Supervised Learning via Redundancy Reduction.

Reference: "Barlow Twins: Self-Supervised Learning via Redundancy
Reduction" (Zbontar et al., 2021)

Barlow Twins learns representations by making the cross-correlation
matrix between embeddings of two augmented views approach the identity
matrix. This naturally avoids representational collapse by decorrelating
embedding dimensions, inspired by the neuroscience principle of
redundancy reduction proposed by Barlow (1961).

Architecture:
    - ProjectionHead: MLP projector mapping backbone features to
      the embedding space where the cross-correlation is computed
    - BarlowTwinsModel: Complete model with backbone encoder and
      projection head

Key properties:
    - No negative pairs, no momentum encoder, no asymmetric architecture
    - Simple and elegant: push cross-correlation toward identity
    - Redundancy reduction prevents collapse
    - Works with any backbone encoder
    - Lambda controls trade-off between invariance and decorrelation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from ...core.base import NexusModule


class ProjectionHead(NexusModule):
    """MLP projection head for Barlow Twins.

    Projects backbone features into the space where the cross-correlation
    objective is applied. Following the original paper, this is a 3-layer
    MLP with batch normalization.

    Args:
        config: Configuration dictionary with:
            - input_dim: Input feature dimension from backbone.
            - proj_dim: Output projection dimension. Default: 8192.
            - hidden_dim: Hidden layer dimension. Default: 8192.
            - num_layers: Number of MLP layers. Default: 3.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        input_dim = config["input_dim"]
        proj_dim = config.get("proj_dim", 8192)
        hidden_dim = config.get("hidden_dim", 8192)
        num_layers = config.get("num_layers", 3)

        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            out_dim = proj_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim, bias=False))

            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))
            else:
                # Last layer: BN without ReLU
                layers.append(nn.BatchNorm1d(out_dim, affine=False))

            in_dim = out_dim

        self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features.

        Args:
            x: Input features (B, D_in).

        Returns:
            Projected features (B, D_proj).
        """
        return self.projector(x)


class BarlowTwinsModel(NexusModule):
    """Barlow Twins self-supervised learning model.

    Wraps an encoder backbone with a projection head and implements
    the Barlow Twins cross-correlation loss. The model takes two
    augmented views of the same image and learns representations by
    pushing their cross-correlation matrix toward the identity.

    Loss = sum_i (1 - C_ii)^2 + lambda * sum_{i!=j} C_ij^2

    where C is the cross-correlation matrix computed over the batch.

    Args:
        config: Configuration dictionary with:
            - encoder: Pre-built encoder (nn.Module) producing features.
            - encoder_dim: Output dimension of the encoder. Required.
            - proj_dim: Projection dimension. Default: 8192.
            - lambd: Lambda for redundancy reduction term. Default: 0.005.

    Example:
        >>> encoder = torchvision.models.resnet50(pretrained=False)
        >>> encoder.fc = nn.Identity()
        >>> config = {"encoder": encoder, "encoder_dim": 2048, "proj_dim": 8192}
        >>> model = BarlowTwinsModel(config)
        >>> loss, metrics = model(view1, view2)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.lambd = config.get("lambd", 0.005)
        encoder_dim = config["encoder_dim"]
        proj_dim = config.get("proj_dim", 8192)

        # Encoder backbone
        if "encoder" in config:
            self.encoder = config["encoder"]
        else:
            # Default simple encoder for testing
            self.encoder = nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim),
                nn.ReLU(),
                nn.Linear(encoder_dim, encoder_dim),
            )

        # Projection head
        proj_config = {
            "input_dim": encoder_dim,
            "proj_dim": proj_dim,
            "hidden_dim": proj_dim,
        }
        self.projector = ProjectionHead(proj_config)

    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass computing Barlow Twins loss.

        Args:
            view1: First augmented view (B, C, H, W) or (B, D).
            view2: Second augmented view (B, C, H, W) or (B, D).

        Returns:
            Tuple of (loss, metrics_dict).
        """
        # Encode both views
        h1 = self.encoder(view1)
        h2 = self.encoder(view2)

        # Flatten if needed
        if h1.dim() > 2:
            h1 = h1.flatten(1)
            h2 = h2.flatten(1)

        # Project
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        # Compute cross-correlation matrix
        batch_size = z1.shape[0]

        # Normalize along the batch dimension
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-4)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-4)

        # Cross-correlation: (D, D) matrix
        cross_corr = (z1_norm.T @ z2_norm) / batch_size

        # Loss computation
        dim = cross_corr.shape[0]
        identity = torch.eye(dim, device=cross_corr.device)

        # On-diagonal: invariance term - (1 - C_ii)^2
        on_diag = ((identity - cross_corr) * identity).pow(2).sum()

        # Off-diagonal: redundancy reduction - lambda * C_ij^2
        off_diag = (cross_corr * (1 - identity)).pow(2).sum()

        loss = on_diag + self.lambd * off_diag

        metrics = {
            "loss": loss.item(),
            "on_diag_loss": on_diag.item(),
            "off_diag_loss": off_diag.item(),
            "cross_corr_diag_mean": cross_corr.diag().mean().item(),
        }

        return loss, metrics
