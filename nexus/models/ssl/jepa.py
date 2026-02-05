"""JEPA: Joint-Embedding Predictive Architecture.

Reference: "Self-Supervised Learning from Images with a Joint-Embedding
Predictive Architecture" (Assran et al., 2023)

I-JEPA (Image-based JEPA) learns representations by predicting the
representations of masked image patches from visible context patches
in a shared embedding space. Unlike pixel-reconstruction methods (MAE),
JEPA predicts in representation space, which encourages learning of
higher-level semantic features.

Architecture:
    - ContextEncoder: Encodes visible (unmasked) image patches
    - TargetEncoder: EMA copy of context encoder, encodes target patches
    - Predictor: Predicts target representations from context representations

Key properties:
    - Predicts in representation space, not pixel space
    - Target encoder is an exponential moving average of context encoder
    - Multi-block masking strategy for spatially coherent targets
    - No data augmentation or negative pairs required
    - Learns semantic features without reconstruction artifacts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Dict, Any, Optional, Tuple, List
from ...core.base import NexusModule


class ContextEncoder(NexusModule):
    """Vision Transformer encoder for visible context patches.

    Processes only the unmasked (visible) patches of an image, producing
    contextualized representations that capture the visible scene content.

    Args:
        config: Configuration dictionary with:
            - encoder_dim: Embedding dimension. Default: 768.
            - num_heads: Number of attention heads. Default: 12.
            - num_layers: Number of transformer layers. Default: 12.
            - mlp_ratio: MLP hidden dim ratio. Default: 4.0.
            - patch_size: Size of image patches. Default: 16.
            - img_size: Input image size. Default: 224.
            - dropout: Dropout rate. Default: 0.0.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.encoder_dim = config.get("encoder_dim", 768)
        self.num_heads = config.get("num_heads", 12)
        self.num_layers = config.get("num_layers", 12)
        self.mlp_ratio = config.get("mlp_ratio", 4.0)
        self.patch_size = config.get("patch_size", 16)
        self.img_size = config.get("img_size", 224)
        self.dropout = config.get("dropout", 0.0)

        self.num_patches = (self.img_size // self.patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, self.encoder_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.encoder_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
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
    ) -> torch.Tensor:
        """Encode visible image patches.

        Args:
            x: Input images (B, C, H, W).
            mask: Boolean mask indicating visible patches (B, N).
                True = visible, False = masked.

        Returns:
            Encoded representations of visible patches (B, N_vis, D).
        """
        # Patch embedding: (B, C, H, W) -> (B, D, H/P, W/P) -> (B, N, D)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply mask: keep only visible patches
        if mask is not None:
            # mask: (B, N) - True for visible patches
            batch_size = x.shape[0]
            visible_tokens = []
            for i in range(batch_size):
                visible_tokens.append(x[i][mask[i]])

            # Pad to same length and stack
            max_vis = max(t.shape[0] for t in visible_tokens)
            padded = torch.zeros(
                batch_size, max_vis, self.encoder_dim, device=x.device
            )
            for i, t in enumerate(visible_tokens):
                padded[i, : t.shape[0]] = t
            x = padded

        # Transformer encoding
        x = self.transformer(x)
        x = self.norm(x)

        return x


class TargetEncoder(NexusModule):
    """EMA-updated target encoder for I-JEPA.

    This encoder has the same architecture as the ContextEncoder but
    its weights are updated via exponential moving average (EMA) of
    the context encoder weights. This provides stable target
    representations for the predictor to match.

    Args:
        config: Same configuration as ContextEncoder.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Build the same architecture as ContextEncoder
        self.encoder = ContextEncoder(config)

        # Freeze all parameters - updated only via EMA
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_from_context(
        self, context_encoder: ContextEncoder, momentum: float = 0.996
    ) -> None:
        """Update target encoder weights via EMA from context encoder.

        Args:
            context_encoder: The context encoder to copy from.
            momentum: EMA momentum factor. Higher = slower update.
                Default: 0.996.
        """
        for target_param, context_param in zip(
            self.encoder.parameters(), context_encoder.parameters()
        ):
            target_param.data.mul_(momentum).add_(
                context_param.data, alpha=1.0 - momentum
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode target patches (no gradient).

        Args:
            x: Input images (B, C, H, W).
            mask: Boolean mask for target patches (B, N).

        Returns:
            Encoded representations of target patches.
        """
        return self.encoder(x, mask)


class Predictor(NexusModule):
    """Lightweight predictor for I-JEPA.

    Given context representations and positional information about
    target patches, predicts the target encoder representations.
    The predictor is intentionally kept smaller than the encoders
    to prevent shortcut solutions.

    Args:
        config: Configuration dictionary with:
            - encoder_dim: Context encoder output dim. Default: 768.
            - predictor_dim: Predictor hidden dim. Default: 384.
            - predictor_heads: Number of attention heads. Default: 12.
            - predictor_layers: Number of transformer layers. Default: 6.
            - num_patches: Total number of patches. Required.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.encoder_dim = config.get("encoder_dim", 768)
        self.predictor_dim = config.get("predictor_dim", 384)
        self.num_heads = config.get("predictor_heads", 12)
        self.num_layers = config.get("predictor_layers", 6)

        num_patches = config.get(
            "num_patches",
            (config.get("img_size", 224) // config.get("patch_size", 16)) ** 2,
        )

        # Project from encoder dim to predictor dim
        self.input_proj = nn.Linear(self.encoder_dim, self.predictor_dim)

        # Positional embedding for target locations
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.predictor_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Mask token for target positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Transformer
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=self.predictor_dim,
            nhead=self.num_heads,
            dim_feedforward=self.predictor_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            predictor_layer, num_layers=self.num_layers
        )

        self.norm = nn.LayerNorm(self.predictor_dim)

        # Project back to encoder dim
        self.output_proj = nn.Linear(self.predictor_dim, self.encoder_dim)

    def forward(
        self,
        context_encodings: torch.Tensor,
        context_positions: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target representations from context.

        Args:
            context_encodings: Encoded context patches (B, N_ctx, D_enc).
            context_positions: Indices of context patches (B, N_ctx).
            target_positions: Indices of target patches (B, N_tgt).

        Returns:
            Predicted target representations (B, N_tgt, D_enc).
        """
        batch_size = context_encodings.shape[0]
        num_targets = target_positions.shape[1]

        # Project context to predictor dimension
        context = self.input_proj(context_encodings)

        # Add positional embeddings for context
        ctx_pos = torch.gather(
            self.pos_embed.expand(batch_size, -1, -1),
            1,
            context_positions.unsqueeze(-1).expand(-1, -1, self.predictor_dim),
        )
        context = context + ctx_pos

        # Create mask tokens with target positional embeddings
        mask_tokens = self.mask_token.expand(batch_size, num_targets, -1)
        tgt_pos = torch.gather(
            self.pos_embed.expand(batch_size, -1, -1),
            1,
            target_positions.unsqueeze(-1).expand(-1, -1, self.predictor_dim),
        )
        mask_tokens = mask_tokens + tgt_pos

        # Concatenate context and mask tokens
        x = torch.cat([context, mask_tokens], dim=1)

        # Transformer prediction
        x = self.transformer(x)
        x = self.norm(x)

        # Extract only the target predictions
        target_preds = x[:, -num_targets:]

        # Project back to encoder dim
        target_preds = self.output_proj(target_preds)

        return target_preds


class JEPAModel(NexusModule):
    """Full I-JEPA (Image Joint-Embedding Predictive Architecture) model.

    Combines the context encoder, target encoder, and predictor into a
    complete self-supervised learning system. The model learns by predicting
    target patch representations from context patch representations.

    Training procedure:
        1. Split image patches into context (visible) and target (masked) sets
        2. Context encoder processes visible patches
        3. Target encoder (EMA) processes target patches (no gradient)
        4. Predictor predicts target representations from context
        5. Loss = MSE between predicted and actual target representations
        6. Update context encoder and predictor via gradient descent
        7. Update target encoder via EMA

    Args:
        config: Configuration dictionary with:
            - encoder_dim: Embedding dimension. Default: 768.
            - predictor_dim: Predictor hidden dimension. Default: 384.
            - num_targets: Number of target blocks. Default: 4.
            - mask_ratio: Fraction of patches to mask. Default: 0.75.
            - ema_momentum: EMA momentum for target encoder. Default: 0.996.
            - Additional ContextEncoder and Predictor config options.

    Example:
        >>> config = {"encoder_dim": 768, "predictor_dim": 384, "mask_ratio": 0.75}
        >>> model = JEPAModel(config)
        >>> images = torch.randn(8, 3, 224, 224)
        >>> loss, metrics = model(images)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.encoder_dim = config.get("encoder_dim", 768)
        self.mask_ratio = config.get("mask_ratio", 0.75)
        self.num_targets = config.get("num_targets", 4)
        self.ema_momentum = config.get("ema_momentum", 0.996)
        self.patch_size = config.get("patch_size", 16)
        self.img_size = config.get("img_size", 224)

        self.num_patches = (self.img_size // self.patch_size) ** 2

        # Context encoder (trainable)
        self.context_encoder = ContextEncoder(config)

        # Target encoder (EMA of context encoder)
        self.target_encoder = TargetEncoder(config)

        # Initialize target encoder from context encoder
        self.target_encoder.update_from_context(
            self.context_encoder, momentum=0.0
        )

        # Predictor
        predictor_config = dict(config)
        predictor_config["num_patches"] = self.num_patches
        self.predictor = Predictor(predictor_config)

    def _generate_masks(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate context and target masks.

        Args:
            batch_size: Number of samples in the batch.
            device: Device to create tensors on.

        Returns:
            Tuple of (context_mask, target_mask, context_indices, target_indices).
        """
        num_mask = int(self.num_patches * self.mask_ratio)
        num_visible = self.num_patches - num_mask

        # Random permutation for each sample
        context_masks = torch.zeros(
            batch_size, self.num_patches, dtype=torch.bool, device=device
        )
        target_masks = torch.zeros(
            batch_size, self.num_patches, dtype=torch.bool, device=device
        )

        context_indices_list = []
        target_indices_list = []

        for i in range(batch_size):
            perm = torch.randperm(self.num_patches, device=device)
            visible_idx = perm[:num_visible]
            masked_idx = perm[num_visible:]

            context_masks[i, visible_idx] = True
            target_masks[i, masked_idx] = True

            context_indices_list.append(visible_idx.sort()[0])
            target_indices_list.append(masked_idx.sort()[0])

        context_indices = torch.stack(context_indices_list)
        target_indices = torch.stack(target_indices_list)

        return context_masks, target_masks, context_indices, target_indices

    def forward(
        self,
        images: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass for I-JEPA training.

        Args:
            images: Input images (B, C, H, W).
            context_mask: Optional pre-computed context mask.
            target_mask: Optional pre-computed target mask.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        batch_size = images.shape[0]
        device = images.device

        # Generate masks if not provided
        if context_mask is None or target_mask is None:
            context_mask, target_mask, ctx_idx, tgt_idx = self._generate_masks(
                batch_size, device
            )
        else:
            # Compute indices from masks
            ctx_idx = torch.stack(
                [m.nonzero(as_tuple=False).squeeze(-1) for m in context_mask]
            )
            tgt_idx = torch.stack(
                [m.nonzero(as_tuple=False).squeeze(-1) for m in target_mask]
            )

        # Context encoder: encode visible patches
        context_encodings = self.context_encoder(images, context_mask)

        # Target encoder: encode target patches (no gradient)
        with torch.no_grad():
            target_encodings = self.target_encoder(images, target_mask)

        # Predictor: predict target representations from context
        predicted_targets = self.predictor(context_encodings, ctx_idx, tgt_idx)

        # Compute loss: MSE in representation space
        # Match the sizes (target_encodings may have different length)
        min_len = min(predicted_targets.shape[1], target_encodings.shape[1])
        loss = F.mse_loss(
            predicted_targets[:, :min_len],
            target_encodings[:, :min_len].detach(),
        )

        # Update target encoder via EMA
        if self.training:
            self.target_encoder.update_from_context(
                self.context_encoder, self.ema_momentum
            )

        metrics = {
            "loss": loss.item(),
            "pred_std": predicted_targets.std().item(),
            "target_std": target_encodings.std().item(),
        }

        return loss, metrics
