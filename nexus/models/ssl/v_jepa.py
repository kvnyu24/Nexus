"""V-JEPA 2: Video Joint-Embedding Predictive Architecture.

Reference: "Revisiting Feature Prediction for Learning Visual Representations
from Video" (Meta AI, 2025)

V-JEPA 2 is a video world model trained on 1M+ hours of video data that learns
spatiotemporal representations by predicting future frame representations in a
latent space. Unlike pixel-level prediction, V-JEPA operates in the representation
space, enabling more semantic and robust learning for downstream tasks including
robot control and video understanding.

Architecture:
    - VideoContextEncoder: Spatiotemporal encoder for visible video frames
    - VideoTargetEncoder: EMA copy of context encoder for target frames
    - VideoPredictor: Predicts future frame representations from context
    - VJEPAModel: Complete video world model for self-supervised learning

Key properties:
    - Predicts future frame representations, not pixels
    - Spatiotemporal transformer backbone (factorized or joint)
    - Target encoder updated via EMA for stability
    - Temporal masking strategies for learning dynamics
    - Enables zero-shot transfer to downstream tasks
    - Scales to very long videos (up to 1M+ hours of training data)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple, List
from ...core.base import NexusModule


class VideoContextEncoder(NexusModule):
    """Spatiotemporal encoder for visible (context) video frames.

    Processes unmasked video frames to produce contextualized representations
    that capture both spatial and temporal information.

    Args:
        config: Configuration dictionary with:
            - encoder_dim: Embedding dimension. Default: 768.
            - num_heads: Number of attention heads. Default: 12.
            - num_spatial_layers: Number of spatial transformer layers. Default: 8.
            - num_temporal_layers: Number of temporal transformer layers. Default: 4.
            - mlp_ratio: MLP hidden dim ratio. Default: 4.0.
            - patch_size: Spatial patch size. Default: 16.
            - tubelet_size: Temporal tubelet size. Default: 2.
            - img_size: Input frame size. Default: 224.
            - num_frames: Number of frames per clip. Default: 16.
            - dropout: Dropout rate. Default: 0.0.
            - factorized: Use factorized space-time attention. Default: True.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.encoder_dim = config.get("encoder_dim", 768)
        self.num_heads = config.get("num_heads", 12)
        self.num_spatial_layers = config.get("num_spatial_layers", 8)
        self.num_temporal_layers = config.get("num_temporal_layers", 4)
        self.mlp_ratio = config.get("mlp_ratio", 4.0)
        self.patch_size = config.get("patch_size", 16)
        self.tubelet_size = config.get("tubelet_size", 2)
        self.img_size = config.get("img_size", 224)
        self.num_frames = config.get("num_frames", 16)
        self.dropout = config.get("dropout", 0.0)
        self.factorized = config.get("factorized", True)

        self.num_spatial_patches = (self.img_size // self.patch_size) ** 2
        self.num_temporal_patches = self.num_frames // self.tubelet_size
        self.num_patches = self.num_spatial_patches * self.num_temporal_patches

        # 3D patch embedding (T x H x W -> tokens)
        self.patch_embed = nn.Conv3d(
            3,
            self.encoder_dim,
            kernel_size=(self.tubelet_size, self.patch_size, self.patch_size),
            stride=(self.tubelet_size, self.patch_size, self.patch_size)
        )

        # Positional embeddings
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spatial_patches, self.encoder_dim)
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_temporal_patches, self.encoder_dim)
        )
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        if self.factorized:
            # Factorized space-time attention
            # First process spatial, then temporal
            spatial_layer = nn.TransformerEncoderLayer(
                d_model=self.encoder_dim,
                nhead=self.num_heads,
                dim_feedforward=int(self.encoder_dim * self.mlp_ratio),
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.spatial_transformer = nn.TransformerEncoder(
                spatial_layer, num_layers=self.num_spatial_layers
            )

            temporal_layer = nn.TransformerEncoderLayer(
                d_model=self.encoder_dim,
                nhead=self.num_heads,
                dim_feedforward=int(self.encoder_dim * self.mlp_ratio),
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal_transformer = nn.TransformerEncoder(
                temporal_layer, num_layers=self.num_temporal_layers
            )
        else:
            # Joint spatiotemporal attention
            joint_layer = nn.TransformerEncoderLayer(
                d_model=self.encoder_dim,
                nhead=self.num_heads,
                dim_feedforward=int(self.encoder_dim * self.mlp_ratio),
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.joint_transformer = nn.TransformerEncoder(
                joint_layer,
                num_layers=self.num_spatial_layers + self.num_temporal_layers
            )

        self.norm = nn.LayerNorm(self.encoder_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode visible video frames.

        Args:
            x: Input video (B, C, T, H, W).
            mask: Boolean mask indicating visible spatiotemporal patches (B, T*H*W).
                True = visible, False = masked.

        Returns:
            Encoded representations (B, N_vis, D).
        """
        # Patch embedding: (B, C, T, H, W) -> (B, D, T', H', W')
        x = self.patch_embed(x)

        # Reshape to sequence: (B, D, T', H', W') -> (B, T'*H'*W', D)
        B, D, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', D)

        # Add positional embeddings
        # Spatial pos: (1, H'*W', D) -> (1, T', H'*W', D) -> (B, T'*H'*W', D)
        spatial_pos = self.spatial_pos_embed.unsqueeze(1).expand(-1, T, -1, -1)
        spatial_pos = spatial_pos.reshape(1, T * H * W, D).expand(B, -1, -1)

        # Temporal pos: (1, T', D) -> (1, T', H'*W', D) -> (B, T'*H'*W', D)
        temporal_pos = self.temporal_pos_embed.unsqueeze(2).expand(-1, -1, H * W, -1)
        temporal_pos = temporal_pos.reshape(1, T * H * W, D).expand(B, -1, -1)

        x = x + spatial_pos + temporal_pos

        # Apply mask: keep only visible patches
        if mask is not None:
            # mask: (B, N) - True for visible patches
            visible_tokens = []
            for i in range(B):
                visible_tokens.append(x[i][mask[i]])

            # Pad to same length and stack
            max_vis = max(t.shape[0] for t in visible_tokens)
            padded = torch.zeros(B, max_vis, self.encoder_dim, device=x.device)
            for i, t in enumerate(visible_tokens):
                padded[i, :t.shape[0]] = t
            x = padded

        if self.factorized:
            # Factorized space-time processing
            # Spatial: process each frame independently
            BT = x.shape[0] if mask is None else B * T
            x_spatial = x.reshape(BT, H * W, -1)
            x_spatial = self.spatial_transformer(x_spatial)

            # Temporal: process each spatial location across time
            x_spatial = x_spatial.reshape(B, T, H * W, -1)
            x_temporal = x_spatial.permute(0, 2, 1, 3).reshape(B * H * W, T, -1)
            x_temporal = self.temporal_transformer(x_temporal)
            x = x_temporal.reshape(B, H * W, T, -1).permute(0, 2, 1, 3)
            x = x.reshape(B, T * H * W, -1)
        else:
            # Joint spatiotemporal processing
            x = self.joint_transformer(x)

        x = self.norm(x)

        return x


class VideoTargetEncoder(NexusModule):
    """EMA-updated target encoder for V-JEPA.

    Same architecture as VideoContextEncoder but updated via exponential
    moving average for stable target representations.

    Args:
        config: Same configuration as VideoContextEncoder.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.encoder = VideoContextEncoder(config)

        # Freeze all parameters - updated only via EMA
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_from_context(
        self,
        context_encoder: VideoContextEncoder,
        momentum: float = 0.996
    ) -> None:
        """Update target encoder weights via EMA.

        Args:
            context_encoder: The context encoder to copy from.
            momentum: EMA momentum factor. Default: 0.996.
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
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode target frames (no gradient).

        Args:
            x: Input video (B, C, T, H, W).
            mask: Boolean mask for target patches.

        Returns:
            Encoded target representations.
        """
        return self.encoder(x, mask)


class VideoPredictor(NexusModule):
    """Lightweight predictor for V-JEPA.

    Predicts future frame representations from visible context frames.
    Uses a smaller transformer to prevent trivial solutions.

    Args:
        config: Configuration dictionary with:
            - encoder_dim: Context encoder output dim. Default: 768.
            - predictor_dim: Predictor hidden dim. Default: 384.
            - predictor_heads: Number of attention heads. Default: 12.
            - predictor_layers: Number of layers. Default: 6.
            - num_patches: Total number of spatiotemporal patches.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.encoder_dim = config.get("encoder_dim", 768)
        self.predictor_dim = config.get("predictor_dim", 384)
        self.num_heads = config.get("predictor_heads", 12)
        self.num_layers = config.get("predictor_layers", 6)

        # Calculate num_patches from config
        patch_size = config.get("patch_size", 16)
        tubelet_size = config.get("tubelet_size", 2)
        img_size = config.get("img_size", 224)
        num_frames = config.get("num_frames", 16)

        num_spatial_patches = (img_size // patch_size) ** 2
        num_temporal_patches = num_frames // tubelet_size
        self.num_patches = num_spatial_patches * num_temporal_patches

        # Project from encoder dim to predictor dim
        self.input_proj = nn.Linear(self.encoder_dim, self.predictor_dim)

        # Positional embedding for target locations
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.predictor_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Mask token for target positions
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, self.predictor_dim)
        )
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
        target_positions: torch.Tensor
    ) -> torch.Tensor:
        """Predict target frame representations from context.

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
            context_positions.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
        )
        context = context + ctx_pos

        # Create mask tokens with target positional embeddings
        mask_tokens = self.mask_token.expand(batch_size, num_targets, -1)
        tgt_pos = torch.gather(
            self.pos_embed.expand(batch_size, -1, -1),
            1,
            target_positions.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
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


class VJEPAModel(NexusModule):
    """V-JEPA 2: Video Joint-Embedding Predictive Architecture.

    Complete video world model for self-supervised learning from videos.
    Learns spatiotemporal representations by predicting future frame
    representations in latent space.

    Training procedure:
        1. Split video patches into context (visible) and target (future) sets
        2. Context encoder processes visible frames
        3. Target encoder (EMA) processes target frames (no gradient)
        4. Predictor predicts target representations from context
        5. Loss = MSE between predicted and actual target representations
        6. Update context encoder and predictor via gradient descent
        7. Update target encoder via EMA

    Args:
        config: Configuration dictionary with:
            - encoder_dim: Embedding dimension. Default: 768.
            - predictor_dim: Predictor hidden dimension. Default: 384.
            - num_frames: Number of frames per clip. Default: 16.
            - mask_ratio: Fraction of patches to mask. Default: 0.7.
            - temporal_mask_ratio: Fraction of frames to mask. Default: 0.5.
            - ema_momentum: EMA momentum for target encoder. Default: 0.996.
            - Additional encoder and predictor config options.

    Example:
        >>> config = {
        ...     "encoder_dim": 768,
        ...     "predictor_dim": 384,
        ...     "num_frames": 16,
        ...     "mask_ratio": 0.7
        ... }
        >>> model = VJEPAModel(config)
        >>> video = torch.randn(4, 3, 16, 224, 224)  # (B, C, T, H, W)
        >>> loss, metrics = model(video)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.encoder_dim = config.get("encoder_dim", 768)
        self.mask_ratio = config.get("mask_ratio", 0.7)
        self.temporal_mask_ratio = config.get("temporal_mask_ratio", 0.5)
        self.ema_momentum = config.get("ema_momentum", 0.996)

        self.patch_size = config.get("patch_size", 16)
        self.tubelet_size = config.get("tubelet_size", 2)
        self.img_size = config.get("img_size", 224)
        self.num_frames = config.get("num_frames", 16)

        num_spatial_patches = (self.img_size // self.patch_size) ** 2
        num_temporal_patches = self.num_frames // self.tubelet_size
        self.num_patches = num_spatial_patches * num_temporal_patches

        # Context encoder (trainable)
        self.context_encoder = VideoContextEncoder(config)

        # Target encoder (EMA of context encoder)
        self.target_encoder = VideoTargetEncoder(config)

        # Initialize target encoder from context encoder
        self.target_encoder.update_from_context(
            self.context_encoder, momentum=0.0
        )

        # Predictor
        self.predictor = VideoPredictor(config)

    def _generate_temporal_masks(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate context and target masks with temporal bias.

        Masks future frames more aggressively to encourage temporal prediction.

        Args:
            batch_size: Number of samples in the batch.
            device: Device to create tensors on.

        Returns:
            Tuple of (context_mask, target_mask, context_indices, target_indices).
        """
        num_mask = int(self.num_patches * self.mask_ratio)
        num_visible = self.num_patches - num_mask

        context_masks = torch.zeros(
            batch_size, self.num_patches, dtype=torch.bool, device=device
        )
        target_masks = torch.zeros(
            batch_size, self.num_patches, dtype=torch.bool, device=device
        )

        context_indices_list = []
        target_indices_list = []

        for i in range(batch_size):
            # Temporal masking: mask later frames more
            # Split patches by temporal location
            temporal_patches = self.num_frames // self.tubelet_size
            spatial_patches = self.num_patches // temporal_patches

            # Select visible frames (earlier frames more likely)
            num_visible_frames = int(temporal_patches * (1 - self.temporal_mask_ratio))
            visible_frame_indices = torch.arange(num_visible_frames, device=device)

            # For visible frames, randomly select spatial patches
            visible_indices = []
            for t_idx in visible_frame_indices:
                frame_start = t_idx * spatial_patches
                frame_end = (t_idx + 1) * spatial_patches
                frame_patches = torch.arange(frame_start, frame_end, device=device)

                # Keep most spatial patches from visible frames
                num_keep = int(spatial_patches * 0.8)
                kept = frame_patches[torch.randperm(spatial_patches, device=device)[:num_keep]]
                visible_indices.append(kept)

            visible_idx = torch.cat(visible_indices) if visible_indices else torch.tensor([], dtype=torch.long, device=device)

            # Remaining patches are targets
            all_indices = torch.arange(self.num_patches, device=device)
            masked_idx = torch.tensor([idx for idx in all_indices if idx not in visible_idx], device=device)

            context_masks[i, visible_idx] = True
            target_masks[i, masked_idx] = True

            context_indices_list.append(visible_idx.sort()[0])
            target_indices_list.append(masked_idx.sort()[0])

        context_indices = torch.nn.utils.rnn.pad_sequence(
            context_indices_list, batch_first=True, padding_value=0
        )
        target_indices = torch.nn.utils.rnn.pad_sequence(
            target_indices_list, batch_first=True, padding_value=0
        )

        return context_masks, target_masks, context_indices, target_indices

    def forward(
        self,
        video: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass for V-JEPA training.

        Args:
            video: Input video (B, C, T, H, W).
            context_mask: Optional pre-computed context mask.
            target_mask: Optional pre-computed target mask.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        batch_size = video.shape[0]
        device = video.device

        # Generate masks if not provided
        if context_mask is None or target_mask is None:
            context_mask, target_mask, ctx_idx, tgt_idx = self._generate_temporal_masks(
                batch_size, device
            )
        else:
            # Compute indices from masks
            ctx_idx = torch.stack([
                m.nonzero(as_tuple=False).squeeze(-1) for m in context_mask
            ])
            tgt_idx = torch.stack([
                m.nonzero(as_tuple=False).squeeze(-1) for m in target_mask
            ])

        # Context encoder: encode visible frames
        context_encodings = self.context_encoder(video, context_mask)

        # Target encoder: encode target frames (no gradient)
        with torch.no_grad():
            target_encodings = self.target_encoder(video, target_mask)

        # Predictor: predict target representations from context
        predicted_targets = self.predictor(context_encodings, ctx_idx, tgt_idx)

        # Compute loss: MSE in representation space
        min_len = min(predicted_targets.shape[1], target_encodings.shape[1])
        loss = F.mse_loss(
            predicted_targets[:, :min_len],
            target_encodings[:, :min_len].detach()
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
            "context_frames": context_mask.float().sum().item() / batch_size,
            "target_frames": target_mask.float().sum().item() / batch_size,
        }

        return loss, metrics
