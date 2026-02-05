"""data2vec 2.0: Efficient Multimodal Self-Supervised Learning.

Reference: "Efficient Self-supervised Learning with Contextualized Target
Representations for Vision, Speech and Language" (Baevski et al., Meta AI, ICML 2023)

data2vec 2.0 is an improved version of data2vec that uses a unified framework
for self-supervised learning across vision, speech, and text modalities. Instead
of predicting raw inputs, it predicts contextualized latent representations from
a teacher model. Version 2.0 introduces key efficiency improvements:
    - Shared encoder-decoder architecture
    - Inverse block masking (only process visible tokens)
    - Fast convolutional decoder
    - Multi-masking for improved sample efficiency

Architecture:
    - StudentEncoder: Transformer encoder processing masked inputs
    - TeacherEncoder: EMA-updated encoder processing full inputs
    - ContextualizedDecoder: Lightweight decoder for reconstruction
    - Data2VecModel: Complete multimodal self-supervised learning system

Key properties:
    - Modality-agnostic: works on images, audio, and text
    - Predicts contextualized representations, not raw features
    - Teacher model updated via EMA for stability
    - Efficient inverse block masking strategy
    - 2x training speedup over data2vec 1.0
    - Strong performance on ImageNet, speech, and NLP benchmarks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from ...core.base import NexusModule


class StudentEncoder(NexusModule):
    """Student encoder for data2vec 2.0.

    Processes masked inputs (only visible tokens) to produce representations
    that will be compared against teacher representations.

    Args:
        config: Configuration dictionary with:
            - encoder_dim: Embedding dimension. Default: 768.
            - num_heads: Number of attention heads. Default: 12.
            - num_layers: Number of transformer layers. Default: 12.
            - mlp_ratio: MLP hidden dim ratio. Default: 4.0.
            - dropout: Dropout rate. Default: 0.1.
            - modality: Input modality ('vision', 'audio', 'text'). Default: 'vision'.
            - patch_size: For vision: patch size. Default: 16.
            - img_size: For vision: image size. Default: 224.
            - vocab_size: For text: vocabulary size. Default: 50000.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.encoder_dim = config.get("encoder_dim", 768)
        self.num_heads = config.get("num_heads", 12)
        self.num_layers = config.get("num_layers", 12)
        self.mlp_ratio = config.get("mlp_ratio", 4.0)
        self.dropout = config.get("dropout", 0.1)
        self.modality = config.get("modality", "vision")

        # Modality-specific input projection
        if self.modality == "vision":
            patch_size = config.get("patch_size", 16)
            img_size = config.get("img_size", 224)
            self.num_patches = (img_size // patch_size) ** 2

            self.patch_embed = nn.Conv2d(
                3, self.encoder_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
        elif self.modality == "text":
            vocab_size = config.get("vocab_size", 50000)
            self.token_embed = nn.Embedding(vocab_size, self.encoder_dim)
            self.num_patches = config.get("max_seq_len", 512)
        elif self.modality == "audio":
            # Simple conv for audio spectrograms
            self.audio_embed = nn.Conv2d(
                1, self.encoder_dim,
                kernel_size=(16, 16),
                stride=(10, 10)
            )
            self.num_patches = config.get("num_audio_patches", 512)
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.encoder_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Mask token (for inverse masking - tokens that were masked)
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, self.encoder_dim)
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)

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
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode masked input.

        Args:
            x: Input tensor (modality-dependent shape).
                - Vision: (B, C, H, W)
                - Text: (B, L) token IDs
                - Audio: (B, 1, T, F) spectrogram
            mask: Boolean mask (B, N). True = visible, False = masked.

        Returns:
            Tuple of (encoded_visible, encoded_full_with_masks).
                - encoded_visible: Only visible tokens (B, N_vis, D)
                - encoded_full_with_masks: All positions with mask tokens (B, N, D)
        """
        # Modality-specific embedding
        if self.modality == "vision":
            x = self.patch_embed(x)  # (B, D, H', W')
            x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        elif self.modality == "text":
            x = self.token_embed(x)  # (B, L, D)
        elif self.modality == "audio":
            x = self.audio_embed(x)  # (B, D, T', F')
            x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        batch_size, num_tokens, dim = x.shape

        # Add positional embedding
        pos_embed = self.pos_embed[:, :num_tokens, :]
        x = x + pos_embed

        # Inverse masking: process only visible tokens (efficient!)
        if mask is not None:
            # Extract visible tokens
            visible_tokens = []
            visible_pos = []
            for i in range(batch_size):
                vis_idx = mask[i].nonzero(as_tuple=False).squeeze(-1)
                visible_tokens.append(x[i, vis_idx])
                visible_pos.append(vis_idx)

            # Stack visible tokens
            max_vis = max(t.shape[0] for t in visible_tokens)
            x_vis = torch.zeros(
                batch_size, max_vis, self.encoder_dim,
                device=x.device
            )
            for i, t in enumerate(visible_tokens):
                x_vis[i, :t.shape[0]] = t

            # Process only visible tokens through transformer
            x_vis = self.transformer(x_vis)
            x_vis = self.norm(x_vis)

            # Reconstruct full sequence with mask tokens
            x_full = torch.zeros(
                batch_size, num_tokens, self.encoder_dim,
                device=x.device
            )
            for i in range(batch_size):
                vis_idx = visible_pos[i]
                mask_idx = (~mask[i]).nonzero(as_tuple=False).squeeze(-1)

                # Place visible encodings
                x_full[i, vis_idx] = x_vis[i, :len(vis_idx)]

                # Place mask tokens at masked positions
                x_full[i, mask_idx] = self.mask_token.expand(len(mask_idx), -1)

            return x_vis, x_full
        else:
            # No masking: process all tokens
            x = self.transformer(x)
            x = self.norm(x)
            return x, x


class TeacherEncoder(NexusModule):
    """Teacher encoder for data2vec 2.0.

    EMA-updated encoder that processes full (unmasked) inputs to generate
    target representations for the student to predict.

    Args:
        config: Same configuration as StudentEncoder.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.encoder = StudentEncoder(config)

        # Freeze all parameters - updated only via EMA
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_from_student(
        self,
        student_encoder: StudentEncoder,
        momentum: float = 0.999
    ) -> None:
        """Update teacher weights via EMA from student.

        Args:
            student_encoder: The student encoder to copy from.
            momentum: EMA momentum factor. Default: 0.999.
        """
        for teacher_param, student_param in zip(
            self.encoder.parameters(),
            student_encoder.parameters()
        ):
            teacher_param.data.mul_(momentum).add_(
                student_param.data, alpha=1.0 - momentum
            )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Encode full input (no masking, no gradient).

        Args:
            x: Input tensor (same format as StudentEncoder).

        Returns:
            Encoded representations for all tokens (B, N, D).
        """
        with torch.no_grad():
            # Process without masking
            _, x_full = self.encoder(x, mask=None)
            return x_full


class ContextualizedDecoder(NexusModule):
    """Fast convolutional decoder for data2vec 2.0.

    Lightweight decoder that processes the full sequence (visible + mask tokens)
    to produce final predictions. Uses depthwise separable convolutions for
    efficiency.

    Args:
        config: Configuration dictionary with:
            - encoder_dim: Input dimension from encoder. Default: 768.
            - decoder_dim: Decoder hidden dimension. Default: 384.
            - decoder_layers: Number of decoder layers. Default: 4.
            - kernel_size: Convolution kernel size. Default: 3.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.encoder_dim = config.get("encoder_dim", 768)
        self.decoder_dim = config.get("decoder_dim", 384)
        self.decoder_layers = config.get("decoder_layers", 4)
        self.kernel_size = config.get("kernel_size", 3)

        # Project to decoder dimension
        self.input_proj = nn.Linear(self.encoder_dim, self.decoder_dim)

        # Depthwise separable conv layers
        self.conv_layers = nn.ModuleList()
        for _ in range(self.decoder_layers):
            # Depthwise conv
            depthwise = nn.Conv1d(
                self.decoder_dim,
                self.decoder_dim,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                groups=self.decoder_dim,
            )
            # Pointwise conv
            pointwise = nn.Conv1d(
                self.decoder_dim,
                self.decoder_dim,
                kernel_size=1,
            )
            self.conv_layers.append(
                nn.Sequential(
                    depthwise,
                    nn.GELU(),
                    pointwise,
                    nn.LayerNorm(self.decoder_dim),
                )
            )

        # Output projection
        self.output_proj = nn.Linear(self.decoder_dim, self.encoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode to target space.

        Args:
            x: Encoder output with mask tokens (B, N, D_enc).

        Returns:
            Decoded representations (B, N, D_enc).
        """
        # Project to decoder dim
        x = self.input_proj(x)  # (B, N, D_dec)

        # Transpose for conv: (B, N, D) -> (B, D, N)
        x = x.transpose(1, 2)

        # Apply conv layers
        for conv in self.conv_layers:
            identity = x
            x = conv(x)
            x = x + identity  # Residual connection

        # Transpose back: (B, D, N) -> (B, N, D)
        x = x.transpose(1, 2)

        # Project to encoder dim
        x = self.output_proj(x)

        return x


class Data2VecModel(NexusModule):
    """data2vec 2.0: Efficient multimodal self-supervised learning.

    Unified framework for self-supervised learning across vision, speech,
    and text. The model learns by predicting contextualized representations
    from a teacher network.

    Training procedure:
        1. Generate masked input (mask random subset of tokens)
        2. Student encoder processes only visible tokens (inverse masking)
        3. Reconstruct full sequence with mask tokens at masked positions
        4. Decoder processes full sequence
        5. Teacher encoder processes unmasked input (no gradient)
        6. Loss = smooth L1 between student predictions and teacher targets
        7. Update student encoder and decoder via gradient descent
        8. Update teacher encoder via EMA

    Args:
        config: Configuration dictionary with:
            - encoder_dim: Embedding dimension. Default: 768.
            - decoder_dim: Decoder dimension. Default: 384.
            - modality: 'vision', 'audio', or 'text'. Default: 'vision'.
            - mask_ratio: Fraction of tokens to mask. Default: 0.6.
            - ema_momentum: EMA momentum for teacher. Default: 0.999.
            - ema_end_momentum: Final EMA momentum. Default: 0.9999.
            - loss_beta: Smooth L1 loss beta. Default: 2.0.
            - multi_mask: Number of masks per sample. Default: 2.
            - Additional encoder/decoder config options.

    Example:
        >>> config = {
        ...     "encoder_dim": 768,
        ...     "modality": "vision",
        ...     "mask_ratio": 0.6
        ... }
        >>> model = Data2VecModel(config)
        >>> images = torch.randn(8, 3, 224, 224)
        >>> loss, metrics = model(images)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.encoder_dim = config.get("encoder_dim", 768)
        self.mask_ratio = config.get("mask_ratio", 0.6)
        self.ema_momentum = config.get("ema_momentum", 0.999)
        self.ema_end_momentum = config.get("ema_end_momentum", 0.9999)
        self.loss_beta = config.get("loss_beta", 2.0)
        self.multi_mask = config.get("multi_mask", 2)
        self.modality = config.get("modality", "vision")

        # Student encoder (trainable)
        self.student_encoder = StudentEncoder(config)

        # Teacher encoder (EMA of student)
        self.teacher_encoder = TeacherEncoder(config)

        # Initialize teacher from student
        self.teacher_encoder.update_from_student(
            self.student_encoder, momentum=0.0
        )

        # Contextualized decoder
        self.decoder = ContextualizedDecoder(config)

        # Layer normalization for targets (improves stability)
        self.target_norm = nn.LayerNorm(self.encoder_dim)

        # Track EMA momentum schedule
        self.register_buffer(
            "ema_momentum_schedule",
            torch.tensor(self.ema_momentum)
        )

    def _generate_mask(
        self,
        batch_size: int,
        num_tokens: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate random mask.

        Args:
            batch_size: Number of samples.
            num_tokens: Number of tokens per sample.
            device: Device for tensors.

        Returns:
            Boolean mask (B, N). True = visible, False = masked.
        """
        num_visible = int(num_tokens * (1 - self.mask_ratio))

        mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)

        for i in range(batch_size):
            # Random permutation
            perm = torch.randperm(num_tokens, device=device)
            visible_idx = perm[:num_visible]
            mask[i, visible_idx] = True

        return mask

    def update_ema_momentum(self, step: int, max_steps: int) -> None:
        """Update EMA momentum using cosine schedule.

        Args:
            step: Current training step.
            max_steps: Total training steps.
        """
        # Cosine schedule from ema_momentum to ema_end_momentum
        progress = step / max_steps
        momentum = (
            self.ema_end_momentum
            - (self.ema_end_momentum - self.ema_momentum)
            * (math.cos(math.pi * progress) + 1) / 2
        )
        self.ema_momentum_schedule.fill_(momentum)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass for data2vec 2.0 training.

        Args:
            x: Input tensor (modality-dependent shape).
            mask: Optional pre-computed mask.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        batch_size = x.shape[0]
        device = x.device

        # Determine number of tokens based on modality
        if self.modality == "vision":
            num_tokens = self.student_encoder.num_patches
        elif self.modality == "text":
            num_tokens = x.shape[1]
        elif self.modality == "audio":
            num_tokens = self.student_encoder.num_patches

        # Multi-masking: generate multiple masks per sample
        total_loss = 0.0
        for _ in range(self.multi_mask):
            # Generate or use provided mask
            if mask is None:
                current_mask = self._generate_mask(batch_size, num_tokens, device)
            else:
                current_mask = mask

            # Student forward pass (only visible tokens)
            _, student_full = self.student_encoder(x, current_mask)

            # Decoder: process full sequence with mask tokens
            student_preds = self.decoder(student_full)

            # Teacher forward pass (no masking, no gradient)
            with torch.no_grad():
                teacher_targets = self.teacher_encoder(x)

                # Normalize targets for stability
                teacher_targets = self.target_norm(teacher_targets)

                # Average over top K layers (in practice, use last layer here)
                # In full implementation, could average over multiple transformer layers

            # Compute loss only on masked positions
            mask_indices = ~current_mask  # Masked positions
            masked_preds = student_preds[mask_indices]
            masked_targets = teacher_targets[mask_indices]

            # Smooth L1 loss (Huber loss)
            loss = F.smooth_l1_loss(
                masked_preds,
                masked_targets,
                beta=self.loss_beta
            )

            total_loss += loss

        # Average loss over multi-masks
        total_loss = total_loss / self.multi_mask

        # Update teacher via EMA
        if self.training:
            momentum = self.ema_momentum_schedule.item()
            self.teacher_encoder.update_from_student(
                self.student_encoder, momentum
            )

        metrics = {
            "loss": total_loss.item(),
            "pred_std": student_preds.std().item(),
            "target_std": teacher_targets.std().item(),
            "ema_momentum": self.ema_momentum_schedule.item(),
        }

        return total_loss, metrics
