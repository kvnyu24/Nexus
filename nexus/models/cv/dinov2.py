"""
DINOv2: Learning Robust Visual Features without Supervision

Implementation of DINOv2, a self-supervised Vision Transformer that learns visual
features through self-distillation with no labels. Extends the original DINO framework
with improved training recipes, larger models, and better augmentation strategies.

Reference:
    Oquab, M., Darcet, T., Moutakanni, T., et al. (2023).
    "DINOv2: Learning Robust Visual Features without Supervision."
    arXiv:2304.07193

Key Components:
    - DINOHead: MLP projection head with L2 normalization, centering, and sharpening
    - StudentTeacher: EMA-based student-teacher framework with asymmetric crops
    - DINOv2Loss: Cross-entropy loss between sharpened teacher and student distributions
    - DINOv2: Full self-supervised ViT model supporting S/B/L/G variants

Architecture Details:
    - Student receives both global (224x224) and local (96x96) crops
    - Teacher receives only global crops and is updated via EMA
    - Centering prevents mode collapse in the teacher output
    - Sharpening controls the entropy of the output distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from typing import Dict, Any, Optional, List, Tuple

from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin


# ============================================================================
# Model variant configurations
# ============================================================================

DINOV2_VARIANTS: Dict[str, Dict[str, Any]] = {
    "vit_small": {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "vit_base": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "vit_large": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
    },
    "vit_giant": {
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "mlp_ratio": 4.0,
    },
}


class PatchEmbedding(NexusModule):
    """Patch embedding layer that converts image patches into token embeddings.

    Splits an image into non-overlapping patches and projects each patch into
    an embedding vector using a convolutional layer.

    Args:
        img_size: Input image resolution (assumed square).
        patch_size: Size of each patch (assumed square).
        in_channels: Number of input image channels.
        embed_dim: Dimension of output embeddings.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim).
        """
        x = self.projection(x)  # (B, E, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        x = self.norm(x)
        return x


class DINOBlock(NexusModule):
    """Transformer block used in DINOv2 with layer scale and stochastic depth.

    Applies multi-head self-attention followed by an MLP, with pre-normalization,
    learnable layer scale parameters, and optional stochastic depth for
    regularization.

    Args:
        dim: Input/output dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dimension to input dimension.
        dropout: Dropout probability.
        drop_path: Stochastic depth rate.
        layer_scale_init: Initial value for layer scale parameters.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

        # Layer scale parameters (DINOv2 uses these for training stability)
        self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(dim))

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, dim).

        Returns:
            Output tensor of shape (B, N, dim).
        """
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop_path(self.gamma1 * attn_out)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization.

    During training, randomly drops entire residual branches with a given
    probability, effectively creating an ensemble of subnetworks.

    Args:
        drop_prob: Probability of dropping the path.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x * mask / keep_prob


class DINOHead(NexusModule):
    """DINO projection head with centering and sharpening.

    Projects the CLS token through an MLP, applies L2 normalization, and
    uses centering and temperature-based sharpening to produce the final
    probability distribution.

    The centering mechanism maintains an exponential moving average of the
    teacher outputs to prevent any single dimension from dominating, thus
    avoiding mode collapse.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        bottleneck_dim: Bottleneck (projection) dimension.
        out_dim: Output dimension (number of prototypes).
        nlayers: Number of MLP layers (minimum 2).
        use_bn: Whether to use batch normalization in the MLP.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        out_dim: int = 65536,
        nlayers: int = 3,
        use_bn: bool = True,
    ):
        super().__init__()

        nlayers = max(nlayers, 2)

        # Build MLP layers
        layers: List[nn.Module] = []
        for i in range(nlayers - 1):
            dim_in = in_dim if i == 0 else hidden_dim
            dim_out = hidden_dim
            layers.append(nn.Linear(dim_in, dim_out))
            if use_bn:
                layers.append(nn.BatchNorm1d(dim_out))
            layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

        # Last layer projects to bottleneck and then to output prototypes
        self.last_layer = nn.Linear(hidden_dim, bottleneck_dim)
        self.last_layer_norm = nn.LayerNorm(bottleneck_dim)
        self.prototypes = nn.Linear(bottleneck_dim, out_dim, bias=False)

        # Center buffer for teacher centering (prevents collapse)
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """Forward pass producing a sharpened probability distribution.

        Args:
            x: Input features of shape (B, in_dim).
            temperature: Temperature for sharpening the output distribution.

        Returns:
            Log-softmax probabilities of shape (B, out_dim).
        """
        x = self.mlp(x)
        x = self.last_layer(x)
        x = self.last_layer_norm(x)
        x = F.normalize(x, dim=-1)
        logits = self.prototypes(x)
        # Center and sharpen
        logits = (logits - self.center) / temperature
        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor, momentum: float = 0.9) -> None:
        """Update the centering buffer with the teacher's current output statistics.

        Uses exponential moving average to smoothly track the mean teacher output.

        Args:
            teacher_output: Raw teacher logits of shape (B, out_dim).
            momentum: EMA momentum for the center update.
        """
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * momentum + batch_center * (1.0 - momentum)


class DINOv2Loss(NexusModule):
    """Cross-entropy loss for DINO self-distillation.

    Computes the cross-entropy between the teacher's sharpened probability
    distribution and the student's predictions. The loss is computed over
    all pairs of teacher global crops and student crops (both global and local),
    excluding same-view pairs.

    Args:
        student_temp: Temperature for sharpening the student distribution.
        teacher_temp: Temperature for sharpening the teacher distribution.
        center_momentum: EMA momentum for the teacher center update.
    """

    def __init__(
        self,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum

    def forward(
        self,
        student_outputs: List[torch.Tensor],
        teacher_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the DINO self-distillation loss.

        The teacher produces soft targets with low temperature (sharp distribution),
        and the student tries to match these targets with higher temperature.

        Args:
            student_outputs: List of student log-probabilities, one per crop.
            teacher_outputs: List of teacher log-probabilities, one per global crop.

        Returns:
            Scalar loss value.
        """
        total_loss = torch.tensor(0.0, device=student_outputs[0].device)
        n_loss_terms = 0

        for t_idx, t_out in enumerate(teacher_outputs):
            # Teacher targets: apply softmax to get probabilities (already log-softmax from head)
            teacher_probs = torch.exp(t_out).detach()

            for s_idx, s_out in enumerate(student_outputs):
                # Skip same-view pairs (student crop == teacher crop)
                if s_idx == t_idx:
                    continue
                # Cross-entropy: -sum(teacher_prob * student_log_prob)
                loss = -torch.sum(teacher_probs * s_out, dim=-1).mean()
                total_loss = total_loss + loss
                n_loss_terms += 1

        if n_loss_terms > 0:
            total_loss = total_loss / n_loss_terms
        return total_loss


class StudentTeacher(NexusModule):
    """Student-Teacher framework with Exponential Moving Average (EMA) update.

    The student is trained with gradient descent while the teacher is updated
    as an exponential moving average of the student's parameters. This creates
    a self-distillation loop where the teacher provides increasingly refined
    targets for the student.

    Both networks share the same architecture but the teacher is never directly
    trained. The EMA schedule typically ramps from a lower momentum to a higher
    one using a cosine schedule.

    Args:
        student: The student network (trained with backprop).
        student_head: The student projection head.
        teacher_head: The teacher projection head (initialized from student_head).
        ema_momentum: Base EMA momentum for teacher updates.
    """

    def __init__(
        self,
        student: NexusModule,
        student_head: DINOHead,
        teacher_head: DINOHead,
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        self.student = student
        self.student_head = student_head

        # Teacher is a detached copy of the student backbone
        self.teacher = copy.deepcopy(student)
        self.teacher.requires_grad_(False)

        self.teacher_head = teacher_head
        self.teacher_head.requires_grad_(False)

        self.ema_momentum = ema_momentum

    @torch.no_grad()
    def update_teacher(self, momentum: Optional[float] = None) -> None:
        """Update the teacher network parameters using EMA.

        Args:
            momentum: Optional override for the EMA momentum. If not provided,
                uses the default self.ema_momentum.
        """
        m = momentum if momentum is not None else self.ema_momentum

        # Update backbone
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.mul_(m).add_(param_s.data, alpha=1.0 - m)

        # Update head
        for param_s, param_t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            param_t.data.mul_(m).add_(param_s.data, alpha=1.0 - m)

    def forward_student(
        self,
        crops: List[torch.Tensor],
        temperature: float = 0.1,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass through the student for all crops.

        Args:
            crops: List of image crop tensors.
            temperature: Student temperature for sharpening.

        Returns:
            Tuple of (student_outputs, student_features) where each is a list
            with one entry per crop.
        """
        outputs = []
        features = []
        for crop in crops:
            feat = self.student(crop)
            cls_token = feat["embeddings"]
            proj = self.student_head(cls_token, temperature=temperature)
            outputs.append(proj)
            features.append(cls_token)
        return outputs, features

    @torch.no_grad()
    def forward_teacher(
        self,
        global_crops: List[torch.Tensor],
        temperature: float = 0.04,
    ) -> List[torch.Tensor]:
        """Forward pass through the teacher for global crops only.

        Args:
            global_crops: List of global crop tensors.
            temperature: Teacher temperature for sharpening.

        Returns:
            List of teacher log-probability outputs.
        """
        outputs = []
        for crop in global_crops:
            feat = self.teacher(crop)
            cls_token = feat["embeddings"]
            proj = self.teacher_head(cls_token, temperature=temperature)
            outputs.append(proj)
        return outputs


class DINOv2(WeightInitMixin, NexusModule):
    """DINOv2: Self-Supervised Vision Transformer with Self-Distillation.

    Full DINOv2 model combining a ViT backbone with student-teacher self-distillation.
    Supports ViT-S/B/L/G variants. During training, uses multi-crop augmentation
    with global and local views. During inference, only the student backbone
    is used to extract features.

    Config:
        img_size (int): Input image resolution. Default: 224.
        patch_size (int): Patch size for tokenization. Default: 14.
        in_channels (int): Number of input channels. Default: 3.
        embed_dim (int): Transformer embedding dimension. Default: 768.
        depth (int): Number of transformer blocks. Default: 12.
        num_heads (int): Number of attention heads. Default: 12.
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1.
        student_temp (float): Student temperature. Default: 0.1.
        teacher_temp (float): Teacher temperature. Default: 0.04.
        ema_momentum (float): Teacher EMA momentum. Default: 0.996.
        out_dim (int): Number of output prototypes. Default: 65536.
        head_hidden_dim (int): Projection head hidden dim. Default: 2048.
        head_bottleneck_dim (int): Projection head bottleneck dim. Default: 256.
        variant (str): Model variant name (vit_small, vit_base, vit_large, vit_giant).
            Overrides embed_dim, depth, num_heads, mlp_ratio if specified.

    Example:
        >>> config = {"img_size": 224, "patch_size": 14, "variant": "vit_base"}
        >>> model = DINOv2(config)
        >>> images = torch.randn(2, 3, 224, 224)
        >>> output = model(images)
        >>> output["embeddings"].shape
        torch.Size([2, 768])
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Apply variant config if specified
        variant = config.get("variant", None)
        if variant is not None and variant in DINOV2_VARIANTS:
            for k, v in DINOV2_VARIANTS[variant].items():
                config.setdefault(k, v)

        # Core configuration
        self.img_size = config.get("img_size", 224)
        self.patch_size = config.get("patch_size", 14)
        self.in_channels = config.get("in_channels", 3)
        self.embed_dim = config.get("embed_dim", 768)
        self.depth = config.get("depth", 12)
        self.num_heads = config.get("num_heads", 12)
        self.mlp_ratio = config.get("mlp_ratio", 4.0)
        self.dropout = config.get("dropout", 0.0)
        self.drop_path_rate = config.get("drop_path_rate", 0.1)

        # Temperature and EMA config
        self.student_temp = config.get("student_temp", 0.1)
        self.teacher_temp = config.get("teacher_temp", 0.04)
        self.ema_momentum = config.get("ema_momentum", 0.996)

        # Projection head config
        self.out_dim = config.get("out_dim", 65536)
        self.head_hidden_dim = config.get("head_hidden_dim", 2048)
        self.head_bottleneck_dim = config.get("head_bottleneck_dim", 256)

        # Build backbone
        self.patch_embed = PatchEmbedding(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
        )

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.dropout)

        # Stochastic depth schedule (linear increase)
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        self.blocks = nn.ModuleList([
            DINOBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                drop_path=dpr[i],
            )
            for i in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        # Build projection heads for student-teacher framework
        self.student_head = DINOHead(
            in_dim=self.embed_dim,
            hidden_dim=self.head_hidden_dim,
            bottleneck_dim=self.head_bottleneck_dim,
            out_dim=self.out_dim,
        )
        self.teacher_head = DINOHead(
            in_dim=self.embed_dim,
            hidden_dim=self.head_hidden_dim,
            bottleneck_dim=self.head_bottleneck_dim,
            out_dim=self.out_dim,
        )

        # Loss function
        self.loss_fn = DINOv2Loss(
            student_temp=self.student_temp,
            teacher_temp=self.teacher_temp,
        )

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.init_weights_vit()

        # Build student-teacher wrapper (must be after weight init)
        self._student_teacher: Optional[StudentTeacher] = None

    def _get_student_teacher(self) -> StudentTeacher:
        """Lazily initialize the student-teacher framework."""
        if self._student_teacher is None:
            # Create a backbone-only wrapper for the teacher to copy
            backbone = _DINOv2Backbone(self)
            self._student_teacher = StudentTeacher(
                student=backbone,
                student_head=self.student_head,
                teacher_head=self.teacher_head,
                ema_momentum=self.ema_momentum,
            )
        return self._student_teacher

    def interpolate_pos_encoding(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Interpolate position embeddings for different input resolutions.

        Args:
            x: Token sequence of shape (B, N+1, embed_dim).
            h: Grid height of the patch embedding.
            w: Grid width of the patch embedding.

        Returns:
            Interpolated position embeddings matching the token count.
        """
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        patch_pos = patch_pos.reshape(1, int(N ** 0.5), int(N ** 0.5), dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(h, w), mode="bicubic", align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((cls_pos, patch_pos), dim=1)

    def forward_backbone(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the backbone only (no projection head).

        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            Dictionary with 'embeddings' (CLS token) and 'features' (all tokens).
        """
        B = x.shape[0]

        # Patch embedding
        tokens = self.patch_embed(x)
        h = w = int(tokens.shape[1] ** 0.5)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        # Add position embedding
        tokens = tokens + self.interpolate_pos_encoding(tokens, h, w)
        tokens = self.pos_drop(tokens)

        # Transformer blocks
        intermediate_features = []
        for block in self.blocks:
            tokens = block(tokens)
            intermediate_features.append(tokens)

        tokens = self.norm(tokens)

        return {
            "embeddings": tokens[:, 0],
            "patch_tokens": tokens[:, 1:],
            "features": intermediate_features,
        }

    def forward(
        self,
        images: torch.Tensor,
        global_crops: Optional[List[torch.Tensor]] = None,
        local_crops: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for both training and inference.

        During training (when global_crops and local_crops are provided):
            Runs the student-teacher framework and returns the self-distillation loss.

        During inference (when only images is provided):
            Extracts features from the backbone.

        Args:
            images: Input images of shape (B, C, H, W). Used for inference.
            global_crops: List of global crop tensors for training.
            local_crops: List of local crop tensors for training.

        Returns:
            Dictionary containing embeddings and optionally loss values.
        """
        if global_crops is not None and self.training:
            return self._forward_train(global_crops, local_crops or [])

        # Inference mode: extract features
        return self.forward_backbone(images)

    def _forward_train(
        self,
        global_crops: List[torch.Tensor],
        local_crops: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with student-teacher self-distillation.

        Args:
            global_crops: List of global crop tensors (typically 2).
            local_crops: List of local crop tensors (typically 6-8).

        Returns:
            Dictionary with loss and training diagnostics.
        """
        all_crops = global_crops + local_crops
        n_global = len(global_crops)

        # Student forward on all crops
        student_outputs = []
        for crop in all_crops:
            feat = self.forward_backbone(crop)
            cls_token = feat["embeddings"]
            proj = self.student_head(cls_token, temperature=self.student_temp)
            student_outputs.append(proj)

        # Teacher forward on global crops only (no grad)
        teacher_outputs = []
        with torch.no_grad():
            for crop in global_crops:
                feat = self.forward_backbone(crop)
                cls_token = feat["embeddings"]
                proj = self.teacher_head(cls_token, temperature=self.teacher_temp)
                teacher_outputs.append(proj)

        # Update teacher center
        with torch.no_grad():
            all_teacher_logits = torch.cat(
                [torch.exp(t) for t in teacher_outputs], dim=0
            )
            self.teacher_head.update_center(all_teacher_logits)

        # Compute loss
        loss = self.loss_fn(student_outputs, teacher_outputs)

        return {
            "loss": loss,
            "embeddings": student_outputs[0].detach(),
            "n_crops": len(all_crops),
        }


class _DINOv2Backbone(NexusModule):
    """Lightweight wrapper around DINOv2 that exposes only backbone forward.

    Used internally by StudentTeacher to create a teacher copy without
    duplicating the projection heads.
    """

    def __init__(self, dinov2: DINOv2):
        super().__init__()
        self.patch_embed = dinov2.patch_embed
        self.cls_token = dinov2.cls_token
        self.pos_embed = dinov2.pos_embed
        self.pos_drop = dinov2.pos_drop
        self.blocks = dinov2.blocks
        self.norm = dinov2.norm
        self._parent = dinov2

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._parent.forward_backbone(x)


__all__ = [
    "DINOv2",
    "DINOHead",
    "DINOBlock",
    "DINOv2Loss",
    "StudentTeacher",
    "DINOV2_VARIANTS",
]
