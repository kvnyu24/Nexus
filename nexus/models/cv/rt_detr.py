"""
RT-DETR: Real-Time Detection Transformer

Implementation of RT-DETR, the first real-time end-to-end object detector that
surpasses YOLO-series detectors in both speed and accuracy. RT-DETR introduces
a hybrid encoder for efficient multi-scale feature processing and an IoU-aware
query selection mechanism for high-quality initial decoder queries.

Reference:
    Zhao, Y., Lv, W., Xu, S., et al. (2024).
    "DETRs Beat YOLOs on Real-time Object Detection."
    arXiv:2304.08069 (CVPR 2024)

Key Components:
    - HybridEncoder: Efficient intra-scale self-attention + cross-scale feature fusion
    - IoUAwareQuerySelection: Selects top-K queries based on predicted IoU quality
    - RTDETRDecoder: Transformer decoder with deformable cross-attention
    - RTDETR: Full real-time detection model

Architecture Details:
    - Hybrid encoder decouples intra-scale interaction and cross-scale fusion
      for computational efficiency
    - Intra-scale interaction uses efficient self-attention within each scale
    - Cross-scale fusion merges features across scales via top-down/bottom-up paths
    - IoU-aware query selection replaces random initialization with content-aware queries
    - Decoder uses deformable attention for efficient cross-attention to multi-scale features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, List, Tuple

from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin


class RepVGGBlock(NexusModule):
    """Re-parameterizable VGG-style block for efficient feature extraction.

    During training, uses a multi-branch architecture (3x3 conv + 1x1 conv +
    identity). During inference, can be fused into a single 3x3 convolution
    for speed.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Convolution stride. Default: 1.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn3x3 = nn.BatchNorm2d(out_channels)

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

        self.identity_bn = nn.BatchNorm2d(out_channels) if in_channels == out_channels and stride == 1 else None

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-branch addition.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H', W').
        """
        out = self.bn3x3(self.conv3x3(x)) + self.bn1x1(self.conv1x1(x))
        if self.identity_bn is not None:
            out = out + self.identity_bn(x)
        return self.act(out)


class EfficientSelfAttention(NexusModule):
    """Efficient self-attention for intra-scale feature interaction.

    Uses a lightweight self-attention mechanism with reduced spatial resolution
    for computational efficiency. Spatial features are first projected to a
    lower-dimensional sequence before applying standard multi-head attention.

    Args:
        embed_dim: Feature dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape (B, N, C).

        Returns:
            Updated features of shape (B, N, C).
        """
        shortcut = x
        x = self.norm(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return shortcut + x


class HybridEncoder(WeightInitMixin, NexusModule):
    """Hybrid encoder with intra-scale interaction and cross-scale fusion.

    Processes multi-scale features in two stages:
    1. Intra-scale interaction: efficient self-attention within each feature level
    2. Cross-scale fusion: top-down and bottom-up feature pyramid for merging
       information across different spatial resolutions

    This decoupling reduces computational cost compared to applying full
    multi-scale attention across all levels simultaneously.

    Args:
        config: Configuration dictionary with keys:
            hidden_dim (int): Hidden dimension for the encoder. Default: 256.
            num_encoder_layers (int): Number of self-attention layers per scale. Default: 1.
            num_heads (int): Number of attention heads. Default: 8.
            dim_feedforward (int): FFN hidden dimension. Default: 1024.
            dropout (float): Dropout rate. Default: 0.0.
            num_feature_levels (int): Number of multi-scale feature levels. Default: 3.
            in_channels (list): Input channel dimensions for each level.
                Default: [512, 1024, 2048].
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_layers = config.get("num_encoder_layers", 1)
        self.num_heads = config.get("num_heads", 8)
        self.dim_feedforward = config.get("dim_feedforward", 1024)
        self.dropout = config.get("dropout", 0.0)
        self.num_feature_levels = config.get("num_feature_levels", 3)
        in_channels = config.get("in_channels", [512, 1024, 2048])

        # Input projection for each scale
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, self.hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.hidden_dim),
            )
            for in_ch in in_channels
        ])

        # Intra-scale self-attention layers (applied to each scale independently)
        self.intra_scale_layers = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            scale_layers = nn.ModuleList()
            for _ in range(self.num_layers):
                scale_layers.append(
                    nn.ModuleList([
                        EfficientSelfAttention(self.hidden_dim, self.num_heads, self.dropout),
                        nn.Sequential(
                            nn.LayerNorm(self.hidden_dim),
                            nn.Linear(self.hidden_dim, self.dim_feedforward),
                            nn.GELU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(self.dim_feedforward, self.hidden_dim),
                            nn.Dropout(self.dropout),
                        ),
                    ])
                )
            self.intra_scale_layers.append(scale_layers)

        # Cross-scale fusion: top-down pathway
        self.top_down_lateral = nn.ModuleList([
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
            for _ in range(self.num_feature_levels - 1)
        ])
        self.top_down_smooth = nn.ModuleList([
            RepVGGBlock(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_feature_levels - 1)
        ])

        # Cross-scale fusion: bottom-up pathway
        self.bottom_up_downsample = nn.ModuleList([
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, stride=2, padding=1)
            for _ in range(self.num_feature_levels - 1)
        ])
        self.bottom_up_smooth = nn.ModuleList([
            RepVGGBlock(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_feature_levels - 1)
        ])

        self.init_weights_vision()

    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """Encode multi-scale features.

        Args:
            multi_scale_features: List of feature maps at different scales,
                each of shape (B, C_i, H_i, W_i) from coarse to fine or
                ascending order.

        Returns:
            Tuple of:
                encoded_features: List of encoded features, each (B, C, H_i, W_i).
                spatial_shapes: List of (H_i, W_i) for each level.
        """
        # Project inputs to hidden_dim
        projected = []
        spatial_shapes = []
        for i, feat in enumerate(multi_scale_features):
            proj = self.input_proj[i](feat)
            projected.append(proj)
            spatial_shapes.append((proj.shape[2], proj.shape[3]))

        # Intra-scale interaction (self-attention within each level)
        for i in range(self.num_feature_levels):
            B, C, H, W = projected[i].shape
            x = projected[i].flatten(2).transpose(1, 2)  # (B, H*W, C)

            for attn_layer, ffn_layer in self.intra_scale_layers[i]:
                x = attn_layer(x)
                x = x + ffn_layer(x)

            projected[i] = x.transpose(1, 2).view(B, C, H, W)

        # Cross-scale fusion: top-down
        for i in range(self.num_feature_levels - 2, -1, -1):
            lateral = self.top_down_lateral[i](projected[i])
            upsampled = F.interpolate(
                projected[i + 1], size=spatial_shapes[i], mode="bilinear", align_corners=False,
            )
            projected[i] = self.top_down_smooth[i](lateral + upsampled)

        # Cross-scale fusion: bottom-up
        for i in range(self.num_feature_levels - 1):
            downsampled = self.bottom_up_downsample[i](projected[i])
            projected[i + 1] = self.bottom_up_smooth[i](projected[i + 1] + downsampled)

        return projected, spatial_shapes


class IoUAwareQuerySelection(NexusModule):
    """IoU-aware query selection for high-quality initial decoder queries.

    Instead of using learnable or random queries, selects the top-K encoder
    features as initial queries based on their predicted classification
    confidence and IoU quality. This provides content-aware initialization
    that significantly improves convergence and accuracy.

    Args:
        hidden_dim: Feature dimension.
        num_queries: Number of queries to select.
        num_classes: Number of object classes.
    """

    def __init__(self, hidden_dim: int, num_queries: int = 300, num_classes: int = 80):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # Classification score predictor
        self.class_embed = nn.Linear(hidden_dim, num_classes)

        # IoU predictor
        self.iou_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Bounding box predictor for initial reference points
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

    def forward(
        self,
        memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top-K queries based on predicted quality.

        Args:
            memory: Encoder output features (B, N, hidden_dim).

        Returns:
            Tuple of:
                selected_features: Top-K features (B, num_queries, hidden_dim).
                reference_points: Initial bbox predictions (B, num_queries, 4).
                selection_scores: Quality scores (B, num_queries).
        """
        B, N, C = memory.shape

        # Predict class scores and IoU quality
        class_scores = self.class_embed(memory)  # (B, N, num_classes)
        iou_scores = self.iou_embed(memory).squeeze(-1)  # (B, N)

        # Combine classification confidence and IoU for ranking
        max_class_scores = class_scores.max(dim=-1).values  # (B, N)
        quality_scores = max_class_scores.sigmoid() * iou_scores

        # Select top-K
        topk_scores, topk_indices = quality_scores.topk(
            min(self.num_queries, N), dim=1
        )

        # Gather selected features and reference points
        topk_indices_expand = topk_indices.unsqueeze(-1).expand(-1, -1, C)
        selected_features = torch.gather(memory, 1, topk_indices_expand)
        reference_points = self.bbox_embed(selected_features)

        return selected_features, reference_points, topk_scores


class DeformableAttention(NexusModule):
    """Deformable attention for efficient cross-attention to multi-scale features.

    Instead of attending to all spatial positions, samples a small set of
    key positions around a reference point. The sampling offsets are learned
    and predicted from the query features, making attention data-dependent.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_levels: Number of feature levels.
        num_points: Number of sampling points per head per level.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 3,
        num_points: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Initialize sampling offsets to focus around reference points
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """Forward pass with deformable attention.

        Args:
            query: Query features (B, num_queries, embed_dim).
            reference_points: Reference coordinates (B, num_queries, num_levels, 2)
                in normalized [0, 1] coordinates.
            value: Multi-scale value features (B, total_tokens, embed_dim).
            spatial_shapes: List of (H, W) for each level.

        Returns:
            Attended features (B, num_queries, embed_dim).
        """
        B, Nq, C = query.shape
        _, Nv, _ = value.shape

        value = self.value_proj(value)
        value = value.reshape(B, Nv, self.num_heads, C // self.num_heads)

        # Predict sampling offsets and attention weights
        offsets = self.sampling_offsets(query).reshape(
            B, Nq, self.num_heads, self.num_levels, self.num_points, 2
        )
        attn_weights = self.attention_weights(query).reshape(
            B, Nq, self.num_heads, self.num_levels * self.num_points
        )
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = attn_weights.reshape(
            B, Nq, self.num_heads, self.num_levels, self.num_points
        )

        # Sample from each level
        offset_normalizer = torch.tensor(
            [s[1] for s in spatial_shapes] + [s[0] for s in spatial_shapes],
            device=query.device, dtype=query.dtype,
        ).reshape(1, 1, 1, -1, 1)

        # Simplified sampling: use bilinear grid_sample for each level
        output = torch.zeros(B, Nq, self.num_heads, C // self.num_heads, device=query.device)

        level_start = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            level_end = level_start + H * W
            value_lvl = value[:, level_start:level_end].reshape(B, H, W, self.num_heads, -1)
            value_lvl = value_lvl.permute(0, 3, 4, 1, 2)  # (B, heads, C//heads, H, W)

            # Reference points for this level
            if reference_points.dim() == 3:
                ref_pts = reference_points[:, :, :2].unsqueeze(2)  # (B, Nq, 1, 2)
            else:
                ref_pts = reference_points[:, :, lvl:lvl+1, :2]  # (B, Nq, 1, 2)

            # Add offsets (normalized by spatial size)
            lvl_offsets = offsets[:, :, :, lvl, :, :]  # (B, Nq, heads, points, 2)
            sampling_locs = ref_pts.unsqueeze(2) + lvl_offsets / torch.tensor(
                [W, H], device=query.device, dtype=query.dtype
            )

            # Flatten and use grid_sample
            for h in range(self.num_heads):
                for p in range(self.num_points):
                    grid = sampling_locs[:, :, h, p, :].unsqueeze(1)  # (B, 1, Nq, 2)
                    grid = grid * 2 - 1  # Normalize to [-1, 1]
                    sampled = F.grid_sample(
                        value_lvl[:, h], grid, mode="bilinear",
                        padding_mode="zeros", align_corners=False,
                    )  # (B, C//heads, 1, Nq)
                    output[:, :, h] += (
                        sampled.squeeze(2).transpose(1, 2) * attn_weights[:, :, h, lvl, p].unsqueeze(-1)
                    )

            level_start = level_end

        output = output.reshape(B, Nq, C)
        return self.output_proj(output)


class RTDETRDecoderLayer(NexusModule):
    """Single decoder layer for RT-DETR.

    Consists of self-attention among queries, deformable cross-attention
    to multi-scale encoder features, and a feed-forward network.

    Args:
        hidden_dim: Feature dimension.
        num_heads: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout rate.
        num_feature_levels: Number of multi-scale levels.
        num_points: Deformable attention sampling points.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_feature_levels: int = 3,
        num_points: int = 4,
    ):
        super().__init__()

        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Deformable cross-attention to encoder features
        self.cross_attn = DeformableAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            num_levels=num_feature_levels,
            num_points=num_points,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: Decoder queries (B, num_queries, hidden_dim).
            reference_points: Query reference points (B, num_queries, 4).
            memory: Flattened multi-scale encoder features (B, N_total, hidden_dim).
            spatial_shapes: List of (H, W) per level.

        Returns:
            Updated queries (B, num_queries, hidden_dim).
        """
        # Self-attention
        q = k = query
        attn_out, _ = self.self_attn(q, k, query)
        query = self.norm1(query + self.dropout1(attn_out))

        # Deformable cross-attention
        cross_out = self.cross_attn(query, reference_points, memory, spatial_shapes)
        query = self.norm2(query + self.dropout2(cross_out))

        # FFN
        query = self.norm3(query + self.ffn(query))

        return query


class RTDETRDecoder(WeightInitMixin, NexusModule):
    """Transformer decoder for RT-DETR with iterative bounding box refinement.

    Uses deformable attention for efficient cross-attention and iteratively
    refines bounding box predictions across decoder layers. Each layer
    predicts offsets to the current reference points, progressively
    improving detection accuracy.

    Args:
        config: Configuration dictionary with keys:
            hidden_dim (int): Decoder dimension. Default: 256.
            num_decoder_layers (int): Number of decoder layers. Default: 6.
            num_heads (int): Number of attention heads. Default: 8.
            dim_feedforward (int): FFN hidden dimension. Default: 1024.
            dropout (float): Dropout rate. Default: 0.0.
            num_classes (int): Number of object classes. Default: 80.
            num_feature_levels (int): Number of multi-scale levels. Default: 3.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_layers = config.get("num_decoder_layers", 6)
        self.num_heads = config.get("num_heads", 8)
        self.dim_feedforward = config.get("dim_feedforward", 1024)
        self.dropout = config.get("dropout", 0.0)
        self.num_classes = config.get("num_classes", 80)
        self.num_feature_levels = config.get("num_feature_levels", 3)

        # Decoder layers
        self.layers = nn.ModuleList([
            RTDETRDecoderLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                num_feature_levels=self.num_feature_levels,
            )
            for _ in range(self.num_layers)
        ])

        # Classification heads (shared or per-layer)
        self.class_heads = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.num_classes)
            for _ in range(self.num_layers)
        ])

        # Box refinement heads (per-layer for iterative refinement)
        self.bbox_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 4),
            )
            for _ in range(self.num_layers)
        ])

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.init_weights_vit()

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with iterative box refinement.

        Args:
            query: Initial queries (B, num_queries, hidden_dim).
            reference_points: Initial reference boxes (B, num_queries, 4).
            memory: Flattened encoder features (B, N_total, hidden_dim).
            spatial_shapes: List of (H, W) per level.

        Returns:
            Dictionary with:
                pred_logits: Class predictions from last layer (B, num_queries, num_classes).
                pred_boxes: Box predictions from last layer (B, num_queries, 4).
                aux_outputs: List of dicts with intermediate layer predictions.
        """
        intermediate_classes = []
        intermediate_boxes = []

        current_ref = reference_points

        for i, layer in enumerate(self.layers):
            query = layer(query, current_ref, memory, spatial_shapes)

            # Predict class and box refinement
            output = self.norm(query)
            cls_pred = self.class_heads[i](output)
            box_delta = self.bbox_heads[i](output)

            # Iterative box refinement: add delta to reference points
            refined_ref = (current_ref + box_delta).sigmoid()
            current_ref = refined_ref.detach()

            intermediate_classes.append(cls_pred)
            intermediate_boxes.append(refined_ref)

        # Build auxiliary outputs for deep supervision
        aux_outputs = []
        for cls_pred, box_pred in zip(intermediate_classes[:-1], intermediate_boxes[:-1]):
            aux_outputs.append({
                "pred_logits": cls_pred,
                "pred_boxes": box_pred,
            })

        return {
            "pred_logits": intermediate_classes[-1],
            "pred_boxes": intermediate_boxes[-1],
            "aux_outputs": aux_outputs,
        }


class RTDETR(WeightInitMixin, NexusModule):
    """RT-DETR: Real-Time Detection Transformer.

    End-to-end real-time object detection model combining a CNN backbone,
    hybrid encoder for multi-scale feature processing, IoU-aware query
    selection, and a transformer decoder with deformable attention.

    Config:
        num_classes (int): Number of object classes. Required.
        hidden_dim (int): Hidden dimension. Default: 256.
        num_queries (int): Number of detection queries. Default: 300.
        num_encoder_layers (int): Encoder self-attention layers per scale. Default: 1.
        num_decoder_layers (int): Decoder transformer layers. Default: 6.
        num_heads (int): Number of attention heads. Default: 8.
        dim_feedforward (int): FFN hidden dimension. Default: 1024.
        dropout (float): Dropout rate. Default: 0.0.
        num_feature_levels (int): Number of multi-scale levels. Default: 3.
        in_channels (list): Backbone output channels per level. Default: [512, 1024, 2048].
        backbone_type (str): Backbone type identifier. Default: "resnet50".

    Example:
        >>> config = {"num_classes": 80, "hidden_dim": 256, "num_queries": 300}
        >>> model = RTDETR(config)
        >>> features = [
        ...     torch.randn(2, 512, 40, 40),
        ...     torch.randn(2, 1024, 20, 20),
        ...     torch.randn(2, 2048, 10, 10),
        ... ]
        >>> output = model(features)
        >>> output["pred_logits"].shape
        torch.Size([2, 300, 80])
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.num_classes = config["num_classes"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_queries = config.get("num_queries", 300)
        self.num_feature_levels = config.get("num_feature_levels", 3)

        # Hybrid encoder
        self.encoder = HybridEncoder(config)

        # IoU-aware query selection
        self.query_selection = IoUAwareQuerySelection(
            hidden_dim=self.hidden_dim,
            num_queries=self.num_queries,
            num_classes=self.num_classes,
        )

        # Decoder
        self.decoder = RTDETRDecoder(config)

    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for object detection.

        Args:
            multi_scale_features: List of multi-scale feature maps from backbone,
                each of shape (B, C_i, H_i, W_i). Typically 3 levels from a
                CNN backbone (e.g., ResNet C3, C4, C5).

        Returns:
            Dictionary with:
                pred_logits: Class predictions (B, num_queries, num_classes).
                pred_boxes: Box predictions (B, num_queries, 4).
                aux_outputs: Intermediate predictions for deep supervision.
                encoder_features: Encoded multi-scale features.
        """
        # Encode multi-scale features
        encoded_features, spatial_shapes = self.encoder(multi_scale_features)

        # Flatten encoder features for decoder
        memory = torch.cat(
            [feat.flatten(2).transpose(1, 2) for feat in encoded_features], dim=1
        )  # (B, N_total, hidden_dim)

        # Select high-quality queries
        selected_features, reference_points, selection_scores = self.query_selection(memory)

        # Decode
        decoder_output = self.decoder(
            query=selected_features,
            reference_points=reference_points,
            memory=memory,
            spatial_shapes=spatial_shapes,
        )

        return {
            "pred_logits": decoder_output["pred_logits"],
            "pred_boxes": decoder_output["pred_boxes"],
            "aux_outputs": decoder_output["aux_outputs"],
            "encoder_features": encoded_features,
            "selection_scores": selection_scores,
        }


__all__ = [
    "RTDETR",
    "HybridEncoder",
    "RTDETRDecoder",
    "IoUAwareQuerySelection",
    "DeformableAttention",
]
