"""
YOLO-World: Real-Time Open-Vocabulary Object Detection

Implementation of YOLO-World, an open-vocabulary object detector that combines
YOLO's real-time detection with CLIP-style vision-language learning to detect
any object described by text prompts.

Reference:
    Cheng, T., Song, L., Ge, Y., et al. (2024).
    "YOLO-World: Real-Time Open-Vocabulary Object Detection."
    CVPR 2024
    arXiv:2401.17270

Key Components:
    - YOLOWorldBackbone: CSPDarknet or similar backbone
    - TextEncoder: CLIP-style text encoder for class embeddings
    - RepVL-PAN: Re-parameterizable Vision-Language Path Aggregation Network
    - YOLOWorld: Full open-vocabulary detector

Architecture Details:
    - Vision-language alignment through contrastive learning
    - Text prompts encode object categories dynamically
    - No fixed vocabulary - can detect arbitrary object classes
    - Maintains real-time inference speed (~50 FPS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple

from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin


class YOLOWorldTextEncoder(WeightInitMixin, NexusModule):
    """CLIP-style text encoder for YOLO-World.

    Encodes text prompts (class names) into embeddings that are used
    for open-vocabulary detection.

    Args:
        config: Configuration dictionary with keys:
            vocab_size (int): Vocabulary size. Default: 49408.
            max_seq_len (int): Maximum sequence length. Default: 77.
            embed_dim (int): Embedding dimension. Default: 512.
            depth (int): Number of transformer layers. Default: 6.
            num_heads (int): Number of attention heads. Default: 8.
            output_dim (int): Output projection dimension. Default: 512.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.vocab_size = config.get("vocab_size", 49408)
        self.max_seq_len = config.get("max_seq_len", 77)
        self.embed_dim = config.get("embed_dim", 512)
        self.depth = config.get("depth", 6)
        self.num_heads = config.get("num_heads", 8)
        self.output_dim = config.get("output_dim", 512)

        # Token and position embeddings
        self.token_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, self.embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                dim_feedforward=self.embed_dim * 4,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)
        self.projection = nn.Linear(self.embed_dim, self.output_dim, bias=False)

        # Initialize
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text to embeddings.

        Args:
            input_ids: Token indices (B, seq_len).
            attention_mask: Attention mask (B, seq_len).

        Returns:
            Text embeddings (B, output_dim).
        """
        B, seq_len = input_ids.shape

        # Token and position embeddings
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :seq_len]

        # Prepare mask for transformer
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()

        # Transformer encoding
        for block in self.blocks:
            x = block(x, src_key_padding_mask=src_key_padding_mask)

        x = self.norm(x)

        # Take [EOS] token (last valid token)
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
            pooled = x[torch.arange(B, device=x.device), seq_lengths]
        else:
            pooled = x[:, -1]

        # Project and normalize
        features = self.projection(pooled)
        features = F.normalize(features, dim=-1)

        return features


class RepVLPAN(NexusModule):
    """Re-parameterizable Vision-Language Path Aggregation Network.

    Fuses multi-scale visual features with text embeddings for open-vocabulary
    detection.

    Args:
        in_channels (List[int]): Input channels for each scale.
        out_channels (int): Output channels.
        text_dim (int): Text embedding dimension.
        num_scales (int): Number of feature scales.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        text_dim: int = 512,
        num_scales: int = 3,
    ):
        super().__init__()

        self.num_scales = num_scales

        # Text-to-vision projection for each scale
        self.text_projections = nn.ModuleList([
            nn.Linear(text_dim, out_channels)
            for _ in range(num_scales)
        ])

        # Vision feature convolutions
        self.vision_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for in_ch in in_channels
        ])

        # Fusion layers
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_scales)
        ])

    def forward(
        self,
        vision_features: List[torch.Tensor],
        text_features: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Fuse vision and text features.

        Args:
            vision_features: List of multi-scale features [(B, C_i, H_i, W_i), ...].
            text_features: Text embeddings (B, num_classes, text_dim).

        Returns:
            List of fused features [(B, out_channels, H_i, W_i), ...].
        """
        outputs = []

        for i in range(self.num_scales):
            # Project vision features
            vis_feat = self.vision_convs[i](vision_features[i])

            # Project text features to vision dimension
            text_proj = self.text_projections[i](text_features)  # (B, num_classes, out_channels)

            # Text-guided attention (simplified)
            # In practice, this would use cross-attention
            B, C, H, W = vis_feat.shape
            vis_flat = vis_feat.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)

            # Compute similarity scores
            scores = torch.bmm(vis_flat, text_proj.transpose(1, 2))  # (B, H*W, num_classes)
            attn_weights = F.softmax(scores, dim=-1)

            # Apply attention
            attended = torch.bmm(attn_weights, text_proj)  # (B, H*W, C)
            attended = attended.permute(0, 2, 1).view(B, C, H, W)

            # Fuse with vision features
            fused = vis_feat + attended
            fused = self.fusion_convs[i](fused)

            outputs.append(fused)

        return outputs


class YOLOWorldHead(NexusModule):
    """Detection head for YOLO-World.

    Predicts bounding boxes and classification scores using text embeddings
    as dynamic class prototypes.

    Args:
        in_channels (int): Input channels.
        num_classes (int): Number of classes (dynamic based on text prompts).
        text_dim (int): Text embedding dimension.
        num_anchors (int): Number of anchors per location.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        text_dim: int = 512,
        num_anchors: int = 3,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Box regression head
        self.bbox_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1),
        )

        # Objectness head
        self.obj_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 1, kernel_size=1),
        )

        # Classification via text similarity
        self.cls_proj = nn.Conv2d(in_channels, text_dim, kernel_size=1)

    def forward(
        self,
        features: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict detections.

        Args:
            features: Feature map (B, in_channels, H, W).
            text_embeddings: Class embeddings (B, num_classes, text_dim).

        Returns:
            Dictionary with bbox, objectness, and class predictions.
        """
        # Bounding box regression
        bbox_pred = self.bbox_head(features)  # (B, num_anchors*4, H, W)

        # Objectness prediction
        obj_pred = self.obj_head(features)  # (B, num_anchors, H, W)

        # Classification via text similarity
        cls_features = self.cls_proj(features)  # (B, text_dim, H, W)
        B, C, H, W = cls_features.shape

        # Reshape for similarity computation
        cls_features = cls_features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        cls_features = F.normalize(cls_features, dim=-1)

        # Compute similarity with text embeddings
        text_norm = F.normalize(text_embeddings, dim=-1)
        cls_pred = torch.bmm(cls_features, text_norm.transpose(1, 2))  # (B, H*W, num_classes)
        cls_pred = cls_pred.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, num_classes, H, W)

        return {
            "bbox": bbox_pred,
            "objectness": obj_pred,
            "classification": cls_pred,
        }


class YOLOWorld(WeightInitMixin, NexusModule):
    """YOLO-World: Real-Time Open-Vocabulary Object Detector.

    Combines YOLO architecture with vision-language learning for detecting
    arbitrary object classes specified by text prompts.

    Config:
        # Backbone config
        backbone_channels (List[int]): Output channels for each backbone scale.

        # Text encoder config
        vocab_size (int): Vocabulary size. Default: 49408.
        text_embed_dim (int): Text embedding dimension. Default: 512.
        text_depth (int): Text encoder depth. Default: 6.

        # Neck config
        neck_channels (int): PAN output channels. Default: 256.

        # Head config
        num_classes (int): Default number of classes. Default: 80.
        num_anchors (int): Number of anchors per location. Default: 3.

    Example:
        >>> config = {
        ...     "backbone_channels": [256, 512, 1024],
        ...     "text_embed_dim": 512,
        ...     "neck_channels": 256,
        ...     "num_classes": 80,
        ... }
        >>> model = YOLOWorld(config)
        >>> images = torch.randn(2, 3, 640, 640)
        >>> text_ids = torch.randint(0, 49408, (2, 10, 77))  # 10 classes
        >>> output = model(images, text_ids)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.backbone_channels = config.get("backbone_channels", [256, 512, 1024])
        self.neck_channels = config.get("neck_channels", 256)
        self.text_dim = config.get("text_embed_dim", 512)
        self.num_classes = config.get("num_classes", 80)

        # Text encoder
        text_config = {
            "vocab_size": config.get("vocab_size", 49408),
            "embed_dim": config.get("text_embed_dim", 512),
            "depth": config.get("text_depth", 6),
            "num_heads": config.get("text_num_heads", 8),
            "output_dim": self.text_dim,
        }
        self.text_encoder = YOLOWorldTextEncoder(text_config)

        # Simple backbone (in practice, use CSPDarknet or similar)
        self.backbone = self._build_simple_backbone()

        # RepVL-PAN neck
        self.neck = RepVLPAN(
            in_channels=self.backbone_channels,
            out_channels=self.neck_channels,
            text_dim=self.text_dim,
            num_scales=len(self.backbone_channels),
        )

        # Detection heads (one per scale)
        self.heads = nn.ModuleList([
            YOLOWorldHead(
                in_channels=self.neck_channels,
                num_classes=self.num_classes,
                text_dim=self.text_dim,
                num_anchors=config.get("num_anchors", 3),
            )
            for _ in range(len(self.backbone_channels))
        ])

    def _build_simple_backbone(self) -> nn.ModuleList:
        """Build a simple multi-scale backbone (placeholder)."""
        # In practice, use CSPDarknet53 or similar
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),
            ),
            nn.Sequential(
                nn.Conv2d(64, 256, 3, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, 3, 2, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 1024, 3, 2, 1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
            ),
        ])

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text prompts to class embeddings.

        Args:
            input_ids: Token indices (B, num_classes, seq_len).
            attention_mask: Attention mask (B, num_classes, seq_len).

        Returns:
            Class embeddings (B, num_classes, text_dim).
        """
        B, num_classes, seq_len = input_ids.shape

        # Flatten batch and classes
        input_ids_flat = input_ids.view(B * num_classes, seq_len)
        mask_flat = attention_mask.view(B * num_classes, seq_len) if attention_mask is not None else None

        # Encode
        embeddings = self.text_encoder(input_ids_flat, mask_flat)

        # Reshape back
        embeddings = embeddings.view(B, num_classes, -1)

        return embeddings

    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass.

        Args:
            images: Input images (B, 3, H, W).
            text_input_ids: Text token IDs (B, num_classes, seq_len).
            text_attention_mask: Text attention mask (B, num_classes, seq_len).

        Returns:
            Dictionary with multi-scale predictions.
        """
        # Encode text
        text_embeddings = self.encode_text(text_input_ids, text_attention_mask)

        # Backbone forward
        x = images
        backbone_features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i >= len(self.backbone) - len(self.backbone_channels):
                backbone_features.append(x)

        # Neck forward
        neck_features = self.neck(backbone_features, text_embeddings)

        # Head predictions
        predictions = []
        for i, head in enumerate(self.heads):
            pred = head(neck_features[i], text_embeddings)
            predictions.append(pred)

        return {
            "predictions": predictions,
            "text_embeddings": text_embeddings,
            "features": neck_features,
        }


__all__ = [
    "YOLOWorld",
    "YOLOWorldTextEncoder",
    "RepVLPAN",
    "YOLOWorldHead",
]
