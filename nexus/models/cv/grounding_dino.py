"""
Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

Implementation of Grounding DINO, an open-set object detection model that combines
the DINO detector with grounded pre-training to detect arbitrary objects described
by human language inputs. It fuses language and vision features at multiple scales
via cross-modality attention and uses language-guided query selection.

Reference:
    Liu, S., Zeng, Z., Ren, T., et al. (2023).
    "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection."
    arXiv:2303.05499 (ECCV 2024)

Key Components:
    - TextBackbone: Transformer-based text feature extractor
    - ImageBackbone: ViT/Swin-based image feature extractor with multi-scale outputs
    - CrossModalFusion: Bidirectional cross-attention between image and text features
    - LanguageGuidedQuerySelection: Text-conditioned query initialization
    - GroundingDINO: Full open-set detection model

Architecture Details:
    - Image features are extracted at multiple scales via a backbone + neck
    - Text features are extracted via a transformer encoder
    - Cross-modal fusion applies bidirectional attention at each feature level
    - Language-guided query selection scores encoder features against text to
      select the most relevant queries for detection
    - Decoder refines queries with deformable cross-attention to fused features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, List, Tuple

from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin


class TextBackbone(WeightInitMixin, NexusModule):
    """Text feature extractor using a transformer encoder.

    Encodes text token sequences into contextual feature representations.
    Uses a lightweight transformer encoder with positional embeddings and
    causal-free (bidirectional) attention.

    Args:
        config: Configuration dictionary with keys:
            text_dim (int): Text embedding dimension. Default: 256.
            text_vocab_size (int): Vocabulary size. Default: 30522.
            text_max_length (int): Maximum sequence length. Default: 256.
            text_num_layers (int): Number of transformer layers. Default: 6.
            text_num_heads (int): Number of attention heads. Default: 8.
            text_mlp_ratio (float): MLP expansion ratio. Default: 4.0.
            text_dropout (float): Dropout rate. Default: 0.1.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.text_dim = config.get("text_dim", 256)
        self.vocab_size = config.get("text_vocab_size", 30522)
        self.max_length = config.get("text_max_length", 256)
        self.num_layers = config.get("text_num_layers", 6)
        self.num_heads = config.get("text_num_heads", 8)
        self.mlp_ratio = config.get("text_mlp_ratio", 4.0)
        self.dropout = config.get("text_dropout", 0.1)

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.text_dim)
        self.position_embedding = nn.Embedding(self.max_length, self.text_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.text_dim,
            nhead=self.num_heads,
            dim_feedforward=int(self.text_dim * self.mlp_ratio),
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.final_norm = nn.LayerNorm(self.text_dim)

        self.init_weights_vit()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode text tokens into feature representations.

        Args:
            input_ids: Token IDs of shape (B, L).
            attention_mask: Binary mask (B, L) with 1 for valid tokens, 0 for padding.

        Returns:
            Dictionary with:
                text_features: Encoded features (B, L, text_dim).
                text_mask: Attention mask (B, L).
        """
        B, L = input_ids.shape

        # Embeddings
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Build key padding mask for transformer (True = ignore)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        # Encode
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.final_norm(x)

        if attention_mask is None:
            attention_mask = torch.ones(B, L, device=input_ids.device)

        return {
            "text_features": x,
            "text_mask": attention_mask,
        }


class ImageBackbone(WeightInitMixin, NexusModule):
    """Multi-scale image feature extractor.

    Extracts image features at multiple spatial resolutions using a backbone
    network followed by a Feature Pyramid Network (FPN)-style neck. Produces
    feature maps at different scales for multi-scale detection.

    Args:
        config: Configuration dictionary with keys:
            hidden_dim (int): Output feature dimension. Default: 256.
            img_size (int): Input image size. Default: 800.
            backbone_channels (list): Channel dimensions for each backbone stage.
                Default: [256, 512, 1024, 2048].
            num_feature_levels (int): Number of output feature levels. Default: 4.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_feature_levels = config.get("num_feature_levels", 4)
        backbone_channels = config.get("backbone_channels", [256, 512, 1024, 2048])

        # Simple backbone with progressive downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Feature extraction stages
        self.stages = nn.ModuleList()
        in_ch = 64
        for ch in backbone_channels:
            self.stages.append(nn.Sequential(
                nn.Conv2d(in_ch, ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = ch

        # Lateral connections (project backbone features to hidden_dim)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, self.hidden_dim, kernel_size=1)
            for ch in backbone_channels[-self.num_feature_levels:]
        ])

        # Smooth layers after lateral merge
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
            for _ in range(self.num_feature_levels)
        ])

        self.init_weights_vision()

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale image features.

        Args:
            images: Input images (B, 3, H, W).

        Returns:
            List of feature maps from fine to coarse, each (B, hidden_dim, H_i, W_i).
        """
        x = self.stem(images)

        # Extract backbone stage features
        stage_features = []
        for stage in self.stages:
            x = stage(x)
            stage_features.append(x)

        # FPN: apply lateral connections on the last num_feature_levels stages
        used_features = stage_features[-self.num_feature_levels:]
        laterals = [
            conv(feat) for conv, feat in zip(self.lateral_convs, used_features)
        ]

        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:],
                mode="bilinear", align_corners=False,
            )
            laterals[i] = laterals[i] + upsampled

        # Smooth
        output = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        return output


class CrossModalFusion(NexusModule):
    """Bidirectional cross-attention fusion between image and text features.

    Applies cross-modal attention in both directions:
    1. Image attends to text (text-guided image feature enhancement)
    2. Text attends to image (image-aware text feature refinement)

    This bidirectional fusion is applied at each feature level to create
    language-aware visual features for open-set detection.

    Args:
        hidden_dim (int): Feature dimension. Default: 256.
        text_dim (int): Text feature dimension. Default: 256.
        num_heads (int): Number of attention heads. Default: 8.
        dropout (float): Dropout rate. Default: 0.0.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        text_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim

        # Project text dim to hidden_dim if they differ
        self.text_proj = nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity()
        self.text_proj_back = nn.Linear(hidden_dim, text_dim) if text_dim != hidden_dim else nn.Identity()

        # Image attends to text (vision queries, text keys/values)
        self.image_to_text_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_img = nn.LayerNorm(hidden_dim)

        # Text attends to image (text queries, vision keys/values)
        self.text_to_image_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_text = nn.LayerNorm(hidden_dim)

        # FFN for both modalities
        self.ffn_img = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_img_ffn = nn.LayerNorm(hidden_dim)

        self.ffn_text = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_text_ffn = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional cross-modal fusion.

        Args:
            image_features: Flattened image features (B, N_img, hidden_dim).
            text_features: Text features (B, N_text, text_dim).
            text_mask: Text attention mask (B, N_text), 1=valid, 0=padding.

        Returns:
            Tuple of (fused_image_features, fused_text_features).
        """
        # Project text to hidden dim
        text_proj = self.text_proj(text_features)

        # Build key padding mask for cross-attention
        text_key_padding_mask = None
        if text_mask is not None:
            text_key_padding_mask = (text_mask == 0)

        # Image attends to text
        img_attn_out, _ = self.image_to_text_attn(
            query=image_features,
            key=text_proj,
            value=text_proj,
            key_padding_mask=text_key_padding_mask,
        )
        image_features = self.norm_img(image_features + img_attn_out)
        image_features = self.norm_img_ffn(image_features + self.ffn_img(image_features))

        # Text attends to image
        text_attn_out, _ = self.text_to_image_attn(
            query=text_proj,
            key=image_features,
            value=image_features,
        )
        text_proj = self.norm_text(text_proj + text_attn_out)
        text_proj = self.norm_text_ffn(text_proj + self.ffn_text(text_proj))

        # Project text back to original dim
        text_features = self.text_proj_back(text_proj)

        return image_features, text_features


class LanguageGuidedQuerySelection(NexusModule):
    """Language-guided query selection for open-set detection.

    Selects initial decoder queries by scoring encoder features against text
    features. Queries that strongly correspond to the language description
    receive higher scores and are selected as initial decoder inputs.

    This enables open-set detection by conditioning the query selection on
    arbitrary text descriptions rather than fixed class embeddings.

    Args:
        hidden_dim (int): Feature dimension. Default: 256.
        text_dim (int): Text feature dimension. Default: 256.
        num_queries (int): Number of queries to select. Default: 900.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        text_dim: int = 256,
        num_queries: int = 900,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # Project text features for scoring
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Feature scoring head
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Reference point prediction
        self.ref_point_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

    def forward(
        self,
        memory: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select queries conditioned on text features.

        Args:
            memory: Encoder features (B, N, hidden_dim).
            text_features: Text features (B, L, text_dim).
            text_mask: Text mask (B, L), 1=valid.

        Returns:
            Tuple of:
                selected_queries: Selected features (B, num_queries, hidden_dim).
                reference_points: Initial bbox predictions (B, num_queries, 4).
                selection_scores: Per-query scores (B, num_queries).
        """
        B, N, C = memory.shape

        # Project text features
        text_proj = self.text_proj(text_features)  # (B, L, hidden_dim)

        # Apply mask to text features
        if text_mask is not None:
            text_proj = text_proj * text_mask.unsqueeze(-1)

        # Score encoder features against text
        # Compute similarity between each spatial feature and pooled text
        text_pooled = text_proj.mean(dim=1, keepdim=True)  # (B, 1, hidden_dim)

        memory_scored = self.score_head(memory)
        # Dot product similarity
        scores = (memory_scored * text_pooled).sum(dim=-1)  # (B, N)
        scores = scores / math.sqrt(C)

        # Select top-K
        num_select = min(self.num_queries, N)
        topk_scores, topk_indices = scores.topk(num_select, dim=1)

        # Gather selected features
        topk_indices_expand = topk_indices.unsqueeze(-1).expand(-1, -1, C)
        selected_queries = torch.gather(memory, 1, topk_indices_expand)

        # Predict reference points
        reference_points = self.ref_point_head(selected_queries)

        return selected_queries, reference_points, topk_scores


class GroundingDINODecoder(NexusModule):
    """Decoder for Grounding DINO with text-enhanced cross-attention.

    Each decoder layer performs self-attention among queries, cross-attention
    to fused image-text features, and text-enhanced classification. The
    decoder iteratively refines both detection boxes and text alignment scores.

    Args:
        hidden_dim (int): Feature dimension. Default: 256.
        num_heads (int): Number of attention heads. Default: 8.
        num_layers (int): Number of decoder layers. Default: 6.
        dim_feedforward (int): FFN hidden dimension. Default: 2048.
        dropout (float): Dropout rate. Default: 0.0.
        num_classes (int): Number of output classes (unused in open-set, kept for
            compatibility). Default: 256.
        text_dim (int): Text feature dimension. Default: 256.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        text_dim: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Decoder layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True,
                ),
                "norm1": nn.LayerNorm(hidden_dim),
                "cross_attn": nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True,
                ),
                "norm2": nn.LayerNorm(hidden_dim),
                "text_cross_attn": nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True,
                ),
                "norm_text": nn.LayerNorm(hidden_dim),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, dim_feedforward),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, hidden_dim),
                    nn.Dropout(dropout),
                ),
                "norm3": nn.LayerNorm(hidden_dim),
            }))

        # Text projection for alignment scoring
        self.text_proj = nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity()

        # Box prediction heads for iterative refinement
        self.bbox_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4),
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Decode queries with text-enhanced cross-attention.

        Args:
            query: Initial queries (B, num_queries, hidden_dim).
            reference_points: Initial reference boxes (B, num_queries, 4).
            memory: Fused image features (B, N_img, hidden_dim).
            text_features: Text features (B, L, text_dim).
            text_mask: Text attention mask (B, L).

        Returns:
            Dictionary with:
                query_features: Final query features (B, num_queries, hidden_dim).
                pred_boxes: Refined box predictions (B, num_queries, 4).
                aux_boxes: Intermediate box predictions.
        """
        text_proj = self.text_proj(text_features)
        text_key_padding_mask = (text_mask == 0) if text_mask is not None else None

        intermediate_boxes = []
        current_ref = reference_points

        for i, layer in enumerate(self.layers):
            # Self-attention
            attn_out, _ = layer["self_attn"](query, query, query)
            query = layer["norm1"](query + attn_out)

            # Cross-attention to image features
            attn_out, _ = layer["cross_attn"](query, memory, memory)
            query = layer["norm2"](query + attn_out)

            # Cross-attention to text features
            attn_out, _ = layer["text_cross_attn"](
                query, text_proj, text_proj,
                key_padding_mask=text_key_padding_mask,
            )
            query = layer["norm_text"](query + attn_out)

            # FFN
            query = layer["norm3"](query + layer["ffn"](query))

            # Iterative box refinement
            box_delta = self.bbox_heads[i](query)
            refined_ref = (current_ref + box_delta).sigmoid()
            current_ref = refined_ref.detach()
            intermediate_boxes.append(refined_ref)

        query = self.norm(query)

        return {
            "query_features": query,
            "pred_boxes": intermediate_boxes[-1],
            "aux_boxes": intermediate_boxes[:-1],
        }


class GroundingDINO(WeightInitMixin, NexusModule):
    """Grounding DINO: Open-Set Object Detection with Language Grounding.

    Combines image and text understanding for detecting arbitrary objects
    described by natural language. The model fuses visual and linguistic
    features at multiple scales and uses language-guided query selection
    for open-vocabulary detection.

    Config:
        num_queries (int): Number of detection queries. Default: 900.
        hidden_dim (int): Hidden dimension for fusion and decoder. Default: 256.
        num_feature_levels (int): Number of multi-scale levels. Default: 4.
        text_dim (int): Text feature dimension. Default: 256.
        text_vocab_size (int): Text vocabulary size. Default: 30522.
        text_num_layers (int): Text encoder layers. Default: 6.
        text_num_heads (int): Text attention heads. Default: 8.
        num_decoder_layers (int): Decoder layers. Default: 6.
        num_heads (int): Decoder attention heads. Default: 8.
        dim_feedforward (int): FFN hidden dimension. Default: 2048.
        dropout (float): Dropout rate. Default: 0.0.
        num_fusion_layers (int): Number of cross-modal fusion layers per level. Default: 1.

    Example:
        >>> config = {
        ...     "num_queries": 900,
        ...     "hidden_dim": 256,
        ...     "text_dim": 256,
        ... }
        >>> model = GroundingDINO(config)
        >>> images = torch.randn(2, 3, 800, 800)
        >>> input_ids = torch.randint(0, 30522, (2, 20))
        >>> attention_mask = torch.ones(2, 20)
        >>> output = model(images, input_ids=input_ids, attention_mask=attention_mask)
        >>> output["pred_boxes"].shape
        torch.Size([2, 900, 4])
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.num_queries = config.get("num_queries", 900)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_feature_levels = config.get("num_feature_levels", 4)
        self.text_dim = config.get("text_dim", 256)
        num_fusion_layers = config.get("num_fusion_layers", 1)

        # Text backbone
        self.text_backbone = TextBackbone(config)

        # Image backbone
        self.image_backbone = ImageBackbone(config)

        # Cross-modal fusion (one per feature level, potentially multiple layers)
        self.fusion_layers = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            level_fusion = nn.ModuleList([
                CrossModalFusion(
                    hidden_dim=self.hidden_dim,
                    text_dim=self.text_dim,
                    num_heads=config.get("num_heads", 8),
                    dropout=config.get("dropout", 0.0),
                )
                for _ in range(num_fusion_layers)
            ])
            self.fusion_layers.append(level_fusion)

        # Language-guided query selection
        self.query_selection = LanguageGuidedQuerySelection(
            hidden_dim=self.hidden_dim,
            text_dim=self.text_dim,
            num_queries=self.num_queries,
        )

        # Decoder
        self.decoder = GroundingDINODecoder(
            hidden_dim=self.hidden_dim,
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_decoder_layers", 6),
            dim_feedforward=config.get("dim_feedforward", 2048),
            dropout=config.get("dropout", 0.0),
            text_dim=self.text_dim,
        )

        # Text-image alignment head (for classification via text similarity)
        self.alignment_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.text_alignment_proj = nn.Linear(self.text_dim, self.hidden_dim)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for open-set object detection.

        Args:
            images: Input images (B, 3, H, W).
            input_ids: Text token IDs (B, L). Required if text_features not provided.
            attention_mask: Text attention mask (B, L).
            text_features: Pre-computed text features (B, L, text_dim). If provided,
                skips the text backbone.

        Returns:
            Dictionary with:
                pred_boxes: Box predictions (B, num_queries, 4).
                pred_logits: Text alignment scores (B, num_queries, L).
                aux_boxes: Intermediate box predictions from decoder layers.
                text_features: Encoded text features.
                image_features: Multi-scale fused image features.
        """
        # Extract text features
        if text_features is None:
            assert input_ids is not None, "Must provide either input_ids or text_features"
            text_output = self.text_backbone(input_ids, attention_mask)
            text_features = text_output["text_features"]
            text_mask = text_output["text_mask"]
        else:
            text_mask = attention_mask

        # Extract multi-scale image features
        image_features = self.image_backbone(images)

        # Cross-modal fusion at each feature level
        fused_image_features = []
        current_text_features = text_features
        for lvl in range(min(len(image_features), self.num_feature_levels)):
            B, C, H, W = image_features[lvl].shape
            img_flat = image_features[lvl].flatten(2).transpose(1, 2)  # (B, H*W, C)

            for fusion in self.fusion_layers[lvl]:
                img_flat, current_text_features = fusion(
                    img_flat, current_text_features, text_mask,
                )

            fused_img = img_flat.transpose(1, 2).view(B, C, H, W)
            fused_image_features.append(fused_img)

        # Flatten all levels for query selection and decoding
        memory = torch.cat(
            [feat.flatten(2).transpose(1, 2) for feat in fused_image_features], dim=1
        )

        # Language-guided query selection
        selected_queries, reference_points, selection_scores = self.query_selection(
            memory, current_text_features, text_mask,
        )

        # Decode with text-enhanced cross-attention
        decoder_output = self.decoder(
            query=selected_queries,
            reference_points=reference_points,
            memory=memory,
            text_features=current_text_features,
            text_mask=text_mask,
        )

        # Compute text alignment scores (open-set classification)
        query_features = decoder_output["query_features"]  # (B, Nq, hidden_dim)
        query_aligned = self.alignment_head(query_features)
        text_aligned = self.text_alignment_proj(current_text_features)  # (B, L, hidden_dim)

        # Dot product between queries and text tokens gives per-token alignment
        pred_logits = torch.einsum(
            "bqd,bld->bql", query_aligned, text_aligned,
        ) / math.sqrt(self.hidden_dim)

        return {
            "pred_boxes": decoder_output["pred_boxes"],
            "pred_logits": pred_logits,
            "aux_boxes": decoder_output["aux_boxes"],
            "text_features": current_text_features,
            "image_features": fused_image_features,
            "selection_scores": selection_scores,
        }


__all__ = [
    "GroundingDINO",
    "TextBackbone",
    "ImageBackbone",
    "CrossModalFusion",
    "LanguageGuidedQuerySelection",
    "GroundingDINODecoder",
]
