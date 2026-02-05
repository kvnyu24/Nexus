"""
SigLIP: Sigmoid Loss for Language-Image Pre-training

Implementation of SigLIP, an efficient vision-language pre-training approach that
replaces the softmax-based contrastive loss in CLIP with a sigmoid-based loss.
This eliminates the need for global batch statistics and enables better scaling.

Reference:
    Zhai, X., Mustafa, B., Kolesnikov, A., et al. (2023).
    "Sigmoid Loss for Language Image Pre-Training."
    ICCV 2023
    arXiv:2303.15343

Key Components:
    - SigLIPVisionEncoder: ViT-based image encoder
    - SigLIPTextEncoder: Transformer-based text encoder
    - SigLIPLoss: Sigmoid contrastive loss (no global softmax normalization)
    - SigLIP: Full vision-language model

Architecture Details:
    - Vision encoder: Standard ViT with optional modifications
    - Text encoder: Transformer encoder with causal or bidirectional attention
    - Sigmoid loss: Per-sample binary classification, no batch normalization needed
    - More memory-efficient than CLIP due to lack of global normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple

from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin


class SigLIPVisionEncoder(WeightInitMixin, NexusModule):
    """Vision Transformer encoder for SigLIP.

    Encodes images into fixed-dimensional feature vectors using a Vision Transformer.

    Args:
        config: Configuration dictionary with keys:
            img_size (int): Input image resolution. Default: 224.
            patch_size (int): Patch size. Default: 16.
            in_channels (int): Number of input channels. Default: 3.
            embed_dim (int): Embedding dimension. Default: 768.
            depth (int): Number of transformer layers. Default: 12.
            num_heads (int): Number of attention heads. Default: 12.
            mlp_ratio (float): MLP hidden dim ratio. Default: 4.0.
            dropout (float): Dropout rate. Default: 0.0.
            output_dim (int): Output projection dimension. Default: 512.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.img_size = config.get("img_size", 224)
        self.patch_size = config.get("patch_size", 16)
        self.in_channels = config.get("in_channels", 3)
        self.embed_dim = config.get("embed_dim", 768)
        self.depth = config.get("depth", 12)
        self.num_heads = config.get("num_heads", 12)
        self.mlp_ratio = config.get("mlp_ratio", 4.0)
        self.dropout = config.get("dropout", 0.0)
        self.output_dim = config.get("output_dim", 512)

        self.grid_size = self.img_size // self.patch_size
        self.num_patches = self.grid_size ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            self.in_channels, self.embed_dim,
            kernel_size=self.patch_size, stride=self.patch_size,
        )

        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim)
        )
        self.pos_drop = nn.Dropout(p=self.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
            )
            for _ in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        # Projection head to output dimension
        self.head = nn.Linear(self.embed_dim, self.output_dim, bias=False)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.init_weights_vit()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors.

        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            Normalized feature vectors of shape (B, output_dim).
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract CLS token and project
        cls_output = x[:, 0]
        features = self.head(cls_output)

        # L2 normalization
        features = F.normalize(features, dim=-1)

        return features


class VisionTransformerBlock(NexusModule):
    """Standard transformer block for vision encoder.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim expansion ratio.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, dim).

        Returns:
            Output tensor of shape (B, N, dim).
        """
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class SigLIPTextEncoder(WeightInitMixin, NexusModule):
    """Transformer-based text encoder for SigLIP.

    Encodes text sequences into fixed-dimensional feature vectors.

    Args:
        config: Configuration dictionary with keys:
            vocab_size (int): Vocabulary size. Default: 49408.
            max_seq_len (int): Maximum sequence length. Default: 77.
            embed_dim (int): Embedding dimension. Default: 512.
            depth (int): Number of transformer layers. Default: 12.
            num_heads (int): Number of attention heads. Default: 8.
            mlp_ratio (float): MLP hidden dim ratio. Default: 4.0.
            dropout (float): Dropout rate. Default: 0.0.
            output_dim (int): Output projection dimension. Default: 512.
            causal (bool): Use causal attention. Default: False.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.vocab_size = config.get("vocab_size", 49408)
        self.max_seq_len = config.get("max_seq_len", 77)
        self.embed_dim = config.get("embed_dim", 512)
        self.depth = config.get("depth", 12)
        self.num_heads = config.get("num_heads", 8)
        self.mlp_ratio = config.get("mlp_ratio", 4.0)
        self.dropout = config.get("dropout", 0.0)
        self.output_dim = config.get("output_dim", 512)
        self.causal = config.get("causal", False)

        # Token and position embeddings
        self.token_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.max_seq_len, self.embed_dim)
        )
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TextTransformerBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                causal=self.causal,
            )
            for _ in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        # Projection head
        self.head = nn.Linear(self.embed_dim, self.output_dim, bias=False)

        # Initialize weights
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)
        self.init_weights_vit()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text to feature vectors.

        Args:
            input_ids: Token indices of shape (B, seq_len).
            attention_mask: Attention mask of shape (B, seq_len).

        Returns:
            Normalized feature vectors of shape (B, output_dim).
        """
        B, seq_len = input_ids.shape

        # Token and position embeddings
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :seq_len]
        x = self.dropout_layer(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.norm(x)

        # Take features from the last token (EOS token) or pooled
        if attention_mask is not None:
            # Find last non-padding token for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1
            pooled = x[torch.arange(B, device=x.device), seq_lengths]
        else:
            # Take last token
            pooled = x[:, -1]

        # Project and normalize
        features = self.head(pooled)
        features = F.normalize(features, dim=-1)

        return features


class TextTransformerBlock(NexusModule):
    """Transformer block for text encoder.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim expansion ratio.
        dropout: Dropout rate.
        causal: Use causal attention mask.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()

        self.causal = causal
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

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, seq_len, dim).
            attention_mask: Attention mask of shape (B, seq_len).

        Returns:
            Output tensor of shape (B, seq_len, dim).
        """
        # Prepare attention mask
        attn_mask = None
        if self.causal:
            seq_len = x.shape[1]
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )

        # Prepare key padding mask
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class SigLIPLoss(NexusModule):
    """Sigmoid contrastive loss for vision-language pre-training.

    Unlike CLIP's softmax-based InfoNCE loss, SigLIP uses a sigmoid loss that
    treats each image-text pair as an independent binary classification problem.
    This removes the need for large batch sizes and global normalization.

    The loss is computed as:
        L = -log(sigmoid(t * logits)) for positive pairs
        L = -log(sigmoid(-t * logits)) for negative pairs

    where t is a learnable temperature parameter and logits are the dot products
    between image and text features.

    Args:
        init_temperature (float): Initial temperature value. Default: 10.0.
        init_bias (float): Initial bias value. Default: -10.0.
        learnable (bool): Whether temperature and bias are learnable. Default: True.
    """

    def __init__(
        self,
        init_temperature: float = 10.0,
        init_bias: float = -10.0,
        learnable: bool = True,
    ):
        super().__init__()

        if learnable:
            self.temperature = nn.Parameter(torch.tensor(init_temperature))
            self.bias = nn.Parameter(torch.tensor(init_bias))
        else:
            self.register_buffer("temperature", torch.tensor(init_temperature))
            self.register_buffer("bias", torch.tensor(init_bias))

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sigmoid contrastive loss.

        Args:
            image_features: Normalized image features (B, D).
            text_features: Normalized text features (B, D).

        Returns:
            Scalar loss value.
        """
        B = image_features.shape[0]
        device = image_features.device

        # Compute logits (dot product between all image-text pairs)
        logits = image_features @ text_features.T  # (B, B)

        # Apply temperature and bias
        logits = logits * self.temperature + self.bias

        # Create labels: positive pairs are on the diagonal
        labels = torch.eye(B, device=device, dtype=torch.float32)

        # Sigmoid loss: -log(sigmoid(logits)) for positive, -log(sigmoid(-logits)) for negative
        # Equivalent to binary cross-entropy with logits
        # Positive pairs: target=1, Negative pairs: target=0
        loss = F.binary_cross_entropy_with_logits(
            logits.flatten(),
            labels.flatten(),
            reduction="mean",
        )

        return loss


class SigLIP(WeightInitMixin, NexusModule):
    """SigLIP: Sigmoid Loss for Language-Image Pre-training.

    Full vision-language model that learns joint embeddings of images and text
    using sigmoid contrastive loss instead of softmax-based contrastive loss.

    Config:
        # Vision encoder config
        img_size (int): Input image resolution. Default: 224.
        patch_size (int): Patch size. Default: 16.
        vision_embed_dim (int): Vision encoder embedding dimension. Default: 768.
        vision_depth (int): Vision encoder depth. Default: 12.
        vision_num_heads (int): Vision encoder attention heads. Default: 12.

        # Text encoder config
        vocab_size (int): Vocabulary size. Default: 49408.
        max_seq_len (int): Maximum text sequence length. Default: 77.
        text_embed_dim (int): Text encoder embedding dimension. Default: 512.
        text_depth (int): Text encoder depth. Default: 12.
        text_num_heads (int): Text encoder attention heads. Default: 8.

        # Shared config
        output_dim (int): Projection output dimension. Default: 512.
        dropout (float): Dropout rate. Default: 0.0.

        # Loss config
        init_temperature (float): Initial loss temperature. Default: 10.0.
        init_bias (float): Initial loss bias. Default: -10.0.

    Example:
        >>> config = {
        ...     "img_size": 224,
        ...     "vision_embed_dim": 768,
        ...     "text_embed_dim": 512,
        ...     "output_dim": 512,
        ... }
        >>> model = SigLIP(config)
        >>> images = torch.randn(4, 3, 224, 224)
        >>> input_ids = torch.randint(0, 49408, (4, 77))
        >>> output = model(images, input_ids)
        >>> output["loss"]
        tensor(...)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.output_dim = config.get("output_dim", 512)

        # Vision encoder
        vision_config = {
            "img_size": config.get("img_size", 224),
            "patch_size": config.get("patch_size", 16),
            "in_channels": config.get("in_channels", 3),
            "embed_dim": config.get("vision_embed_dim", 768),
            "depth": config.get("vision_depth", 12),
            "num_heads": config.get("vision_num_heads", 12),
            "mlp_ratio": config.get("mlp_ratio", 4.0),
            "dropout": config.get("dropout", 0.0),
            "output_dim": self.output_dim,
        }
        self.vision_encoder = SigLIPVisionEncoder(vision_config)

        # Text encoder
        text_config = {
            "vocab_size": config.get("vocab_size", 49408),
            "max_seq_len": config.get("max_seq_len", 77),
            "embed_dim": config.get("text_embed_dim", 512),
            "depth": config.get("text_depth", 12),
            "num_heads": config.get("text_num_heads", 8),
            "mlp_ratio": config.get("mlp_ratio", 4.0),
            "dropout": config.get("dropout", 0.0),
            "output_dim": self.output_dim,
            "causal": config.get("text_causal", False),
        }
        self.text_encoder = SigLIPTextEncoder(text_config)

        # Loss function
        self.loss_fn = SigLIPLoss(
            init_temperature=config.get("init_temperature", 10.0),
            init_bias=config.get("init_bias", -10.0),
            learnable=config.get("learnable_temperature", True),
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors.

        Args:
            images: Images of shape (B, C, H, W).

        Returns:
            Normalized features of shape (B, output_dim).
        """
        return self.vision_encoder(images)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text to feature vectors.

        Args:
            input_ids: Token indices of shape (B, seq_len).
            attention_mask: Attention mask of shape (B, seq_len).

        Returns:
            Normalized features of shape (B, output_dim).
        """
        return self.text_encoder(input_ids, attention_mask)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through vision and text encoders.

        Args:
            images: Images of shape (B, C, H, W).
            input_ids: Token indices of shape (B, seq_len).
            attention_mask: Attention mask of shape (B, seq_len).
            return_loss: Whether to compute and return the loss.

        Returns:
            Dictionary containing:
                image_features: Encoded image features (B, output_dim).
                text_features: Encoded text features (B, output_dim).
                loss: Sigmoid contrastive loss (if return_loss=True).
                logits: Similarity logits (B, B).
        """
        # Encode images and text
        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)

        # Compute similarity logits
        logits = image_features @ text_features.T

        output = {
            "image_features": image_features,
            "text_features": text_features,
            "logits": logits,
        }

        # Compute loss if requested
        if return_loss:
            loss = self.loss_fn(image_features, text_features)
            output["loss"] = loss

        return output


__all__ = [
    "SigLIP",
    "SigLIPVisionEncoder",
    "SigLIPTextEncoder",
    "SigLIPLoss",
    "VisionTransformerBlock",
    "TextTransformerBlock",
]
