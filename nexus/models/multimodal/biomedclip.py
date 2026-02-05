"""BiomedCLIP: Biomedical vision-language model.

BiomedCLIP is a contrastive vision-language model specifically trained on
biomedical image-text pairs. It extends CLIP to the biomedical domain with
domain-specific pretraining data.

Key features:
- Biomedical domain specialization (PubMed, medical imaging datasets)
- Contrastive learning between medical images and descriptions
- Zero-shot medical image classification
- Medical image-text retrieval
- Supports radiology, pathology, and other medical imaging modalities

References:
    - BiomedCLIP: https://arxiv.org/abs/2303.00915 (Microsoft, 2023)
    - PMC-OA dataset for training
    - BioViL predecessor work

Authors: Microsoft Research (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
from nexus.core.base import NexusModule


class BiomedicalImageEncoder(NexusModule):
    """Vision encoder specialized for biomedical images.

    Optimized for medical imaging modalities including:
    - Radiology (X-rays, CT, MRI)
    - Pathology (histology slides)
    - Dermatology
    - Ophthalmology

    Args:
        in_channels: Input channels (1 for grayscale, 3 for RGB)
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        patch_size: Patch size for vision transformer
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 768,
        num_layers: int = 12,
        patch_size: int = 16,
        num_heads: int = 12,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # Patch embedding (support both RGB and grayscale)
        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Class token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + 196, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # Projection head for contrastive learning
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode biomedical images.

        Args:
            images: [batch_size, in_channels, H, W]

        Returns:
            Tuple of:
                - image_features: Global features [batch_size, hidden_dim]
                - patch_features: Patch-level features [batch_size, num_patches+1, hidden_dim]
        """
        B = images.shape[0]

        # Extract patches
        x = self.patch_embed(images)  # [B, hidden_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        num_tokens = x.shape[1]
        if num_tokens <= self.pos_embed.shape[1]:
            pos_embed = self.pos_embed[:, :num_tokens, :]
        else:
            # Interpolate positional embeddings if needed
            cls_pos = self.pos_embed[:, :1, :]
            patch_pos = self.pos_embed[:, 1:, :]
            patch_pos = F.interpolate(
                patch_pos.transpose(1, 2),
                size=num_tokens - 1,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            pos_embed = torch.cat([cls_pos, patch_pos], dim=1)

        x = x + pos_embed

        # Apply transformer
        x = self.encoder(x)
        x = self.norm(x)

        # Extract global and patch features
        image_features = x[:, 0, :]  # CLS token
        image_features = self.projection(image_features)
        patch_features = x

        return image_features, patch_features


class BiomedicalTextEncoder(NexusModule):
    """Text encoder specialized for biomedical terminology.

    Handles medical terminology, anatomical terms, and clinical descriptions.

    Args:
        vocab_size: Vocabulary size
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        max_length: int = 77,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # Projection head
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode biomedical text.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Tuple of:
                - text_features: Sentence features [batch_size, hidden_dim]
                - token_features: Token-level features [batch_size, seq_len, hidden_dim]
        """
        B, L = input_ids.shape

        # Embed tokens
        x = self.token_embed(input_ids)

        # Add positional embeddings
        x = x + self.pos_embed[:, :L, :]

        # Convert attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (0 = attend, -inf = mask)
            mask = (1.0 - attention_mask) * -10000.0
        else:
            mask = None

        # Apply transformer
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0) if attention_mask is not None else None)
        x = self.norm(x)

        # Extract sentence feature (use EOS token or last non-padded token)
        if attention_mask is not None:
            # Use the last non-padded token
            seq_lengths = attention_mask.sum(dim=1) - 1
            text_features = x[torch.arange(B), seq_lengths]
        else:
            # Use the last token
            text_features = x[:, -1, :]

        text_features = self.projection(text_features)
        token_features = x

        return text_features, token_features


class BiomedCLIP(NexusModule):
    """BiomedCLIP: Contrastive vision-language model for biomedical domain.

    Pretrained on large-scale biomedical image-text pairs (PMC-OA dataset)
    for medical image understanding and retrieval tasks.

    Key applications:
    - Zero-shot medical image classification
    - Medical image-text retrieval
    - Cross-modal medical search
    - Clinical decision support
    - Radiology report generation (when combined with decoder)

    Args:
        image_encoder_dim: Dimension of image encoder
        text_encoder_dim: Dimension of text encoder
        projection_dim: Dimension of projection space
        num_image_layers: Number of image encoder layers
        num_text_layers: Number of text encoder layers
        temperature: Temperature for contrastive loss

    Example:
        >>> model = BiomedCLIP(
        ...     image_encoder_dim=768,
        ...     text_encoder_dim=512,
        ...     projection_dim=512
        ... )
        >>> images = torch.randn(4, 3, 224, 224)  # Medical images
        >>> input_ids = torch.randint(0, 49408, (4, 77))  # Medical descriptions
        >>> outputs = model(images, input_ids)
        >>> # Compute similarity for retrieval
        >>> similarity = outputs['image_features'] @ outputs['text_features'].T
    """

    def __init__(
        self,
        image_encoder_dim: int = 768,
        text_encoder_dim: int = 512,
        projection_dim: int = 512,
        num_image_layers: int = 12,
        num_text_layers: int = 12,
        temperature: float = 0.07,
        in_channels: int = 3,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.image_encoder_dim = image_encoder_dim
        self.text_encoder_dim = text_encoder_dim
        self.projection_dim = projection_dim

        # Image encoder
        self.image_encoder = BiomedicalImageEncoder(
            in_channels=in_channels,
            hidden_dim=image_encoder_dim,
            num_layers=num_image_layers
        )

        # Text encoder
        self.text_encoder = BiomedicalTextEncoder(
            hidden_dim=text_encoder_dim,
            num_layers=num_text_layers
        )

        # Additional projections to shared space if dimensions differ
        if image_encoder_dim != projection_dim:
            self.image_projection = nn.Linear(image_encoder_dim, projection_dim)
        else:
            self.image_projection = nn.Identity()

        if text_encoder_dim != projection_dim:
            self.text_projection = nn.Linear(text_encoder_dim, projection_dim)
        else:
            self.text_projection = nn.Identity()

        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature space.

        Args:
            images: [batch_size, in_channels, H, W]

        Returns:
            Normalized image features [batch_size, projection_dim]
        """
        image_features, _ = self.image_encoder(images)
        image_features = self.image_projection(image_features)

        # L2 normalize
        image_features = F.normalize(image_features, dim=-1)

        return image_features

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode text to feature space.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Normalized text features [batch_size, projection_dim]
        """
        text_features, _ = self.text_encoder(input_ids, attention_mask)
        text_features = self.text_projection(text_features)

        # L2 normalize
        text_features = F.normalize(text_features, dim=-1)

        return text_features

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through BiomedCLIP.

        Args:
            images: Input images [batch_size, in_channels, H, W]
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Text attention mask [batch_size, seq_len]
            return_loss: Whether to compute contrastive loss

        Returns:
            Dictionary containing:
                - image_features: Normalized image features
                - text_features: Normalized text features
                - logits_per_image: Image-to-text similarity scores
                - logits_per_text: Text-to-image similarity scores
                - loss: Contrastive loss (if return_loss=True)
        """
        outputs = {}

        # Encode images
        if images is not None:
            image_features = self.encode_image(images)
            outputs['image_features'] = image_features

        # Encode text
        if input_ids is not None:
            text_features = self.encode_text(input_ids, attention_mask)
            outputs['text_features'] = text_features

        # Compute similarity if both modalities present
        if images is not None and input_ids is not None:
            # Scaled dot product similarity
            logit_scale = self.logit_scale.exp()

            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logits_per_image.T

            outputs['logits_per_image'] = logits_per_image
            outputs['logits_per_text'] = logits_per_text

            # Compute contrastive loss if requested
            if return_loss:
                batch_size = images.shape[0]
                labels = torch.arange(batch_size, device=images.device)

                loss_i = F.cross_entropy(logits_per_image, labels)
                loss_t = F.cross_entropy(logits_per_text, labels)
                loss = (loss_i + loss_t) / 2

                outputs['loss'] = loss

        return outputs

    def zero_shot_classification(
        self,
        images: torch.Tensor,
        class_descriptions: List[str],
        tokenizer: Any,
        device: torch.device
    ) -> torch.Tensor:
        """Perform zero-shot classification on medical images.

        Args:
            images: Input images [batch_size, in_channels, H, W]
            class_descriptions: List of text descriptions for each class
            tokenizer: Tokenizer for encoding text
            device: Device to use

        Returns:
            Class probabilities [batch_size, num_classes]
        """
        # Encode images
        image_features = self.encode_image(images)

        # Encode class descriptions
        # This is a placeholder - actual implementation would use real tokenizer
        num_classes = len(class_descriptions)
        text_features_list = []

        for desc in class_descriptions:
            # Tokenize (placeholder - would use actual tokenizer)
            # tokens = tokenizer(desc, return_tensors='pt', padding='max_length', truncation=True)
            # For now, use dummy tokens
            input_ids = torch.randint(0, 49408, (1, 77), device=device)
            text_feat = self.encode_text(input_ids)
            text_features_list.append(text_feat)

        text_features = torch.cat(text_features_list, dim=0)  # [num_classes, projection_dim]

        # Compute similarity
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T  # [batch_size, num_classes]

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        return probs


# Export
__all__ = [
    'BiomedCLIP',
    'BiomedicalImageEncoder',
    'BiomedicalTextEncoder'
]
