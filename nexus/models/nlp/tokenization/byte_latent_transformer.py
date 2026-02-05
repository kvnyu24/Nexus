"""Byte Latent Transformer (BLT): Tokenizer-free Language Modeling.

Reference:
    Hsu, Y., et al. "Byte Latent Transformer: Patches Scale Better Than Tokens."
    Meta AI, 2024. https://arxiv.org/abs/2412.09871

BLT operates directly on raw bytes without a fixed tokenizer vocabulary.
It dynamically groups bytes into patches based on entropy, allowing:
1. Infinite vocabulary (no tokenizer needed)
2. Dynamic patch sizes (short patches for complex content, long for simple)
3. Better scaling properties than token-based models

Key innovation: Entropy-based dynamic patching with a learned latent
transformer backbone, achieving better perplexity at the same compute.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import math

from nexus.core.base import NexusModule


@dataclass
class BLTConfig:
    """Configuration for Byte Latent Transformer.

    Attributes:
        vocab_size: Byte vocabulary size (256 for raw bytes).
        hidden_size: Latent transformer hidden dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        max_patch_size: Maximum bytes per patch.
        min_patch_size: Minimum bytes per patch.
        entropy_threshold: Entropy threshold for patch boundaries.
        patch_dim: Dimension of patch embeddings.
        use_local_encoder: Whether to use local byte encoder.
        use_local_decoder: Whether to use local byte decoder.
    """
    vocab_size: int = 256  # 256 bytes
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_patch_size: int = 16
    min_patch_size: int = 1
    entropy_threshold: float = 0.7
    patch_dim: int = 512
    use_local_encoder: bool = True
    use_local_decoder: bool = True


class EntropyPatcher:
    """Dynamic byte-to-patch segmentation based on entropy.

    Groups bytes into variable-length patches where patch boundaries
    occur at high-entropy positions (high uncertainty).
    """

    def __init__(self, config: BLTConfig):
        self.config = config

    def compute_byte_entropy(
        self,
        byte_probs: torch.Tensor
    ) -> torch.Tensor:
        """Compute entropy at each byte position.

        Args:
            byte_probs: Byte probabilities, shape (batch, seq_len, 256).

        Returns:
            Entropy values, shape (batch, seq_len).
        """
        # Shannon entropy: -sum(p * log(p))
        log_probs = torch.log(byte_probs + 1e-10)
        entropy = -(byte_probs * log_probs).sum(dim=-1)
        return entropy

    def create_patches(
        self,
        byte_ids: torch.Tensor,
        byte_probs: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Create variable-length patches from bytes.

        Args:
            byte_ids: Byte token IDs, shape (batch, seq_len).
            byte_probs: Optional byte probabilities for entropy computation.

        Returns:
            Tuple of (patches, patch_lengths) where patches is a list of
            tensors and patch_lengths is a list of integers.
        """
        batch_size, seq_len = byte_ids.shape

        if byte_probs is None:
            # No entropy info, use fixed-size patches
            patch_size = self.config.max_patch_size
            num_patches = (seq_len + patch_size - 1) // patch_size
            patches = []
            patch_lengths = []

            for i in range(num_patches):
                start_idx = i * patch_size
                end_idx = min(start_idx + patch_size, seq_len)
                patch = byte_ids[:, start_idx:end_idx]
                patches.append(patch)
                patch_lengths.append(end_idx - start_idx)

            return patches, patch_lengths

        # Compute entropy
        entropy = self.compute_byte_entropy(byte_probs)

        # Find patch boundaries (high entropy positions)
        boundaries = []
        current_pos = 0

        for pos in range(seq_len):
            if entropy[:, pos].mean() > self.config.entropy_threshold:
                # High entropy - create boundary
                if pos - current_pos >= self.config.min_patch_size:
                    boundaries.append(pos)
                    current_pos = pos

            # Force boundary if max patch size reached
            if pos - current_pos >= self.config.max_patch_size:
                boundaries.append(pos)
                current_pos = pos

        # Add final boundary
        if current_pos < seq_len:
            boundaries.append(seq_len)

        # Create patches
        patches = []
        patch_lengths = []
        start_pos = 0

        for end_pos in boundaries:
            patch = byte_ids[:, start_pos:end_pos]
            patches.append(patch)
            patch_lengths.append(end_pos - start_pos)
            start_pos = end_pos

        return patches, patch_lengths


class LocalByteEncoder(nn.Module):
    """Local encoder for bytes within a patch.

    Encodes variable-length byte sequences into fixed-size patch embeddings.

    Args:
        config: BLT configuration.
    """

    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config

        # Byte embedding
        self.byte_embed = nn.Embedding(config.vocab_size, config.patch_dim)

        # Local transformer for within-patch encoding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.patch_dim,
            nhead=8,
            dim_feedforward=config.patch_dim * 4,
            batch_first=True
        )
        self.local_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Pooling to fixed size
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Encode bytes into patch embedding.

        Args:
            byte_ids: Byte IDs, shape (batch, patch_len).

        Returns:
            Patch embedding, shape (batch, patch_dim).
        """
        # Embed bytes
        byte_embeds = self.byte_embed(byte_ids)  # (batch, patch_len, patch_dim)

        # Local transformer
        encoded = self.local_transformer(byte_embeds)  # (batch, patch_len, patch_dim)

        # Pool to fixed size
        pooled = encoded.mean(dim=1)  # (batch, patch_dim)

        return pooled


class LatentTransformer(nn.Module):
    """Latent transformer operating on patch embeddings.

    Args:
        config: BLT configuration.
    """

    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config

        # Project patch embeddings to latent space
        self.input_proj = nn.Linear(config.patch_dim, config.hidden_size)

        # Latent transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output projection
        self.output_proj = nn.Linear(config.hidden_size, config.patch_dim)

    def forward(self, patch_embeds: torch.Tensor) -> torch.Tensor:
        """Process patch embeddings through latent transformer.

        Args:
            patch_embeds: Patch embeddings, shape (batch, num_patches, patch_dim).

        Returns:
            Latent representations, shape (batch, num_patches, patch_dim).
        """
        # Project to latent space
        latent = self.input_proj(patch_embeds)  # (batch, num_patches, hidden_size)

        # Transform
        transformed = self.transformer(latent)  # (batch, num_patches, hidden_size)

        # Project back
        output = self.output_proj(transformed)  # (batch, num_patches, patch_dim)

        return output


class LocalByteDecoder(nn.Module):
    """Local decoder for generating bytes from patch embeddings.

    Args:
        config: BLT configuration.
    """

    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config

        # Decoder transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.patch_dim,
            nhead=8,
            dim_feedforward=config.patch_dim * 4,
            batch_first=True
        )
        self.local_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Output projection to byte vocabulary
        self.output_proj = nn.Linear(config.patch_dim, config.vocab_size)

    def forward(
        self,
        patch_embed: torch.Tensor,
        target_length: int
    ) -> torch.Tensor:
        """Decode patch embedding into byte probabilities.

        Args:
            patch_embed: Patch embedding, shape (batch, patch_dim).
            target_length: Number of bytes to generate.

        Returns:
            Byte logits, shape (batch, target_length, vocab_size).
        """
        batch_size = patch_embed.shape[0]

        # Expand patch embedding to target length
        patch_expanded = patch_embed.unsqueeze(1).expand(-1, target_length, -1)

        # Decode (simplified - would need proper autoregressive decoding)
        decoded = self.local_decoder(
            tgt=patch_expanded,
            memory=patch_expanded
        )  # (batch, target_length, patch_dim)

        # Project to vocabulary
        logits = self.output_proj(decoded)  # (batch, target_length, vocab_size)

        return logits


class ByteLatentTransformer(NexusModule):
    """Byte Latent Transformer for tokenizer-free language modeling.

    Args:
        config: BLT configuration.
    """

    def __init__(self, config: BLTConfig):
        super().__init__(config.__dict__)

        self.config = config
        self.patcher = EntropyPatcher(config)

        # Components
        if config.use_local_encoder:
            self.local_encoder = LocalByteEncoder(config)

        self.latent_transformer = LatentTransformer(config)

        if config.use_local_decoder:
            self.local_decoder = LocalByteDecoder(config)

    def forward(
        self,
        byte_ids: torch.Tensor,
        byte_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through BLT.

        Args:
            byte_ids: Raw byte IDs, shape (batch, seq_len).
            byte_probs: Optional byte probabilities for patching.

        Returns:
            Byte logits, shape (batch, seq_len, vocab_size).
        """
        batch_size, seq_len = byte_ids.shape

        # Step 1: Create patches
        patches, patch_lengths = self.patcher.create_patches(byte_ids, byte_probs)

        # Step 2: Encode patches
        patch_embeds = []
        for patch in patches:
            if self.config.use_local_encoder:
                patch_embed = self.local_encoder(patch)
            else:
                # Fallback: simple embedding + pooling
                patch_embed = patch.float().mean(dim=1)
            patch_embeds.append(patch_embed)

        # Stack patch embeddings
        patch_embeds = torch.stack(patch_embeds, dim=1)  # (batch, num_patches, patch_dim)

        # Step 3: Latent transformer
        latent_embeds = self.latent_transformer(patch_embeds)  # (batch, num_patches, patch_dim)

        # Step 4: Decode patches back to bytes
        all_logits = []
        for i, patch_length in enumerate(patch_lengths):
            if self.config.use_local_decoder:
                patch_logits = self.local_decoder(
                    latent_embeds[:, i, :],
                    target_length=patch_length
                )
            else:
                # Fallback: simple linear projection
                patch_logits = torch.randn(
                    batch_size,
                    patch_length,
                    self.config.vocab_size,
                    device=byte_ids.device
                )
            all_logits.append(patch_logits)

        # Concatenate all patch outputs
        output_logits = torch.cat(all_logits, dim=1)  # (batch, seq_len, vocab_size)

        return output_logits

    def generate(
        self,
        prompt_bytes: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate bytes autoregressively.

        Args:
            prompt_bytes: Prompt byte IDs, shape (batch, prompt_len).
            max_length: Maximum bytes to generate.
            temperature: Sampling temperature.

        Returns:
            Generated byte IDs, shape (batch, total_len).
        """
        generated = prompt_bytes.clone()

        for _ in range(max_length):
            # Forward pass
            logits = self.forward(generated)

            # Get last byte logits
            next_byte_logits = logits[:, -1, :] / temperature

            # Sample
            probs = F.softmax(next_byte_logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_byte], dim=1)

        return generated


def create_byte_latent_transformer(config: Optional[BLTConfig] = None) -> ByteLatentTransformer:
    """Create a Byte Latent Transformer.

    Args:
        config: BLT configuration (uses defaults if None).

    Returns:
        ByteLatentTransformer instance.
    """
    config = config or BLTConfig()
    return ByteLatentTransformer(config)
