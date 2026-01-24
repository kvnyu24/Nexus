"""
Relative Positional Bias.

T5-style relative positional bias that learns bias terms based on
relative distance between positions.
"""
import torch
import torch.nn as nn
import math
from typing import Optional
from nexus.core.base import NexusModule


class RelativePositionalBias(NexusModule):
    """
    Relative positional bias (T5-style).

    Learns bias terms based on relative distance between positions.
    Distances are bucketed logarithmically to handle long sequences
    efficiently while maintaining fine-grained distinctions for
    nearby positions.

    Used by: T5, LongT5, mT5, FLAN-T5

    Reference: https://arxiv.org/abs/1910.10683 (Exploring the Limits of Transfer Learning)

    Args:
        num_heads: Number of attention heads
        num_buckets: Number of distance buckets (default: 32)
        max_distance: Maximum distance to consider (default: 128)
        bidirectional: Whether to use bidirectional bias (default: True)
            - True for encoder (BERT-style, looks both ways)
            - False for decoder (GPT-style, causal)
    """

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        # Learnable bias embeddings per bucket per head
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        # Initialize with small values
        nn.init.normal_(self.relative_attention_bias.weight, std=0.02)

    def _relative_position_bucket(
        self,
        relative_position: torch.Tensor
    ) -> torch.Tensor:
        """
        Map relative positions to bucket indices.

        Uses logarithmic bucketing: exact for nearby positions,
        coarser for distant positions.

        Args:
            relative_position: Relative position tensor (any shape)

        Returns:
            Bucket indices (same shape as input)
        """
        num_buckets = self.num_buckets
        max_distance = self.max_distance

        relative_buckets = 0

        if self.bidirectional:
            # Half buckets for positive, half for negative
            num_buckets //= 2
            # Positive positions get offset
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            # Clamp to non-positive for causal
            relative_position = -torch.min(
                relative_position,
                torch.zeros_like(relative_position)
            )

        # Half of buckets are for exact positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Other half are for logarithmically larger bins
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) /
            math.log(max_distance / max_exact) *
            (num_buckets - max_exact)
        ).to(torch.long)

        # Clamp to valid bucket range
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(
            is_small,
            relative_position.to(torch.long),
            relative_position_if_large
        )

        return relative_buckets

    def forward(
        self,
        query_length: int,
        key_length: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compute relative positional bias.

        Args:
            query_length: Length of query sequence
            key_length: Length of key sequence
            device: Target device

        Returns:
            Bias tensor of shape (1, num_heads, query_length, key_length)
        """
        if device is None:
            device = self.relative_attention_bias.weight.device

        # Compute relative positions
        # query_position[i] - key_position[j]
        query_positions = torch.arange(query_length, device=device)
        key_positions = torch.arange(key_length, device=device)
        relative_position = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)

        # Map to buckets
        relative_position_bucket = self._relative_position_bucket(relative_position)

        # Get bias values from embeddings
        # Shape: (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)

        # Reshape to (1, num_heads, query_length, key_length)
        values = values.permute(2, 0, 1).unsqueeze(0)

        return values

    def get_bias(
        self,
        query_length: int,
        key_length: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Alias for forward() for compatibility."""
        return self.forward(query_length, key_length, device)

    def compute_with_cache(
        self,
        query_length: int,
        key_length: int,
        cached_key_length: int = 0,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compute bias with KV cache support for incremental decoding.

        Args:
            query_length: Current query length (typically 1 during generation)
            key_length: Total key length (including cached)
            cached_key_length: Length of cached keys
            device: Target device

        Returns:
            Bias tensor for the new query positions
        """
        if device is None:
            device = self.relative_attention_bias.weight.device

        # Query positions are the new positions
        query_positions = torch.arange(
            cached_key_length,
            cached_key_length + query_length,
            device=device
        )
        # Key positions are all positions (cached + new)
        key_positions = torch.arange(key_length, device=device)

        relative_position = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)
        relative_position_bucket = self._relative_position_bucket(relative_position)

        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute(2, 0, 1).unsqueeze(0)

        return values
