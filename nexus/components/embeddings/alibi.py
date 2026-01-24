"""
ALiBi: Attention with Linear Biases.

ALiBi adds linear biases based on token distance instead of learned
position embeddings, enabling extrapolation to longer sequences.
"""
import torch
import torch.nn as nn
import math
from typing import Optional
from nexus.core.base import NexusModule


class ALiBi(NexusModule):
    """Attention with Linear Biases (ALiBi).

    Instead of position embeddings, adds a linear bias to attention scores
    based on the distance between query and key positions.

    bias[i,j] = -slope * |i - j|

    Slopes are head-specific, decreasing geometrically.

    Used by: BLOOM, MPT, Falcon

    Reference: https://arxiv.org/abs/2108.12409

    Args:
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length for precomputation (can extrapolate beyond)
    """

    def __init__(
        self,
        num_heads: int,
        max_seq_len: int = 8192
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Compute slopes for each head
        # Slopes follow geometric sequence: 2^(-8/n), 2^(-8*2/n), ...
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)

        # Precompute bias matrix for efficiency
        bias = self._build_alibi_bias(max_seq_len, slopes)
        self.register_buffer('bias', bias)

    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """
        Compute head-specific slopes.

        For power-of-2 heads, slopes are: 2^(-8/n * i) for i in 1..n
        For non-power-of-2, uses closest power of 2 and interpolates.
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # For non-power-of-2, combine closest power of 2
            closest_power = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power)
            extra_slopes = get_slopes_power_of_2(2 * closest_power)
            slopes = slopes + extra_slopes[0::2][:num_heads - closest_power]

        return torch.tensor(slopes, dtype=torch.float32)

    def _build_alibi_bias(
        self,
        seq_len: int,
        slopes: torch.Tensor
    ) -> torch.Tensor:
        """Build the ALiBi bias matrix."""
        # Create position difference matrix
        # bias[i,j] = |i - j|
        positions = torch.arange(seq_len)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)

        # For causal attention, only negative (previous) positions matter
        # But we'll compute full matrix for flexibility
        relative_positions = relative_positions.float()

        # Shape: (num_heads, seq_len, seq_len)
        # Each head has different slope
        bias = -slopes.view(-1, 1, 1) * relative_positions.abs().unsqueeze(0)

        return bias

    def forward(
        self,
        attention_scores: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Add ALiBi bias to attention scores.

        Args:
            attention_scores: Attention scores of shape
                (batch, num_heads, query_len, key_len)
            seq_len: Optional sequence length (for dynamic computation)

        Returns:
            Attention scores with ALiBi bias added
        """
        batch_size, num_heads, query_len, key_len = attention_scores.shape

        if key_len <= self.max_seq_len:
            # Use precomputed bias
            # Handle query_len != key_len (e.g., during incremental decoding)
            bias = self.bias[:, :query_len, :key_len]
        else:
            # Compute bias on-the-fly for longer sequences
            bias = self._build_alibi_bias(key_len, self.slopes)
            bias = bias[:, :query_len, :key_len].to(attention_scores.device)

        return attention_scores + bias.unsqueeze(0)

    def get_bias(
        self,
        query_len: int,
        key_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Get ALiBi bias for given dimensions."""
        if key_len <= self.max_seq_len:
            return self.bias[:, :query_len, :key_len]
        else:
            bias = self._build_alibi_bias(key_len, self.slopes)
            return bias[:, :query_len, :key_len].to(device)


class ALiBiPositionalEncoding(ALiBi):
    """Alias for ALiBi for compatibility."""
    pass
