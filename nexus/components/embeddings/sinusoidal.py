"""
Sinusoidal Positional Encoding.

Fixed sinusoidal positional encoding from the original Transformer paper.
"""
import torch
import torch.nn as nn
import math
from typing import Optional
from nexus.core.base import NexusModule


class SinusoidalPositionalEncoding(NexusModule):
    """
    Fixed sinusoidal positional encoding from original Transformer.

    Uses sine and cosine functions of different frequencies to encode
    position information. The encoding is fixed (not learned) and allows
    the model to extrapolate to longer sequences than seen during training.

    Formula:
        PE(pos, 2i) = sin(pos / base^(2i/dim))
        PE(pos, 2i+1) = cos(pos / base^(2i/dim))

    Reference: https://arxiv.org/abs/1706.03762 (Attention Is All You Need)

    Args:
        dim: Embedding dimension (must be even)
        max_seq_len: Maximum sequence length for precomputation
        base: Base for frequency computation (default: 10000)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 5000,
        base: float = 10000.0,
        dropout: float = 0.1
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {dim}")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.dropout = nn.Dropout(p=dropout)

        # Precompute positional encodings
        pe = self._compute_encodings(max_seq_len, dim, base)
        self.register_buffer('pe', pe)

    def _compute_encodings(
        self,
        max_seq_len: int,
        dim: int,
        base: float
    ) -> torch.Tensor:
        """
        Compute sinusoidal positional encodings.

        Args:
            max_seq_len: Maximum sequence length
            dim: Embedding dimension
            base: Base frequency

        Returns:
            Positional encodings of shape (1, max_seq_len, dim)
        """
        # Create position and dimension indices
        positions = torch.arange(max_seq_len).unsqueeze(1)  # (max_seq_len, 1)
        dim_indices = torch.arange(0, dim, 2).float()  # (dim/2,)

        # Compute frequencies: 1 / base^(2i/dim)
        frequencies = 1.0 / (base ** (dim_indices / dim))

        # Compute angles: position * frequency
        angles = positions * frequencies  # (max_seq_len, dim/2)

        # Apply sin to even indices, cos to odd indices
        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        # Add batch dimension
        return pe.unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0
    ) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            offset: Starting position offset (for incremental decoding)

        Returns:
            Tensor with positional encoding added (batch, seq_len, dim)
        """
        seq_len = x.shape[1]

        if seq_len + offset > self.max_seq_len:
            # Compute encodings on-the-fly for longer sequences
            pe = self._compute_encodings(
                seq_len + offset, self.dim, self.base
            ).to(x.device)
            output = x + pe[:, offset:offset + seq_len, :]
        else:
            output = x + self.pe[:, offset:offset + seq_len, :]

        return self.dropout(output)

    def get_encoding(
        self,
        seq_len: int,
        offset: int = 0,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Get positional encoding without adding to input.

        Args:
            seq_len: Sequence length
            offset: Starting position offset
            device: Target device

        Returns:
            Positional encoding of shape (1, seq_len, dim)
        """
        if seq_len + offset > self.max_seq_len:
            pe = self._compute_encodings(
                seq_len + offset, self.dim, self.base
            )
            if device is not None:
                pe = pe.to(device)
            return pe[:, offset:offset + seq_len, :]

        pe = self.pe[:, offset:offset + seq_len, :]
        if device is not None:
            pe = pe.to(device)
        return pe

    @staticmethod
    def compute_fixed(
        positions: torch.Tensor,
        dim: int,
        base: float = 10000.0
    ) -> torch.Tensor:
        """
        Compute sinusoidal encoding for arbitrary positions.

        Useful for continuous or non-integer positions.

        Args:
            positions: Position tensor of any shape
            dim: Embedding dimension
            base: Base frequency

        Returns:
            Encoding of shape (*positions.shape, dim)
        """
        dim_indices = torch.arange(0, dim, 2, device=positions.device).float()
        frequencies = 1.0 / (base ** (dim_indices / dim))

        # Expand positions for broadcasting
        angles = positions.unsqueeze(-1) * frequencies

        # Interleave sin and cos
        pe = torch.zeros(*positions.shape, dim, device=positions.device)
        pe[..., 0::2] = torch.sin(angles)
        pe[..., 1::2] = torch.cos(angles)

        return pe
