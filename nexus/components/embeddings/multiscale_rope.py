"""
Multi-Scale Rotary Position Embedding.

Multi-scale RoPE with different frequencies per head for capturing
patterns at different granularities.
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class MultiScaleRotaryEmbedding(NexusModule):
    """
    Multi-scale RoPE with different frequencies per head.

    Different heads use different frequency scales for capturing patterns
    at different granularities. Low-frequency heads capture long-range
    dependencies while high-frequency heads capture local patterns.

    This is useful for models that need to attend to both local context
    (e.g., syntax) and global context (e.g., document structure).

    Reference:
        - RoPE: https://arxiv.org/abs/2104.09864
        - Multi-scale concept inspired by wavelet analysis and
          multi-resolution transformers

    Args:
        dim: Embedding dimension per head
        num_heads: Number of attention heads
        base_range: (min_base, max_base) for frequency bases (default: (1000, 100000))
            - Smaller base = higher frequency = local patterns
            - Larger base = lower frequency = global patterns
        max_seq_len: Maximum sequence length (default: 8192)
        scaling_factor: Optional linear scaling factor for context extension
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        base_range: Tuple[float, float] = (1000.0, 100000.0),
        max_seq_len: int = 8192,
        scaling_factor: float = 1.0
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {dim}")

        self.dim = dim
        self.num_heads = num_heads
        self.base_range = base_range
        self.max_seq_len = max_seq_len
        self.scaling_factor = scaling_factor

        # Compute different base frequencies for each head
        # Using log-space interpolation for better distribution
        min_base, max_base = base_range
        if num_heads > 1:
            bases = torch.exp(
                torch.linspace(
                    math.log(min_base),
                    math.log(max_base),
                    num_heads
                )
            )
        else:
            bases = torch.tensor([math.sqrt(min_base * max_base)])

        self.register_buffer('bases', bases)

        # Precompute inverse frequencies for each head
        # Shape: (num_heads, dim/2)
        inv_freqs = self._compute_inv_freqs()
        self.register_buffer('inv_freqs', inv_freqs)

        # Cache for cos/sin embeddings
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _compute_inv_freqs(self) -> torch.Tensor:
        """Compute inverse frequencies for all heads."""
        inv_freqs = []
        dim_indices = torch.arange(0, self.dim, 2).float()

        for base in self.bases:
            inv_freq = 1.0 / (base ** (dim_indices / self.dim))
            if self.scaling_factor > 1.0:
                inv_freq = inv_freq / self.scaling_factor
            inv_freqs.append(inv_freq)

        return torch.stack(inv_freqs, dim=0)  # (num_heads, dim/2)

    def _compute_embeddings(
        self,
        seq_len: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos/sin embeddings for given sequence length.

        Returns:
            cos: Shape (num_heads, seq_len, dim)
            sin: Shape (num_heads, seq_len, dim)
        """
        positions = torch.arange(seq_len, device=device).float()

        # Compute angles for each head
        # (num_heads, seq_len, dim/2)
        angles = torch.einsum('h d, s -> h s d', self.inv_freqs.to(device), positions)

        # Expand to full dimension by duplicating
        angles = torch.cat([angles, angles], dim=-1)  # (num_heads, seq_len, dim)

        return angles.cos(), angles.sin()

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings for all heads.

        Args:
            x: Input tensor (used for device and dtype)
            seq_len: Optional sequence length
            position_ids: Optional position indices (batch, seq_len)

        Returns:
            cos: Cosine embeddings (num_heads, seq_len, dim)
            sin: Sine embeddings (num_heads, seq_len, dim)
        """
        if seq_len is None:
            if position_ids is not None:
                seq_len = position_ids.max().item() + 1
            else:
                seq_len = x.shape[1] if x.dim() > 1 else x.shape[0]

        # Use cache if available
        if seq_len <= self._seq_len_cached and self._cos_cached is not None:
            return (
                self._cos_cached[:, :seq_len, :],
                self._sin_cached[:, :seq_len, :]
            )

        cos, sin = self._compute_embeddings(seq_len, x.device)

        # Update cache
        self._cos_cached = cos
        self._sin_cached = sin
        self._seq_len_cached = seq_len

        return cos, sin

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to Q and K.

        Args:
            q: Query tensor (batch, heads, seq, head_dim)
            k: Key tensor (batch, heads, seq, head_dim)
            cos: Cosine embeddings (num_heads, seq_len, dim)
            sin: Sine embeddings (num_heads, seq_len, dim)
            head_indices: Optional mapping from layer heads to base heads

        Returns:
            Rotated (q, k) tensors
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Select embeddings for each head
        if head_indices is not None:
            cos = cos[head_indices]
            sin = sin[head_indices]
        else:
            # Assume num_heads matches or cycle
            if cos.shape[0] < num_heads:
                repeats = (num_heads + cos.shape[0] - 1) // cos.shape[0]
                cos = cos.repeat(repeats, 1, 1)[:num_heads]
                sin = sin.repeat(repeats, 1, 1)[:num_heads]

        # Reshape for broadcasting: (1, num_heads, seq_len, dim)
        cos = cos[:, :seq_len, :].unsqueeze(0)
        sin = sin[:, :seq_len, :].unsqueeze(0)

        # Apply rotation
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the hidden dims."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def get_head_info(self) -> dict:
        """
        Get information about frequency scales per head.

        Returns:
            Dictionary with head index to base frequency mapping
        """
        return {
            f"head_{i}": {
                "base": self.bases[i].item(),
                "frequency_range": (
                    1.0 / self.bases[i].item(),
                    self.dim / (2 * math.pi * self.bases[i].item())
                )
            }
            for i in range(self.num_heads)
        }
