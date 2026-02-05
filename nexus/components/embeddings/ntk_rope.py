"""
NTK-Aware RoPE: Non-uniform frequency scaling for context extension.

Standard RoPE position interpolation scales all frequency dimensions
uniformly, which degrades high-frequency (local) pattern recognition.
NTK-Aware RoPE applies non-uniform scaling based on Neural Tangent Kernel
theory: high frequencies (which capture local patterns) are scaled less,
while low frequencies (which capture global patterns) are scaled more.

This preserves the model's ability to distinguish nearby tokens while
enabling extrapolation to longer sequences.

Theory:
    In standard RoPE, the frequency for dimension i is:
        theta_i = base^(-2i/d)

    With NTK-aware scaling by factor s, the base is modified:
        new_base = base * s^(d/(d-2))

    This effectively applies non-uniform scaling:
        - High freq dimensions (small i): scaled by ~1 (preserved)
        - Low freq dimensions (large i): scaled by ~s (interpolated)

Reference: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
           (NTK-Aware Scaled RoPE allows LLaMA models to have extended context)
           https://arxiv.org/abs/2306.15595 (Scaling Laws of RoPE-based Extrapolation)

See Also:
    - rotary_embedding.py: Standard RoPE implementation
    - yarn.py: YaRN (NTK-by-parts interpolation)
    - long_rope.py: LongRoPE with searched scaling factors
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class NTKAwareRoPE(NexusModule):
    """NTK-Aware Rotary Position Embedding.

    Scales the RoPE base frequency to achieve non-uniform frequency scaling
    across dimensions, enabling context extension without fine-tuning.

    The key insight from NTK theory is that the information content of
    different frequency bands is not equal. High-frequency components
    encode local relationships (nearby tokens) and should be preserved,
    while low-frequency components encode global structure and can be
    interpolated more aggressively.

    Supports two modes:
        - Static: Base is scaled once based on target scale_factor
        - Dynamic: Base is scaled at runtime based on actual sequence length

    Args:
        dim: Embedding dimension (must be even)
        max_position: Maximum position for the extended context
        base: Base frequency for computing RoPE frequencies (default 10000)
        scale_factor: Target context extension factor (e.g., 4.0 for 4x extension).
            If 1.0, behaves like standard RoPE.
        dynamic: If True, adjusts base dynamically based on sequence length
            at runtime. If False, uses static scaling based on scale_factor.

    Example:
        >>> rope = NTKAwareRoPE(dim=128, max_position=16384, scale_factor=4.0)
        >>> x = torch.randn(2, 8192, 1, 128)  # Beyond original 4096 training length
        >>> cos, sin = rope(x)
        >>> cos.shape
        torch.Size([1, 8192, 128])
    """

    def __init__(
        self,
        dim: int,
        max_position: int = 8192,
        base: float = 10000.0,
        scale_factor: float = 1.0,
        dynamic: bool = False
    ):
        super().__init__()

        assert dim % 2 == 0, f"dim must be even, got {dim}"
        assert scale_factor >= 1.0, f"scale_factor must be >= 1.0, got {scale_factor}"

        self.dim = dim
        self.max_position = max_position
        self.base = base
        self.scale_factor = scale_factor
        self.dynamic = dynamic

        if not dynamic:
            # Static NTK-aware scaling: modify base once
            scaled_base = self._compute_ntk_base(scale_factor)
            inv_freq = 1.0 / (scaled_base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer('inv_freq', inv_freq)
        else:
            # Dynamic: store base frequencies, adjust at runtime
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer('inv_freq', inv_freq)
            # Store the original max position for dynamic scaling
            self.original_max_position = max_position

        # Cache for cos/sin embeddings
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _compute_ntk_base(self, scale: float) -> float:
        """Compute NTK-scaled base frequency.

        The NTK-aware formula scales the base as:
            new_base = base * scale^(dim / (dim - 2))

        This results in non-uniform frequency scaling where high frequencies
        are scaled less than low frequencies.

        Args:
            scale: Context extension scale factor

        Returns:
            Scaled base frequency
        """
        if scale <= 1.0:
            return self.base
        return self.base * (scale ** (self.dim / (self.dim - 2)))

    def _compute_inv_freq_dynamic(self, seq_len: int) -> torch.Tensor:
        """Compute inverse frequencies with dynamic NTK scaling.

        Adjusts the base frequency based on actual sequence length,
        only when the sequence exceeds the original training length.

        Args:
            seq_len: Current sequence length

        Returns:
            Inverse frequency tensor
        """
        if seq_len <= self.original_max_position:
            return self.inv_freq

        scale = seq_len / self.original_max_position
        scaled_base = self._compute_ntk_base(scale)
        inv_freq = 1.0 / (
            scaled_base ** (
                torch.arange(0, self.dim, 2, device=self.inv_freq.device).float() / self.dim
            )
        )
        return inv_freq

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute NTK-aware rotary position embeddings.

        Args:
            x: Input tensor (used for shape/device inference)
            position_ids: Explicit position indices (batch, seq_len)
            seq_len: Sequence length (if position_ids not provided)

        Returns:
            cos: Cosine embeddings (1, seq_len, dim) or (batch, seq_len, dim)
            sin: Sine embeddings (1, seq_len, dim) or (batch, seq_len, dim)
        """
        if seq_len is None:
            if position_ids is not None:
                seq_len = position_ids.max().item() + 1
            else:
                seq_len = x.shape[1]

        # Check cache
        if (seq_len <= self._seq_len_cached
                and self._cos_cached is not None
                and not self.dynamic):
            return (
                self._cos_cached[:, :seq_len, :],
                self._sin_cached[:, :seq_len, :]
            )

        # Get inverse frequencies (possibly dynamically scaled)
        if self.dynamic:
            inv_freq = self._compute_inv_freq_dynamic(seq_len)
        else:
            inv_freq = self.inv_freq

        # Compute position indices
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=x.device
            ).unsqueeze(0).float()

        # Compute angles: position * inv_freq
        freqs = position_ids.unsqueeze(-1).float() * inv_freq.unsqueeze(0)

        # Duplicate for full dimension
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        # Update cache (only for static mode)
        if not self.dynamic:
            self._cos_cached = cos
            self._sin_cached = sin
            self._seq_len_cached = seq_len

        return cos, sin

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply NTK-aware rotary embeddings to Q and K.

        Args:
            q: Query tensor (batch, heads, seq, head_dim)
            k: Key tensor (batch, heads, seq, head_dim)
            cos: Cosine embeddings
            sin: Sine embeddings

        Returns:
            Rotated (q, k) tensors
        """
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        # Expand for heads dimension
        cos = cos.unsqueeze(1)  # (batch, 1, seq, dim)
        sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed


class NTKRoPE(NTKAwareRoPE):
    """Alias for NTKAwareRoPE."""
    pass
