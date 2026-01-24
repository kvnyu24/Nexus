"""
YaRN: Yet another RoPE extensioN.

YaRN extends RoPE to longer contexts through NTK-by-parts interpolation,
which treats different frequency dimensions differently.
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class YaRN(NexusModule):
    """Yet another RoPE extensioN (YaRN).

    Extends RoPE to longer sequences using NTK-by-parts interpolation:
    - High frequencies (local patterns): No interpolation
    - Low frequencies (global patterns): Full interpolation
    - Middle frequencies: Gradual interpolation

    Reference: https://arxiv.org/abs/2309.00071

    Args:
        dim: Embedding dimension (must be even)
        max_position_embeddings: Maximum position for base model
        base: Base for computing frequencies (default 10000)
        scale: Context extension scale factor
        original_max_position_embeddings: Original training length
        beta_fast: Fast (high frequency) dimension cutoff (default 32)
        beta_slow: Slow (low frequency) dimension cutoff (default 1)
        mscale: Attention scaling factor adjustment
        mscale_all_dim: Apply mscale to all dimensions
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scale: float = 1.0,
        original_max_position_embeddings: int = 2048,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

        # Compute YaRN frequencies
        inv_freq = self._compute_yarn_frequencies()
        self.register_buffer('inv_freq', inv_freq)

        # Compute attention scaling
        self._mscale = self._compute_mscale()

        # Cache for cos/sin embeddings
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _compute_yarn_frequencies(self) -> torch.Tensor:
        """Compute YaRN-modified inverse frequencies."""
        # Base frequencies
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        inv_freq_base = 1.0 / pos_freqs

        # Compute wavelengths
        wavelengths = 2 * math.pi * pos_freqs

        # Find dimension boundaries based on wavelength
        # low_dim: wavelength > beta_slow * original_length (low freq, needs interpolation)
        # high_dim: wavelength < beta_fast * original_length (high freq, no interpolation)
        low_bound = self.original_max_position_embeddings / self.beta_slow
        high_bound = self.original_max_position_embeddings / self.beta_fast

        # Compute interpolation ramp
        # Linear ramp from 0 (high freq) to 1 (low freq)
        ramp = torch.clamp(
            (wavelengths - high_bound) / (low_bound - high_bound),
            0.0, 1.0
        )

        # Interpolation factor per dimension
        # gamma = 0 means no interpolation (keep original freq)
        # gamma = 1 means full interpolation (scale down freq)
        gamma = ramp

        # Apply NTK-by-parts: scale frequencies based on gamma
        # new_freq = base_freq * scale^(-gamma)
        # In practice, we adjust the base: new_base = base * scale^(-gamma * dim / (dim-2))
        inv_freq = inv_freq_base / (self.scale ** gamma)

        return inv_freq

    def _compute_mscale(self) -> float:
        """Compute attention scaling factor."""
        if self.scale <= 1.0:
            return 1.0

        # mscale adjusts attention scores to account for longer sequences
        # Following the YaRN paper formula
        return (
            0.1 * math.log(self.scale) + 1.0
        ) ** self.mscale_all_dim * self.mscale

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings with YaRN scaling.

        Args:
            x: Input tensor (used for shape and device)
            position_ids: Position indices (batch, seq_len)
            seq_len: Sequence length (if position_ids not provided)

        Returns:
            cos: Cosine embeddings (1, seq_len, dim)
            sin: Sine embeddings (1, seq_len, dim)
        """
        if seq_len is None:
            if position_ids is not None:
                seq_len = position_ids.max().item() + 1
            else:
                seq_len = x.shape[1]

        # Use cache if possible
        if seq_len <= self._seq_len_cached and self._cos_cached is not None:
            return (
                self._cos_cached[:, :seq_len, :],
                self._sin_cached[:, :seq_len, :]
            )

        # Compute position indices
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Compute angles: position * inv_freq
        # Shape: (batch, seq_len, dim/2)
        freqs = position_ids.unsqueeze(-1).float() * self.inv_freq.unsqueeze(0)

        # Expand to full dimension by duplicating
        # This matches the standard RoPE implementation
        emb = torch.cat([freqs, freqs], dim=-1)

        # Apply mscale to attention (handled in attention computation)
        cos = emb.cos() * self._mscale
        sin = emb.sin() * self._mscale

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
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to Q and K.

        Args:
            q: Query tensor (batch, heads, seq, head_dim)
            k: Key tensor (batch, heads, seq, head_dim)
            cos: Cosine embeddings
            sin: Sine embeddings
            position_ids: Optional position indices

        Returns:
            Rotated (q, k) tensors
        """
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        # Expand cos/sin for heads dimension
        cos = cos.unsqueeze(1)  # (batch, 1, seq, dim)
        sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed


class DynamicNTKScaling(NexusModule):
    """Dynamic NTK-Aware RoPE Scaling.

    Adjusts the RoPE base frequency dynamically based on sequence length,
    enabling context extension without fine-tuning.

    Used by: CodeLlama, various extended-context models

    Reference: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/

    Args:
        dim: Embedding dimension
        base: Base frequency (default 10000)
        max_position_embeddings: Max positions for base model
        scaling_factor: Target context extension factor
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_position_embeddings: int = 2048,
        scaling_factor: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.scaling_factor = scaling_factor

        # Base frequencies (will be modified dynamically)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _compute_dynamic_base(self, seq_len: int) -> float:
        """Compute dynamic base for given sequence length."""
        if seq_len <= self.max_position_embeddings:
            return self.base

        # Scale base to accommodate longer sequences
        # new_base = base * (scale_factor)^(dim/(dim-2))
        scale = seq_len / self.max_position_embeddings
        return self.base * (scale ** (self.dim / (self.dim - 2)))

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings with dynamic NTK scaling.

        Args:
            x: Input tensor
            seq_len: Sequence length

        Returns:
            cos, sin: Position embeddings
        """
        if seq_len is None:
            seq_len = x.shape[1]

        # Compute dynamic base if sequence is longer than training length
        if seq_len > self.max_position_embeddings:
            base = self._compute_dynamic_base(seq_len)
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
        else:
            inv_freq = self.inv_freq

        # Compute position embeddings
        positions = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos().unsqueeze(0)
        sin = emb.sin().unsqueeze(0)

        return cos, sin


class RotaryEmbeddingExtended(NexusModule):
    """Extended RoPE with multiple scaling options.

    Unified interface for various RoPE extensions.

    Args:
        dim: Embedding dimension
        base: Base frequency
        max_position_embeddings: Maximum positions
        scaling_type: Type of scaling ('none', 'linear', 'dynamic', 'yarn')
        scaling_factor: Scale factor for context extension
        yarn_config: Additional config for YaRN scaling
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_position_embeddings: int = 2048,
        scaling_type: str = 'none',
        scaling_factor: float = 1.0,
        yarn_config: Optional[dict] = None
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor

        if scaling_type == 'yarn':
            yarn_config = yarn_config or {}
            self.rope = YaRN(
                dim=dim,
                max_position_embeddings=max_position_embeddings,
                base=base,
                scale=scaling_factor,
                original_max_position_embeddings=max_position_embeddings,
                **yarn_config
            )
        elif scaling_type == 'dynamic':
            self.rope = DynamicNTKScaling(
                dim=dim,
                base=base,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=scaling_factor
            )
        else:
            # Standard or linear RoPE
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            if scaling_type == 'linear' and scaling_factor > 1.0:
                inv_freq = inv_freq / scaling_factor
            self.register_buffer('inv_freq', inv_freq)
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings."""
        if self.rope is not None:
            return self.rope(x, seq_len=seq_len)

        if seq_len is None:
            seq_len = x.shape[1]

        positions = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)
