"""
LongRoPE: Extending LLM context to 2M+ tokens via evolutionary-searched
non-uniform rescaling factors.

LongRoPE extends RoPE context length through:
    1. Non-uniform rescaling: Each frequency dimension has its own scaling
       factor, found via evolutionary search to minimize perplexity.
    2. Progressive extension: Context is extended in stages (e.g., 4k -> 128k
       -> 2M) rather than in one jump, with fine-tuning at each stage.
    3. Readjusted short-context factors: After long extension, factors are
       re-optimized for short-context performance to avoid degradation.

The key insight is that uniform frequency scaling (as in PI or NTK) is
suboptimal -- different frequency dimensions contribute differently to
attention patterns and should be scaled independently.

Reference: https://arxiv.org/abs/2402.13753 (LongRoPE: Extending LLM Context
           Window Beyond 2 Million Tokens)

See Also:
    - ntk_rope.py: NTK-Aware RoPE (uniform non-linear scaling)
    - yarn.py: YaRN (piecewise interpolation)
    - resonance_rope.py: Resonance RoPE (integer wavelength snapping)
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List
from nexus.core.base import NexusModule


class LongRoPE(NexusModule):
    """LongRoPE with non-uniform per-dimension rescaling.

    Each pair of frequency dimensions has an independent scaling factor.
    These factors can be:
        - Provided directly (from evolutionary search results)
        - Initialized from a heuristic (NTK-like progressive scaling)

    Progressive Extension Strategy:
        Stage 1: Extend from original to 8x with searched factors
        Stage 2: Fine-tune, then extend to 64x with new factors
        Stage 3: Continue to 2M+ tokens

    Args:
        dim: Embedding dimension (must be even)
        max_position: Target maximum position (extended context length)
        original_max_position: Original training context length
        base: Base frequency for RoPE (default 10000)
        search_factors: Pre-computed per-dimension scaling factors from
            evolutionary search. Shape: (dim // 2,). If None, uses
            a heuristic initialization based on NTK-aware progressive scaling.
        short_factors: Optional separate factors optimized for short contexts.
            Shape: (dim // 2,). Applied when seq_len <= original_max_position.
        mscale: Attention scale correction factor for extended sequences
        mscale_all_dim: Scale for mscale computation

    Example:
        >>> # Using default heuristic factors
        >>> rope = LongRoPE(dim=128, max_position=2097152, original_max_position=4096)
        >>> x = torch.randn(1, 65536, 1, 128)
        >>> cos, sin = rope(x)
        >>>
        >>> # Using searched factors from paper
        >>> factors = torch.ones(64) * 2.0  # example factors
        >>> rope = LongRoPE(dim=128, max_position=2097152,
        ...                 original_max_position=4096, search_factors=factors)
    """

    def __init__(
        self,
        dim: int,
        max_position: int = 2097152,
        original_max_position: int = 4096,
        base: float = 10000.0,
        search_factors: Optional[torch.Tensor] = None,
        short_factors: Optional[torch.Tensor] = None,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0
    ):
        super().__init__()

        assert dim % 2 == 0, f"dim must be even, got {dim}"

        self.dim = dim
        self.max_position = max_position
        self.original_max_position = original_max_position
        self.base = base
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

        half_dim = dim // 2

        # Base inverse frequencies
        inv_freq_base = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        # Extension ratio
        self.scale = max_position / original_max_position

        # Per-dimension scaling factors (long context)
        if search_factors is not None:
            assert search_factors.shape == (half_dim,), \
                f"search_factors shape {search_factors.shape} must be ({half_dim},)"
            self.register_buffer('long_factors', search_factors.float())
        else:
            # Heuristic initialization: progressive NTK-like factors
            long_factors = self._compute_progressive_factors(inv_freq_base)
            self.register_buffer('long_factors', long_factors)

        # Per-dimension scaling factors (short context, optional)
        if short_factors is not None:
            assert short_factors.shape == (half_dim,), \
                f"short_factors shape {short_factors.shape} must be ({half_dim},)"
            self.register_buffer('short_factors', short_factors.float())
        else:
            # Default short factors: minimal scaling (near 1.0)
            self.register_buffer('short_factors', torch.ones(half_dim))

        # Store base frequencies
        self.register_buffer('inv_freq_base', inv_freq_base)

        # Compute attention scale correction
        self._mscale = self._compute_mscale()

        # Caches
        self._cos_cached_long = None
        self._sin_cached_long = None
        self._cos_cached_short = None
        self._sin_cached_short = None
        self._seq_len_cached_long = 0
        self._seq_len_cached_short = 0

    def _compute_progressive_factors(
        self,
        inv_freq_base: torch.Tensor
    ) -> torch.Tensor:
        """Compute heuristic progressive scaling factors.

        Uses a smooth interpolation between no scaling (for high frequencies)
        and full scaling (for low frequencies), similar to NTK-by-parts
        but with a smoother transition.

        Args:
            inv_freq_base: Base inverse frequencies

        Returns:
            Per-dimension scaling factors
        """
        half_dim = self.dim // 2

        # Compute wavelengths for each frequency dimension
        wavelengths = 2 * math.pi / inv_freq_base

        # Ratio of wavelength to original training length
        ratio = wavelengths / self.original_max_position

        # Progressive scaling: smooth interpolation
        # Dimensions with short wavelengths (< original_max) need less scaling
        # Dimensions with long wavelengths (> original_max) need more scaling
        factors = torch.where(
            ratio < 1.0,
            torch.ones_like(ratio),
            torch.clamp(ratio, 1.0, self.scale)
        )

        # Apply smoothing
        # Use a sigmoid-like ramp for smooth transition
        ramp = torch.sigmoid(4.0 * (torch.log(ratio) / math.log(self.scale) - 0.5))
        factors = 1.0 + ramp * (self.scale - 1.0)

        return factors

    def _compute_mscale(self) -> float:
        """Compute attention scaling correction for extended context."""
        if self.scale <= 1.0:
            return 1.0

        return (
            0.1 * math.log(self.scale) + 1.0
        ) ** self.mscale_all_dim * self.mscale

    def _compute_cos_sin(
        self,
        seq_len: int,
        factors: torch.Tensor,
        device: torch.device,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin embeddings with given scaling factors.

        Args:
            seq_len: Sequence length
            factors: Per-dimension scaling factors (dim // 2,)
            device: Device for computation
            position_ids: Optional explicit position indices

        Returns:
            cos, sin embeddings
        """
        # Apply per-dimension scaling to base frequencies
        inv_freq = self.inv_freq_base / factors

        # Compute position indices
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).float()

        # Compute angles
        freqs = position_ids.unsqueeze(-1).float() * inv_freq.unsqueeze(0).to(device)

        # Duplicate for full dimension
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos() * self._mscale
        sin = emb.sin() * self._mscale

        return cos, sin

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute LongRoPE position embeddings.

        Automatically selects between short-context and long-context
        scaling factors based on the sequence length.

        Args:
            x: Input tensor (used for shape/device)
            position_ids: Explicit position indices (batch, seq_len)
            seq_len: Sequence length override

        Returns:
            cos: Cosine embeddings
            sin: Sine embeddings
        """
        if seq_len is None:
            if position_ids is not None:
                seq_len = position_ids.max().item() + 1
            else:
                seq_len = x.shape[1]

        # Select factors based on context length
        if seq_len <= self.original_max_position:
            # Short context: use short factors (minimal distortion)
            factors = self.short_factors

            if (seq_len <= self._seq_len_cached_short
                    and self._cos_cached_short is not None):
                return (
                    self._cos_cached_short[:, :seq_len, :],
                    self._sin_cached_short[:, :seq_len, :]
                )

            cos, sin = self._compute_cos_sin(
                seq_len, factors, x.device, position_ids
            )

            self._cos_cached_short = cos
            self._sin_cached_short = sin
            self._seq_len_cached_short = seq_len
        else:
            # Long context: use searched/heuristic factors
            factors = self.long_factors

            if (seq_len <= self._seq_len_cached_long
                    and self._cos_cached_long is not None):
                return (
                    self._cos_cached_long[:, :seq_len, :],
                    self._sin_cached_long[:, :seq_len, :]
                )

            cos, sin = self._compute_cos_sin(
                seq_len, factors, x.device, position_ids
            )

            self._cos_cached_long = cos
            self._sin_cached_long = sin
            self._seq_len_cached_long = seq_len

        return cos, sin

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply LongRoPE rotary embeddings to Q and K.

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

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed

    def get_scaling_info(self) -> dict:
        """Return information about the current scaling configuration.

        Returns:
            Dict with scaling metadata
        """
        return {
            'dim': self.dim,
            'scale': self.scale,
            'original_max_position': self.original_max_position,
            'max_position': self.max_position,
            'long_factors_range': (
                self.long_factors.min().item(),
                self.long_factors.max().item()
            ),
            'short_factors_range': (
                self.short_factors.min().item(),
                self.short_factors.max().item()
            ),
            'mscale': self._mscale,
        }
