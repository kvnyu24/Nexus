"""
Resonance RoPE: Snapping RoPE frequencies to integer wavelengths.

Standard RoPE assigns frequencies theta_i = base^(-2i/d), which generally
produce non-integer wavelengths. When extending to longer contexts, these
non-integer wavelengths create destructive interference patterns, degrading
position encoding quality.

Resonance RoPE fixes this by "snapping" each frequency to the nearest
integer wavelength, ensuring that the position encoding repeats cleanly
at extended lengths. This is analogous to tuning a musical instrument
to resonant frequencies.

Key properties:
    - Preserves RoPE's rotation structure and relative position encoding
    - Eliminates destructive interference from non-integer wavelengths
    - Can be applied on top of YaRN or other scaling methods
    - Minimal computation overhead (just frequency adjustment)

The snapping operation:
    Original: lambda_i = 2*pi / theta_i  (may be non-integer)
    Snapped:  lambda_i' = round(lambda_i) (integer wavelength)
    New freq: theta_i' = 2*pi / lambda_i'

Reference: https://arxiv.org/abs/2403.00071 (Resonance RoPE: Improving Context
           Length Generalization of Large Language Models)

See Also:
    - rotary_embedding.py: Standard RoPE
    - yarn.py: YaRN (can be combined with Resonance RoPE)
    - ntk_rope.py: NTK-Aware scaling
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class ResonanceRoPE(NexusModule):
    """Resonance RoPE with integer wavelength snapping.

    Adjusts RoPE frequencies so that each has an integer wavelength,
    ensuring clean periodicity at extended context lengths.

    This can be used standalone or layered on top of other scaling
    methods (YaRN, NTK, etc.) by providing pre-scaled frequencies.

    Args:
        dim: Embedding dimension (must be even)
        max_position: Maximum position embeddings
        base: Base frequency for RoPE (default 10000)
        pre_scaled_inv_freq: Optional pre-scaled inverse frequencies from
            another method (e.g., YaRN). If provided, resonance snapping
            is applied on top of these frequencies instead of the base ones.
        snap_threshold: Minimum wavelength to snap. Frequencies with
            wavelengths below this are left unmodified (they represent
            very local patterns that don't benefit from snapping).

    Example:
        >>> # Standalone Resonance RoPE
        >>> rope = ResonanceRoPE(dim=128, max_position=32768)
        >>> x = torch.randn(2, 8192, 1, 128)
        >>> cos, sin = rope(x)
        >>>
        >>> # Combined with YaRN
        >>> yarn = YaRN(dim=128, scale=4.0, ...)
        >>> yarn_inv_freq = yarn.inv_freq
        >>> rope = ResonanceRoPE(dim=128, pre_scaled_inv_freq=yarn_inv_freq)
    """

    def __init__(
        self,
        dim: int,
        max_position: int = 8192,
        base: float = 10000.0,
        pre_scaled_inv_freq: Optional[torch.Tensor] = None,
        snap_threshold: float = 2.0
    ):
        super().__init__()

        assert dim % 2 == 0, f"dim must be even, got {dim}"

        self.dim = dim
        self.max_position = max_position
        self.base = base
        self.snap_threshold = snap_threshold

        if pre_scaled_inv_freq is not None:
            # Apply resonance snapping on top of pre-scaled frequencies
            inv_freq = self._snap_to_resonance(pre_scaled_inv_freq)
        else:
            # Compute base frequencies and snap
            base_inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2).float() / dim)
            )
            inv_freq = self._snap_to_resonance(base_inv_freq)

        self.register_buffer('inv_freq', inv_freq)

        # Also store the original (unsnapped) frequencies for comparison
        if pre_scaled_inv_freq is not None:
            self.register_buffer('original_inv_freq', pre_scaled_inv_freq.clone())
        else:
            original = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer('original_inv_freq', original)

        # Cache
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _snap_to_resonance(self, inv_freq: torch.Tensor) -> torch.Tensor:
        """Snap inverse frequencies to produce integer wavelengths.

        For each frequency theta_i:
            wavelength_i = 2*pi / theta_i
            snapped_wavelength_i = round(wavelength_i)
            snapped_theta_i = 2*pi / snapped_wavelength_i

        Frequencies with wavelengths below snap_threshold are left unchanged
        since they represent very local patterns.

        Args:
            inv_freq: Inverse frequency tensor (dim // 2,)

        Returns:
            Snapped inverse frequencies (dim // 2,)
        """
        # Compute wavelengths
        wavelengths = 2 * math.pi / inv_freq

        # Only snap wavelengths above threshold
        should_snap = wavelengths >= self.snap_threshold

        # Snap to nearest integer
        snapped_wavelengths = torch.where(
            should_snap,
            wavelengths.round().clamp(min=1.0),
            wavelengths
        )

        # Convert back to inverse frequencies
        snapped_inv_freq = 2 * math.pi / snapped_wavelengths

        return snapped_inv_freq

    def get_wavelength_info(self) -> dict:
        """Return information about original vs snapped wavelengths.

        Useful for debugging and understanding the snapping behavior.

        Returns:
            Dict with wavelength comparison information
        """
        original_wavelengths = 2 * math.pi / self.original_inv_freq
        snapped_wavelengths = 2 * math.pi / self.inv_freq
        adjustment = (snapped_wavelengths - original_wavelengths).abs()

        return {
            'original_wavelengths': original_wavelengths,
            'snapped_wavelengths': snapped_wavelengths,
            'max_adjustment': adjustment.max().item(),
            'mean_adjustment': adjustment.mean().item(),
            'num_snapped': (adjustment > 1e-6).sum().item(),
            'num_total': self.dim // 2,
        }

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Resonance RoPE position embeddings.

        Args:
            x: Input tensor (for shape/device inference)
            position_ids: Explicit position indices (batch, seq_len)
            seq_len: Sequence length override

        Returns:
            cos: Cosine embeddings (1 or batch, seq_len, dim)
            sin: Sine embeddings (1 or batch, seq_len, dim)
        """
        if seq_len is None:
            if position_ids is not None:
                seq_len = position_ids.max().item() + 1
            else:
                seq_len = x.shape[1]

        # Check cache
        if seq_len <= self._seq_len_cached and self._cos_cached is not None:
            return (
                self._cos_cached[:, :seq_len, :],
                self._sin_cached[:, :seq_len, :]
            )

        # Compute position indices
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=x.device
            ).unsqueeze(0).float()

        # Compute angles
        freqs = position_ids.unsqueeze(-1).float() * self.inv_freq.unsqueeze(0).to(x.device)

        # Duplicate for full dimension
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()
        sin = emb.sin()

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
        sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Resonance RoPE rotary embeddings to Q and K.

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


class ResonanceYaRN(NexusModule):
    """Resonance RoPE applied on top of YaRN scaling.

    Convenience class that first computes YaRN-scaled frequencies,
    then applies resonance snapping. This combination provides:
        - YaRN's NTK-by-parts interpolation for smooth scaling
        - Resonance's integer wavelength guarantee for clean periodicity

    Args:
        dim: Embedding dimension
        max_position: Maximum position (extended context)
        base: Base frequency
        scale: YaRN context extension scale
        original_max_position: Original training length
        beta_fast: YaRN fast dimension cutoff
        beta_slow: YaRN slow dimension cutoff
        snap_threshold: Minimum wavelength to snap
    """

    def __init__(
        self,
        dim: int,
        max_position: int = 32768,
        base: float = 10000.0,
        scale: float = 4.0,
        original_max_position: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        snap_threshold: float = 2.0
    ):
        super().__init__()

        self.dim = dim
        self.scale = scale

        # Step 1: Compute YaRN-scaled frequencies
        pos_freqs = base ** (torch.arange(0, dim, 2).float() / dim)
        inv_freq_base = 1.0 / pos_freqs
        wavelengths = 2 * math.pi * pos_freqs

        low_bound = original_max_position / beta_slow
        high_bound = original_max_position / beta_fast

        ramp = torch.clamp(
            (wavelengths - high_bound) / (low_bound - high_bound),
            0.0, 1.0
        )

        yarn_inv_freq = inv_freq_base / (scale ** ramp)

        # Step 2: Apply Resonance snapping
        self.resonance_rope = ResonanceRoPE(
            dim=dim,
            max_position=max_position,
            base=base,
            pre_scaled_inv_freq=yarn_inv_freq,
            snap_threshold=snap_threshold
        )

        # YaRN mscale
        if scale > 1.0:
            self._mscale = 0.1 * math.log(scale) + 1.0
        else:
            self._mscale = 1.0

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Resonance YaRN position embeddings.

        Args:
            x: Input tensor
            position_ids: Optional position indices
            seq_len: Sequence length

        Returns:
            cos, sin embeddings (with mscale applied)
        """
        cos, sin = self.resonance_rope(x, position_ids, seq_len)

        # Apply YaRN mscale
        cos = cos * self._mscale
        sin = sin * self._mscale

        return cos, sin

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings."""
        return self.resonance_rope.apply_rotary_pos_emb(q, k, cos, sin)
