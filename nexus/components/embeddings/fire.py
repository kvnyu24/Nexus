"""
FIRE: Functional Interpolation for Relative Positional Encoding.

FIRE provides a general framework for relative position encoding that can
represent many existing methods (ALiBi, T5 RPE, Kerple) as special cases.
It works by:
    1. Progressive interpolation: Maps arbitrary relative positions to a
       bounded domain [0, 1] using a learnable threshold
    2. Learned mapping: A small MLP transforms the interpolated positions
       into position bias values

This approach naturally handles length generalization because all positions
are mapped to the same bounded domain, regardless of absolute distance.

The progressive interpolation ensures that:
    - Nearby positions retain fine-grained distinctions
    - Distant positions are smoothly compressed
    - The mapping adapts to the data during training

Reference: https://arxiv.org/abs/2310.04418 (FIRE: Functional Interpolation for
           Relative Positional Encoding, Enables Length Generalization)

See Also:
    - alibi.py: ALiBi (linear relative bias, special case of FIRE)
    - relative_bias.py: T5-style learned relative position bias
    - cope.py: Contextual position encoding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class ProgressiveInterpolation(NexusModule):
    """Progressive interpolation mapping for relative positions.

    Maps relative positions from [0, infinity) to [0, 1] using a
    learnable threshold parameter. Positions below the threshold map
    linearly, while positions above are progressively compressed.

    The mapping function is:
        f(d) = d / L              if d <= L
        f(d) = 1 - c * exp(-d/L)  if d > L (asymptotically approaches 1)

    where L is the learnable threshold and c is a normalization constant.

    In practice, we use a smooth version:
        f(d) = log(1 + d) / log(1 + L)  (clamped to [0, 1])

    Args:
        num_heads: Number of attention heads (each gets independent mapping)
        init_threshold: Initial threshold value (learnable)
    """

    def __init__(
        self,
        num_heads: int,
        init_threshold: float = 512.0
    ):
        super().__init__()
        self.num_heads = num_heads

        # Learnable threshold per head (in log space for stability)
        self.log_threshold = nn.Parameter(
            torch.full((num_heads,), math.log(init_threshold))
        )

    @property
    def threshold(self) -> torch.Tensor:
        """Current threshold values."""
        return self.log_threshold.exp()

    def forward(self, relative_positions: torch.Tensor) -> torch.Tensor:
        """Map relative positions to [0, 1].

        Args:
            relative_positions: Non-negative relative distances
                Shape: (seq_len, seq_len) or (batch, seq_len, seq_len)

        Returns:
            Interpolated positions in [0, 1]
                Shape: (num_heads, seq_len, seq_len)
        """
        # Ensure non-negative
        positions = relative_positions.float().abs()

        # Get thresholds per head
        threshold = self.threshold  # (num_heads,)

        # Add head dimension
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)  # (1, S, S)

        # Progressive interpolation: log mapping normalized by threshold
        # f(d) = log(1 + d) / log(1 + L), clamped to [0, 1]
        threshold = threshold.view(-1, 1, 1)  # (H, 1, 1)
        interpolated = torch.log1p(positions) / torch.log1p(threshold)
        interpolated = interpolated.clamp(0.0, 1.0)

        return interpolated


class FIRE(NexusModule):
    """Functional Interpolation for Relative Positional Encoding.

    Combines progressive interpolation with a learned MLP to produce
    relative position biases. The MLP transforms bounded [0,1] inputs
    into per-head attention biases.

    Generality:
        - With linear MLP and high threshold: approximates ALiBi
        - With lookup-table MLP: approximates T5 RPE
        - With specific nonlinearity: approximates Kerple

    Args:
        dim: Model dimension (used for initialization scaling)
        num_heads: Number of attention heads
        max_position: Maximum relative position (for threshold init)
        num_layers: Number of MLP layers (default 2)
        mlp_width: Hidden width of the MLP. Defaults to 32.
        init_threshold: Initial progressive interpolation threshold
        bias: Whether to use bias in MLP layers

    Example:
        >>> fire = FIRE(dim=512, num_heads=8, max_position=8192)
        >>> # Use as attention bias
        >>> bias = fire(seq_len=1024)
        >>> bias.shape
        torch.Size([8, 1024, 1024])
        >>> # Add to attention scores
        >>> attn_scores = attn_scores + bias.unsqueeze(0)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_position: int = 8192,
        num_layers: int = 2,
        mlp_width: int = 32,
        init_threshold: Optional[float] = None,
        bias: bool = True
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.max_position = max_position
        self.num_layers = num_layers

        # Progressive interpolation
        threshold = init_threshold or float(max_position)
        self.interpolation = ProgressiveInterpolation(
            num_heads=num_heads,
            init_threshold=threshold
        )

        # Learned mapping MLP: [0, 1] -> R (per head)
        # Input: 1 (interpolated position) per head
        # Output: 1 (bias value) per head
        layers = []
        in_features = 1

        for i in range(num_layers):
            out_features = mlp_width if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_features, out_features, bias=bias))
            if i < num_layers - 1:
                layers.append(nn.GELU())
            in_features = out_features

        self.mlp = nn.Sequential(*layers)

        # Initialize for stable training
        self._init_weights()

    def _init_weights(self):
        """Initialize MLP weights for near-zero initial bias."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _compute_relative_positions(
        self,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device
    ) -> torch.Tensor:
        """Compute relative position matrix.

        Args:
            seq_len_q: Query sequence length
            seq_len_k: Key sequence length
            device: Device for tensors

        Returns:
            Relative positions (seq_len_q, seq_len_k)
        """
        q_pos = torch.arange(seq_len_q, device=device).float()
        k_pos = torch.arange(seq_len_k, device=device).float()

        # Relative distances (absolute value)
        relative_pos = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs()

        return relative_pos

    def forward(
        self,
        seq_len: Optional[int] = None,
        seq_len_q: Optional[int] = None,
        seq_len_k: Optional[int] = None,
        relative_positions: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Compute FIRE position bias.

        Can be called with either:
            - seq_len (for self-attention, square bias matrix)
            - seq_len_q + seq_len_k (for cross-attention or asymmetric)
            - relative_positions (pre-computed distances)

        Args:
            seq_len: Sequence length (for square self-attention bias)
            seq_len_q: Query sequence length
            seq_len_k: Key sequence length
            relative_positions: Pre-computed relative distances
            device: Device for computation

        Returns:
            Position bias tensor (num_heads, seq_len_q, seq_len_k)
        """
        if relative_positions is None:
            if seq_len is not None:
                seq_len_q = seq_len_q or seq_len
                seq_len_k = seq_len_k or seq_len
            assert seq_len_q is not None and seq_len_k is not None, \
                "Must provide seq_len, (seq_len_q, seq_len_k), or relative_positions"

            if device is None:
                device = next(self.parameters()).device

            relative_positions = self._compute_relative_positions(
                seq_len_q, seq_len_k, device
            )

        # Progressive interpolation: map to [0, 1]
        # Shape: (num_heads, seq_len_q, seq_len_k)
        interpolated = self.interpolation(relative_positions)

        # Apply learned MLP per head
        # We need to process each head's interpolated values through the shared MLP
        # interpolated: (num_heads, S_q, S_k) -> reshape for MLP
        num_heads, s_q, s_k = interpolated.shape

        # Reshape: (num_heads * S_q * S_k, 1)
        mlp_input = interpolated.reshape(-1, 1)

        # Pass through MLP
        bias_values = self.mlp(mlp_input)  # (num_heads * S_q * S_k, 1)

        # Reshape back: (num_heads, S_q, S_k)
        bias = bias_values.view(num_heads, s_q, s_k)

        return bias

    def forward_incremental(
        self,
        query_pos: int,
        seq_len_k: int,
        device: torch.device
    ) -> torch.Tensor:
        """Compute FIRE bias for a single query position (incremental decoding).

        More efficient than computing the full matrix during generation.

        Args:
            query_pos: Position of the current query token
            seq_len_k: Total key sequence length
            device: Device for computation

        Returns:
            Position bias (num_heads, 1, seq_len_k)
        """
        k_pos = torch.arange(seq_len_k, device=device).float()
        relative_pos = (query_pos - k_pos).abs().unsqueeze(0)  # (1, S_k)

        # Interpolate
        interpolated = self.interpolation(relative_pos)  # (H, 1, S_k)

        # MLP
        H, _, S_k = interpolated.shape
        mlp_input = interpolated.reshape(-1, 1)
        bias_values = self.mlp(mlp_input)
        bias = bias_values.view(H, 1, S_k)

        return bias
