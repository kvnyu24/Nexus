"""
Hyena - Sub-Quadratic Attention Replacement via Long Convolutions.

Hyena replaces the attention mechanism with a hierarchy of long convolutions
and element-wise gating operations. The key insight is that attention can be
decomposed into a series of element-wise multiplicative interactions and
linear operators -- Hyena replaces the costly softmax attention with
implicitly parametrized long convolutions.

Key innovations:
1. Implicit convolution filters: Instead of materializing a full N x N attention
   matrix, Hyena parametrizes the convolution filters implicitly using a small
   feed-forward network that maps position encodings to filter values.

2. Data-controlled gating: Short linear projections of the input provide
   multiplicative gates that control information flow, similar to how attention
   values weight the attention output.

3. Sub-quadratic complexity: The combination of FFT-based convolution (O(N log N))
   and element-wise gating (O(N)) gives overall sub-quadratic complexity.

4. Order parameter: The "order" N of Hyena controls the depth of the gating
   hierarchy. Order 2 approximates standard attention (Q, K, V), while higher
   orders increase expressivity.

Reference: Poli et al., "Hyena Hierarchy: Towards Larger Convolutional Language
    Models", ICML 2023. https://arxiv.org/abs/2302.10866
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class PositionalEncoding(NexusModule):
    """Sinusoidal positional encoding for implicit filter parametrization.

    Generates sinusoidal position embeddings that are fed into the filter
    FFN to produce position-dependent convolution weights.

    Args:
        d_model: Embedding dimension.
        max_seq_len: Maximum sequence length.
    """

    def __init__(self, d_model: int, max_seq_len: int = 8192):
        super().__init__()
        self.d_model = d_model

        # Precompute sinusoidal embeddings
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        self.register_buffer('pe', pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Get positional encoding.

        Args:
            seq_len: Sequence length.

        Returns:
            Positional encoding of shape (seq_len, d_model).
        """
        return self.pe[:seq_len]


class ImplicitFilter(NexusModule):
    """Implicitly Parametrized Convolution Filter.

    Instead of storing a full convolution kernel of length L, this module
    parametrizes the filter implicitly using a small FFN that maps position
    encodings to filter values. This enables:
    - Parameter efficiency: O(filter_order) parameters instead of O(L)
    - Length generalization: can generate filters for any sequence length

    The filter is generated as:
        h(t) = FFN(pos_encoding(t)) * window(t)

    where window(t) provides exponential decay for causality.

    Args:
        d_model: Output filter dimension (number of channels).
        filter_order: Hidden dimension of the filter FFN (default: 64).
        max_seq_len: Maximum sequence length (default: 8192).
        num_inner_mlps: Number of hidden layers in filter FFN (default: 1).
    """

    def __init__(
        self,
        d_model: int,
        filter_order: int = 64,
        max_seq_len: int = 8192,
        num_inner_mlps: int = 1
    ):
        super().__init__()
        self.d_model = d_model
        self.filter_order = filter_order

        # Positional encoding for filter input
        self.pos_encoding = PositionalEncoding(filter_order, max_seq_len)

        # Filter FFN: pos_encoding -> filter values
        layers = [nn.Linear(filter_order, filter_order), nn.SiLU()]
        for _ in range(num_inner_mlps):
            layers.extend([nn.Linear(filter_order, filter_order), nn.SiLU()])
        layers.append(nn.Linear(filter_order, d_model))
        self.filter_ffn = nn.Sequential(*layers)

        # Exponential decay window for causality
        self.decay = nn.Parameter(torch.linspace(0.1, 2.0, d_model))

    def forward(self, seq_len: int) -> torch.Tensor:
        """Generate the implicit convolution filter.

        Args:
            seq_len: Length of the filter to generate.

        Returns:
            filter: Convolution filter of shape (d_model, seq_len).
        """
        # Get positional encoding: (seq_len, filter_order)
        pos = self.pos_encoding(seq_len)

        # Generate filter via FFN: (seq_len, d_model)
        h = self.filter_ffn(pos)

        # Apply exponential decay window
        t = torch.arange(seq_len, device=h.device, dtype=h.dtype)
        window = torch.exp(-self.decay.unsqueeze(1) * t.unsqueeze(0))  # (d_model, seq_len)

        # Transpose and apply window
        h = h.t() * window  # (d_model, seq_len)

        return h


class HyenaOperator(NexusModule):
    """Hyena Operator - replaces attention with long convolution + gating.

    The Hyena operator of order N computes:
        y = h_N * (v_N . (h_{N-1} * (v_{N-1} . ... (h_1 * (v_1 . x)) ...)))

    where:
    - v_i are data-dependent projections (gates) from the input
    - h_i are implicitly parametrized convolution filters
    - * denotes (causal) convolution
    - . denotes element-wise multiplication

    For order=2 (default), this simplifies to:
        y = h_2 * (v_2 . (h_1 * (v_1 . x)))

    which is analogous to attention: v_1 ~ Q, v_2 ~ V, h_i ~ attention pattern.

    Args:
        d_model: Model dimension.
        max_seq_len: Maximum sequence length (default: 8192).
        order: Hyena order N (default: 2). Higher = more expressive.
        filter_order: Hidden dim for implicit filter FFN (default: 64).
        dropout: Dropout probability (default: 0.0).
        short_filter_order: Short convolution kernel size (default: 3).
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 8192,
        order: int = 2,
        filter_order: int = 64,
        dropout: float = 0.0,
        short_filter_order: int = 3
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.order = order
        self.filter_order = filter_order

        # Input projection: produces (order + 1) projections
        # First projection is the "value" (x), rest are gates (v_1, ..., v_N)
        self.in_proj = nn.Linear(d_model, d_model * (order + 1), bias=False)

        # Short depthwise convolutions for each projection (local context)
        self.short_convs = nn.ModuleList([
            nn.Conv1d(
                d_model, d_model,
                kernel_size=short_filter_order,
                padding=short_filter_order - 1,
                groups=d_model,
                bias=True
            )
            for _ in range(order + 1)
        ])

        # Implicit long convolution filters (one per order)
        self.long_filters = nn.ModuleList([
            ImplicitFilter(
                d_model=d_model,
                filter_order=filter_order,
                max_seq_len=max_seq_len
            )
            for _ in range(order)
        ])

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass of Hyena operator.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Not used (included for interface compatibility).

        Returns:
            output: Shape (batch, seq_len, d_model).
            state: None.
        """
        batch_size, seq_len, _ = x.shape

        # Project input to (order + 1) branches
        projections = self.in_proj(x)  # (batch, seq, d_model * (order + 1))
        projections = projections.view(
            batch_size, seq_len, self.order + 1, self.d_model
        )

        # Apply short convolutions to each projection
        branches = []
        for i in range(self.order + 1):
            p_i = projections[:, :, i, :]  # (batch, seq, d_model)
            p_i = p_i.transpose(1, 2)  # (batch, d_model, seq)
            p_i = self.short_convs[i](p_i)[:, :, :seq_len]
            p_i = p_i.transpose(1, 2)  # (batch, seq, d_model)
            branches.append(p_i)

        # branches[0] is the "value" x, branches[1:] are gates v_1, ..., v_N
        y = branches[0]  # Start with x (the value)

        # Iteratively apply: y = h_i * (v_i . y)
        for i in range(self.order):
            # Element-wise gating
            y = y * branches[i + 1]

            # Long convolution via FFT
            h = self.long_filters[i](seq_len)  # (d_model, seq_len)
            y = self._fft_conv(y, h)

        y = self.dropout(y)
        output = self.out_proj(y)

        return output, None

    def _fft_conv(
        self,
        x: torch.Tensor,
        h: torch.Tensor
    ) -> torch.Tensor:
        """Causal convolution via FFT.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            h: Filter of shape (d_model, seq_len).

        Returns:
            y: Convolved output of shape (batch, seq_len, d_model).
        """
        seq_len = x.shape[1]
        fft_size = 2 * seq_len  # Pad for linear (non-circular) convolution

        # FFT of input: (batch, d_model, fft_size)
        x_t = x.transpose(1, 2)
        x_fft = torch.fft.rfft(x_t, n=fft_size, dim=-1)

        # FFT of filter: (d_model, fft_size)
        h_fft = torch.fft.rfft(h, n=fft_size, dim=-1)

        # Multiply in frequency domain
        y_fft = x_fft * h_fft.unsqueeze(0)

        # IFFT
        y = torch.fft.irfft(y_fft, n=fft_size, dim=-1)
        y = y[:, :, :seq_len]  # Truncate (causal)

        return y.transpose(1, 2)


class HyenaBlock(NexusModule):
    """Hyena block with pre-normalization, residual connection, and FFN.

    Standard transformer-style block with Hyena replacing attention:
        x -> norm -> HyenaOperator -> dropout -> + residual
        x -> norm -> FFN -> + residual

    Args:
        d_model: Model dimension.
        max_seq_len: Maximum sequence length.
        order: Hyena order (default: 2).
        filter_order: Filter FFN hidden dim (default: 64).
        ffn_expand: FFN expansion factor (default: 4).
        dropout: Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 8192,
        order: int = 2,
        filter_order: int = 64,
        ffn_expand: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model

        # Hyena operator branch
        self.norm1 = nn.LayerNorm(d_model)
        self.hyena = HyenaOperator(
            d_model=d_model,
            max_seq_len=max_seq_len,
            order=order,
            filter_order=filter_order,
            dropout=dropout
        )
        self.dropout1 = nn.Dropout(dropout)

        # FFN branch
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expand, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Not used.

        Returns:
            output: Shape (batch, seq_len, d_model).
            state: None.
        """
        # Hyena with residual
        residual = x
        x = self.norm1(x)
        x, _ = self.hyena(x)
        x = self.dropout1(x)
        x = x + residual

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x, None


class HyenaModel(NexusModule):
    """Full Hyena language model.

    Stacks multiple HyenaBlocks with embedding and language model head.
    Achieves sub-quadratic complexity for long sequences while maintaining
    competitive performance with attention-based models.

    Reference: Poli et al., "Hyena Hierarchy: Towards Larger Convolutional
        Language Models", ICML 2023. https://arxiv.org/abs/2302.10866

    Args:
        d_model: Model dimension.
        num_layers: Number of Hyena blocks.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length (default: 8192).
        order: Hyena order (default: 2).
        filter_order: Filter FFN hidden dim (default: 64).
        ffn_expand: FFN expansion factor (default: 4).
        dropout: Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        vocab_size: int,
        max_seq_len: int = 8192,
        order: int = 2,
        filter_order: int = 64,
        ffn_expand: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Hyena blocks
        self.blocks = nn.ModuleList([
            HyenaBlock(
                d_model=d_model,
                max_seq_len=max_seq_len,
                order=order,
                filter_order=filter_order,
                ffn_expand=ffn_expand,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Output head
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size).
        """
        x = self.embedding(input_ids)
        x = self.embed_dropout(x)

        for block in self.blocks:
            x, _ = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits
