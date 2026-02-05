"""
StripedHyena - Attention-Hyena Hybrid for Long Context (128K tokens).

StripedHyena alternates between:
1. Hyena operators: Long convolutions with data-controlled gating
2. Standard attention blocks: For precise token retrieval

This "striped" pattern (alternating layers) combines the efficiency of
Hyena for long-range processing with the precision of attention for
complex reasoning, achieving 128K context length support.

Key innovations:
- Hyena layers handle bulk of long-range dependencies efficiently
- Attention layers provide precise token-level interactions where needed
- Frequency-domain convolution for O(N log N) computation
- Data-controlled filters adapt to input statistics

The architecture achieves strong performance on long-context tasks while
maintaining efficiency comparable to pure Hyena models.

Reference: Poli et al., "Hyena Hierarchy: Towards Larger Convolutional
    Language Models", Together AI, 2023. https://arxiv.org/abs/2302.10866
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class HyenaFilter(NexusModule):
    """Implicit long convolution filter for Hyena.

    Uses a small MLP to generate position-dependent filter coefficients,
    enabling data-controlled long convolutions.

    Args:
        d_model: Model dimension.
        seq_len: Maximum sequence length.
        order: Order of Hyena (number of filters).
        filter_hidden_dim: Hidden dimension for filter MLP.
    """
    def __init__(
        self,
        d_model: int,
        seq_len: int = 8192,
        order: int = 2,
        filter_hidden_dim: int = 64
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.order = order

        # Positional encoding for filter generation
        self.register_buffer('positions', torch.linspace(0, 1, seq_len).unsqueeze(0))

        # Filter MLPs (one per order)
        self.filter_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, filter_hidden_dim),
                nn.GELU(),
                nn.Linear(filter_hidden_dim, d_model)
            )
            for _ in range(order)
        ])

    def forward(self, L: int) -> list:
        """Generate implicit filters.

        Args:
            L: Sequence length.

        Returns:
            filters: List of order filters, each shape (d_model, L).
        """
        # Get positions
        pos = self.positions[:, :L].transpose(0, 1)  # (L, 1)

        # Generate filters
        filters = []
        for mlp in self.filter_mlps:
            filt = mlp(pos)  # (L, d_model)
            filt = filt.transpose(0, 1)  # (d_model, L)
            filters.append(filt)

        return filters


class HyenaOperator(NexusModule):
    """Hyena operator with data-controlled implicit convolutions.

    Implements the Hyena recurrence:
        v_0 = input
        v_i = v_{i-1} * h_i(v_{i-1})    for i = 1, ..., order

    where h_i is a data-controlled long convolution filter.

    Args:
        d_model: Model dimension.
        seq_len: Maximum sequence length.
        order: Hyena order (typically 2 or 3).
    """
    def __init__(
        self,
        d_model: int,
        seq_len: int = 8192,
        order: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.order = order
        self.seq_len = seq_len

        # Implicit filter generator
        self.filter_gen = HyenaFilter(d_model, seq_len, order)

        # Input projections (short convolutions for data control)
        self.in_proj = nn.Conv1d(d_model, d_model * (order + 1), kernel_size=3, padding=1, groups=d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def _fft_conv(self, u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Fast convolution via FFT.

        Args:
            u: Input signal of shape (batch, d_model, L).
            k: Convolution kernel of shape (d_model, L).

        Returns:
            output: Convolved signal of shape (batch, d_model, L).
        """
        L = u.shape[-1]

        # FFT-based convolution
        u_fft = torch.fft.rfft(u, n=2*L, dim=-1)
        k_fft = torch.fft.rfft(k.unsqueeze(0), n=2*L, dim=-1)

        output_fft = u_fft * k_fft
        output = torch.fft.irfft(output_fft, n=2*L, dim=-1)[..., :L]

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Hyena operator.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            output: Output of shape (batch, seq_len, d_model).
        """
        batch, seq_len, d_model = x.shape

        # Short convolution for data control
        x_transpose = x.transpose(1, 2)  # (batch, d_model, seq_len)
        v = self.in_proj(x_transpose)  # (batch, d_model * (order + 1), seq_len)

        # Split into order + 1 branches
        v = v.view(batch, self.order + 1, d_model, seq_len)
        v_list = [v[:, i] for i in range(self.order + 1)]

        # Generate implicit filters
        filters = self.filter_gen(seq_len)

        # Apply Hyena hierarchy
        output = v_list[0]
        for i in range(self.order):
            # Data-controlled gating
            gate = torch.sigmoid(v_list[i + 1])

            # Long convolution with implicit filter
            conv_out = self._fft_conv(output, filters[i])

            # Combine
            output = gate * conv_out

        # Transpose back and project
        output = output.transpose(1, 2)  # (batch, seq_len, d_model)
        output = self.out_proj(output)

        return output


class AttentionBlock(NexusModule):
    """Standard multi-head attention block for StripedHyena.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model

        # QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            mask: Optional attention mask.

        Returns:
            output: Output of shape (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Transpose: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2) * self.scale
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        if mask is not None:
            attn_scores = attn_scores + mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        # Reshape
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.out_proj(output)

        return output


class StripedHyenaBlock(NexusModule):
    """StripedHyena block (either Hyena or Attention).

    Args:
        d_model: Model dimension.
        block_type: Either 'hyena' or 'attention'.
        num_heads: Number of heads (for attention).
        seq_len: Max sequence length (for Hyena).
        hyena_order: Hyena order.
        d_ff: Feedforward dimension.
        dropout: Dropout probability.
    """
    def __init__(
        self,
        d_model: int,
        block_type: str = 'hyena',
        num_heads: int = 8,
        seq_len: int = 8192,
        hyena_order: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.block_type = block_type

        if d_ff is None:
            d_ff = 4 * d_model

        # Main block
        self.norm1 = nn.LayerNorm(d_model)
        if block_type == 'hyena':
            self.main_block = HyenaOperator(d_model, seq_len, hyena_order)
        elif block_type == 'attention':
            self.main_block = AttentionBlock(d_model, num_heads, dropout)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        # Feedforward
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            mask: Optional attention mask (only for attention blocks).

        Returns:
            x: Output of shape (batch, seq_len, d_model).
        """
        # Main block (Hyena or Attention)
        if self.block_type == 'attention':
            x = x + self.dropout(self.main_block(self.norm1(x), mask))
        else:
            x = x + self.dropout(self.main_block(self.norm1(x)))

        # Feedforward
        x = x + self.ffn(self.norm2(x))

        return x


class StripedHyenaModel(NexusModule):
    """Complete StripedHyena Model with Alternating Hyena-Attention Layers.

    Creates a "striped" pattern of Hyena and attention layers.

    Args:
        d_model: Model dimension.
        n_layers: Number of layers (total).
        num_heads: Number of attention heads.
        seq_len: Maximum sequence length.
        hyena_order: Hyena operator order.
        d_ff: Feedforward dimension.
        dropout: Dropout probability.
        attention_every_n: Place attention layer every N layers (e.g., 4).
    """
    def __init__(
        self,
        d_model: int,
        n_layers: int = 32,
        num_heads: int = 8,
        seq_len: int = 131072,  # 128K context
        hyena_order: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        attention_every_n: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        if d_ff is None:
            d_ff = 4 * d_model

        # Create alternating layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # Every N-th layer is attention, rest are Hyena
            block_type = 'attention' if (i + 1) % attention_every_n == 0 else 'hyena'

            self.layers.append(
                StripedHyenaBlock(
                    d_model=d_model,
                    block_type=block_type,
                    num_heads=num_heads,
                    seq_len=seq_len,
                    hyena_order=hyena_order,
                    d_ff=d_ff,
                    dropout=dropout
                )
            )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through all layers.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            mask: Optional attention mask.

        Returns:
            x: Output of shape (batch, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        return x
