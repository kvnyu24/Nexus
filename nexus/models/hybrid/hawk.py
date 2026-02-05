"""
Hawk - Pure Gated Linear Recurrence Model.

Hawk is the pure recurrence variant of Griffin, using only RGLRU blocks without
any attention layers. It represents the extreme efficiency point of the
recurrence-attention tradeoff, offering:

1. O(1) memory complexity during inference (no KV cache)
2. Fully parallel training through associative scan
3. Strong performance on tasks with long-range dependencies
4. Significant throughput advantages over transformer models

While Griffin interleaves attention for precision, Hawk demonstrates that
pure gated linear recurrence can be competitive on many tasks, especially
those emphasizing efficiency over absolute quality.

Reference: De et al., "Griffin: Mixing Gated Linear Recurrences with Local
    Attention for Efficient Language Models", Google DeepMind, 2024.
    https://arxiv.org/abs/2402.19427

Note: Hawk is the recurrence-only component extracted from Griffin.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class RMSNorm(NexusModule):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class RGLRU(NexusModule):
    """Real-Gated Linear Recurrent Unit.

    The core Hawk recurrent component with diagonal gating and
    magnitude-preserving updates.

    Args:
        d_model: Model dimension.
        d_recurrence: Recurrence state dimension.
    """
    def __init__(self, d_model: int, d_recurrence: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.d_recurrence = d_recurrence or d_model

        # Input projection
        self.x_proj = nn.Linear(d_model, self.d_recurrence, bias=False)

        # Recurrence gate (sigmoid activation)
        self.a_proj = nn.Linear(d_model, self.d_recurrence, bias=True)

        # Initialize gate bias to favor remembering (high a values)
        nn.init.constant_(self.a_proj.bias, 2.0)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with recurrence.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Previous state of shape (batch, d_recurrence).

        Returns:
            output: Output of shape (batch, seq_len, d_recurrence).
            state: Final state of shape (batch, d_recurrence).
        """
        batch_size, seq_len, _ = x.shape

        # Compute gate and input projection
        a = torch.sigmoid(self.a_proj(x))  # (batch, seq, d_rec)
        x_in = self.x_proj(x)  # (batch, seq, d_rec)

        # Scale input to preserve magnitude (unitary-like property)
        # This ensures h[t] has similar magnitude to h[t-1]
        x_in = x_in * torch.sqrt(1 - a ** 2 + 1e-8)

        # Initialize state if needed
        if state is None:
            state = torch.zeros(
                batch_size, self.d_recurrence,
                device=x.device, dtype=x.dtype
            )

        # Recurrence: h[t] = a[t] * h[t-1] + sqrt(1 - a[t]^2) * x[t]
        outputs = []
        for t in range(seq_len):
            state = a[:, t] * state + x_in[:, t]
            outputs.append(state)

        output = torch.stack(outputs, dim=1)

        return output, state


class TemporalConvolution(NexusModule):
    """Short temporal convolution for local context.

    Adds local receptive field before recurrence for better modeling.

    Args:
        d_model: Model dimension.
        kernel_size: Convolution kernel size.
    """
    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model  # Depthwise
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            output: Output of shape (batch, seq_len, d_model).
        """
        # Transpose for conv1d: (batch, d_model, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)

        # Remove extra padding (causal)
        x = x[:, :, :x.shape[-1] - self.conv.kernel_size + 1]

        # Transpose back
        x = x.transpose(1, 2)

        return x


class SwiGLU(NexusModule):
    """Swish-Gated Linear Unit for feedforward.

    Commonly used in modern efficient LLMs (e.g., LLaMA, PaLM).

    Args:
        d_model: Input dimension.
        d_ff: Hidden dimension.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (..., d_model).

        Returns:
            output: Output of shape (..., d_model).
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class HawkBlock(NexusModule):
    """Hawk block with RGLRU and feedforward.

    Args:
        d_model: Model dimension.
        d_recurrence: Recurrence state dimension.
        d_ff: Feedforward dimension.
        kernel_size: Temporal convolution kernel size.
        dropout: Dropout probability.
    """
    def __init__(
        self,
        d_model: int,
        d_recurrence: Optional[int] = None,
        d_ff: Optional[int] = None,
        kernel_size: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model

        if d_ff is None:
            d_ff = 4 * d_model

        # Pre-norm for recurrence
        self.norm1 = RMSNorm(d_model)

        # Temporal convolution (optional, for local context)
        self.temporal_conv = TemporalConvolution(d_model, kernel_size)

        # RGLRU
        self.rglru = RGLRU(d_model, d_recurrence)

        # Output projection
        self.out_proj = nn.Linear(d_recurrence or d_model, d_model, bias=False)

        # Pre-norm for FFN
        self.norm2 = RMSNorm(d_model)

        # Feedforward with SwiGLU
        self.ffn = SwiGLU(d_model, d_ff)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Recurrence state of shape (batch, d_recurrence).

        Returns:
            x: Output of shape (batch, seq_len, d_model).
            state: Updated state of shape (batch, d_recurrence).
        """
        # Recurrence block
        residual = x
        x = self.norm1(x)

        # Temporal convolution for local context
        x = self.temporal_conv(x)

        # RGLRU
        x, new_state = self.rglru(x, state)

        # Output projection and residual
        x = self.out_proj(x)
        x = residual + self.dropout(x)

        # Feedforward block
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x, new_state


class HawkModel(NexusModule):
    """Complete Hawk Model - Pure Gated Linear Recurrence.

    Stacks multiple Hawk blocks for efficient sequence modeling without attention.

    Args:
        d_model: Model dimension.
        n_layers: Number of layers.
        d_recurrence: Recurrence state dimension.
        d_ff: Feedforward dimension.
        kernel_size: Temporal convolution kernel size.
        dropout: Dropout probability.
    """
    def __init__(
        self,
        d_model: int,
        n_layers: int = 24,
        d_recurrence: Optional[int] = None,
        d_ff: Optional[int] = None,
        kernel_size: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Layers
        self.layers = nn.ModuleList([
            HawkBlock(
                d_model=d_model,
                d_recurrence=d_recurrence,
                d_ff=d_ff,
                kernel_size=kernel_size,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """Forward pass through all layers.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            states: List of recurrence states per layer.

        Returns:
            x: Output of shape (batch, seq_len, d_model).
            states: Updated recurrence states per layer.
        """
        if states is None:
            states = [None] * self.n_layers

        new_states = []

        for i, layer in enumerate(self.layers):
            x, state = layer(x, states[i])
            new_states.append(state)

        x = self.norm(x)

        return x, new_states

    def init_states(self, batch_size: int, device: torch.device) -> list:
        """Initialize recurrence states for generation.

        Args:
            batch_size: Batch size.
            device: Device to create states on.

        Returns:
            states: List of initialized states per layer.
        """
        d_rec = self.layers[0].rglru.d_recurrence
        states = [
            torch.zeros(batch_size, d_rec, device=device, dtype=torch.float32)
            for _ in range(self.n_layers)
        ]
        return states
