"""
RecurrentGemma - Griffin-Based Open Language Model.

RecurrentGemma is Google's open-source implementation of the Griffin architecture,
combining gated linear recurrences with local attention. It provides an efficient
alternative to pure transformer models with strong performance.

Key components:
1. RGLRU (Real-Gated Linear Recurrent Unit): Diagonal gated recurrence
2. Local sliding window attention: Precise short-range modeling
3. Pre-normalization with RMSNorm: Stable training
4. GeGLU activations: Improved feedforward blocks

The model demonstrates that hybrid recurrence-attention architectures can
match transformer quality while offering better inference efficiency.

Reference: Google DeepMind, "RecurrentGemma: Moving Past Transformers for
    Efficient Open Language Models", 2024.
    https://arxiv.org/abs/2404.07839
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class RMSNorm(NexusModule):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm, commonly used in modern LLMs.

    Args:
        d_model: Model dimension.
        eps: Small constant for numerical stability.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (..., d_model).

        Returns:
            output: Normalized output of same shape.
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class RGLRU(NexusModule):
    """Real-Gated Linear Recurrent Unit.

    The core recurrent component with diagonal gating.

    Args:
        d_model: Model dimension.
        d_recurrence: Recurrence dimension (default: same as d_model).
    """
    def __init__(self, d_model: int, d_recurrence: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.d_recurrence = d_recurrence or d_model

        # Input projection
        self.x_proj = nn.Linear(d_model, self.d_recurrence, bias=False)

        # Recurrence gate
        self.a_proj = nn.Linear(d_model, self.d_recurrence, bias=True)

        # Initialize gate bias to favor remembering
        nn.init.constant_(self.a_proj.bias, 2.0)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Previous state of shape (batch, d_recurrence).

        Returns:
            output: Output of shape (batch, seq_len, d_recurrence).
            state: Updated state of shape (batch, d_recurrence).
        """
        batch_size, seq_len, _ = x.shape

        # Compute gate and input
        a = torch.sigmoid(self.a_proj(x))  # (batch, seq, d_rec)
        x_in = self.x_proj(x)  # (batch, seq, d_rec)

        # Scale input to preserve magnitude (unitary-like)
        x_in = x_in * torch.sqrt(1 - a ** 2 + 1e-8)

        # Initialize state
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


class LocalSlidingWindowAttention(NexusModule):
    """Local sliding window multi-query attention.

    Uses multi-query attention (MQA) with a sliding window for efficiency.

    Args:
        d_model: Model dimension.
        num_heads: Number of query heads.
        window_size: Size of sliding window.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        window_size: int = 2048
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size

        assert self.head_dim * num_heads == d_model

        # Multi-query: multiple query heads, single key/value head
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

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

        # Project queries (multi-head)
        q = self.q_proj(x)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2) * self.scale  # (batch, num_heads, seq_len, head_dim)

        # Project keys and values (single head, broadcast to all query heads)
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)  # Each: (batch, seq_len, head_dim)

        # Expand for multi-query
        k = k.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (batch, num_heads, seq_len, head_dim)
        v = v.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # Create sliding window mask
        attn_mask = torch.full(
            (seq_len, seq_len), float('-inf'),
            device=x.device, dtype=x.dtype
        )
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            attn_mask[i, start:i+1] = 0  # Causal + window

        # Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores + attn_mask.unsqueeze(0).unsqueeze(0)

        if mask is not None:
            attn_scores = attn_scores + mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.out_proj(output)

        return output


class GeGLU(NexusModule):
    """GELU-Gated Linear Unit.

    Used in Gemma models for improved feedforward blocks.

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
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class RecurrentGemmaBlock(NexusModule):
    """RecurrentGemma block alternating RGLRU and local attention.

    Args:
        d_model: Model dimension.
        block_type: 'recurrence' or 'attention'.
        num_heads: Number of attention heads.
        window_size: Attention window size.
        d_ff: Feedforward dimension.
    """
    def __init__(
        self,
        d_model: int,
        block_type: str = 'recurrence',
        num_heads: int = 8,
        window_size: int = 2048,
        d_ff: Optional[int] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.block_type = block_type

        if d_ff is None:
            d_ff = int(d_model * 8 / 3)  # Gemma uses 8/3 expansion

        # Pre-normalization
        self.norm1 = RMSNorm(d_model)

        # Main block
        if block_type == 'recurrence':
            self.main_block = RGLRU(d_model)
        elif block_type == 'attention':
            self.main_block = LocalSlidingWindowAttention(d_model, num_heads, window_size)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        # Feedforward with GeGLU
        self.norm2 = RMSNorm(d_model)
        self.ffn = GeGLU(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Recurrence state (if block_type='recurrence').
            mask: Attention mask (if block_type='attention').

        Returns:
            x: Output of shape (batch, seq_len, d_model).
            state: Updated state (or None for attention blocks).
        """
        # Main block
        if self.block_type == 'recurrence':
            out, new_state = self.main_block(self.norm1(x), state)
            x = x + out
        else:  # attention
            x = x + self.main_block(self.norm1(x), mask)
            new_state = None

        # Feedforward
        x = x + self.ffn(self.norm2(x))

        return x, new_state


class RecurrentGemmaModel(NexusModule):
    """Complete RecurrentGemma Model.

    Alternates between RGLRU and local attention layers.

    Args:
        d_model: Model dimension.
        n_layers: Number of layers.
        num_heads: Number of attention heads.
        window_size: Attention window size.
        d_ff: Feedforward dimension.
        recurrence_every_n: Use recurrence every N layers, else attention.
    """
    def __init__(
        self,
        d_model: int,
        n_layers: int = 26,
        num_heads: int = 8,
        window_size: int = 2048,
        d_ff: Optional[int] = None,
        recurrence_every_n: int = 3
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Create alternating layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # Every N-th layer uses attention, others use recurrence
            block_type = 'attention' if (i + 1) % recurrence_every_n == 0 else 'recurrence'

            self.layers.append(
                RecurrentGemmaBlock(
                    d_model=d_model,
                    block_type=block_type,
                    num_heads=num_heads,
                    window_size=window_size,
                    d_ff=d_ff
                )
            )

        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[list] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            states: List of recurrence states per layer.
            mask: Attention mask.

        Returns:
            x: Output of shape (batch, seq_len, d_model).
            states: Updated recurrence states.
        """
        if states is None:
            states = [None] * self.n_layers

        new_states = []

        for i, layer in enumerate(self.layers):
            x, state = layer(x, states[i], mask)
            new_states.append(state)

        x = self.norm(x)

        return x, new_states
