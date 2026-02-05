"""
Gated Delta Network (GDN) - Combining Mamba2 Gating with DeltaNet Delta Rule.

The Gated Delta Network combines two key innovations from the SSM/linear attention
literature:

1. Mamba-2 gating: The exponential gating mechanism from Mamba-2 enables rapid
   memory erasure. When the gate value is large, previous memory is exponentially
   decayed, allowing the model to quickly "forget" irrelevant information.

2. DeltaNet delta rule: The delta rule from DeltaNet enables precise, targeted
   updates to the associative memory. Instead of simply accumulating key-value
   pairs (as in linear attention), the delta rule computes an error signal and
   corrects the memory, enabling one-shot association.

The combination is synergistic: the gate handles coarse-grained memory management
(what to forget), while the delta rule handles fine-grained memory updates (what
to precisely remember). This addresses a key limitation of each method alone:
- DeltaNet without gating struggles to erase old memories quickly
- Mamba-2 without delta rule lacks precise memory write capability

Reference: Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule",
    2024. https://arxiv.org/abs/2412.06464
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class GatedDeltaNetCore(NexusModule):
    """Core Gated Delta Network computation.

    Implements the combined gating + delta rule recurrence:
        h[t] = alpha[t] * h[t-1] + beta[t] * (v[t] - h[t-1] @ k[t]) @ k[t]^T

    where:
    - alpha[t] = exp(dt[t] * A) is the exponential decay (Mamba-2 style gating)
    - beta[t] is the delta rule learning rate (data-dependent)
    - k[t], v[t] are key and value projections of the input
    - The outer product update follows the delta rule for associative memory

    Args:
        d_model: Model dimension.
        d_state: State dimension (head_dim for the KV state, default: 64).
        num_heads: Number of attention-like heads (default: 8).
        head_dim: Dimension per head. If None, computed as d_model * expand // num_heads.
        expand: Expansion factor for inner dimension (default: 2).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        expand: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_inner = d_model * expand
        self.head_dim = head_dim or self.d_inner // num_heads
        self.hidden_dim = self.num_heads * self.head_dim

        assert self.hidden_dim == self.num_heads * self.head_dim

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.hidden_dim, bias=False)

        # Gate projection (output gate, Mamba-2 style)
        self.g_proj = nn.Linear(d_model, self.hidden_dim, bias=False)

        # Beta projection (delta rule learning rate)
        self.beta_proj = nn.Linear(d_model, num_heads, bias=True)

        # Exponential decay parameter A (per head, Mamba-2 style)
        self.A_log = nn.Parameter(torch.randn(num_heads))

        # Delta (discretization step) projection
        self.dt_proj = nn.Linear(d_model, num_heads, bias=True)

        # Short convolutions for local context
        self.conv_q = nn.Conv1d(
            self.hidden_dim, self.hidden_dim,
            kernel_size=4, padding=3,
            groups=self.hidden_dim, bias=True
        )
        self.conv_k = nn.Conv1d(
            self.hidden_dim, self.hidden_dim,
            kernel_size=4, padding=3,
            groups=self.hidden_dim, bias=True
        )
        self.conv_v = nn.Conv1d(
            self.hidden_dim, self.hidden_dim,
            kernel_size=4, padding=3,
            groups=self.hidden_dim, bias=True
        )

        # QK normalization
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        # Output normalization
        self.out_norm = nn.LayerNorm(self.hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.g_proj.weight)
        # Beta initialized to encourage moderate learning rate
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.ones_(self.beta_proj.bias)
        # dt initialized for moderate decay
        dt_init = torch.exp(
            torch.rand(self.num_heads) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def _apply_conv(self, x: torch.Tensor, conv: nn.Conv1d, seq_len: int) -> torch.Tensor:
        """Apply causal convolution."""
        x = x.transpose(1, 2)
        x = conv(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        return F.silu(x)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of Gated Delta Network core.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Optional recurrent state of shape
                (batch, num_heads, head_dim, head_dim).

        Returns:
            output: Output of shape (batch, seq_len, d_model).
            state: Updated state.
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        # Apply short convolutions
        q = self._apply_conv(q, self.conv_q, seq_len)
        k = self._apply_conv(k, self.conv_k, seq_len)
        v = self._apply_conv(v, self.conv_v, seq_len)

        # Reshape to multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Normalize Q, K
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Compute beta (delta rule learning rate)
        beta = torch.sigmoid(self.beta_proj(x))  # (batch, seq, num_heads)

        # Compute exponential decay (Mamba-2 gating)
        dt = F.softplus(self.dt_proj(x))  # (batch, seq, num_heads)
        A = -torch.exp(self.A_log)  # (num_heads,)
        alpha = torch.exp(dt * A.unsqueeze(0).unsqueeze(0))  # (batch, seq, num_heads)

        # Initialize state
        if state is None:
            state = torch.zeros(
                batch_size, self.num_heads, self.head_dim, self.head_dim,
                device=x.device, dtype=x.dtype
            )

        # Run gated delta rule recurrence
        if self.training:
            output, state = self._forward_parallel(q, k, v, alpha, beta, state)
        else:
            output, state = self._forward_recurrent(q, k, v, alpha, beta, state)

        # Reshape output
        output = output.view(batch_size, seq_len, self.hidden_dim)

        # Normalize and gate
        output = self.out_norm(output)
        output = output * F.silu(g)

        # Project output
        output = self.out_proj(output)

        return output, state

    def _forward_parallel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel (chunk-wise) forward for training.

        Processes chunks sequentially but within each chunk uses the
        delta rule recurrence. The chunked approach balances parallelism
        and memory.

        Args:
            q: Query (batch, seq, heads, head_dim).
            k: Key (batch, seq, heads, head_dim).
            v: Value (batch, seq, heads, head_dim).
            alpha: Decay gates (batch, seq, heads).
            beta: Delta rule learning rates (batch, seq, heads).
            state: Initial state (batch, heads, head_dim, head_dim).

        Returns:
            output: (batch, seq, heads, head_dim).
            state: Updated state.
        """
        batch_size, seq_len, num_heads, head_dim = q.shape

        # Normalize keys for stable delta rule
        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        outputs = []
        chunk_size = min(64, seq_len)

        for i in range(0, seq_len, chunk_size):
            end = min(i + chunk_size, seq_len)

            for t in range(i, end):
                q_t = q[:, t]
                k_t = k_norm[:, t]
                v_t = v[:, t]
                alpha_t = alpha[:, t]  # (batch, heads)
                beta_t = beta[:, t]    # (batch, heads)

                # Step 1: Apply exponential decay (Mamba-2 gating)
                state = alpha_t.unsqueeze(-1).unsqueeze(-1) * state

                # Step 2: Delta rule update
                # Retrieve current memory for this key
                retrieved = torch.einsum('bhij,bhj->bhi', state, k_t)

                # Compute error
                error = v_t - retrieved

                # Update with delta rule: state += beta * error outer k
                delta = torch.einsum('bhi,bhj->bhij', error, k_t)
                state = state + beta_t.unsqueeze(-1).unsqueeze(-1) * delta

                # Step 3: Query the state
                output_t = torch.einsum('bhij,bhj->bhi', state, q_t)
                outputs.append(output_t)

        output = torch.stack(outputs, dim=1)
        return output, state

    def _forward_recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recurrent forward for inference.

        Args:
            q: Query (batch, seq, heads, head_dim).
            k: Key (batch, seq, heads, head_dim).
            v: Value (batch, seq, heads, head_dim).
            alpha: Decay gates (batch, seq, heads).
            beta: Delta rule learning rates (batch, seq, heads).
            state: State (batch, heads, head_dim, head_dim).

        Returns:
            output: (batch, seq, heads, head_dim).
            state: Updated state.
        """
        batch_size, seq_len, num_heads, head_dim = q.shape

        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        outputs = []
        for t in range(seq_len):
            q_t = q[:, t]
            k_t = k_norm[:, t]
            v_t = v[:, t]
            alpha_t = alpha[:, t]
            beta_t = beta[:, t]

            # Exponential decay
            state = alpha_t.unsqueeze(-1).unsqueeze(-1) * state

            # Delta rule update
            retrieved = torch.einsum('bhij,bhj->bhi', state, k_t)
            error = v_t - retrieved
            delta = torch.einsum('bhi,bhj->bhij', error, k_t)
            state = state + beta_t.unsqueeze(-1).unsqueeze(-1) * delta

            # Query
            output_t = torch.einsum('bhij,bhj->bhi', state, q_t)
            outputs.append(output_t)

        output = torch.stack(outputs, dim=1)
        return output, state

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Initialize recurrent state.

        Args:
            batch_size: Batch size.
            device: Device.
            dtype: Data type.

        Returns:
            Initial state of shape (batch, num_heads, head_dim, head_dim).
        """
        return torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.head_dim,
            device=device, dtype=dtype
        )


class GatedDeltaNetBlock(NexusModule):
    """Full Gated Delta Network block with pre-norm and residual.

    Combines the GatedDeltaNetCore with pre-normalization, residual connection,
    and a feed-forward network following the standard transformer block pattern.

    Reference: Yang et al., "Gated Delta Networks: Improving Mamba2 with
        Delta Rule", 2024. https://arxiv.org/abs/2412.06464

    Args:
        d_model: Model dimension.
        d_state: State dimension (default: 64).
        expand: Expansion factor (default: 2).
        num_heads: Number of heads (default: 8).
        ffn_expand: FFN expansion factor (default: 4).
        dropout: Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        num_heads: int = 8,
        ffn_expand: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model

        # GDN block with pre-norm
        self.norm1 = nn.LayerNorm(d_model)
        self.gdn = GatedDeltaNetCore(
            d_model=d_model,
            d_state=d_state,
            num_heads=num_heads,
            expand=expand
        )
        self.dropout1 = nn.Dropout(dropout)

        # FFN block with pre-norm
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with residual connections.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Optional recurrent state.

        Returns:
            output: Shape (batch, seq_len, d_model).
            state: Updated state.
        """
        # GDN with residual
        residual = x
        x = self.norm1(x)
        x, state = self.gdn(x, state)
        x = self.dropout1(x)
        x = x + residual

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x, state
