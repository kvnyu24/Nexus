"""
Mamba-2: State Space Duality (SSD) Framework.

Mamba-2 introduces the State Space Duality framework, showing that structured
state space models (SSMs) are equivalent to a form of structured masked attention.
This duality enables more efficient computation via matrix multiplications and
tensor cores, while the multi-head SSM structure provides improved expressivity.

Key innovations over Mamba-1:
- SSD algorithm: frames SSM computation as structured masked attention
- Multi-head SSM: each head has its own A, B, C parameters (like multi-head attention)
- Head dimension: controls the granularity of state transitions per head
- 2-8x faster than Mamba-1 on modern hardware due to better use of matrix multiplications

Reference: Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient
    Algorithms Through Structured State Space Duality", 2024.
    https://arxiv.org/abs/2405.21060
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class SSDLayer(NexusModule):
    """State Space Duality (SSD) Layer.

    Core component of Mamba-2 that implements the SSD algorithm. The key insight
    is that the selective SSM can be written as:
        Y = (L * QK^T) V
    where L is a lower-triangular mask encoding cumulative products of the scalar
    decay factors, Q = C, K = B, and V = X (the SSM input). This is equivalent
    to a structured masked attention with a specific causal mask.

    In multi-head mode, each head has its own scalar decay parameter A_h, shared
    across the head dimension. B and C are per-head projections of the input.

    Args:
        d_model: Model dimension (input/output).
        d_state: SSM state dimension N (default: 128). Larger than Mamba-1 due
            to improved efficiency from the SSD algorithm.
        num_heads: Number of SSM heads (default: 8). Each head operates on
            a d_inner // num_heads dimensional subspace.
        head_dim: Dimension per head. If None, computed as d_inner // num_heads.
        dt_min: Minimum discretization step (default: 0.001).
        dt_max: Maximum discretization step (default: 0.1).
        bias: Whether to use bias in projections (default: False).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bias: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_heads = num_heads
        self.head_dim = head_dim or d_model // num_heads
        self.d_inner = self.num_heads * self.head_dim

        # Per-head scalar decay parameter A (log-space for positivity)
        # In Mamba-2, A is a scalar per head (not per dimension)
        self.A_log = nn.Parameter(torch.randn(num_heads))

        # Skip connection parameter D (per head)
        self.D = nn.Parameter(torch.ones(num_heads))

        # Projections for B, C (per head, mapping to d_state)
        # B: input -> state; C: state -> output
        self.B_proj = nn.Linear(d_model, num_heads * d_state, bias=bias)
        self.C_proj = nn.Linear(d_model, num_heads * d_state, bias=bias)

        # Delta (discretization step) projection
        self.dt_proj = nn.Linear(d_model, num_heads, bias=True)

        # Initialize dt bias for proper range
        dt = torch.exp(
            torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        self._init_weights()

    def _init_weights(self):
        """Initialize projections with proper scaling."""
        nn.init.xavier_uniform_(self.B_proj.weight)
        nn.init.xavier_uniform_(self.C_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of SSD layer.

        Automatically selects parallel (training) or recurrent (inference) mode.

        Args:
            x: Input tensor of shape (batch, seq_len, d_inner).
                This is the expanded hidden representation, not raw input.
            state: Optional recurrent state of shape
                (batch, num_heads, head_dim, d_state) for recurrent mode.

        Returns:
            y: Output tensor of shape (batch, seq_len, d_inner).
            state: Updated state (returned in recurrent mode, None in parallel).
        """
        if self.training:
            return self.forward_ssd(x)
        else:
            return self.forward_recurrent(x, state)

    def forward_ssd(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """Parallel SSD computation for training.

        Implements the dual form: Y = (L * QK^T) V where
        L[i,j] = prod_{k=j+1}^{i} a_k for i >= j (0 otherwise),
        Q = C @ x, K = B @ x, V = x reshaped into heads.

        Args:
            x: Input of shape (batch, seq_len, d_inner).

        Returns:
            y: Output of shape (batch, seq_len, d_inner).
            state: None (no persistent state in parallel mode).
        """
        batch_size, seq_len, _ = x.shape

        # Project to get B, C, dt from the original input dimension
        # For SSD, we use x as the "value" and project to get B (key) and C (query)
        B = self.B_proj(x).view(batch_size, seq_len, self.num_heads, self.d_state)
        C = self.C_proj(x).view(batch_size, seq_len, self.num_heads, self.d_state)
        dt = F.softplus(self.dt_proj(x))  # (batch, seq, num_heads)

        # Get scalar decay per head
        A = -torch.exp(self.A_log)  # (num_heads,)

        # Reshape x into multi-head format: (batch, seq, heads, head_dim)
        x_heads = x.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Build the structured mask L:
        # L[i,j] = prod_{k=j+1}^{i} exp(dt_k * A_h) for head h
        # = exp(sum_{k=j+1}^{i} dt_k * A_h)
        # = exp(A_h * (cumsum_dt[i] - cumsum_dt[j]))

        # Cumulative dt * A: (batch, seq, heads)
        dt_A = dt * A.unsqueeze(0).unsqueeze(0)  # (batch, seq, heads)
        cumsum_dtA = torch.cumsum(dt_A, dim=1)  # (batch, seq, heads)

        # L[i,j] = exp(cumsum_dtA[i] - cumsum_dtA[j]) for i >= j
        # Compute pairwise differences: (batch, heads, seq_i, seq_j)
        cumsum_dtA_t = cumsum_dtA.transpose(1, 2)  # (batch, heads, seq)
        L = cumsum_dtA_t.unsqueeze(-1) - cumsum_dtA_t.unsqueeze(-2)
        # L shape: (batch, heads, seq, seq), L[b,h,i,j] = cumsum[i] - cumsum[j]
        L = torch.exp(L)

        # Apply causal mask (L[i,j] = 0 for j > i)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype)
        )
        L = L * causal_mask.unsqueeze(0).unsqueeze(0)

        # SSD attention: Y = (L * (C @ B^T)) @ X
        # Q = C: (batch, seq, heads, d_state) -> (batch, heads, seq, d_state)
        # K = B: same shape
        Q = C.transpose(1, 2)  # (batch, heads, seq, d_state)
        K = B.transpose(1, 2)  # (batch, heads, seq, d_state)
        V = x_heads.transpose(1, 2)  # (batch, heads, seq, head_dim)

        # Attention scores: (batch, heads, seq, seq)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_state)

        # Apply structured mask
        attn = L * attn

        # Apply attention to values
        y = torch.matmul(attn, V)  # (batch, heads, seq, head_dim)

        # Add skip connection: D * x
        y = y + self.D.view(1, self.num_heads, 1, 1) * V

        # Reshape back: (batch, seq, d_inner)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_inner)

        return y, None

    def forward_recurrent(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recurrent computation for inference.

        Standard SSM recurrence:
            h[t] = exp(dt * A) * h[t-1] + dt * B[t] * x[t]
            y[t] = C[t] @ h[t] + D * x[t]

        Args:
            x: Input of shape (batch, seq_len, d_inner).
            state: Recurrent state of shape (batch, num_heads, head_dim, d_state).

        Returns:
            y: Output of shape (batch, seq_len, d_inner).
            state: Updated state of shape (batch, num_heads, head_dim, d_state).
        """
        batch_size, seq_len, _ = x.shape

        if state is None:
            state = torch.zeros(
                batch_size, self.num_heads, self.head_dim, self.d_state,
                device=x.device, dtype=x.dtype
            )

        B = self.B_proj(x).view(batch_size, seq_len, self.num_heads, self.d_state)
        C = self.C_proj(x).view(batch_size, seq_len, self.num_heads, self.d_state)
        dt = F.softplus(self.dt_proj(x))  # (batch, seq, num_heads)

        A = -torch.exp(self.A_log)  # (num_heads,)

        x_heads = x.view(batch_size, seq_len, self.num_heads, self.head_dim)

        outputs = []
        for t in range(seq_len):
            x_t = x_heads[:, t]  # (batch, heads, head_dim)
            B_t = B[:, t]  # (batch, heads, d_state)
            C_t = C[:, t]  # (batch, heads, d_state)
            dt_t = dt[:, t]  # (batch, heads)

            # Discretize: dA = exp(dt * A), per head scalar
            dA = torch.exp(
                dt_t.unsqueeze(-1).unsqueeze(-1) * A.view(1, -1, 1, 1)
            )  # (batch, heads, 1, 1)

            # State update: h = dA * h + dt * outer(x_t, B_t)
            dB_x = dt_t.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
                'bhd,bhn->bhdn', x_t, B_t
            )  # (batch, heads, head_dim, d_state)
            state = dA * state + dB_x

            # Output: y = C @ h + D * x
            y_t = torch.einsum(
                'bhn,bhdn->bhd', C_t, state
            ) + self.D.view(1, -1, 1) * x_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq, heads, head_dim)
        y = y.contiguous().view(batch_size, seq_len, self.d_inner)

        return y, state


class Mamba2Block(NexusModule):
    """Mamba-2 Block with State Space Duality.

    Full Mamba-2 block combining:
    1. Input projection with expansion and gating split
    2. Depthwise convolution for local context
    3. SSD layer for sequence modeling (the core innovation)
    4. Gated output with normalization

    The block architecture is:
        x -> in_proj -> [x_branch, z]
        x_branch -> conv1d -> SiLU -> SSD -> norm -> * SiLU(z) -> out_proj -> y

    Compared to Mamba-1:
    - Uses SSD instead of selective scan for 2-8x speedup
    - Multi-head structure with per-head scalar decay
    - Larger default state dimension (128 vs 16)

    Reference: Dao & Gu, "Transformers are SSMs", 2024.
        https://arxiv.org/abs/2405.21060

    Args:
        d_model: Model dimension (input/output dimension).
        d_state: SSM state dimension N (default: 128).
        d_conv: Depthwise convolution kernel size (default: 4).
        expand: Expansion factor for inner dimension (default: 2).
            Inner dimension = d_model * expand.
        num_heads: Number of SSM heads (default: 8).
        bias: Whether to use bias in linear projections (default: False).
        conv_bias: Whether to use bias in convolution (default: True).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        num_heads: int = 8,
        bias: bool = False,
        conv_bias: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.num_heads = num_heads
        self.d_inner = d_model * expand
        self.head_dim = self.d_inner // num_heads

        assert self.d_inner % num_heads == 0, (
            f"d_inner ({self.d_inner}) must be divisible by num_heads ({num_heads})"
        )

        # Input projection: splits into x_branch and gate z
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Depthwise causal convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias
        )

        # SSD core
        self.ssd = SSDLayer(
            d_model=self.d_inner,
            d_state=d_state,
            num_heads=num_heads,
            head_dim=self.head_dim
        )

        # Output normalization and projection
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of Mamba-2 block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            state: Optional recurrent state for inference.

        Returns:
            output: Output tensor of shape (batch, seq_len, d_model).
            state: Updated recurrent state (None during training).
        """
        batch_size, seq_len, _ = x.shape

        # Project and split into two branches
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Causal convolution for local context
        x_branch = x_branch.transpose(1, 2)  # (batch, d_inner, seq)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]
        x_branch = x_branch.transpose(1, 2)  # (batch, seq, d_inner)
        x_branch = F.silu(x_branch)

        # Core SSD computation
        y, state = self.ssd(x_branch, state)

        # Normalize, gate, and project
        y = self.norm(y)
        y = y * F.silu(z)
        output = self.out_proj(y)

        return output, state

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Initialize recurrent state for inference.

        Args:
            batch_size: Batch size.
            device: Device for state tensor.
            dtype: Data type for state tensor.

        Returns:
            Initial state of shape (batch, num_heads, head_dim, d_state).
        """
        return torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.d_state,
            device=device, dtype=dtype
        )


class Mamba2Layer(NexusModule):
    """Full Mamba-2 layer with pre-norm, residual connection, and FFN.

    Stacks a Mamba2Block with a feed-forward network, using pre-normalization
    and residual connections following the standard transformer block pattern.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension (default: 128).
        d_conv: Convolution kernel size (default: 4).
        expand: Expansion factor (default: 2).
        num_heads: Number of SSM heads (default: 8).
        ffn_expand: FFN expansion factor (default: 4).
        dropout: Dropout probability (default: 0.0).
        bias: Whether to use bias (default: False).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        num_heads: int = 8,
        ffn_expand: int = 4,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        self.d_model = d_model

        # Mamba-2 block with pre-norm
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba2 = Mamba2Block(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_heads=num_heads,
            bias=bias
        )
        self.dropout1 = nn.Dropout(dropout)

        # FFN block with pre-norm
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expand, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expand, d_model, bias=bias),
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
            output: Output of shape (batch, seq_len, d_model).
            state: Updated state.
        """
        # Mamba-2 with residual
        residual = x
        x = self.norm1(x)
        x, state = self.mamba2(x, state)
        x = self.dropout1(x)
        x = x + residual

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x, state
