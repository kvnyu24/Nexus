"""
Mamba: Linear-Time Sequence Modeling with Selective State Spaces.

Mamba uses selective state space models (SSMs) to achieve linear complexity
while maintaining strong performance. Key innovation is input-dependent
state transitions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class SelectiveSSM(NexusModule):
    """Selective State Space Model (S6).

    Core component of Mamba that implements input-dependent state transitions.
    Unlike traditional SSMs with fixed A, B, C matrices, selective SSM makes
    these matrices depend on the input.

    Args:
        d_model: Model dimension
        d_state: SSM state dimension (N in paper, typically 16)
        dt_rank: Rank for Δ projection ('auto' uses d_model // 16)
        dt_min: Minimum value for Δ
        dt_max: Maximum value for Δ
        dt_init: Initialization method for Δ ('random' or 'constant')
        dt_scale: Scale for Δ initialization
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: str = 'auto',
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = 'random',
        dt_scale: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = d_model // 16 if dt_rank == 'auto' else dt_rank

        # A parameter (diagonal, initialized to HiPPO)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))  # Store log for numerical stability

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(d_model))

        # Projections for input-dependent B, C, Δ
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)

        # Δ projection
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # Initialize Δ projection bias
        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of selective SSM.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            state: Optional initial state of shape (batch, d_model, d_state)

        Returns:
            y: Output tensor of shape (batch, seq_len, d_model)
            state: Final state of shape (batch, d_model, d_state)
        """
        batch_size, seq_len, d_model = x.shape

        # Initialize state if not provided
        if state is None:
            state = torch.zeros(
                batch_size, d_model, self.d_state,
                device=x.device, dtype=x.dtype
            )

        # Project x to get Δ, B, C
        x_dbl = self.x_proj(x)  # (batch, seq, dt_rank + 2*d_state)

        # Split into Δ, B, C components
        delta, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Project and softplus Δ
        delta = self.dt_proj(delta)  # (batch, seq, d_model)
        delta = F.softplus(delta)

        # Get A from log
        A = -torch.exp(self.A_log)  # (d_model, d_state)

        # Run selective scan
        y, state = self.selective_scan(x, delta, A, B, C, self.D, state)

        return y, state

    def selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selective scan algorithm (sequential for reference, can be parallelized).

        Computes:
            h[t] = Ā h[t-1] + B̄ u[t]
            y[t] = C h[t] + D u[t]

        where Ā = exp(Δ A), B̄ = Δ B
        """
        batch_size, seq_len, d_model = u.shape

        outputs = []

        for t in range(seq_len):
            u_t = u[:, t, :]  # (batch, d_model)
            delta_t = delta[:, t, :]  # (batch, d_model)
            B_t = B[:, t, :]  # (batch, d_state)
            C_t = C[:, t, :]  # (batch, d_state)

            # Discretize A and B
            # Ā = exp(Δ * A)
            deltaA = torch.exp(delta_t.unsqueeze(-1) * A)  # (batch, d_model, d_state)
            # B̄ = Δ * B
            deltaB = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (batch, d_model, d_state)

            # State update: h = Ā * h + B̄ * u
            state = deltaA * state + deltaB * u_t.unsqueeze(-1)

            # Output: y = C * h + D * u
            y_t = torch.einsum('bdn,bn->bd', state, C_t) + D * u_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y, state


class MambaBlock(NexusModule):
    """Mamba Block combining SSM with gated MLP.

    The Mamba block consists of:
    1. Linear projection to expand dimension
    2. 1D convolution for local context
    3. Selective SSM for sequence modeling
    4. Gated output with SiLU activation

    Reference: https://arxiv.org/abs/2312.00752

    Args:
        d_model: Model dimension
        d_state: SSM state dimension (typically 16)
        d_conv: Convolution kernel size (typically 4)
        expand: Expansion factor for inner dimension (typically 2)
        dt_rank: Rank for Δ projection
        bias: Whether to use bias in linear layers
        conv_bias: Whether to use bias in convolution
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = 'auto',
        bias: bool = False,
        conv_bias: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        # Input projection (expands to 2x for gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias
        )

        # SSM
        self.ssm = SelectiveSSM(
            d_model=self.d_inner,
            d_state=d_state,
            dt_rank=dt_rank
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Mamba block.

        Args:
            x: Input of shape (batch, seq_len, d_model)
            state: Optional SSM state

        Returns:
            output: Shape (batch, seq_len, d_model)
            state: Updated SSM state
        """
        batch_size, seq_len, _ = x.shape

        # Project and split into two branches
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Convolution branch
        x_branch = x_branch.transpose(1, 2)  # (batch, d_inner, seq)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]  # Causal conv
        x_branch = x_branch.transpose(1, 2)  # (batch, seq, d_inner)

        # Activation
        x_branch = F.silu(x_branch)

        # SSM
        y, state = self.ssm(x_branch, state)

        # Gated output
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        return output, state


class Mamba2Block(NexusModule):
    """Mamba-2 Block with State Space Duality (SSD).

    Mamba-2 improves on Mamba with:
    1. Better hardware efficiency via SSD algorithm
    2. Multi-head structure similar to attention
    3. Improved expressivity

    Reference: https://arxiv.org/abs/2405.21060

    Args:
        d_model: Model dimension
        d_state: SSM state dimension (typically 64-128 for Mamba-2)
        d_conv: Convolution kernel size
        expand: Expansion factor
        num_heads: Number of heads (Mamba-2 uses multi-head SSM)
        bias: Whether to use bias
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        num_heads: int = 8,
        bias: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_heads = num_heads
        self.d_inner = d_model * expand
        self.head_dim = self.d_inner // num_heads

        assert self.d_inner % num_heads == 0

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        # SSM parameters (per head)
        self.A_log = nn.Parameter(torch.randn(num_heads))
        self.D = nn.Parameter(torch.ones(num_heads))

        # Projections for B, C, dt
        self.x_proj = nn.Linear(self.d_inner, num_heads * (d_state * 2 + 1), bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # Normalization
        self.norm = nn.LayerNorm(self.d_inner)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of Mamba-2 block."""
        batch_size, seq_len, _ = x.shape

        # Project and split
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Conv
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        # Get SSM parameters
        x_ssm = self.x_proj(x_branch)

        # Split into B, C, dt for each head
        x_ssm = x_ssm.view(batch_size, seq_len, self.num_heads, -1)
        B, C, dt = x_ssm.split([self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(dt.squeeze(-1))  # (batch, seq, heads)

        # Get A
        A = -torch.exp(self.A_log)  # (heads,)

        # Run multi-head SSM
        x_branch = x_branch.view(batch_size, seq_len, self.num_heads, self.head_dim)
        y = self._multi_head_ssm(x_branch, dt, A, B, C, self.D)
        y = y.view(batch_size, seq_len, self.d_inner)

        # Normalize and gate
        y = self.norm(y)
        y = y * F.silu(z)

        return self.out_proj(y), None

    def _multi_head_ssm(self, x, dt, A, B, C, D):
        """Multi-head selective scan (simplified sequential version)."""
        batch_size, seq_len, num_heads, head_dim = x.shape
        d_state = B.shape[-1]

        state = torch.zeros(
            batch_size, num_heads, head_dim, d_state,
            device=x.device, dtype=x.dtype
        )

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]  # (batch, heads, head_dim)
            dt_t = dt[:, t]  # (batch, heads)
            B_t = B[:, t]  # (batch, heads, d_state)
            C_t = C[:, t]  # (batch, heads, d_state)

            # Discretize
            dA = torch.exp(dt_t.unsqueeze(-1).unsqueeze(-1) * A.view(1, -1, 1, 1))
            dB = dt_t.unsqueeze(-1).unsqueeze(-1) * B_t.unsqueeze(2)

            # Update state
            state = dA * state + dB * x_t.unsqueeze(-1)

            # Output
            y_t = torch.einsum('bhdn,bhn->bhd', state, C_t) + D.view(1, -1, 1) * x_t
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)
