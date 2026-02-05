"""
Liquid-S4 - Input-Dependent State Space Models.

Liquid-S4 extends S4 by making the state transitions input-dependent, allowing
the model to dynamically adapt its recurrence based on the input signal. Key features:

1. Input-dependent dynamics: The state matrix A and input matrix B are modulated
   by the input, creating adaptive state transitions.

2. Liquid Time Constants (LTC): Inspired by liquid neural networks, the timescale
   dt becomes input-dependent, allowing the model to speed up or slow down based
   on input complexity.

3. Continuous-time formulation: Maintains the continuous-time SSM framework but
   with input-modulated parameters.

4. Enhanced expressivity: Input-dependent transitions allow the model to capture
   more complex temporal patterns than fixed-parameter SSMs.

The architecture combines the efficiency of S4's structured parameterization with
the flexibility of input-dependent neural ODEs, yielding a powerful sequence model
that can adapt its dynamics on-the-fly.

Reference: Hasani et al., "Liquid Structural State-Space Models", ICLR 2023.
    https://arxiv.org/abs/2209.12951
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


def hippo_diagonal_initializer(N: int) -> torch.Tensor:
    """Construct diagonal approximation of HiPPO matrix."""
    return -(torch.arange(N, dtype=torch.float32) + 1)


class LiquidS4Kernel(NexusModule):
    """Liquid S4 Kernel with Input-Dependent State Transitions.

    Generates SSM kernels where the state matrix A and timescale dt are
    modulated by the input signal, creating adaptive dynamics.

    Args:
        d_model: Model dimension.
        d_state: State dimension.
        dt_min: Minimum timescale.
        dt_max: Maximum timescale.
        modulation_rank: Rank of input modulation (smaller = more efficient).
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        modulation_rank: int = 16,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.modulation_rank = modulation_rank

        # Base diagonal state matrix A (fixed component)
        A_diag = hippo_diagonal_initializer(d_state)
        self.register_buffer('A_diag', A_diag)

        # Base input/output matrices
        B = torch.randn(d_model, d_state) / math.sqrt(d_state)
        self.B = nn.Parameter(B)

        C = torch.randn(d_model, d_state) / math.sqrt(d_state)
        self.C = nn.Parameter(C)

        # Base timescale (will be modulated)
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        self.D = nn.Parameter(torch.randn(d_model))

        # Input modulation networks (low-rank for efficiency)
        # Modulate A: small perturbation to diagonal
        self.A_modulator = nn.Sequential(
            nn.Linear(d_model, modulation_rank),
            nn.Tanh(),
            nn.Linear(modulation_rank, d_state)
        )

        # Modulate B: scale input transformation
        self.B_modulator = nn.Sequential(
            nn.Linear(d_model, modulation_rank),
            nn.Tanh(),
            nn.Linear(modulation_rank, d_state)
        )

        # Modulate dt: liquid time constant (key innovation)
        self.dt_modulator = nn.Sequential(
            nn.Linear(d_model, modulation_rank),
            nn.Tanh(),
            nn.Linear(modulation_rank, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def _modulate_parameters(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute input-dependent SSM parameters.

        Args:
            u: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            A_modulated: Input-dependent A matrix (batch, seq_len, d_model, d_state).
            B_modulated: Input-dependent B matrix (batch, seq_len, d_model, d_state).
            dt_modulated: Input-dependent timescale (batch, seq_len, d_model).
        """
        batch, seq_len, _ = u.shape

        # Compute modulations
        A_delta = self.A_modulator(u)  # (batch, seq_len, d_state)
        B_scale = 1.0 + 0.1 * self.B_modulator(u)  # (batch, seq_len, d_state)
        dt_gate = self.dt_modulator(u).squeeze(-1)  # (batch, seq_len)

        # Base parameters
        A_base = self.A_diag.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, d_state)
        B_base = self.B.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model, d_state)
        dt_base = torch.exp(self.log_dt).unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)

        # Apply modulations
        # A: Add small perturbation (keep diagonal dominant)
        A_modulated = A_base + 0.1 * A_delta.unsqueeze(2)  # (batch, seq_len, d_model, d_state)

        # B: Scale by modulation
        B_modulated = B_base * B_scale.unsqueeze(2)  # (batch, seq_len, d_model, d_state)

        # dt: Liquid time constant - interpolate between fast and slow
        dt_modulated = dt_base * (self.dt_min / torch.exp(self.log_dt.mean()) +
                                   dt_gate.unsqueeze(-1) * (self.dt_max / torch.exp(self.log_dt.mean())))

        return A_modulated, B_modulated, dt_modulated

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Apply Liquid S4 with input-dependent recurrence.

        Args:
            u: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            y: Output tensor of shape (batch, seq_len, d_model).
        """
        batch, seq_len, d_model = u.shape

        # Get modulated parameters
        A_mod, B_mod, dt_mod = self._modulate_parameters(u)

        # Initialize hidden state
        h = torch.zeros(batch, d_model, self.d_state, device=u.device, dtype=u.dtype)

        outputs = []
        for t in range(seq_len):
            # Current input-dependent parameters
            A_t = A_mod[:, t]  # (batch, d_model, d_state)
            B_t = B_mod[:, t]  # (batch, d_model, d_state)
            dt_t = dt_mod[:, t]  # (batch, d_model)
            u_t = u[:, t]  # (batch, d_model)

            # Discretize with input-dependent dt
            A_discrete = torch.exp(A_t * dt_t.unsqueeze(-1))  # (batch, d_model, d_state)
            B_discrete = (A_discrete - 1) / (A_t + 1e-8) * B_t  # (batch, d_model, d_state)

            # Recurrence: h_new = A * h + B * u
            h = A_discrete * h + B_discrete * u_t.unsqueeze(-1)

            # Output: y = C * h + D * u
            y_t = (self.C.unsqueeze(0) * h).sum(dim=-1) + self.D.unsqueeze(0) * u_t

            outputs.append(y_t)

        # Stack outputs
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)

        return y


class LiquidS4Layer(NexusModule):
    """Liquid S4 Layer with Pre/Post Processing.

    Args:
        d_model: Model dimension.
        d_state: State dimension.
        dropout: Dropout probability.
        activation: Activation function.
        modulation_rank: Rank for input modulation networks.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        activation: str = 'gelu',
        modulation_rank: int = 16,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model

        # Liquid S4 kernel
        self.kernel = LiquidS4Kernel(d_model, d_state, modulation_rank=modulation_rank, **kwargs)

        # Activation
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            y: Output tensor of shape (batch, seq_len, d_model).
        """
        # Apply Liquid S4 kernel
        y = self.kernel(x)

        # Activation and dropout
        y = self.activation(y)
        y = self.dropout(y)

        # Output projection
        y = self.output_linear(y)

        return y


class LiquidS4Block(NexusModule):
    """Complete Liquid S4 Block with Residual and Normalization.

    Args:
        d_model: Model dimension.
        d_state: State dimension.
        d_ff: Feedforward dimension.
        dropout: Dropout probability.
        prenorm: Whether to use pre-normalization.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        prenorm: bool = True,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.prenorm = prenorm

        self.norm1 = nn.LayerNorm(d_model)
        self.liquid_s4 = LiquidS4Layer(d_model, d_state, dropout, **kwargs)

        self.use_ffn = d_ff is not None
        if self.use_ffn:
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            y: Output tensor of shape (batch, seq_len, d_model).
        """
        # Liquid S4 block with residual
        if self.prenorm:
            x = x + self.liquid_s4(self.norm1(x))
        else:
            x = self.norm1(x + self.liquid_s4(x))

        # Optional FFN block
        if self.use_ffn:
            if self.prenorm:
                x = x + self.ffn(self.norm2(x))
            else:
                x = self.norm2(x + self.ffn(x))

        return x


class LiquidS4Model(NexusModule):
    """Complete Liquid S4 Model for Sequence Processing.

    Stacks multiple Liquid S4 blocks for deep sequence modeling with
    input-dependent dynamics.

    Args:
        d_model: Model dimension.
        n_layers: Number of Liquid S4 blocks.
        d_state: State dimension.
        d_ff: Feedforward dimension.
        dropout: Dropout probability.
    """
    def __init__(
        self,
        d_model: int,
        n_layers: int = 6,
        d_state: int = 64,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        if d_ff is None:
            d_ff = 4 * d_model

        self.layers = nn.ModuleList([
            LiquidS4Block(d_model, d_state, d_ff, dropout, **kwargs)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """Forward pass through all layers.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            return_features: Whether to return intermediate features.

        Returns:
            y: Output tensor of shape (batch, seq_len, d_model).
            features (optional): List of intermediate features if return_features=True.
        """
        features = []

        for layer in self.layers:
            x = layer(x)
            if return_features:
                features.append(x)

        x = self.norm(x)

        if return_features:
            return x, features
        return x
