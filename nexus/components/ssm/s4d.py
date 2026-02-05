"""
S4D - Diagonal State Spaces for Sequence Modeling.

S4D simplifies the S4 architecture by restricting the state matrix A to be purely
diagonal, removing the need for complex low-rank decompositions. This yields:

1. Simplified parameterization: A is diagonal, so computation is significantly
   simplified while retaining most of the performance of full S4.

2. Efficient computation: Both recurrence and convolution modes benefit from the
   diagonal structure, with memory and compute improvements.

3. Practical gains: Achieves similar quality to S4 on many tasks while being
   easier to implement and faster to train.

The diagonal parameterization follows the HiPPO initialization principle but
restricts to diagonal matrices, which still capture long-range dependencies
effectively for many practical applications.

Reference: Gu et al., "On the Parameterization and Initialization of Diagonal
    State Space Models", NeurIPS 2022. https://arxiv.org/abs/2206.11893
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


def hippo_diagonal_initializer(N: int) -> torch.Tensor:
    """Construct diagonal approximation of HiPPO-LegS matrix.

    For the diagonal S4D variant, we extract only the diagonal elements
    of the HiPPO matrix, which are:
        A[n,n] = -(n+1)

    This simplification maintains the essential spectral properties for
    many sequence modeling tasks while dramatically simplifying computation.

    Args:
        N: State dimension.

    Returns:
        A_diag: Diagonal entries of shape (N,).
    """
    return -(torch.arange(N, dtype=torch.float32) + 1)


class S4DKernel(NexusModule):
    """S4D Convolution Kernel Generator with Diagonal State Matrix.

    Generates the SSM convolution kernel using a diagonal state matrix A.
    The transfer function simplifies to:
        K_hat(omega) = C * (1 / (i*omega - A)) * B
    where A is diagonal, allowing element-wise operations.

    Args:
        d_model: Model dimension (input/output).
        d_state: State dimension (N in SSM literature).
        dt_min: Minimum value for learnable timescale dt.
        dt_max: Maximum value for learnable timescale dt.
        lr_scale: Learning rate scale for SSM parameters (often set < 1).
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr_scale: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_min = dt_min
        self.dt_max = dt_max

        # Diagonal state matrix A (N,)
        A_diag = hippo_diagonal_initializer(d_state)
        self.register_buffer('A_diag', A_diag)

        # Input matrix B (d_model, N)
        B = torch.randn(d_model, d_state) / math.sqrt(d_state)
        self.B = nn.Parameter(B)

        # Output matrix C (d_model, N)
        C = torch.randn(d_model, d_state) / math.sqrt(d_state)
        self.C = nn.Parameter(C)

        # Learnable timescale per feature (d_model,)
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # Feedthrough matrix D (d_model,)
        self.D = nn.Parameter(torch.randn(d_model))

        # Apply learning rate scaling
        if lr_scale != 1.0:
            self.B._lr_scale = lr_scale
            self.C._lr_scale = lr_scale
            self.log_dt._lr_scale = lr_scale

    def _compute_kernel_fft(self, L: int) -> torch.Tensor:
        """Compute SSM convolution kernel via FFT (training mode).

        Uses the frequency-domain formula:
            K(z) = C * (zI - A_discrete)^{-1} * B_discrete + D

        Args:
            L: Sequence length.

        Returns:
            kernel: Convolution kernel of shape (d_model, L).
        """
        dt = torch.exp(self.log_dt)  # (d_model,)

        # Discretize using zero-order hold (Euler method)
        # A_discrete = exp(A * dt), B_discrete = (exp(A * dt) - 1) / A * B
        # For diagonal A, this simplifies:
        A_discrete = torch.exp(self.A_diag.unsqueeze(0) * dt.unsqueeze(1))  # (d_model, d_state)
        B_discrete = (A_discrete - 1) / self.A_diag.unsqueeze(0) * self.B  # (d_model, d_state)

        # Compute via FFT
        # Create frequency grid
        freqs = torch.fft.rfftfreq(L, device=self.A_diag.device) * 2 * math.pi  # (L//2 + 1,)

        # Transfer function: H(omega) = C * (exp(i*omega) - A_discrete)^{-1} * B_discrete
        # For diagonal A_discrete, inversion is element-wise
        z = torch.exp(1j * freqs)  # (L//2 + 1,)

        # Broadcast: z is (L//2+1,), A_discrete is (d_model, d_state)
        # Result: (d_model, d_state, L//2+1)
        H = self.C.unsqueeze(-1) * B_discrete.unsqueeze(-1) / (
            z.unsqueeze(0).unsqueeze(0) - A_discrete.unsqueeze(-1)
        )

        # Sum over state dimension
        H = H.sum(dim=1)  # (d_model, L//2+1)

        # Add feedthrough (D is applied in time domain, acts as DC offset in frequency)
        H = H + self.D.unsqueeze(-1)

        # IFFT to time domain
        kernel = torch.fft.irfft(H, n=L)  # (d_model, L)

        return kernel

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Apply SSM convolution kernel to input.

        Args:
            u: Input tensor of shape (batch, d_model, seq_len).

        Returns:
            y: Output tensor of shape (batch, d_model, seq_len).
        """
        L = u.shape[-1]
        kernel = self._compute_kernel_fft(L)  # (d_model, L)

        # Apply convolution via FFT (circular convolution)
        u_fft = torch.fft.rfft(u, n=2*L)  # Zero-pad to avoid circular artifacts
        k_fft = torch.fft.rfft(kernel, n=2*L)

        y_fft = u_fft * k_fft.unsqueeze(0)
        y = torch.fft.irfft(y_fft, n=2*L)[..., :L]

        return y


class S4DLayer(NexusModule):
    """S4D Layer with Pre/Post Projections and Nonlinearity.

    A complete S4D layer with:
    - Input projection
    - S4D kernel application
    - Activation function
    - Output projection

    Args:
        d_model: Model dimension.
        d_state: State dimension for S4D kernel.
        dropout: Dropout probability.
        activation: Activation function name.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        activation: str = 'gelu',
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model

        # S4D kernel
        self.kernel = S4DKernel(d_model, d_state, **kwargs)

        # Activation
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Output projection
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            y: Output tensor of shape (batch, seq_len, d_model).
        """
        # Transpose for convolution: (batch, d_model, seq_len)
        u = x.transpose(1, 2)

        # Apply S4D kernel
        y = self.kernel(u)

        # Transpose back: (batch, seq_len, d_model)
        y = y.transpose(1, 2)

        # Activation and dropout
        y = self.activation(y)
        y = self.dropout(y)

        # Output projection
        y = self.output_linear(y)

        return y


class S4DBlock(NexusModule):
    """Complete S4D Block with Residual Connection and Normalization.

    A Transformer-style block with:
    - Layer normalization
    - S4D layer
    - Residual connection
    - Optional feedforward network

    Args:
        d_model: Model dimension.
        d_state: State dimension.
        d_ff: Feedforward dimension (if None, no FFN).
        dropout: Dropout probability.
        prenorm: Whether to use pre-normalization (default: True).
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

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)

        # S4D layer
        self.s4d = S4DLayer(d_model, d_state, dropout, **kwargs)

        # Optional feedforward network
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
        # S4D block with residual
        if self.prenorm:
            x = x + self.s4d(self.norm1(x))
        else:
            x = self.norm1(x + self.s4d(x))

        # Optional FFN block with residual
        if self.use_ffn:
            if self.prenorm:
                x = x + self.ffn(self.norm2(x))
            else:
                x = self.norm2(x + self.ffn(x))

        return x


class S4DRecurrentCell(NexusModule):
    """S4D Recurrent Cell for Sequential Inference.

    Maintains hidden state and performs one-step recurrence.
    Useful for autoregressive generation.

    Args:
        d_model: Model dimension.
        d_state: State dimension.
    """
    def __init__(self, d_model: int, d_state: int = 64, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Initialize same as kernel
        A_diag = hippo_diagonal_initializer(d_state)
        self.register_buffer('A_diag', A_diag)

        B = torch.randn(d_model, d_state) / math.sqrt(d_state)
        self.B = nn.Parameter(B)

        C = torch.randn(d_model, d_state) / math.sqrt(d_state)
        self.C = nn.Parameter(C)

        log_dt = torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        self.log_dt = nn.Parameter(log_dt)

        self.D = nn.Parameter(torch.randn(d_model))

        # Discretized parameters (computed once)
        self.register_buffer('A_discrete', None)
        self.register_buffer('B_discrete', None)
        self._discretize()

    def _discretize(self):
        """Pre-compute discretized SSM parameters."""
        dt = torch.exp(self.log_dt)
        self.A_discrete = torch.exp(self.A_diag.unsqueeze(0) * dt.unsqueeze(1))
        self.B_discrete = (self.A_discrete - 1) / self.A_diag.unsqueeze(0) * self.B

    def forward(self, u: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-step recurrence.

        Args:
            u: Input of shape (batch, d_model).
            h: Hidden state of shape (batch, d_model, d_state), or None to initialize.

        Returns:
            y: Output of shape (batch, d_model).
            h_new: Updated hidden state of shape (batch, d_model, d_state).
        """
        batch = u.shape[0]

        if h is None:
            h = torch.zeros(batch, self.d_model, self.d_state, device=u.device, dtype=u.dtype)

        # Recurrence: h_new = A * h + B * u
        h_new = self.A_discrete.unsqueeze(0) * h + self.B_discrete.unsqueeze(0) * u.unsqueeze(-1)

        # Output: y = C * h + D * u
        y = (self.C.unsqueeze(0) * h_new).sum(dim=-1) + self.D.unsqueeze(0) * u

        return y, h_new
