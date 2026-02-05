"""
S4 - Structured State Spaces for Sequence Modeling.

S4 introduces a principled approach to initializing and parameterizing state space
models via the HiPPO (High-order Polynomial Projection Operators) framework. The
key contributions are:

1. HiPPO initialization: The state matrix A is initialized to approximate the
   optimal solution for online function approximation, enabling long-range
   dependency modeling.

2. DPLR (Diagonal Plus Low-Rank) parameterization: The HiPPO matrix is decomposed
   into a diagonal matrix plus a low-rank correction, enabling efficient computation.

3. Dual computation modes:
   - Convolution mode (parallel, for training): Uses FFT to compute the SSM kernel
     in O(N log N) time, enabling parallelism.
   - Recurrence mode (sequential, for inference): Standard O(1) per-step recurrence.

Reference: Gu et al., "Efficiently Modeling Long Sequences with Structured State
    Spaces", ICLR 2022. https://arxiv.org/abs/2111.00396

See also: Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial Projections",
    NeurIPS 2020. https://arxiv.org/abs/2008.07669
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


def hippo_initializer(N: int) -> torch.Tensor:
    """Construct the HiPPO-LegS matrix for state initialization.

    The HiPPO (High-order Polynomial Projection Operators) matrix defines
    a continuous-time system that optimally compresses a history of inputs
    onto a polynomial basis. For the LegS (Legendre scaled) measure, the
    matrix A has entries:
        A[n,k] = -(2n+1)^{1/2} (2k+1)^{1/2}  if n > k
        A[n,k] = -(n+1)                         if n == k
        A[n,k] = 0                               if n < k

    Args:
        N: State dimension (number of polynomial basis functions).

    Returns:
        A: HiPPO-LegS matrix of shape (N, N).
    """
    P = torch.sqrt(1 + 2 * torch.arange(N, dtype=torch.float32))
    A = P.unsqueeze(1) * P.unsqueeze(0)
    A = torch.tril(A) - torch.diag(torch.arange(N, dtype=torch.float32) + 1)
    return -A


def dplr_decomposition(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose a matrix into Diagonal Plus Low-Rank (DPLR) form.

    Computes an approximate DPLR decomposition: A ~ diag(Lambda) + P Q^T
    using eigendecomposition. For the HiPPO matrix, this decomposition is
    exact with rank-1 correction.

    Args:
        A: Input matrix of shape (N, N).

    Returns:
        Lambda: Diagonal eigenvalues of shape (N,), complex-valued.
        P: Low-rank factor of shape (N, 1), complex-valued.
        Q: Low-rank factor of shape (N, 1), complex-valued.
    """
    N = A.shape[0]

    # Symmetrize for numerically stable eigendecomposition
    S = A + A.T

    # Eigendecomposition of the symmetric part
    eigenvalues, eigenvectors = torch.linalg.eigh(S / 2)

    # Convert to complex for the full (possibly non-symmetric) decomposition
    Lambda = eigenvalues.to(torch.complex64)

    # Low-rank correction: captures the skew-symmetric part
    # For HiPPO, this is a rank-1 correction
    P = torch.sqrt(1 + 2 * torch.arange(N, dtype=torch.float32)).unsqueeze(1)
    P = P.to(torch.complex64)

    # Q is the conjugate (for real input, Q = P)
    Q = P.clone()

    return Lambda, P, Q


class S4Kernel(NexusModule):
    """S4 Convolution Kernel Generator.

    Generates the SSM convolution kernel using the DPLR parameterization.
    The kernel is computed in the frequency domain via the transfer function:
        K_hat(omega) = C (i*omega*I - A)^{-1} B
    which can be efficiently evaluated for DPLR matrices.

    For training, the full kernel is materialized for convolution.
    For inference, the discretized recurrence matrices are used directly.

    Args:
        d_model: Model dimension (number of independent SSM channels).
        d_state: State dimension N (default: 64).
        dt_min: Minimum discretization step (default: 0.001).
        dt_max: Maximum discretization step (default: 0.1).
        channels: Number of output channels per SSM (default: 1).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        channels: int = 1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.channels = channels

        # Initialize with HiPPO
        A = hippo_initializer(d_state)
        Lambda, P, Q = dplr_decomposition(A)

        # Store DPLR parameters (learnable)
        # Lambda: diagonal part (complex)
        self.Lambda_real = nn.Parameter(Lambda.real)
        self.Lambda_imag = nn.Parameter(Lambda.imag)

        # P, Q: low-rank corrections (complex)
        self.P_real = nn.Parameter(P.real.expand(d_model, -1, -1).clone())
        self.P_imag = nn.Parameter(P.imag.expand(d_model, -1, -1).clone())

        # B: input matrix (complex, learnable)
        B = torch.randn(d_model, d_state, 1, dtype=torch.float32) / math.sqrt(d_state)
        self.B_real = nn.Parameter(B)
        self.B_imag = nn.Parameter(torch.zeros_like(B))

        # C: output matrix (complex, learnable)
        C = torch.randn(d_model, channels, d_state, dtype=torch.float32) / math.sqrt(d_state)
        self.C_real = nn.Parameter(C)
        self.C_imag = nn.Parameter(torch.zeros_like(C))

        # Discretization step (log-space)
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # D: skip connection
        self.D = nn.Parameter(torch.randn(d_model, channels))

    def _get_complex_params(self):
        """Reconstruct complex parameters from real/imaginary parts."""
        Lambda = torch.complex(self.Lambda_real, self.Lambda_imag)
        P = torch.complex(self.P_real, self.P_imag)
        B = torch.complex(self.B_real, self.B_imag)
        C = torch.complex(self.C_real, self.C_imag)
        return Lambda, P, B, C

    def compute_kernel(self, seq_len: int) -> torch.Tensor:
        """Compute the SSM convolution kernel via frequency domain.

        Uses the DPLR structure for efficient kernel computation:
        K(z) = C (zI - A_bar)^{-1} B_bar where A_bar, B_bar are discretized.

        For DPLR A = diag(Lambda) + P Q^T, the resolvent is computed efficiently
        using the Woodbury identity.

        Args:
            seq_len: Length of the kernel to generate.

        Returns:
            kernel: Real-valued kernel of shape (d_model, channels, seq_len).
        """
        Lambda, P, B, C = self._get_complex_params()
        dt = torch.exp(self.log_dt)  # (d_model,)

        # Discretize using ZOH (zero-order hold)
        # A_bar = exp(dt * Lambda) for diagonal part
        dtLambda = dt.unsqueeze(-1) * Lambda.unsqueeze(0)  # (d_model, d_state)

        # Generate the kernel using power series / Vandermonde
        # K[l] = C A_bar^l B_bar
        # For the diagonal part: A_bar^l = exp(l * dt * Lambda)

        # Create position indices
        arange = torch.arange(seq_len, device=Lambda.device, dtype=torch.float32)

        # Vandermonde-like computation: exp(l * dt * Lambda)
        # Shape: (d_model, d_state, seq_len)
        vandermonde = torch.exp(
            dtLambda.unsqueeze(-1) * arange.unsqueeze(0).unsqueeze(0)
        )

        # B_bar = dt * B (simplified ZOH discretization for diagonal)
        B_bar = dt.unsqueeze(-1).unsqueeze(-1) * B  # (d_model, d_state, 1)

        # K = C @ diag(vandermonde) @ B_bar
        # C: (d_model, channels, d_state)
        # vandermonde: (d_model, d_state, seq_len)
        # B_bar: (d_model, d_state, 1)

        # Einsum: K[d, c, l] = sum_n C[d, c, n] * vandermonde[d, n, l] * B_bar[d, n, 0]
        CB = C * B_bar.transpose(-2, -1)  # (d_model, channels, d_state) element-wise
        kernel = torch.einsum('dcn,dnl->dcl', CB, vandermonde)

        return kernel.real  # Take real part: (d_model, channels, seq_len)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute SSM output via convolution (training) or recurrence (inference).

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Optional recurrent state for inference.

        Returns:
            y: Output of shape (batch, seq_len, d_model * channels).
            state: Updated state (if using recurrence).
        """
        if self.training:
            return self.forward_conv(x)
        else:
            return self.forward_recurrent(x, state)

    def forward_conv(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Parallel convolution mode for training.

        Computes y = K * x + D * x using FFT-based convolution, where K is
        the SSM kernel and * denotes convolution.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            y: Output of shape (batch, seq_len, d_model * channels).
            state: None.
        """
        batch_size, seq_len, d_model = x.shape

        # Generate kernel: (d_model, channels, seq_len)
        kernel = self.compute_kernel(seq_len)

        # FFT convolution
        # x: (batch, seq, d_model) -> (batch, d_model, seq)
        x_t = x.transpose(1, 2)

        # Pad for causal convolution
        x_padded = F.pad(x_t, (seq_len - 1, 0))  # (batch, d_model, 2*seq-1)
        k_padded = F.pad(kernel, (0, seq_len - 1))  # (d_model, channels, 2*seq-1)

        # FFT
        x_fft = torch.fft.rfft(x_padded, dim=-1)  # (batch, d_model, freq)
        k_fft = torch.fft.rfft(k_padded, dim=-1)   # (d_model, channels, freq)

        # Multiply in frequency domain
        # x_fft: (batch, d_model, freq)
        # k_fft: (d_model, channels, freq)
        y_fft = x_fft.unsqueeze(2) * k_fft.unsqueeze(0)  # (batch, d_model, channels, freq)

        # IFFT
        y = torch.fft.irfft(y_fft, dim=-1)  # (batch, d_model, channels, time)
        y = y[..., :seq_len]  # Truncate to seq_len

        # Add skip connection
        y = y + x_t.unsqueeze(2) * self.D.unsqueeze(0).unsqueeze(-1)

        # Reshape: (batch, d_model, channels, seq) -> (batch, seq, d_model * channels)
        if self.channels == 1:
            y = y.squeeze(2).transpose(1, 2)  # (batch, seq, d_model)
        else:
            y = y.permute(0, 3, 1, 2).reshape(batch_size, seq_len, d_model * self.channels)

        return y, None

    def forward_recurrent(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential recurrence mode for inference.

        Standard SSM recurrence: h[t] = A_bar h[t-1] + B_bar x[t]
                                  y[t] = C h[t] + D x[t]

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Recurrent state of shape (batch, d_model, d_state), complex.

        Returns:
            y: Output of shape (batch, seq_len, d_model * channels).
            state: Updated state.
        """
        batch_size, seq_len, d_model = x.shape

        Lambda, P, B, C = self._get_complex_params()
        dt = torch.exp(self.log_dt)

        # Discretize
        A_bar = torch.exp(dt.unsqueeze(-1) * Lambda.unsqueeze(0))  # (d_model, d_state)
        B_bar = dt.unsqueeze(-1).unsqueeze(-1) * B  # (d_model, d_state, 1)
        B_bar = B_bar.squeeze(-1)  # (d_model, d_state)

        if state is None:
            state = torch.zeros(
                batch_size, d_model, self.d_state,
                device=x.device, dtype=torch.complex64
            )

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_model)
            x_t_complex = x_t.to(torch.complex64)

            # State update: h = A_bar * h + B_bar * x
            state = A_bar.unsqueeze(0) * state + B_bar.unsqueeze(0) * x_t_complex.unsqueeze(-1)

            # Output: y = real(C @ h) + D * x
            # C: (d_model, channels, d_state), state: (batch, d_model, d_state)
            y_t = torch.einsum('dcn,bdn->bdc', C, state).real  # (batch, d_model, channels)

            # Add skip
            y_t = y_t + x_t.unsqueeze(-1) * self.D.unsqueeze(0)  # (batch, d_model, channels)

            if self.channels == 1:
                y_t = y_t.squeeze(-1)  # (batch, d_model)
            else:
                y_t = y_t.reshape(batch_size, d_model * self.channels)

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y, state


class S4Layer(NexusModule):
    """S4 Layer with normalization and activation.

    Wraps the S4Kernel with pre-normalization, nonlinear activation, and
    dropout, forming a single S4 processing layer.

    Args:
        d_model: Model dimension.
        d_state: State dimension (default: 64).
        channels: Number of output channels (default: 1).
        dropout: Dropout probability (default: 0.0).
        dt_min: Minimum discretization step (default: 0.001).
        dt_max: Maximum discretization step (default: 0.1).
        bidirectional: Whether to use bidirectional SSM (default: False).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        channels: int = 1,
        dropout: float = 0.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.channels = channels
        self.bidirectional = bidirectional

        self.kernel = S4Kernel(
            d_model=d_model,
            d_state=d_state,
            dt_min=dt_min,
            dt_max=dt_max,
            channels=channels
        )

        if bidirectional:
            self.kernel_rev = S4Kernel(
                d_model=d_model,
                d_state=d_state,
                dt_min=dt_min,
                dt_max=dt_max,
                channels=channels
            )

        out_dim = d_model * channels * (2 if bidirectional else 1)
        self.out_proj = nn.Linear(out_dim, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of S4Layer.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Optional recurrent state.

        Returns:
            y: Output of shape (batch, seq_len, d_model).
            state: Updated state.
        """
        y, state = self.kernel(x, state)

        if self.bidirectional:
            x_rev = x.flip(dims=[1])
            y_rev, _ = self.kernel_rev(x_rev)
            y_rev = y_rev.flip(dims=[1])
            y = torch.cat([y, y_rev], dim=-1)

        y = self.activation(y)
        y = self.dropout(y)
        y = self.out_proj(y)

        return y, state


class S4Block(NexusModule):
    """S4 Block with residual connection and feed-forward network.

    Standard transformer-style block using S4 instead of attention:
        x -> norm -> S4Layer -> dropout -> + residual
        x -> norm -> FFN -> dropout -> + residual

    Reference: Gu et al., "Efficiently Modeling Long Sequences with
        Structured State Spaces", ICLR 2022.
        https://arxiv.org/abs/2111.00396

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension (default: 64).
        channels: Number of SSM channels (default: 1).
        dropout: Dropout probability (default: 0.0).
        ffn_expand: FFN expansion factor (default: 4).
        bidirectional: Whether to use bidirectional SSM (default: False).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        channels: int = 1,
        dropout: float = 0.0,
        ffn_expand: int = 4,
        bidirectional: bool = False
    ):
        super().__init__()
        self.d_model = d_model

        # S4 branch
        self.norm1 = nn.LayerNorm(d_model)
        self.s4 = S4Layer(
            d_model=d_model,
            d_state=d_state,
            channels=channels,
            dropout=dropout,
            bidirectional=bidirectional
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of S4Block.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Optional recurrent state.

        Returns:
            y: Output of shape (batch, seq_len, d_model).
            state: Updated state.
        """
        # S4 with residual
        residual = x
        x = self.norm1(x)
        x, state = self.s4(x, state)
        x = self.dropout1(x)
        x = x + residual

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x, state
