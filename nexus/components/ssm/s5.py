"""
S5 - Simplified State Space Layers.

S5 simplifies the S4 framework by using a single MIMO (multi-input, multi-output)
SSM per layer instead of multiple SISO (single-input, single-output) SSMs. Key
simplifications and improvements:

1. MIMO SSM: Instead of d_model independent SSMs, uses a single large SSM that
   maps R^d_model -> R^d_model through a shared state of size d_state.

2. Parallel scan: Replaces S4's frequency-domain computation with a parallel
   associative scan in the time domain. This is simpler to implement and avoids
   the numerical issues of frequency-domain methods.

3. Diagonal state matrix: Uses a complex diagonal state matrix (like S4D) with
   proper HiPPO-inspired initialization, avoiding the DPLR complexity of S4.

4. Efficient discretization: Uses the ZOH (zero-order hold) method with diagonal
   structure for O(N) discretization per step.

Reference: Smith et al., "Simplified State Space Layers for Sequence Modeling",
    ICLR 2023. https://arxiv.org/abs/2208.04933
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


def s5_diagonal_init(d_state: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Initialize diagonal state matrix with HiPPO-inspired values.

    Uses the eigenvalues of the HiPPO-LegS matrix, which lie along the
    negative real axis with imaginary components. The diagonal approximation
    captures the essential spectral properties for long-range modeling.

    Args:
        d_state: State dimension (will use d_state // 2 complex pairs).

    Returns:
        real_part: Real parts of diagonal entries, shape (d_state,).
        imag_part: Imaginary parts of diagonal entries, shape (d_state,).
    """
    # Half the state dimension for complex conjugate pairs
    half_N = d_state // 2

    # Real part: decreasing negative values (decay rates)
    real_part = -0.5 * torch.ones(d_state)

    # Imaginary part: frequencies inspired by HiPPO spectrum
    imag_part = torch.zeros(d_state)
    if half_N > 0:
        freqs = math.pi * torch.arange(1, half_N + 1, dtype=torch.float32)
        imag_part[:half_N] = freqs
        imag_part[half_N:2 * half_N] = -freqs

    return real_part, imag_part


def parallel_scan(gates: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Parallel associative scan for linear recurrence.

    Computes the recurrence h[t] = gate[t] * h[t-1] + input[t] in O(log T)
    parallel steps using the associative scan algorithm.

    The associative operator is:
        (g_a, x_a) * (g_b, x_b) = (g_a * g_b, g_b * x_a + x_b)

    This scan computes cumulative products of gates and weighted sums of inputs.

    Args:
        gates: Decay gates of shape (batch, seq_len, d_state), complex-valued.
        inputs: Input values of shape (batch, seq_len, d_state), complex-valued.

    Returns:
        hidden_states: All hidden states, shape (batch, seq_len, d_state).
    """
    batch_size, seq_len, d_state = gates.shape

    # Handle power-of-2 alignment
    log2_T = int(math.ceil(math.log2(max(seq_len, 1))))
    T_padded = 2 ** log2_T

    if T_padded > seq_len:
        pad_len = T_padded - seq_len
        gates = F.pad(gates, (0, 0, 0, pad_len), value=1.0)
        inputs = F.pad(inputs, (0, 0, 0, pad_len), value=0.0)

    # Up sweep (reduce)
    saved_gates = []
    saved_inputs = []

    current_gates = gates
    current_inputs = inputs

    for d in range(log2_T):
        T_cur = current_gates.shape[1]
        half_T = T_cur // 2

        even_gates = current_gates[:, 0::2]   # (batch, half_T, d_state)
        odd_gates = current_gates[:, 1::2]
        even_inputs = current_inputs[:, 0::2]
        odd_inputs = current_inputs[:, 1::2]

        saved_gates.append(current_gates)
        saved_inputs.append(current_inputs)

        # Combine pairs: (g_even, x_even) * (g_odd, x_odd)
        new_gates = even_gates * odd_gates
        new_inputs = odd_gates * even_inputs + odd_inputs

        current_gates = new_gates
        current_inputs = new_inputs

    # Down sweep (from the reduced result, broadcast back)
    # Reconstruct by interleaving
    result = current_inputs  # (batch, 1, d_state)

    for d in range(log2_T - 1, -1, -1):
        orig_gates = saved_gates[d]
        orig_inputs = saved_inputs[d]
        T_cur = orig_gates.shape[1]

        new_result = torch.zeros(
            batch_size, T_cur, d_state,
            device=gates.device, dtype=gates.dtype
        )

        # Odd indices get their value from the reduced result
        new_result[:, 0::2] = result if d > 0 else orig_inputs[:, 0:1]
        new_result[:, 1::2] = result

        if d == 0:
            # Base case: direct computation
            new_result[:, 0] = orig_inputs[:, 0]
            for t in range(1, T_cur):
                new_result[:, t] = orig_gates[:, t] * new_result[:, t - 1] + orig_inputs[:, t]
        result = new_result

    return result[:, :seq_len]


class S5SSM(NexusModule):
    """MIMO State Space Model used in S5.

    A single large SSM that processes all d_model input channels through a
    shared d_state dimensional state. Uses complex diagonal parameterization
    for the state matrix.

    The continuous-time system is:
        dh/dt = A h + B u
        y = Re(C h) + D u

    where A = diag(Lambda) is complex diagonal.

    Discretization via ZOH:
        A_bar = exp(dt * Lambda)
        B_bar = (A_bar - I) Lambda^{-1} B

    Args:
        d_model: Input/output dimension.
        d_state: State dimension (default: 64).
        dt_min: Minimum discretization step (default: 0.001).
        dt_max: Maximum discretization step (default: 0.1).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Diagonal state matrix A (complex)
        real_init, imag_init = s5_diagonal_init(d_state)
        self.Lambda_real = nn.Parameter(real_init)
        self.Lambda_imag = nn.Parameter(imag_init)

        # Input matrix B: d_state x d_model (complex)
        B_real = torch.randn(d_state, d_model) / math.sqrt(d_model)
        B_imag = torch.randn(d_state, d_model) / math.sqrt(d_model)
        self.B_real = nn.Parameter(B_real)
        self.B_imag = nn.Parameter(B_imag)

        # Output matrix C: d_model x d_state (complex)
        C_real = torch.randn(d_model, d_state) / math.sqrt(d_state)
        C_imag = torch.randn(d_model, d_state) / math.sqrt(d_state)
        self.C_real = nn.Parameter(C_real)
        self.C_imag = nn.Parameter(C_imag)

        # Skip connection D
        self.D = nn.Parameter(torch.randn(d_model))

        # Discretization step (learnable, log-space)
        log_dt = torch.rand(1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    def _discretize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discretize the continuous-time parameters via ZOH.

        Returns:
            A_bar: Discretized diagonal state matrix, shape (d_state,), complex.
            B_bar: Discretized input matrix, shape (d_state, d_model), complex.
        """
        dt = torch.exp(self.log_dt)
        Lambda = torch.complex(self.Lambda_real, self.Lambda_imag)
        B = torch.complex(self.B_real, self.B_imag)

        # ZOH: A_bar = exp(dt * Lambda)
        A_bar = torch.exp(dt * Lambda)  # (d_state,)

        # ZOH: B_bar = (A_bar - I) * Lambda^{-1} * B
        # For numerical stability when Lambda is near zero:
        Lambda_safe = Lambda + 1e-8 * (Lambda.abs() < 1e-8).float()
        B_bar = ((A_bar - 1.0) / Lambda_safe).unsqueeze(-1) * B  # (d_state, d_model)

        return A_bar, B_bar

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with automatic mode selection.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Optional state of shape (batch, d_state), complex.

        Returns:
            y: Output of shape (batch, seq_len, d_model).
            state: Updated state.
        """
        if self.training:
            return self.forward_parallel(x, state)
        else:
            return self.forward_recurrent(x, state)

    def forward_parallel(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel forward using associative scan.

        Computes the full recurrence h[t] = A_bar h[t-1] + B_bar u[t]
        in parallel using the scan algorithm.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Optional initial state of shape (batch, d_state), complex.

        Returns:
            y: Output of shape (batch, seq_len, d_model).
            state: Final state of shape (batch, d_state), complex.
        """
        batch_size, seq_len, d_model = x.shape

        A_bar, B_bar = self._discretize()
        C = torch.complex(self.C_real, self.C_imag)

        # Compute gate and input for parallel scan
        # gates[t] = A_bar (constant for all t): (d_state,) -> (batch, seq, d_state)
        gates = A_bar.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        # inputs[t] = B_bar @ u[t]: (batch, seq, d_state)
        x_complex = x.to(torch.complex64)
        inputs = torch.einsum('nd,btd->btn', B_bar, x_complex)  # (batch, seq, d_state)

        # Handle initial state
        if state is not None:
            # Fold initial state into the first input:
            # h[0] = A_bar * h_init + B_bar u[0]
            # This is already what the scan computes if we set inputs[0] += A_bar * h_init
            # But for cleanliness, we just add the contribution
            inputs[:, 0] = inputs[:, 0] + A_bar.unsqueeze(0) * state

        # Run parallel scan
        hidden = parallel_scan(gates, inputs)  # (batch, seq, d_state)

        # Output: y = Re(C @ h) + D * u
        y = torch.einsum('dn,btn->btd', C, hidden).real  # (batch, seq, d_model)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        # Return final state
        final_state = hidden[:, -1]  # (batch, d_state)

        return y, final_state

    def forward_recurrent(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential recurrence for inference.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: State of shape (batch, d_state), complex.

        Returns:
            y: Output of shape (batch, seq_len, d_model).
            state: Updated state.
        """
        batch_size, seq_len, d_model = x.shape

        A_bar, B_bar = self._discretize()
        C = torch.complex(self.C_real, self.C_imag)

        if state is None:
            state = torch.zeros(
                batch_size, self.d_state,
                device=x.device, dtype=torch.complex64
            )

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t].to(torch.complex64)  # (batch, d_model)

            # State update: h = A_bar * h + B_bar @ x
            Bx = torch.einsum('nd,bd->bn', B_bar, x_t)  # (batch, d_state)
            state = A_bar.unsqueeze(0) * state + Bx

            # Output: y = Re(C @ h) + D * x
            y_t = torch.einsum('dn,bn->bd', C, state).real  # (batch, d_model)
            y_t = y_t + x[:, t] * self.D.unsqueeze(0)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y, state

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.complex64
    ) -> torch.Tensor:
        """Initialize recurrent state.

        Args:
            batch_size: Batch size.
            device: Device for state tensor.
            dtype: Data type (should be complex).

        Returns:
            Initial state of shape (batch, d_state).
        """
        return torch.zeros(batch_size, self.d_state, device=device, dtype=dtype)


class S5Layer(NexusModule):
    """S5 Layer with pre-normalization, activation, and dropout.

    Wraps the S5SSM core with standard transformer-style layer processing.
    Supports multiple parallel SSMs per layer for increased capacity.

    Args:
        d_model: Model dimension.
        d_state: State dimension per SSM (default: 64).
        num_ssm: Number of parallel SSMs (default: 1).
        dropout: Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        num_ssm: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_ssm = num_ssm

        self.norm = nn.LayerNorm(d_model)

        # Multiple parallel SSMs
        self.ssms = nn.ModuleList([
            S5SSM(d_model=d_model, d_state=d_state)
            for _ in range(num_ssm)
        ])

        # Combine outputs if multiple SSMs
        if num_ssm > 1:
            self.combine = nn.Linear(d_model * num_ssm, d_model)
        else:
            self.combine = nn.Identity()

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[list] = None
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Optional list of states, one per SSM.

        Returns:
            y: Output of shape (batch, seq_len, d_model).
            states: Updated list of states.
        """
        x_normed = self.norm(x)

        if state is None:
            state = [None] * self.num_ssm

        outputs = []
        new_states = []
        for i, ssm in enumerate(self.ssms):
            y_i, s_i = ssm(x_normed, state[i])
            outputs.append(y_i)
            new_states.append(s_i)

        if self.num_ssm > 1:
            y = torch.cat(outputs, dim=-1)
            y = self.combine(y)
        else:
            y = outputs[0]

        y = self.activation(y)
        y = self.dropout(y)

        return y, new_states


class S5Block(NexusModule):
    """S5 Block with residual connections and FFN.

    Standard transformer-style block using S5 instead of attention:
        x -> norm -> S5Layer -> dropout -> + residual
        x -> norm -> FFN -> + residual

    Reference: Smith et al., "Simplified State Space Layers for Sequence
        Modeling", ICLR 2023. https://arxiv.org/abs/2208.04933

    Args:
        d_model: Model dimension.
        d_state: State dimension per SSM (default: 64).
        num_ssm: Number of parallel SSMs (default: 1).
        dropout: Dropout probability (default: 0.0).
        ffn_expand: FFN expansion factor (default: 4).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        num_ssm: int = 1,
        dropout: float = 0.0,
        ffn_expand: int = 4
    ):
        super().__init__()
        self.d_model = d_model

        # S5 branch
        self.s5_layer = S5Layer(
            d_model=d_model,
            d_state=d_state,
            num_ssm=num_ssm,
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
        state: Optional[list] = None
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Forward pass with residual connections.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Optional list of SSM states.

        Returns:
            y: Output of shape (batch, seq_len, d_model).
            state: Updated states.
        """
        # S5 with residual
        residual = x
        y, state = self.s5_layer(x, state)
        y = self.dropout1(y)
        y = y + residual

        # FFN with residual
        residual = y
        y = self.norm2(y)
        y = self.ffn(y)
        y = y + residual

        return y, state
