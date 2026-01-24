"""
HGRN - Hierarchically Gated Recurrent Neural Network.

HGRN uses hierarchical gating for efficient long-range modeling,
achieving strong performance on language modeling while maintaining
linear complexity.

The key innovation is using hierarchical gates that control information
flow at different levels, enabling the model to capture both local
and global patterns efficiently.

Reference: https://arxiv.org/abs/2311.04823
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from nexus.core.base import NexusModule


class HGRNCell(NexusModule):
    """
    Single HGRN cell with hierarchical gating.

    Implements the core HGRN recurrence:
        g_t = sigmoid(W_g @ x_t + U_g @ h_{t-1})
        f_t = sigmoid(W_f @ x_t + U_f @ h_{t-1} - lower_bound)
        i_t = sigmoid(W_i @ x_t + U_i @ h_{t-1})
        h_t = f_t * h_{t-1} + i_t * g_t * x_t

    Args:
        dim: Input dimension
        expand: Expansion factor for hidden dimension
        use_lower_bound: Use lower bound for forget gate
        lower_bound: Value of lower bound (default: 0.5)
        bias: Use bias in projections
    """

    def __init__(
        self,
        dim: int,
        expand: int = 1,
        use_lower_bound: bool = True,
        lower_bound: float = 0.5,
        bias: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.hidden_dim = dim * expand
        self.use_lower_bound = use_lower_bound
        self.lower_bound = lower_bound

        # Input gate projection
        self.i_proj = nn.Linear(dim, self.hidden_dim, bias=bias)

        # Forget gate projection
        self.f_proj = nn.Linear(dim, self.hidden_dim, bias=bias)

        # Output gate projection
        self.g_proj = nn.Linear(dim, self.hidden_dim, bias=bias)

        # Value projection
        self.v_proj = nn.Linear(dim, self.hidden_dim, bias=bias)

        # If expand > 1, need output projection
        if expand > 1:
            self.out_proj = nn.Linear(self.hidden_dim, dim, bias=bias)
        else:
            self.out_proj = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        # Initialize forget gate bias to encourage remembering initially
        if self.f_proj.bias is not None:
            nn.init.constant_(self.f_proj.bias, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of HGRN cell.

        Args:
            x: Input of shape (batch, dim)
            h: Previous hidden state of shape (batch, hidden_dim)

        Returns:
            output: Output of shape (batch, dim)
            h: New hidden state of shape (batch, hidden_dim)
        """
        batch_size = x.size(0)

        # Initialize hidden state if needed
        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        # Compute gates
        i = torch.sigmoid(self.i_proj(x))  # Input gate
        f = self.f_proj(x)  # Forget gate (pre-activation)

        # Apply lower bound to forget gate
        if self.use_lower_bound:
            # Ensures f >= lower_bound by using: f = lower_bound + (1 - lower_bound) * sigmoid(f)
            f = self.lower_bound + (1 - self.lower_bound) * torch.sigmoid(f)
        else:
            f = torch.sigmoid(f)

        g = torch.sigmoid(self.g_proj(x))  # Output gate

        # Compute candidate value
        v = self.v_proj(x)  # No activation on value

        # Update hidden state
        h_new = f * h + i * g * v

        # Project output
        output = self.out_proj(h_new)

        return output, h_new


class HGRN(NexusModule):
    """
    Hierarchically Gated Recurrent Neural Network.

    Uses hierarchical gating for efficient long-range modeling.
    Supports both sequential (recurrent) and parallel (training) modes.

    Reference: https://arxiv.org/abs/2311.04823

    Args:
        dim: Model dimension
        expand: Expansion factor for hidden state
        use_lower_bound: Use lower bound for forget gate
        lower_bound: Value of lower bound (default: 0.5)
        use_short_conv: Use short convolution for local context
        conv_size: Convolution kernel size
        use_output_gate: Use additional output gating
        bias: Use bias in projections
    """

    def __init__(
        self,
        dim: int,
        expand: int = 1,
        use_lower_bound: bool = True,
        lower_bound: float = 0.5,
        use_short_conv: bool = True,
        conv_size: int = 4,
        use_output_gate: bool = True,
        bias: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.hidden_dim = dim * expand
        self.use_lower_bound = use_lower_bound
        self.lower_bound = lower_bound
        self.use_short_conv = use_short_conv
        self.use_output_gate = use_output_gate

        # Input projection
        proj_dim = self.hidden_dim * 3  # i, f, v projections
        if use_output_gate:
            proj_dim += self.hidden_dim  # Add output gate
        self.in_proj = nn.Linear(dim, proj_dim, bias=bias)

        # Short convolution
        if use_short_conv:
            self.conv = nn.Conv1d(
                self.hidden_dim, self.hidden_dim,
                kernel_size=conv_size,
                padding=conv_size - 1,
                groups=self.hidden_dim,
                bias=bias
            )

        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=bias)

        # Layer normalization
        self.norm = nn.LayerNorm(self.hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of HGRN.

        Args:
            x: Input of shape (batch, seq_len, dim)
            state: Optional hidden state of shape (batch, hidden_dim)

        Returns:
            output: Output of shape (batch, seq_len, dim)
            state: Updated hidden state
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        proj = self.in_proj(x)

        if self.use_output_gate:
            ifv, g = proj.split([self.hidden_dim * 3, self.hidden_dim], dim=-1)
            i, f, v = ifv.split(self.hidden_dim, dim=-1)
        else:
            i, f, v = proj.split(self.hidden_dim, dim=-1)
            g = None

        # Apply short convolution to v
        if self.use_short_conv:
            v = v.transpose(1, 2)  # (batch, hidden, seq)
            v = self.conv(v)[:, :, :seq_len]  # Causal
            v = v.transpose(1, 2)  # (batch, seq, hidden)

        # Apply activations
        i = torch.sigmoid(i)  # Input gate

        # Forget gate with optional lower bound
        if self.use_lower_bound:
            f = self.lower_bound + (1 - self.lower_bound) * torch.sigmoid(f)
        else:
            f = torch.sigmoid(f)

        # Run recurrence
        if self.training:
            # Parallel mode for training (using cumsum trick)
            output, state = self.forward_parallel(i, f, v, state)
        else:
            # Recurrent mode for inference
            output, state = self.forward_recurrent(i, f, v, state)

        # Apply normalization
        output = self.norm(output)

        # Apply output gate
        if self.use_output_gate and g is not None:
            output = output * F.silu(g)

        # Project output
        output = self.out_proj(output)

        return output, state

    def forward_parallel(
        self,
        i: torch.Tensor,
        f: torch.Tensor,
        v: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel forward using log-space cumulative sum.

        Uses the associative scan property to compute the recurrence
        in parallel: h_t = f_t * h_{t-1} + i_t * v_t

        This can be rewritten as:
            h_t = sum_{j=0}^{t} (prod_{k=j+1}^{t} f_k) * i_j * v_j

        Args:
            i: Input gate of shape (batch, seq, hidden)
            f: Forget gate of shape (batch, seq, hidden)
            v: Value of shape (batch, seq, hidden)
            state: Initial state of shape (batch, hidden)

        Returns:
            output: Shape (batch, seq, hidden)
            state: Final state
        """
        batch_size, seq_len, hidden_dim = i.shape

        # Initialize state
        if state is None:
            state = torch.zeros(batch_size, hidden_dim, device=i.device, dtype=i.dtype)

        # Log-space computation for numerical stability
        log_f = torch.log(f + 1e-6)  # (batch, seq, hidden)

        # Cumulative sum of log forget gates
        log_f_cumsum = torch.cumsum(log_f, dim=1)  # (batch, seq, hidden)

        # Compute weighted values: i * v * exp(cumsum_log_f)
        # For each position t, we need sum_{j<=t} (prod_{k=j+1}^{t} f_k) * i_j * v_j
        # = sum_{j<=t} exp(sum_{k=j+1}^{t} log_f_k) * i_j * v_j
        # = sum_{j<=t} exp(cumsum_log_f[t] - cumsum_log_f[j]) * i_j * v_j
        # = exp(cumsum_log_f[t]) * sum_{j<=t} exp(-cumsum_log_f[j]) * i_j * v_j

        # Weighted inputs in log space
        weighted_v = i * v * torch.exp(-log_f_cumsum)  # (batch, seq, hidden)

        # Cumulative sum of weighted values
        cumsum_weighted_v = torch.cumsum(weighted_v, dim=1)  # (batch, seq, hidden)

        # Scale back from log space
        output = cumsum_weighted_v * torch.exp(log_f_cumsum)  # (batch, seq, hidden)

        # Handle initial state
        # h_t = output_t + state * prod_{k=1}^{t} f_k
        state_contrib = state.unsqueeze(1) * torch.exp(log_f_cumsum)
        output = output + state_contrib

        # Final state is the last hidden state
        final_state = output[:, -1, :]

        return output, final_state

    def forward_recurrent(
        self,
        i: torch.Tensor,
        f: torch.Tensor,
        v: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent forward for inference.

        Processes one timestep at a time with O(1) complexity per step.

        Args:
            i: Input gate of shape (batch, seq, hidden)
            f: Forget gate of shape (batch, seq, hidden)
            v: Value of shape (batch, seq, hidden)
            state: Hidden state of shape (batch, hidden)

        Returns:
            output: Shape (batch, seq, hidden)
            state: Updated state
        """
        batch_size, seq_len, hidden_dim = i.shape

        # Initialize state
        if state is None:
            state = torch.zeros(batch_size, hidden_dim, device=i.device, dtype=i.dtype)

        outputs = []

        for t in range(seq_len):
            i_t = i[:, t]  # (batch, hidden)
            f_t = f[:, t]
            v_t = v[:, t]

            # Update state: h_t = f_t * h_{t-1} + i_t * v_t
            state = f_t * state + i_t * v_t
            outputs.append(state)

        output = torch.stack(outputs, dim=1)  # (batch, seq, hidden)
        return output, state

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Initialize recurrent state.

        Args:
            batch_size: Batch size
            device: Device for state tensor
            dtype: Data type for state tensor

        Returns:
            Initial state of shape (batch, hidden_dim)
        """
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)


class HGRNLayer(NexusModule):
    """
    Full HGRN layer with normalization, FFN, and residual connections.

    Combines HGRN with feed-forward network for use in
    transformer-style architectures.

    Args:
        dim: Model dimension
        expand: Expansion factor for HGRN
        ffn_expand: Expansion factor for FFN
        dropout: Dropout probability
        **kwargs: Additional arguments for HGRN
    """

    def __init__(
        self,
        dim: int,
        expand: int = 1,
        ffn_expand: int = 4,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim

        # HGRN block
        self.norm1 = nn.LayerNorm(dim)
        self.hgrn = HGRN(dim=dim, expand=expand, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        # FFN block
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_expand, dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with residual connections.

        Args:
            x: Input of shape (batch, seq_len, dim)
            state: Optional recurrent state

        Returns:
            output: Output of shape (batch, seq_len, dim)
            state: Updated state
        """
        # HGRN with residual
        residual = x
        x = self.norm1(x)
        x, state = self.hgrn(x, state)
        x = self.dropout1(x)
        x = x + residual

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x, state


class HGRN2(NexusModule):
    """
    HGRN2 - Enhanced HGRN with state expansion.

    Extends HGRN by expanding the recurrent state dimension,
    allowing for richer representations.

    Reference: https://arxiv.org/abs/2311.04823 (HGRN2 variant)

    Args:
        dim: Model dimension
        expand: Expansion factor for state
        state_expand: Additional expansion for state dimension
        use_lower_bound: Use lower bound for forget gate
        lower_bound: Value of lower bound
        bias: Use bias
    """

    def __init__(
        self,
        dim: int,
        expand: int = 1,
        state_expand: int = 2,
        use_lower_bound: bool = True,
        lower_bound: float = 0.5,
        bias: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.state_expand = state_expand
        self.hidden_dim = dim * expand
        self.state_dim = self.hidden_dim * state_expand
        self.use_lower_bound = use_lower_bound
        self.lower_bound = lower_bound

        # Input projections
        self.i_proj = nn.Linear(dim, self.hidden_dim, bias=bias)
        self.f_proj = nn.Linear(dim, self.hidden_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.state_dim, bias=bias)
        self.o_proj = nn.Linear(dim, self.hidden_dim, bias=bias)

        # State to output projection
        self.state_proj = nn.Linear(self.state_dim, self.hidden_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=bias)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        if self.f_proj.bias is not None:
            nn.init.constant_(self.f_proj.bias, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of HGRN2.

        Args:
            x: Input of shape (batch, seq_len, dim)
            state: Optional hidden state of shape (batch, hidden_dim, state_expand)

        Returns:
            output: Output of shape (batch, seq_len, dim)
            state: Updated hidden state
        """
        batch_size, seq_len, _ = x.shape

        # Compute gates
        i = torch.sigmoid(self.i_proj(x))  # (batch, seq, hidden)
        f_raw = self.f_proj(x)

        if self.use_lower_bound:
            f = self.lower_bound + (1 - self.lower_bound) * torch.sigmoid(f_raw)
        else:
            f = torch.sigmoid(f_raw)

        v = self.v_proj(x)  # (batch, seq, state_dim)
        o = torch.sigmoid(self.o_proj(x))  # (batch, seq, hidden)

        # Reshape for state expansion
        v = v.view(batch_size, seq_len, self.hidden_dim, self.state_expand)
        i = i.unsqueeze(-1)  # (batch, seq, hidden, 1)
        f = f.unsqueeze(-1)  # (batch, seq, hidden, 1)

        # Initialize state
        if state is None:
            state = torch.zeros(
                batch_size, self.hidden_dim, self.state_expand,
                device=x.device, dtype=x.dtype
            )

        # Run recurrence
        outputs = []
        for t in range(seq_len):
            i_t = i[:, t]  # (batch, hidden, 1)
            f_t = f[:, t]
            v_t = v[:, t]  # (batch, hidden, state_expand)

            # Update state
            state = f_t * state + i_t * v_t

            # Project state to output
            h_t = state.view(batch_size, self.state_dim)
            h_t = self.state_proj(h_t)  # (batch, hidden)

            # Apply output gate
            out_t = o[:, t] * h_t
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)  # (batch, seq, hidden)

        # Project output
        output = self.out_proj(output)

        return output, state

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Initialize recurrent state."""
        return torch.zeros(
            batch_size, self.hidden_dim, self.state_expand,
            device=device, dtype=dtype
        )
