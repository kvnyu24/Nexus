"""
Gated DeltaNet - Linear RNN with Delta Rule Updates.

DeltaNet uses the delta rule for memory updates, providing an alternative
to attention that is used in models like Qwen3-Next and Kimi Linear.
The delta rule allows for selective updates to the hidden state, enabling
the model to learn what information to store and forget.

Reference: https://arxiv.org/abs/2412.06464
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List
from nexus.core.base import NexusModule
from nexus.core.initialization import WeightInitMixin


class GatedDeltaNet(WeightInitMixin, NexusModule):
    """
    Gated DeltaNet - Linear RNN with delta rule updates.

    Used by Qwen3-Next, Kimi Linear as alternative to attention.
    Combines benefits of linear attention with gated recurrence.

    The delta rule updates the hidden state as:
        h_t = h_{t-1} + beta_t * (v_t - h_{t-1} @ k_t) @ k_t^T

    This allows the model to selectively update specific parts of
    the hidden state based on the input, similar to how attention
    selectively retrieves information.

    Reference: https://arxiv.org/abs/2412.06464

    Args:
        dim: Model dimension
        expand: Expansion factor for hidden state (default: 2)
        num_heads: Number of heads (default: 4)
        head_dim: Dimension per head (default: None, computed from dim/num_heads)
        use_gate: Whether to use output gating (default: True)
        use_short_conv: Whether to use short convolution (default: True)
        conv_size: Convolution kernel size (default: 4)
        use_beta_gate: Whether to use learnable beta gate (default: True)
        qk_norm: Whether to normalize Q and K (default: True)
        bias: Use bias in projections (default: False)
    """

    def __init__(
        self,
        dim: int,
        expand: int = 2,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        use_beta_gate: bool = True,
        qk_norm: bool = True,
        bias: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim * expand) // num_heads
        self.hidden_dim = self.num_heads * self.head_dim
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_beta_gate = use_beta_gate
        self.qk_norm = qk_norm

        # Input projection: q, k, v (and gate if enabled)
        proj_dim = self.hidden_dim * 3
        if use_gate:
            proj_dim += self.hidden_dim  # Add gate dimension
        self.in_proj = nn.Linear(dim, proj_dim, bias=bias)

        # Beta projection (learning rate for delta rule)
        if use_beta_gate:
            self.beta_proj = nn.Linear(dim, self.num_heads, bias=bias)
        else:
            self.beta = nn.Parameter(torch.ones(self.num_heads) * 0.5)

        # Short convolution
        if use_short_conv:
            self.conv_q = nn.Conv1d(
                self.hidden_dim, self.hidden_dim,
                kernel_size=conv_size, padding=conv_size - 1,
                groups=self.hidden_dim, bias=True
            )
            self.conv_k = nn.Conv1d(
                self.hidden_dim, self.hidden_dim,
                kernel_size=conv_size, padding=conv_size - 1,
                groups=self.hidden_dim, bias=True
            )
            self.conv_v = nn.Conv1d(
                self.hidden_dim, self.hidden_dim,
                kernel_size=conv_size, padding=conv_size - 1,
                groups=self.hidden_dim, bias=True
            )

        # QK normalization
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=bias)

        # Output normalization
        self.out_norm = nn.LayerNorm(self.hidden_dim)

        # Initialize weights using mixin (xavier uniform)
        from nexus.core.initialization import InitMethod
        self.apply_weight_init(method=InitMethod.XAVIER_UNIFORM)
        # Special initialization for beta projection
        if self.use_beta_gate:
            nn.init.zeros_(self.beta_proj.weight)
            if self.beta_proj.bias is not None:
                nn.init.ones_(self.beta_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Gated DeltaNet.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            state: Optional hidden state of shape (batch, num_heads, head_dim, head_dim)

        Returns:
            output: Output tensor of shape (batch, seq_len, dim)
            state: Updated hidden state
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        proj = self.in_proj(x)

        if self.use_gate:
            q, k, v, g = proj.split(self.hidden_dim, dim=-1)
        else:
            q, k, v = proj.split(self.hidden_dim, dim=-1)
            g = None

        # Apply short convolutions
        if self.use_short_conv:
            q = self._apply_conv(q, self.conv_q, seq_len)
            k = self._apply_conv(k, self.conv_k, seq_len)
            v = self._apply_conv(v, self.conv_v, seq_len)

        # Reshape to multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply QK normalization
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Compute beta (learning rate for delta rule)
        if self.use_beta_gate:
            beta = torch.sigmoid(self.beta_proj(x))  # (batch, seq, num_heads)
        else:
            beta = self.beta.view(1, 1, -1).expand(batch_size, seq_len, -1)

        # Initialize state if needed
        if state is None:
            state = torch.zeros(
                batch_size, self.num_heads, self.head_dim, self.head_dim,
                device=x.device, dtype=x.dtype
            )

        # Run delta rule recurrence
        if self.training:
            # Parallel mode for training (chunk-wise)
            output, state = self.forward_parallel(q, k, v, beta, state)
        else:
            # Recurrent mode for inference
            output, state = self.forward_recurrent(q, k, v, beta, state)

        # Reshape output
        output = output.view(batch_size, seq_len, self.hidden_dim)

        # Apply output normalization
        output = self.out_norm(output)

        # Apply output gate
        if self.use_gate and g is not None:
            output = output * F.silu(g)

        # Project output
        output = self.out_proj(output)

        return output, state

    def _apply_conv(self, x: torch.Tensor, conv: nn.Conv1d, seq_len: int) -> torch.Tensor:
        """Apply causal convolution."""
        x = x.transpose(1, 2)  # (batch, hidden, seq)
        x = conv(x)[:, :, :seq_len]  # Causal
        x = x.transpose(1, 2)  # (batch, seq, hidden)
        return F.silu(x)

    def forward_parallel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel forward for training.

        Uses chunk-wise processing to balance efficiency and memory.

        Args:
            q: Query of shape (batch, seq, heads, head_dim)
            k: Key of shape (batch, seq, heads, head_dim)
            v: Value of shape (batch, seq, heads, head_dim)
            beta: Learning rate of shape (batch, seq, heads)
            state: Hidden state of shape (batch, heads, head_dim, head_dim)

        Returns:
            output: Shape (batch, seq, heads, head_dim)
            state: Updated state
        """
        batch_size, seq_len, num_heads, head_dim = q.shape

        # Normalize keys (L2 norm for stability)
        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        outputs = []

        # Process in chunks for efficiency
        chunk_size = min(64, seq_len)

        for i in range(0, seq_len, chunk_size):
            end = min(i + chunk_size, seq_len)
            chunk_len = end - i

            q_chunk = q[:, i:end]  # (batch, chunk, heads, dim)
            k_chunk = k_norm[:, i:end]
            v_chunk = v[:, i:end]
            beta_chunk = beta[:, i:end]

            # Process chunk with delta rule
            chunk_outputs = []
            for t in range(chunk_len):
                q_t = q_chunk[:, t]  # (batch, heads, dim)
                k_t = k_chunk[:, t]
                v_t = v_chunk[:, t]
                beta_t = beta_chunk[:, t]  # (batch, heads)

                # Delta rule update:
                # error = v_t - state @ k_t
                # state = state + beta_t * error @ k_t^T

                # Retrieve current memory
                retrieved = torch.einsum('bhij,bhj->bhi', state, k_t)  # (batch, heads, dim)

                # Compute error
                error = v_t - retrieved  # (batch, heads, dim)

                # Update state with delta rule
                delta = torch.einsum('bhi,bhj->bhij', error, k_t)  # (batch, heads, dim, dim)
                state = state + beta_t.unsqueeze(-1).unsqueeze(-1) * delta

                # Query the updated state
                output_t = torch.einsum('bhij,bhj->bhi', state, q_t)
                chunk_outputs.append(output_t)

            outputs.extend(chunk_outputs)

        output = torch.stack(outputs, dim=1)  # (batch, seq, heads, dim)
        return output, state

    def forward_recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent forward for inference.

        Processes one timestep at a time, suitable for autoregressive generation.

        Args:
            q: Query of shape (batch, seq, heads, head_dim)
            k: Key of shape (batch, seq, heads, head_dim)
            v: Value of shape (batch, seq, heads, head_dim)
            beta: Learning rate of shape (batch, seq, heads)
            state: Hidden state of shape (batch, heads, head_dim, head_dim)

        Returns:
            output: Shape (batch, seq, heads, head_dim)
            state: Updated state
        """
        batch_size, seq_len, num_heads, head_dim = q.shape

        # Normalize keys
        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        outputs = []

        for t in range(seq_len):
            q_t = q[:, t]  # (batch, heads, dim)
            k_t = k_norm[:, t]
            v_t = v[:, t]
            beta_t = beta[:, t]

            # Retrieve from memory
            retrieved = torch.einsum('bhij,bhj->bhi', state, k_t)

            # Compute error and update
            error = v_t - retrieved
            delta = torch.einsum('bhi,bhj->bhij', error, k_t)
            state = state + beta_t.unsqueeze(-1).unsqueeze(-1) * delta

            # Query state
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
        """
        Initialize recurrent state.

        Args:
            batch_size: Batch size
            device: Device for state tensor
            dtype: Data type for state tensor

        Returns:
            Initial state of shape (batch, num_heads, head_dim, head_dim)
        """
        return torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.head_dim,
            device=device, dtype=dtype
        )


class DeltaNetLayer(NexusModule):
    """
    Full DeltaNet layer with normalization and residual connection.

    Combines GatedDeltaNet with pre-normalization and residual connection
    for use in transformer-style architectures.

    Args:
        dim: Model dimension
        expand: Expansion factor
        num_heads: Number of heads
        dropout: Dropout probability
        **kwargs: Additional arguments for GatedDeltaNet
    """

    def __init__(
        self,
        dim: int,
        expand: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim

        self.norm = nn.LayerNorm(dim)
        self.deltanet = GatedDeltaNet(
            dim=dim,
            expand=expand,
            num_heads=num_heads,
            **kwargs
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with residual connection.

        Args:
            x: Input of shape (batch, seq_len, dim)
            state: Optional recurrent state

        Returns:
            output: Output of shape (batch, seq_len, dim)
            state: Updated state
        """
        residual = x
        x = self.norm(x)
        x, state = self.deltanet(x, state)
        x = self.dropout(x)
        return x + residual, state
