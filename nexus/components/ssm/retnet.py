"""
RetNet - Retentive Network.

RetNet uses a retention mechanism with exponential decay, providing
an efficient alternative to attention that supports both parallel
(training) and recurrent (inference) modes.

The retention mechanism combines the strengths of:
- Recurrent models: O(1) inference complexity
- Transformers: Parallelizable training
- Linear attention: O(n) training complexity

Reference: https://arxiv.org/abs/2307.08621
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List
from nexus.core.base import NexusModule


class MultiScaleRetention(NexusModule):
    """
    Multi-Scale Retention mechanism.

    Uses different decay factors (gamma) per head to capture
    patterns at different temporal scales.

    Args:
        dim: Model dimension
        num_heads: Number of retention heads
        head_dim: Dimension per head (default: dim // num_heads)
        gamma: Decay factor or list of decay factors per head
        use_group_norm: Whether to use group normalization
        bias: Use bias in projections
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        gamma: Optional[Union[float, List[float]]] = None,
        use_group_norm: bool = True,
        bias: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.hidden_dim = self.num_heads * self.head_dim
        self.use_group_norm = use_group_norm

        # Set decay factors (gamma) per head
        if gamma is None:
            # Default: logarithmically spaced values between 0.8 and 0.99
            gamma = [1 - 2 ** (-5 - i) for i in range(num_heads)]
        elif isinstance(gamma, float):
            gamma = [gamma] * num_heads

        self.register_buffer('gamma', torch.tensor(gamma, dtype=torch.float32))

        # Projections for Q, K, V
        self.q_proj = nn.Linear(dim, self.hidden_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.hidden_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.hidden_dim, bias=bias)

        # Gating
        self.g_proj = nn.Linear(dim, self.hidden_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=bias)

        # Group normalization
        if use_group_norm:
            self.group_norm = nn.GroupNorm(num_heads, self.hidden_dim)

        # Swish gate parameters
        self.swish_scale = nn.Parameter(torch.ones(1))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.g_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        use_recurrent: Optional[bool] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Multi-Scale Retention.

        Automatically selects parallel or recurrent mode based on
        training status unless explicitly specified.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            state: Optional recurrent state of shape (batch, heads, head_dim, head_dim)
            use_recurrent: Force recurrent mode if True

        Returns:
            output: Output tensor of shape (batch, seq_len, dim)
            state: Updated recurrent state
        """
        if use_recurrent is None:
            use_recurrent = not self.training

        if use_recurrent or (state is not None and x.size(1) == 1):
            return self.forward_recurrent(x, state)
        else:
            return self.forward_parallel(x)

    def forward_parallel(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """
        Parallel forward for training.

        Computes retention in parallel over the sequence using
        matrix multiplications, similar to attention.

        Args:
            x: Input of shape (batch, seq_len, dim)

        Returns:
            output: Shape (batch, seq_len, dim)
            state: None (no state in parallel mode)
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)  # (batch, seq, hidden)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        # Reshape to multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for batched matrix multiplication
        q = q.transpose(1, 2)  # (batch, heads, seq, dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build decay mask D
        D = self._build_decay_mask(seq_len, x.device, x.dtype)  # (heads, seq, seq)

        # Compute retention scores: Q @ K^T * D
        # Note: Retention uses simple dot product, no softmax
        retention = torch.einsum('bhid,bhjd->bhij', q, k)  # (batch, heads, seq, seq)
        retention = retention * D.unsqueeze(0)  # Apply decay mask

        # Apply retention to values
        output = torch.einsum('bhij,bhjd->bhid', retention, v)  # (batch, heads, seq, dim)

        # Transpose back
        output = output.transpose(1, 2)  # (batch, seq, heads, dim)
        output = output.reshape(batch_size, seq_len, self.hidden_dim)

        # Apply group normalization
        if self.use_group_norm:
            output = self.group_norm(output.transpose(1, 2)).transpose(1, 2)

        # Apply gating
        output = output * F.silu(g * self.swish_scale)

        # Project output
        output = self.out_proj(output)

        return output, None

    def forward_recurrent(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent forward for inference.

        Processes one timestep at a time with O(1) complexity,
        suitable for autoregressive generation.

        Args:
            x: Input of shape (batch, seq_len, dim)
            state: Recurrent state of shape (batch, heads, head_dim, head_dim)

        Returns:
            output: Shape (batch, seq_len, dim)
            state: Updated state
        """
        batch_size, seq_len, _ = x.shape

        # Initialize state if needed
        if state is None:
            state = torch.zeros(
                batch_size, self.num_heads, self.head_dim, self.head_dim,
                device=x.device, dtype=x.dtype
            )

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        # Reshape to multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        outputs = []
        gamma = self.gamma.view(1, self.num_heads, 1, 1)  # (1, heads, 1, 1)

        for t in range(seq_len):
            q_t = q[:, t]  # (batch, heads, dim)
            k_t = k[:, t]
            v_t = v[:, t]

            # Update state with decay: S_t = gamma * S_{t-1} + k_t @ v_t^T
            kv = torch.einsum('bhd,bhe->bhde', k_t, v_t)  # (batch, heads, dim, dim)
            state = gamma * state + kv

            # Query state: o_t = q_t @ S_t
            output_t = torch.einsum('bhd,bhde->bhe', q_t, state)
            outputs.append(output_t)

        output = torch.stack(outputs, dim=1)  # (batch, seq, heads, dim)
        output = output.reshape(batch_size, seq_len, self.hidden_dim)

        # Apply group normalization
        if self.use_group_norm:
            output = self.group_norm(output.transpose(1, 2)).transpose(1, 2)

        # Apply gating
        output = output * F.silu(g * self.swish_scale)

        # Project output
        output = self.out_proj(output)

        return output, state

    def _build_decay_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build the decay mask D for parallel computation.

        D[i,j] = gamma^(i-j) if i >= j else 0

        Args:
            seq_len: Sequence length
            device: Device for tensor
            dtype: Data type for tensor

        Returns:
            Decay mask of shape (num_heads, seq_len, seq_len)
        """
        # Create position indices
        pos = torch.arange(seq_len, device=device, dtype=dtype)

        # Compute distance matrix
        dist = pos.unsqueeze(1) - pos.unsqueeze(0)  # (seq, seq)

        # Create causal mask
        mask = dist >= 0  # (seq, seq)

        # Compute decay for each head
        gamma = self.gamma.to(dtype)  # (heads,)
        decay = gamma.view(-1, 1, 1) ** dist.unsqueeze(0)  # (heads, seq, seq)

        # Apply causal mask
        decay = decay * mask.unsqueeze(0).to(dtype)

        return decay

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


class RetNet(NexusModule):
    """
    Retentive Network block.

    Uses retention mechanism with exponential decay,
    supporting both parallel (training) and recurrent (inference) modes.

    Reference: https://arxiv.org/abs/2307.08621

    Args:
        dim: Model dimension
        num_heads: Number of retention heads
        head_dim: Dimension per head
        gamma: Decay factor (or list per head)
        expand: FFN expansion factor
        dropout: Dropout probability
        bias: Use bias in projections
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        gamma: Optional[Union[float, List[float]]] = None,
        expand: int = 4,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Multi-scale retention
        self.retention = MultiScaleRetention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            gamma=gamma,
            bias=bias
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expand, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand, dim, bias=bias),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of RetNet block.

        Args:
            x: Input of shape (batch, seq_len, dim)
            state: Optional recurrent state

        Returns:
            output: Shape (batch, seq_len, dim)
            state: Updated state
        """
        # Retention with residual
        residual = x
        x = self.norm1(x)
        x, state = self.retention(x, state)
        x = self.dropout(x)
        x = x + residual

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x, state

    def forward_parallel(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """
        Parallel forward for training.

        Args:
            x: Input of shape (batch, seq_len, dim)

        Returns:
            output: Shape (batch, seq_len, dim)
            state: None
        """
        # Retention with residual
        residual = x
        x = self.norm1(x)
        x, _ = self.retention.forward_parallel(x)
        x = self.dropout(x)
        x = x + residual

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x, None

    def forward_recurrent(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent forward for inference.

        Args:
            x: Input of shape (batch, seq_len, dim)
            state: Recurrent state

        Returns:
            output: Shape (batch, seq_len, dim)
            state: Updated state
        """
        # Retention with residual
        residual = x
        x = self.norm1(x)
        x, state = self.retention.forward_recurrent(x, state)
        x = self.dropout(x)
        x = x + residual

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x, state

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Initialize recurrent state."""
        return self.retention.init_state(batch_size, device, dtype)
