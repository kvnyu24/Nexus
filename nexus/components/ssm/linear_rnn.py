"""
Linear RNN - Base class for linear recurrent architectures.

This module provides the foundational components for linear RNN variants
like RWKV, Mamba, DeltaNet, and RetNet. Linear RNNs achieve linear
complexity in sequence length while maintaining strong modeling capabilities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from nexus.core.base import NexusModule


class LinearRNN(NexusModule):
    """
    General Linear RNN block.

    Base class for linear recurrent architectures like
    RWKV, Mamba, DeltaNet, RetNet. Provides common infrastructure
    for input/output projections and state management.

    Linear RNNs avoid the O(n^2) complexity of attention by using
    recurrent formulations that can be computed in O(n) time while
    still capturing long-range dependencies.

    Args:
        dim: Model dimension
        expand: Hidden expansion factor (default: 2)
        bias: Use bias in linear projections (default: True)
        use_short_conv: Whether to use short convolution for local context (default: True)
        conv_size: Convolution kernel size (default: 4)
    """

    def __init__(
        self,
        dim: int,
        expand: int = 2,
        bias: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4
    ):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.hidden_dim = dim * expand
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size

        # Input projection
        self.in_proj = nn.Linear(dim, self.hidden_dim * 2, bias=bias)

        # Optional short convolution for local context
        if use_short_conv:
            self.conv = nn.Conv1d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=conv_size,
                padding=conv_size - 1,
                groups=self.hidden_dim,
                bias=bias
            )

        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=bias)

        # Layer normalization
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of linear RNN.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            state: Optional recurrent state

        Returns:
            output: Output tensor of shape (batch, seq_len, dim)
            state: Updated recurrent state
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Apply short convolution if enabled
        if self.use_short_conv:
            x_branch = x_branch.transpose(1, 2)  # (batch, hidden, seq)
            x_branch = self.conv(x_branch)[:, :, :seq_len]  # Causal
            x_branch = x_branch.transpose(1, 2)  # (batch, seq, hidden)

        # Apply activation
        x_branch = F.silu(x_branch)

        # Recurrent computation (to be overridden by subclasses)
        y, state = self.recurrent_forward(x_branch, state)

        # Normalize and gate
        y = self.norm(y)
        y = y * F.silu(z)

        # Project output
        output = self.out_proj(y)

        return output, state

    def recurrent_forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Recurrent computation. Override in subclasses.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            state: Optional recurrent state

        Returns:
            output: Output tensor of shape (batch, seq_len, hidden_dim)
            state: Updated recurrent state
        """
        # Default: identity (no recurrence)
        return x, state

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Initialize recurrent state. Override in subclasses.

        Args:
            batch_size: Batch size
            device: Device for state tensor
            dtype: Data type for state tensor

        Returns:
            Initial state tensor
        """
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)


class ShortConvolution(NexusModule):
    """
    Short depthwise convolution for local context.

    Used by many linear RNN architectures to capture local patterns
    before applying the recurrent operation.

    Args:
        dim: Input/output dimension
        kernel_size: Convolution kernel size
        bias: Use bias
        causal: Whether to use causal convolution
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 4,
        bias: bool = True,
        causal: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.causal = causal

        # Depthwise convolution
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1 if causal else kernel_size // 2,
            groups=dim,
            bias=bias
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, seq_len, dim)
            state: Optional conv state for incremental decoding

        Returns:
            output: Convolved output
            state: Updated conv state
        """
        batch_size, seq_len, dim = x.shape

        # Handle incremental decoding
        if state is not None and seq_len == 1:
            # Concatenate with cached state
            x_cache = torch.cat([state, x], dim=1)
            x_cache = x_cache.transpose(1, 2)
            y = self.conv(x_cache)[:, :, -1:]
            y = y.transpose(1, 2)

            # Update state (keep last kernel_size-1 positions)
            new_state = x_cache.transpose(1, 2)[:, -(self.kernel_size - 1):, :]
            return y, new_state

        # Standard forward
        x = x.transpose(1, 2)  # (batch, dim, seq)
        y = self.conv(x)

        if self.causal:
            y = y[:, :, :seq_len]

        y = y.transpose(1, 2)  # (batch, seq, dim)

        # Cache state for incremental decoding
        if seq_len >= self.kernel_size - 1:
            new_state = x.transpose(1, 2)[:, -(self.kernel_size - 1):, :]
        else:
            new_state = x.transpose(1, 2)

        return y, new_state
