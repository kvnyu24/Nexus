"""
Gated Linear Unit (GLU) activation variants for modern LLMs.

These activations are used in the feed-forward layers of transformer models
and have become standard in recent architectures like Llama, Mistral, and PaLM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from nexus.core.base import NexusModule


class GLUVariant(NexusModule):
    """Base class for Gated Linear Unit variants.

    GLU variants use the formula: GLU(x) = activation(xW) ⊗ (xV)
    where ⊗ is element-wise multiplication and activation varies by variant.

    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (if None, uses 4 * dim * 2/3 rounded to multiple of 256)
        activation: Activation function to use for the gate
        bias: Whether to use bias in linear layers
        multiple_of: Round hidden_dim to multiple of this value (for hardware efficiency)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        activation: Callable = F.silu,
        bias: bool = False,
        multiple_of: int = 256
    ):
        super().__init__()
        self.dim = dim
        self.activation = activation

        # Compute hidden dimension (following Llama convention)
        if hidden_dim is None:
            hidden_dim = int(dim * 4 * 2 / 3)
            # Round to multiple for hardware efficiency
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.hidden_dim = hidden_dim

        # Gate projection (produces activation input)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        # Up projection (produces values to be gated)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        # Down projection (projects back to model dimension)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Output tensor of shape (..., dim)
        """
        return self.w_down(self.activation(self.w_gate(x)) * self.w_up(x))

    def extra_repr(self) -> str:
        return f'dim={self.dim}, hidden_dim={self.hidden_dim}'


class SwiGLU(GLUVariant):
    """Swish-Gated Linear Unit.

    SwiGLU(x) = Swish(xW) ⊗ (xV) where Swish(x) = x * sigmoid(x) = SiLU(x)

    Used by: Llama, Llama 2, Llama 3, Mistral, Qwen, PaLM

    Reference: https://arxiv.org/abs/2002.05202

    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (if None, auto-computed)
        bias: Whether to use bias in linear layers
        multiple_of: Round hidden_dim to multiple of this value
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        bias: bool = False,
        multiple_of: int = 256
    ):
        super().__init__(
            dim=dim,
            hidden_dim=hidden_dim,
            activation=F.silu,
            bias=bias,
            multiple_of=multiple_of
        )


class GeGLU(GLUVariant):
    """GELU-Gated Linear Unit.

    GeGLU(x) = GELU(xW) ⊗ (xV)

    Used by: GPT-J, Falcon

    Reference: https://arxiv.org/abs/2002.05202

    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (if None, auto-computed)
        bias: Whether to use bias in linear layers
        multiple_of: Round hidden_dim to multiple of this value
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        bias: bool = False,
        multiple_of: int = 256
    ):
        super().__init__(
            dim=dim,
            hidden_dim=hidden_dim,
            activation=F.gelu,
            bias=bias,
            multiple_of=multiple_of
        )


class ReGLU(GLUVariant):
    """ReLU-Gated Linear Unit.

    ReGLU(x) = ReLU(xW) ⊗ (xV)

    Reference: https://arxiv.org/abs/2002.05202

    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (if None, auto-computed)
        bias: Whether to use bias in linear layers
        multiple_of: Round hidden_dim to multiple of this value
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        bias: bool = False,
        multiple_of: int = 256
    ):
        super().__init__(
            dim=dim,
            hidden_dim=hidden_dim,
            activation=F.relu,
            bias=bias,
            multiple_of=multiple_of
        )


class GEGLU(GeGLU):
    """Alias for GeGLU for backwards compatibility."""
    pass


class SwiGLUFFN(NexusModule):
    """SwiGLU Feed-Forward Network with optional normalization.

    A complete FFN block with SwiGLU activation and optional layer normalization.

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension (if None, auto-computed)
        bias: Whether to use bias in linear layers
        dropout: Dropout probability
        norm_type: Type of normalization ('layer', 'rms', None)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
        norm_type: Optional[str] = None
    ):
        super().__init__()
        self.swiglu = SwiGLU(dim=dim, hidden_dim=hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if norm_type == 'layer':
            self.norm = nn.LayerNorm(dim)
        elif norm_type == 'rms':
            from nexus.components.normalization import RMSNorm
            self.norm = RMSNorm(dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.norm is not None:
            x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        return x + residual
