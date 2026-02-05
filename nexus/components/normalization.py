"""
Normalization layers for modern deep learning architectures.

Includes standard and advanced normalization techniques used across
transformer, vision, and hybrid models.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from nexus.core.base import NexusModule

__all__ = [
    'RMSNorm',
    'QKNorm',
    'LayerNorm2d',
    'GroupNorm2d',
    'DeepNorm',
    'DynamicTanh',
    'HybridNorm',
]


class RMSNorm(NexusModule):
    """Root Mean Square Layer Normalization.

    A simplified normalization layer that normalizes by the root mean square,
    without subtracting the mean. Used by Llama, Mistral, Qwen, Gemma, DeepSeek.

    Reference: https://arxiv.org/abs/1910.07467

    Args:
        dim: The dimension to normalize over
        eps: Small constant for numerical stability
        elementwise_affine: Whether to include learnable scale parameter
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'{self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class QKNorm(NexusModule):
    """Query-Key Normalization for attention.

    Normalizes Q and K vectors before computing attention scores.
    Used by Gemma 2 to stabilize training with large head dimensions.

    Args:
        head_dim: Dimension of each attention head
        eps: Small constant for numerical stability
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.head_dim = head_dim
        self.eps = eps
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim)

        Returns:
            Normalized (q, k) tensors
        """
        return self.q_norm(q), self.k_norm(k)


class LayerNorm2d(NexusModule):
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

class GroupNorm2d(NexusModule):
    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.group_norm(
            x,
            self.num_groups,
            self.weight,
            self.bias,
            self.eps
        )


class DeepNorm(NexusModule):
    """DeepNorm: scaled residual connection with post-layer normalization.

    Enables stable training of very deep transformers (1000+ layers) by
    scaling the residual branch and applying layer normalization *after*
    the residual addition.  The key insight is that the residual scaling
    factor ``alpha`` and weight initialisation factor ``beta`` depend on
    the total number of encoder/decoder layers.

    Formulation::

        output = LayerNorm(x * alpha + sublayer(x))

    Weight initialisation (should be done externally):
        - V, output projections in attention: scale by ``beta``
        - FFN output projection: scale by ``beta``

    Where for a decoder with *N* layers::

        alpha = (2 * N) ** 0.25
        beta  = (8 * N) ** (-0.25)

    Reference:
        DeepNet: Scaling Transformers to 1,000 Layers
        https://arxiv.org/abs/2203.00555

    Args:
        dim: Feature dimension for the layer normalization.
        num_layers: Total number of transformer layers (used to compute
            ``alpha``).
        alpha: Explicit residual scaling factor.  If ``None`` it is
            derived from ``num_layers``.
        eps: Epsilon for layer normalization.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 1,
        alpha: Optional[float] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        if alpha is not None:
            self.alpha = alpha
        else:
            # Decoder formula from the DeepNet paper
            self.alpha = (2.0 * num_layers) ** 0.25

        self.beta = (8.0 * num_layers) ** (-0.25)

        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        sublayer_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply scaled residual + post-layer-norm.

        Args:
            x: Residual / shortcut tensor ``(..., dim)``.
            sublayer_output: Output of the sub-layer (attention or FFN)
                ``(..., dim)``.

        Returns:
            Normalised output ``(..., dim)``.
        """
        return self.norm(x * self.alpha + sublayer_output)

    def get_init_scale(self) -> float:
        """Return the weight initialisation scale ``beta``.

        Sub-layer output projections should be multiplied by this
        factor at initialisation time::

            nn.init.xavier_normal_(layer.weight)
            layer.weight.data.mul_(deep_norm.get_init_scale())
        """
        return self.beta

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, num_layers={self.num_layers}, '
            f'alpha={self.alpha:.4f}, beta={self.beta:.4f}'
        )


class DynamicTanh(NexusModule):
    """DynamicTanh (DyT): normalization-free activation alternative.

    Replaces layer normalization with a learnable element-wise tanh
    transform::

        DyT(x) = gamma * tanh(alpha * x) + beta

    where ``alpha`` is a learnable scalar controlling the saturation
    of the tanh, and ``gamma`` / ``beta`` are learnable affine
    parameters.  This provides implicit normalisation without computing
    statistics over the sequence or batch dimensions, making it fully
    compatible with sequence parallelism and long-context training.

    Reference:
        Transformers without Normalization
        https://arxiv.org/abs/2503.10622

    Args:
        dim: Feature dimension.
        alpha_init: Initial value for the learnable ``alpha`` scalar.
            The paper suggests values around 0.5-1.0.
        elementwise_affine: Whether to include learnable ``gamma`` and
            ``beta`` affine parameters.
    """

    def __init__(
        self,
        dim: int,
        alpha_init: float = 0.5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.dim = dim

        # Learnable scaling factor inside tanh
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(dim))
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DynamicTanh transform.

        Args:
            x: Input tensor ``(..., dim)``.

        Returns:
            Transformed tensor ``(..., dim)``.
        """
        out = torch.tanh(self.alpha * x)
        if self.elementwise_affine:
            out = self.gamma * out + self.beta
        return out

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, alpha_init={self.alpha.item():.4f}, '
            f'elementwise_affine={self.elementwise_affine}'
        )


class HybridNorm(NexusModule):
    """HybridNorm: Pre-Norm for attention, Post-Norm for FFN.

    Combines the benefits of Pre-LayerNorm (stable training) and
    Post-LayerNorm (better representation capacity) by using Pre-Norm
    for the attention sub-layer and Post-Norm for the FFN sub-layer.

    This hybrid approach has been shown to improve both training stability
    and final model performance compared to using either strategy alone.

    The forward pass implements:
        x = x + Attention(PreNorm(x))
        x = PostNorm(x + FFN(x))

    Reference:
        Understanding the Difficulty of Training Transformers
        https://arxiv.org/abs/2004.08249

    Args:
        dim: Feature dimension.
        norm_type: Type of normalization ('layer' or 'rms').
        eps: Epsilon for numerical stability.
        dropout: Dropout probability for residual connections.
    """

    def __init__(
        self,
        dim: int,
        norm_type: str = 'layer',
        eps: float = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim

        # Pre-norm for attention (applied before sublayer)
        if norm_type == 'layer':
            self.attn_norm = nn.LayerNorm(dim, eps=eps)
            self.ffn_norm = nn.LayerNorm(dim, eps=eps)
        elif norm_type == 'rms':
            self.attn_norm = RMSNorm(dim, eps=eps)
            self.ffn_norm = RMSNorm(dim, eps=eps)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward_attn(
        self,
        x: torch.Tensor,
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Pre-Norm for attention sublayer.

        Args:
            x: Input tensor (residual).
            attn_output: Output from attention sublayer (already computed
                on normalized input).

        Returns:
            Output with residual connection.
        """
        if self.dropout is not None:
            attn_output = self.dropout(attn_output)
        return x + attn_output

    def forward_ffn(
        self,
        x: torch.Tensor,
        ffn_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Post-Norm for FFN sublayer.

        Args:
            x: Input tensor (residual).
            ffn_output: Output from FFN sublayer (computed on unnormalized input).

        Returns:
            Normalized output with residual connection.
        """
        if self.dropout is not None:
            ffn_output = self.dropout(ffn_output)
        return self.ffn_norm(x + ffn_output)

    def get_attn_norm_input(self, x: torch.Tensor) -> torch.Tensor:
        """Get normalized input for attention sublayer.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor to pass to attention.
        """
        return self.attn_norm(x)

    def get_ffn_input(self, x: torch.Tensor) -> torch.Tensor:
        """Get input for FFN sublayer (no normalization).

        Args:
            x: Input tensor.

        Returns:
            Unnormalized tensor to pass to FFN (Post-Norm doesn't
            normalize before sublayer).
        """
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}'