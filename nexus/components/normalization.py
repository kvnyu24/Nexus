import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from nexus.core.base import NexusModule


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