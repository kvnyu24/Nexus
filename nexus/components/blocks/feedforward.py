import torch
import torch.nn as nn
from typing import Optional, Callable
from nexus.core.base import NexusModule

class FeedForward(NexusModule):
    """Standard feed-forward network used in transformers and other architectures.

    Implements: Linear -> Activation -> Dropout -> Linear -> Dropout

    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden layer dimension (defaults to 4 * hidden_size)
        dropout: Dropout probability
        activation: Activation function ('gelu', 'relu', 'silu')
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        bias: bool = True
    ):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size * 4

        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
        }

        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.act = activations.get(activation, nn.GELU())
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MLPBlock(NexusModule):
    """Configurable MLP block with optional normalization.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (defaults to hidden_dim)
        num_layers: Number of hidden layers
        dropout: Dropout probability
        activation: Activation function
        use_layernorm: Whether to apply LayerNorm after each layer
        use_batchnorm: Whether to apply BatchNorm1d after each layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_layernorm: bool = False,
        use_batchnorm: bool = False
    ):
        super().__init__()
        output_dim = output_dim or hidden_dim

        activations = {
            'gelu': nn.GELU,
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'tanh': nn.Tanh,
        }
        act_cls = activations.get(activation, nn.ReLU)

        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        if num_layers > 1:
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueHead(NexusModule):
    """Value head for RL models, supports ensemble of heads.

    Args:
        hidden_dim: Input dimension
        num_heads: Number of ensemble heads
        use_layernorm: Whether to use LayerNorm
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        use_layernorm: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_heads == 1:
            return self.heads[0](x)
        return torch.stack([head(x) for head in self.heads], dim=-1)
