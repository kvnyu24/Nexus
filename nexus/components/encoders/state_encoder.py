import torch
import torch.nn as nn
from typing import Optional
from nexus.core.base import NexusModule

class StateEncoder(NexusModule):
    """Reusable state encoder for encoding input states into hidden representations.

    Consolidates the common pattern of:
    Linear -> LayerNorm -> Activation -> Dropout -> (optional second layer)

    Used across agent models, quant models, and reasoning models.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (defaults to hidden_dim if not specified)
        num_layers: Number of encoding layers (1 or 2)
        dropout: Dropout probability
        activation: Activation function ('relu', 'gelu', 'silu')
        use_layernorm: Whether to use LayerNorm
        layernorm_first: If True, applies LayerNorm before activation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_layernorm: bool = True,
        layernorm_first: bool = True
    ):
        super().__init__()
        output_dim = output_dim or hidden_dim

        # Build activation
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
        }
        act_fn = activations.get(activation, nn.ReLU())

        # Build layers
        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_layernorm and layernorm_first:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_fn)
        if not layernorm_first and use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Second layer (if num_layers == 2)
        if num_layers >= 2:
            layers.append(nn.Linear(hidden_dim, output_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
