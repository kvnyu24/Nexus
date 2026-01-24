import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from ....core.base import NexusModule
from ....core.initialization import WeightInitMixin
from ....components.embeddings import PositionalEncoding
import numpy as np

class DensityNetwork(NexusModule):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [1]
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU() if i < len(dims) - 2 else nn.Softplus()
            ])
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ColorNetwork(NexusModule):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) 
              for _ in range(num_layers - 2)],
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class EnhancedNeRF(WeightInitMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Encoding configurations
        self.pos_frequencies = config.get("pos_frequencies", 10)
        self.dir_frequencies = config.get("dir_frequencies", 4)

        # Network configurations
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_layers = config.get("num_layers", 8)
        self.skip_connections = config.get("skip_connections", [4])

        # Positional encodings
        self.pos_encoder = PositionalEncoding(self.pos_frequencies)
        self.dir_encoder = PositionalEncoding(self.dir_frequencies)

        # Calculate encoded dimensions
        pos_channels = 3 * (2 * self.pos_frequencies + 1)
        dir_channels = 3 * (2 * self.dir_frequencies + 1)

        # Networks
        self.density_net = DensityNetwork(
            input_dim=pos_channels,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )

        self.color_net = ColorNetwork(
            input_dim=pos_channels + dir_channels + self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            num_layers=3
        )

        # Initialize weights
        self.init_weights_vision()