import torch
import torch.nn as nn
from typing import Dict, Tuple
from ....core.base import NexusModule

class BaseEncoder(NexusModule):
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class MLPEncoder(BaseEncoder):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.log_var(h)

class ConvEncoder(BaseEncoder):
    def __init__(self, in_channels: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        
        modules = []
        for h_dim in hidden_dims:
            modules.extend([
                nn.Conv2d(in_channels, h_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(inplace=True)
            ])
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*modules)
        self.flatten = nn.Flatten()
        
        # Calculate flattened dimension
        self.flatten_dim = hidden_dims[-1] * 4  # Assumes input is 32x32 or similar
        
        self.mu = nn.Linear(self.flatten_dim, latent_dim)
        self.log_var = nn.Linear(self.flatten_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = self.flatten(h)
        return self.mu(h), self.log_var(h) 