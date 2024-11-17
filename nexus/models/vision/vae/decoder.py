import torch
import torch.nn as nn
from typing import List

class BaseDecoder(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class MLPDecoder(BaseDecoder):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class ConvDecoder(BaseDecoder):
    def __init__(self, latent_dim: int, hidden_dims: List[int], out_channels: int):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        
        # Initial projection
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1] * 4),
            nn.ReLU(inplace=True)
        )
        
        modules = []
        for i in range(len(hidden_dims) - 1, 0, -1):
            modules.extend([
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], 3, 
                                 stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i-1]),
                nn.ReLU(inplace=True)
            ])
            
        modules.append(
            nn.ConvTranspose2d(hidden_dims[0], out_channels, 3, 
                              stride=2, padding=1, output_padding=1)
        )
        modules.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z)
        x = x.view(x.size(0), self.hidden_dims[-1], 2, 2)
        return self.decoder(x) 