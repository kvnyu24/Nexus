import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from .nerf import NeRFNetwork
from ....components.embeddings import PositionalEncoding

class IntegratedPositionalEncoding(PositionalEncoding):
    def __init__(self, num_frequencies: int = 10, min_deg: int = 0, max_deg: int = 16):
        super().__init__(num_frequencies)
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = 2.0 ** torch.linspace(min_deg, max_deg-1, max_deg-min_deg)

    def forward(self, means: torch.Tensor, covs: torch.Tensor) -> torch.Tensor:
        # Compute integrated positional encoding for Gaussian inputs
        scales = self.scales[None, :].to(means.device)
        scaled_means = means[..., None] * scales
        scaled_covs = covs[..., None] * (scales ** 2)
        
        # Compute expected sine and cosine terms
        exp_sin = torch.exp(-0.5 * scaled_covs) * torch.sin(scaled_means)
        exp_cos = torch.exp(-0.5 * scaled_covs) * torch.cos(scaled_means)
        
        # Concatenate encoded features
        encoded = torch.cat([exp_sin, exp_cos], dim=-1)
        return encoded.reshape(means.shape[0], -1)

class MipNeRFNetwork(NeRFNetwork):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        mip_config = config.get("mipnerf", {})
        
        # MipNeRF specific parameters
        self.num_samples = mip_config.get("num_samples", 128)
        self.min_deg = mip_config.get("min_deg", 0)
        self.max_deg = mip_config.get("max_deg", 16)
        self.num_levels = mip_config.get("num_levels", 2)
        
        # Initialize integrated positional encoding
        self.pos_encoder = IntegratedPositionalEncoding(
            min_deg=self.min_deg,
            max_deg=self.max_deg
        )
        
        # Density network (sigma)
        self.density_net = nn.Sequential(
            nn.Linear(self.pos_encoder.get_output_dim(), self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1 + self.hidden_size)  # 1 for density + features
        )
        
        # Color network
        self.color_net = nn.Sequential(
            nn.Linear(self.hidden_size + self.pos_encoder.get_output_dim(), self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 3)
        )

    def compute_density(self, positions: torch.Tensor, covs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.pos_encoder(positions, covs)
        h = self.density_net(encoded)
        density = F.softplus(h[..., 0])
        features = h[..., 1:]
        return density, features

    def compute_color(self, features: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        encoded_dirs = self.pos_encoder(dirs, torch.zeros_like(dirs))
        h = torch.cat([features, encoded_dirs], dim=-1)
        rgb = torch.sigmoid(self.color_net(h))
        return rgb

    def forward(self, positions: torch.Tensor, directions: torch.Tensor, 
                position_covs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if position_covs is None:
            position_covs = torch.zeros_like(positions)
            
        density, features = self.compute_density(positions, position_covs)
        color = self.compute_color(features, directions)
        
        return {
            "density": density.unsqueeze(-1),
            "color": color,
            "features": features
        }