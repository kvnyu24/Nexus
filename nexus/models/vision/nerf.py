import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from ...core.base import NexusModule
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 10):
        super().__init__()
        self.num_frequencies = num_frequencies
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create frequencies for encoding
        frequencies = 2.0 ** torch.arange(self.num_frequencies, device=x.device)
        
        # Apply sin and cos to each frequency
        angles = x[..., None] * frequencies
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        
        return encoding.flatten(start_dim=-2)

class NeRFNetwork(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Configuration
        self.pos_encoding_dims = config.get("pos_encoding_dims", 10)
        self.dir_encoding_dims = config.get("dir_encoding_dims", 4)
        self.hidden_dim = config.get("hidden_dim", 256)
        
        # Positional encodings
        self.position_encoder = PositionalEncoding(self.pos_encoding_dims)
        self.direction_encoder = PositionalEncoding(self.dir_encoding_dims)
        
        # Calculate input dimensions after positional encoding
        pos_channels = 3 * 2 * self.pos_encoding_dims  # xyz * (sin,cos) * frequencies
        dir_channels = 3 * 2 * self.dir_encoding_dims  # xyz * (sin,cos) * frequencies
        
        # Main MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(pos_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        
        # Density prediction
        self.density_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.ReLU()
        )
        
        # Color prediction
        self.color_head = nn.Sequential(
            nn.Linear(self.hidden_dim + dir_channels, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3),
            nn.Sigmoid()
        )
        
    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode inputs
        pos_encoded = self.position_encoder(positions)
        dir_encoded = self.direction_encoder(directions)
        
        # Process through main network
        features = self.mlp(pos_encoded)
        
        # Predict density
        density = self.density_head(features)
        
        # Concatenate features with direction encoding for color prediction
        color_input = torch.cat([features, dir_encoded], dim=-1)
        color = self.color_head(color_input)
        
        return {
            "density": density,
            "color": color
        }
        
    def render_rays(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        near: float,
        far: float,
        num_samples: int = 64,
        noise_std: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        # Generate sample points along rays
        t_vals = torch.linspace(near, far, num_samples, device=ray_origins.device)
        z_vals = t_vals[None, :].expand(ray_origins.shape[0], -1)
        
        if noise_std > 0:
            z_vals = z_vals + torch.randn_like(z_vals) * noise_std
            
        # Get sample positions
        sample_points = ray_origins[..., None, :] + ray_directions[..., None, :] * z_vals[..., :, None]
        
        # Evaluate network at sample points
        directions = ray_directions[:, None].expand(-1, num_samples, -1)
        sample_points_flat = sample_points.reshape(-1, 3)
        directions_flat = directions.reshape(-1, 3)
        
        outputs = self(sample_points_flat, directions_flat)
        density = outputs["density"].reshape(-1, num_samples, 1)
        color = outputs["color"].reshape(-1, num_samples, 3)
        
        # Compute weights for volume rendering
        delta_z = z_vals[..., 1:] - z_vals[..., :-1]
        delta_z = torch.cat([delta_z, torch.tensor([1e10], device=delta_z.device).expand(delta_z.shape[0], 1)], dim=-1)
        alpha = 1 - torch.exp(-density.squeeze(-1) * delta_z)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1 - alpha + 1e-10], dim=-1),
            dim=-1
        )[:, :-1]
        
        # Compute final color and depth
        rgb = (weights[..., None] * color).sum(dim=1)
        depth = (weights * z_vals).sum(dim=1)
        
        return {
            "rgb": rgb,
            "depth": depth,
            "weights": weights,
            "z_vals": z_vals
        } 