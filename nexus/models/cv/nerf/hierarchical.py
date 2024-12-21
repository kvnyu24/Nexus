import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from ....core.base import NexusModule
from ....components.embeddings import PositionalEncoding
from .nerf import NeRFNetwork

class HierarchicalSampling(NexusModule):
    def __init__(self, num_samples: int = 64):
        super().__init__()
        self.num_samples = num_samples
        
    def forward(self, bins: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Sample points hierarchically based on weights from coarse network"""
        # Get PDF from weights
        weights = weights + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        
        # Draw uniform samples
        uniform_samples = torch.linspace(0., 1., self.num_samples, device=bins.device)
        uniform_samples = uniform_samples.expand(list(cdf.shape[:-1]) + [self.num_samples])
        
        # Inverse CDF sampling
        indices = torch.searchsorted(cdf, uniform_samples)
        below = torch.max(torch.zeros_like(indices-1), indices-1)
        above = torch.min(cdf.shape[-1]-1 * torch.ones_like(indices), indices)
        indices_g = torch.stack([below, above], -1)
        
        matched_shape = [indices_g.shape[0], indices_g.shape[1], 2]
        cdf_g = torch.gather(cdf, 2, indices_g)
        bins_g = torch.gather(bins, 2, indices_g)
        
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (uniform_samples - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        
        return samples

class HierarchicalNeRF(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Coarse network configuration
        coarse_config = config.copy()
        coarse_config["hidden_dim"] = config.get("coarse_hidden_dim", 128)
        self.coarse_network = NeRFNetwork(coarse_config)
        
        # Fine network configuration
        fine_config = config.copy()
        fine_config["hidden_dim"] = config.get("fine_hidden_dim", 256)
        self.fine_network = NeRFNetwork(fine_config)
        
        # Hierarchical sampling
        self.hierarchical_sampler = HierarchicalSampling(
            num_samples=config.get("fine_samples", 128)
        )
        
    def forward(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        near: float,
        far: float,
        num_coarse: int = 64,
        num_fine: int = 128,
        noise_std: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        # Coarse sampling
        coarse_outputs = self.coarse_network.render_rays(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            near=near,
            far=far,
            num_samples=num_coarse,
            noise_std=noise_std
        )
        
        # Hierarchical sampling
        fine_samples = self.hierarchical_sampler(
            coarse_outputs["z_vals"],
            coarse_outputs["weights"]
        )
        
        # Combine coarse and fine samples
        z_vals = torch.sort(torch.cat([
            coarse_outputs["z_vals"],
            fine_samples
        ], dim=-1), dim=-1)[0]
        
        # Fine network forward pass
        fine_outputs = self.fine_network.render_rays(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            near=near,
            far=far,
            z_vals=z_vals,
            noise_std=noise_std
        )
        
        return {
            "coarse": coarse_outputs,
            "fine": fine_outputs
        }