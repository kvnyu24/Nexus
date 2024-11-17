import torch
from typing import Dict, Tuple
from .networks import EnhancedNeRF

class NeRFRenderer:
    def __init__(
        self,
        model: EnhancedNeRF,
        near: float = 2.0,
        far: float = 6.0,
        n_samples: int = 64,
        n_importance: int = 128,
        perturb: bool = True,
        noise_std: float = 0.0
    ):
        self.model = model
        self.near = near
        self.far = far
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.perturb = perturb
        self.noise_std = noise_std
        
    def _stratified_sampling(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate sampling points along each ray
        t_vals = torch.linspace(0., 1., self.n_samples, device=rays_o.device)
        z_vals = self.near * (1. - t_vals) + self.far * t_vals
        
        if self.perturb:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand
            
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        return pts, z_vals 