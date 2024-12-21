import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .nerf import NeRFNetwork, PositionalEncoding

class NeRFPlusPlusNetwork(NeRFNetwork):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        nerf_pp_config = config.get("nerf++", {})
        self.scene_bound = nerf_pp_config.get("scene_bound", 4.0)
        self.use_viewdirs = nerf_pp_config.get("use_viewdirs", True)
        
        # Initialize background networks
        self.background_density_net = nn.Sequential(
            nn.Linear(self.pos_encoder.get_output_dim(), self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1 + self.hidden_size)  # 1 for density + features
        )
        
        view_dependent_dim = self.pos_encoder.get_output_dim() if self.use_viewdirs else 0
        self.background_color_net = nn.Sequential(
            nn.Linear(self.hidden_size + view_dependent_dim, self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 3),
            nn.Sigmoid()
        )

    def _inverse_transform_sampling(self, positions: torch.Tensor) -> torch.Tensor:
        """Convert unbounded positions to bounded for background model."""
        norm = torch.norm(positions, dim=-1, keepdim=True)
        normalized_positions = positions / norm
        inv_positions = normalized_positions / (norm - self.scene_bound)
        return inv_positions

    def compute_background_outputs(self, positions: torch.Tensor, directions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute density and color for background (unbounded) points."""
        inv_positions = self._inverse_transform_sampling(positions)
        encoded_positions = self.pos_encoder(inv_positions)
        
        # Compute density and features
        h = self.background_density_net(encoded_positions)
        density = F.softplus(h[..., 0])
        features = h[..., 1:]
        
        # Compute colors with optional view dependence
        if self.use_viewdirs and directions is not None:
            encoded_dirs = self.pos_encoder(directions)
            color_input = torch.cat([features, encoded_dirs], dim=-1)
        else:
            color_input = features
            
        colors = self.background_color_net(color_input)
        
        return {
            "density": density.unsqueeze(-1),
            "color": colors,
            "features": features
        }

    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Detect unbounded positions
        bound_mask = torch.norm(positions, dim=-1) > self.scene_bound
        bounded_positions = positions[~bound_mask]
        unbounded_positions = positions[bound_mask]
        
        outputs = {}
        
        # Process bounded positions with existing NeRF
        if bounded_positions.size(0) > 0:
            bounded_outputs = super().forward(bounded_positions, directions[~bound_mask])
        else:
            bounded_outputs = {
                "density": torch.empty(0, 1, device=positions.device),
                "color": torch.empty(0, 3, device=positions.device)
            }

        # Process unbounded positions with background model
        if unbounded_positions.size(0) > 0:
            unbounded_outputs = self.compute_background_outputs(
                unbounded_positions, 
                directions[bound_mask] if self.use_viewdirs else None
            )
        else:
            unbounded_outputs = {
                "density": torch.empty(0, 1, device=positions.device),
                "color": torch.empty(0, 3, device=positions.device)
            }

        # Combine outputs maintaining original point order
        outputs = {
            "density": torch.zeros_like(positions[..., :1]),
            "color": torch.zeros_like(positions[..., :3])
        }
        outputs["density"][~bound_mask] = bounded_outputs["density"]
        outputs["density"][bound_mask] = unbounded_outputs["density"]
        outputs["color"][~bound_mask] = bounded_outputs["color"]
        outputs["color"][bound_mask] = unbounded_outputs["color"]

        return outputs