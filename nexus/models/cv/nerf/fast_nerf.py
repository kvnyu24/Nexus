import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .nerf import NeRFNetwork
from ....components.embeddings import PositionalEncoding

class FastNeRFNetwork(NeRFNetwork):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        fast_config = config.get("fastnerf", {})
        self.scene_bound = fast_config.get("scene_bound", 4.0)
        self.use_viewdirs = fast_config.get("use_viewdirs", True)
        self.fast_layers = fast_config.get("fast_layers", [2, 4, 6])

        # Fast MLP for bounded scene regions
        self.fast_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )

        # Background networks for unbounded regions
        self.background_density_net = nn.Sequential(
            nn.Linear(self.pos_encoder.get_output_dim(), self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1 + self.hidden_size)
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
        
        h = self.background_density_net(encoded_positions)
        density = F.softplus(h[..., 0])
        features = h[..., 1:]
        
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
        bound_mask = torch.norm(positions, dim=-1) > self.scene_bound
        bounded_positions = positions[~bound_mask]
        unbounded_positions = positions[bound_mask]
        
        outputs = {
            "density": torch.zeros_like(positions[..., :1]),
            "color": torch.zeros_like(positions[..., :3])
        }

        # Process bounded positions with fast MLP
        if bounded_positions.size(0) > 0:
            x = self.mlp[0](bounded_positions)
            for i, layer in enumerate(self.mlp[1:], 1):
                x = layer(x)
                if i in self.fast_layers:
                    x = self.fast_mlp(x)
            
            bounded_density = self.density_head(x)
            bounded_color = self.color_head(x)
            
            outputs["density"][~bound_mask] = bounded_density
            outputs["color"][~bound_mask] = bounded_color

        # Process unbounded positions with background model
        if unbounded_positions.size(0) > 0:
            unbounded_outputs = self.compute_background_outputs(
                unbounded_positions,
                directions[bound_mask] if self.use_viewdirs else None
            )
            outputs["density"][bound_mask] = unbounded_outputs["density"]
            outputs["color"][bound_mask] = unbounded_outputs["color"]

        return outputs