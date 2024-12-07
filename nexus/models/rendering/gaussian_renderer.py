import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from ...core.base import NexusModule

class GaussianRenderer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core parameters
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_samples = config.get("num_samples", 64)
        self.min_depth = config.get("min_depth", 0.1)
        self.max_depth = config.get("max_depth", 100.0)
        
        # Feature processing (following TextureMapper pattern)
        self.feature_processor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Density MLP (similar to NeRF pattern)
        self.density_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Softplus()
        )
        
        # Color MLP with view dependence
        self.color_net = nn.Sequential(
            nn.Linear(self.hidden_dim + 3, self.hidden_dim),  # +3 for view direction
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        gaussians: Dict[str, torch.Tensor],
        camera_params: Dict[str, torch.Tensor],
        features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Process features
        if features is not None:
            processed_features = self.feature_processor(features)
        else:
            processed_features = None
            
        # Compute ray-gaussian intersections
        intersections = self._compute_intersections(
            gaussians["means"],
            gaussians["covariances"],
            camera_params
        )
        
        # Compute densities and colors
        densities = self.density_net(intersections["features"])
        colors = self.color_net(
            torch.cat([
                intersections["features"],
                camera_params["directions"]
            ], dim=-1)
        )
        
        # Alpha compositing
        weights = self._compute_weights(
            densities,
            intersections["depths"]
        )
        
        # Final rendering
        rendered_color = torch.sum(weights[..., None] * colors, dim=1)
        rendered_depth = torch.sum(weights * intersections["depths"], dim=1)
        
        return {
            "color": rendered_color,
            "depth": rendered_depth,
            "weights": weights,
            "intersections": intersections
        }
        
    def _compute_intersections(
        self,
        means: torch.Tensor,
        covariances: torch.Tensor,
        camera_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        batch_size = camera_params["positions"].shape[0]
        num_gaussians = means.shape[0]

        # Project gaussians to camera space
        cam_positions = camera_params["positions"].unsqueeze(1)  # [B, 1, 3]
        cam_directions = camera_params["directions"].unsqueeze(1)  # [B, 1, 3]
        
        # Compute ray-gaussian intersections
        means_expanded = means.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, 3]
        
        # Get vectors from camera to gaussians
        to_gaussian = means_expanded - cam_positions  # [B, N, 3]
        
        # Project vectors onto ray directions
        proj_length = torch.sum(to_gaussian * cam_directions, dim=-1)  # [B, N]
        
        # Compute closest points on rays
        closest_points = (
            cam_positions + 
            cam_directions * proj_length.unsqueeze(-1)
        )  # [B, N, 3]
        
        # Compute intersection depths
        depths = torch.norm(closest_points - cam_positions, dim=-1)  # [B, N]
        
        # Compute features at intersection points
        intersection_features = torch.cat([
            closest_points - means_expanded,  # Offset from gaussian center
            torch.diagonal(covariances, dim1=-2, dim2=-1).unsqueeze(0).expand(batch_size, -1, -1)  # Gaussian properties
        ], dim=-1)
        
        return {
            "points": closest_points,
            "depths": depths,
            "features": intersection_features,
            "proj_length": proj_length
        }