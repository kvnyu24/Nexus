import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from ...core.base import NexusModule

class GaussianRenderer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core parameters with validation
        self.hidden_dim = config.get("hidden_dim", 256)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
            
        self.num_samples = config.get("num_samples", 64)
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
            
        self.min_depth = config.get("min_depth", 0.1)
        self.max_depth = config.get("max_depth", 100.0)
        if self.min_depth >= self.max_depth:
            raise ValueError("min_depth must be less than max_depth")
            
        # Feature processing with layer normalization for stability
        self.feature_processor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Density MLP with skip connections
        self.density_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Softplus()
        )
        
        # Color MLP with enhanced view dependence and material properties
        self.color_net = nn.Sequential(
            nn.Linear(self.hidden_dim + 6, self.hidden_dim),  # +3 for view dir, +3 for light dir
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 4),  # RGB + Alpha
            nn.Sigmoid()
        )
        
    def forward(
        self,
        gaussians: Dict[str, torch.Tensor],
        camera_params: Dict[str, torch.Tensor],
        features: Optional[torch.Tensor] = None,
        light_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        # Input validation
        if not all(k in gaussians for k in ["means", "covariances"]):
            raise ValueError("Missing required gaussian parameters")
        if not all(k in camera_params for k in ["positions", "directions"]):
            raise ValueError("Missing required camera parameters")
            
        # Process features with error checking
        if features is not None:
            if not torch.isfinite(features).all():
                raise ValueError("Features contain NaN or inf values")
            processed_features = self.feature_processor(features)
        else:
            processed_features = None
            
        # Compute ray-gaussian intersections with numerical stability
        intersections = self._compute_intersections(
            gaussians["means"],
            gaussians["covariances"],
            camera_params,
            eps=1e-6
        )
        
        # Compute densities and colors with light interaction
        densities = self.density_net(intersections["features"])
        
        # Prepare color inputs
        color_inputs = [intersections["features"], camera_params["directions"]]
        if light_params is not None:
            color_inputs.append(light_params["directions"])
        else:
            # Default light from camera
            color_inputs.append(camera_params["directions"])
            
        colors_alpha = self.color_net(torch.cat(color_inputs, dim=-1))
        colors, alpha = colors_alpha[..., :3], colors_alpha[..., 3:]
        
        # Alpha compositing with depth-based sorting
        weights = self._compute_weights(
            densities * alpha,
            intersections["depths"],
            eps=1e-8
        )
        
        # Final rendering with accumulated opacity
        rendered_color = torch.sum(weights[..., None] * colors, dim=1)
        rendered_depth = torch.sum(weights * intersections["depths"], dim=1)
        opacity = torch.sum(weights, dim=1)
        
        return {
            "color": rendered_color,
            "depth": rendered_depth,
            "weights": weights,
            "opacity": opacity,
            "intersections": intersections,
            "raw_densities": densities,
            "raw_colors": colors
        }
        
    def _compute_intersections(
        self,
        means: torch.Tensor,
        covariances: torch.Tensor,
        camera_params: Dict[str, torch.Tensor],
        eps: float = 1e-6
    ) -> Dict[str, torch.Tensor]:
        batch_size = camera_params["positions"].shape[0]
        num_gaussians = means.shape[0]

        # Project gaussians to camera space with bounds checking
        cam_positions = camera_params["positions"].unsqueeze(1)  # [B, 1, 3]
        cam_directions = nn.functional.normalize(
            camera_params["directions"].unsqueeze(1),
            dim=-1,
            eps=eps
        )  # [B, 1, 3]
        
        # Compute ray-gaussian intersections with broadcasting
        means_expanded = means.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, 3]
        
        # Get vectors from camera to gaussians with numerical stability
        to_gaussian = means_expanded - cam_positions  # [B, N, 3]
        dist_to_gaussian = torch.norm(to_gaussian, dim=-1, keepdim=True).clamp(min=eps)
        to_gaussian_normalized = to_gaussian / dist_to_gaussian
        
        # Project vectors onto ray directions with robust dot product
        proj_length = torch.sum(to_gaussian * cam_directions, dim=-1)  # [B, N]
        
        # Compute closest points on rays with bounds
        proj_length_clamped = proj_length.clamp(min=self.min_depth, max=self.max_depth)
        closest_points = (
            cam_positions + 
            cam_directions * proj_length_clamped.unsqueeze(-1)
        )  # [B, N, 3]
        
        # Compute intersection depths with safe norm
        depths = torch.norm(closest_points - cam_positions, dim=-1).clamp(min=eps)  # [B, N]
        
        # Compute rich features at intersection points
        intersection_features = torch.cat([
            closest_points - means_expanded,  # Offset from gaussian center
            torch.diagonal(covariances, dim1=-2, dim2=-1).unsqueeze(0).expand(batch_size, -1, -1),  # Gaussian properties
            to_gaussian_normalized,  # View direction relative to gaussian
            depths.unsqueeze(-1)  # Depth information
        ], dim=-1)
        
        return {
            "points": closest_points,
            "depths": depths,
            "features": intersection_features,
            "proj_length": proj_length,
            "dist_to_gaussian": dist_to_gaussian.squeeze(-1)
        }