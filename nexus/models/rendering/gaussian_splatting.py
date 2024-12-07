import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from ...core.base import NexusModule

class GaussianSplatting(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config
        self._validate_config(config)
        
        # Core dimensions
        self.hidden_dim = config["hidden_dim"]
        self.num_gaussians = config.get("num_gaussians", 10000)
        
        # Gaussian parameters
        self.register_parameter(
            "means",
            nn.Parameter(torch.randn(self.num_gaussians, 3))  # 3D positions
        )
        self.register_parameter(
            "scales",
            nn.Parameter(torch.ones(self.num_gaussians, 3))   # 3D scales
        )
        self.register_parameter(
            "rotations",
            nn.Parameter(torch.zeros(self.num_gaussians, 4))  # Quaternions
        )
        self.register_parameter(
            "opacities",
            nn.Parameter(torch.ones(self.num_gaussians, 1))
        )
        
        # Color features
        self.register_parameter(
            "colors",
            nn.Parameter(torch.ones(self.num_gaussians, 3))  # RGB
        )
        
        # Feature bank for view-dependent effects
        self.register_buffer(
            "feature_bank",
            torch.zeros(config.get("bank_size", 10000), self.hidden_dim)
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
        # Neural renderer
        self.renderer = nn.Sequential(
            nn.Linear(self.hidden_dim + 6, self.hidden_dim),  # +6 for view direction
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3)  # RGB output
        )
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = ["hidden_dim"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
    def update_feature_bank(self, features: torch.Tensor):
        """Update feature bank following EnhancedReID pattern"""
        batch_size = features.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.feature_bank.size(0):
            ptr = 0
            
        self.feature_bank[ptr:ptr + batch_size] = features.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.feature_bank.size(0)
        
    def _compute_covariance(self) -> torch.Tensor:
        """Compute covariance matrices from scales and rotations"""
        # Convert quaternions to rotation matrices
        qw, qx, qy, qz = self.rotations.unbind(-1)
        rot_matrices = torch.stack([
            1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
            2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw,
            2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2
        ], dim=-1).view(-1, 3, 3)
        
        # Scale matrices
        scale_matrices = torch.diag_embed(self.scales)
        
        # Compute covariance: R * S * S * R^T
        return rot_matrices @ scale_matrices @ scale_matrices @ rot_matrices.transpose(-1, -2)
        
    def forward(
        self,
        camera_positions: torch.Tensor,
        camera_directions: torch.Tensor,
        image_size: Tuple[int, int],
        near: float = 0.1,
        far: float = 100.0
    ) -> Dict[str, torch.Tensor]:
        batch_size = camera_positions.shape[0]
        H, W = image_size
        
        # Compute ray-gaussian intersections
        covariances = self._compute_covariance()
        
        # Project gaussians to image space
        projected_means = self.means.unsqueeze(0).expand(batch_size, -1, -1)
        view_directions = camera_directions.unsqueeze(1).expand(-1, self.num_gaussians, -1)
        
        # Compute view-dependent features
        view_features = torch.cat([
            projected_means - camera_positions.unsqueeze(1),
            view_directions
        ], dim=-1)
        
        # Render colors with neural renderer
        rendered_colors = self.renderer(view_features)
        
        # Apply opacity
        final_colors = rendered_colors * self.opacities
        
        # Update feature bank
        self.update_feature_bank(view_features.mean(dim=1))
        
        return {
            "colors": final_colors,
            "depths": projected_means[..., 2],
            "opacities": self.opacities,
            "covariances": covariances,
            "view_features": view_features
        } 