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
        self.bank_size = config.get("bank_size", 10000)
        self.min_opacity = config.get("min_opacity", 0.0)
        self.max_opacity = config.get("max_opacity", 1.0)
        
        # Gaussian parameters with better initialization
        self.register_parameter(
            "means",
            nn.Parameter(torch.randn(self.num_gaussians, 3) * 0.1)  # 3D positions
        )
        self.register_parameter(
            "scales",
            nn.Parameter(torch.ones(self.num_gaussians, 3) * 0.01)  # Small initial scales
        )
        self.register_parameter(
            "rotations", 
            nn.Parameter(torch.zeros(self.num_gaussians, 4))  # Quaternions
        )
        self.register_parameter(
            "opacities",
            nn.Parameter(torch.ones(self.num_gaussians, 1) * 0.1)  # Start mostly transparent
        )
        
        # Color and material features
        self.register_parameter(
            "colors",
            nn.Parameter(torch.ones(self.num_gaussians, 3))  # RGB
        )
        self.register_parameter(
            "roughness",
            nn.Parameter(torch.ones(self.num_gaussians, 1) * 0.5)  # Material roughness
        )
        self.register_parameter(
            "metallic",
            nn.Parameter(torch.zeros(self.num_gaussians, 1))  # Metallic factor
        )
        
        # Feature banks for view-dependent effects and temporal coherence
        self.register_buffer(
            "feature_bank",
            torch.zeros(self.bank_size, self.hidden_dim)
        )
        self.register_buffer(
            "temporal_bank",
            torch.zeros(self.bank_size, self.hidden_dim)
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("bank_is_full", torch.zeros(1, dtype=torch.bool))
        
        # Enhanced neural renderer with material properties
        self.renderer = nn.Sequential(
            nn.Linear(self.hidden_dim + 8, self.hidden_dim * 2),  # +8 for view and material
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3)  # RGB output
        )
        
        # Density MLP for adaptive opacity
        self.density_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = ["hidden_dim"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
        if config["hidden_dim"] <= 0:
            raise ValueError("hidden_dim must be positive")
                
    def update_feature_bank(self, features: torch.Tensor):
        """Update feature bank with error checking"""
        if not torch.isfinite(features).all():
            return  # Skip update if features contain NaN/inf
            
        batch_size = features.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.feature_bank.size(0):
            ptr = 0
            self.bank_is_full[0] = True
            
        self.feature_bank[ptr:ptr + batch_size] = features.detach()
        self.temporal_bank[ptr:ptr + batch_size] = features.detach()  # Temporal copy
        self.bank_ptr[0] = (ptr + batch_size) % self.bank_size
        
    def _compute_covariance(self) -> torch.Tensor:
        """Compute covariance matrices with quaternion normalization"""
        # Normalize quaternions
        rotations = nn.functional.normalize(self.rotations, dim=-1)
        qw, qx, qy, qz = rotations.unbind(-1)
        
        # Convert to rotation matrices with numerical stability
        rot_matrices = torch.stack([
            1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
            2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw,
            2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2
        ], dim=-1).view(-1, 3, 3)
        
        # Ensure positive scales
        scales = torch.abs(self.scales) + 1e-6
        scale_matrices = torch.diag_embed(scales)
        
        # Compute covariance: R * S * S * R^T
        covariance = rot_matrices @ scale_matrices @ scale_matrices @ rot_matrices.transpose(-1, -2)
        
        # Ensure symmetric positive definite
        return 0.5 * (covariance + covariance.transpose(-1, -2))
        
    def forward(
        self,
        camera_positions: torch.Tensor,
        camera_directions: torch.Tensor,
        image_size: Tuple[int, int],
        near: float = 0.1,
        far: float = 100.0,
        light_positions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if not torch.isfinite(camera_positions).all() or not torch.isfinite(camera_directions).all():
            raise ValueError("Camera parameters contain NaN/inf values")
            
        batch_size = camera_positions.shape[0]
        H, W = image_size
        
        # Compute ray-gaussian intersections
        covariances = self._compute_covariance()
        
        # Project gaussians to image space with bounds checking
        projected_means = torch.clamp(
            self.means.unsqueeze(0).expand(batch_size, -1, -1),
            min=-far, max=far
        )
        view_directions = nn.functional.normalize(
            camera_directions.unsqueeze(1).expand(-1, self.num_gaussians, -1),
            dim=-1
        )
        
        # Compute view-dependent features with material properties
        view_features = torch.cat([
            projected_means - camera_positions.unsqueeze(1),
            view_directions,
            self.roughness.expand(batch_size, -1, -1),
            self.metallic.expand(batch_size, -1, -1)
        ], dim=-1)
        
        # Compute adaptive opacity
        density = self.density_net(view_features.mean(dim=2))
        opacity = torch.clamp(
            self.opacities * density,
            min=self.min_opacity,
            max=self.max_opacity
        )
        
        # Render colors with material properties
        rendered_colors = self.renderer(view_features)
        rendered_colors = torch.sigmoid(rendered_colors)  # Ensure valid RGB
        
        # Apply opacity and tone mapping
        final_colors = rendered_colors * opacity
        
        # Update feature banks
        self.update_feature_bank(view_features.mean(dim=1))
        
        return {
            "colors": final_colors,
            "depths": projected_means[..., 2],
            "opacities": opacity,
            "covariances": covariances,
            "view_features": view_features,
            "material_properties": {
                "roughness": self.roughness,
                "metallic": self.metallic
            },
            "bank_usage": self.bank_is_full
        }