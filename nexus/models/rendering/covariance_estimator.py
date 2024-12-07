import torch
import torch.nn as nn
from typing import Dict, Any
from ...core.base import NexusModule

class CovarianceEstimator(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core parameters with validation
        self.hidden_dim = config.get("hidden_dim", 256)
        if self.hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even")
            
        # Feature extraction with layer normalization for stability
        self.feature_extractor = nn.Sequential(
            nn.Linear(3, self.hidden_dim),  # 3D coordinates
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Scale prediction with positive constraint
        self.scale_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3),  # 3D scales
            nn.Softplus()
        )
        
        # Rotation prediction with quaternion normalization
        self.rotation_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 4)  # Quaternion
        )
        
    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Input validation
        if points.dim() != 2 or points.size(-1) != 3:
            raise ValueError("Expected points tensor of shape (N, 3)")
            
        # Extract features
        features = self.feature_extractor(points)
        
        # Predict scales and rotations
        scales = self.scale_net(features)
        rotations = self.rotation_net(features)
        
        # Normalize quaternions with numerical stability
        rotations_norm = torch.norm(rotations, dim=-1, keepdim=True).clamp(min=1e-8)
        rotations = rotations / rotations_norm
        
        # Compute covariance matrices
        covariances = self._compute_covariance(scales, rotations)
        
        return {
            "scales": scales,
            "rotations": rotations,
            "covariances": covariances,
            "features": features
        }