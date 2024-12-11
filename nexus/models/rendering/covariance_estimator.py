import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule

class CovarianceEstimator(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core parameters with validation
        self.hidden_dim = config.get("hidden_dim", 256)
        if self.hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even")
            
        self.min_scale = config.get("min_scale", 1e-4)
        self.max_scale = config.get("max_scale", 1.0)
        self.use_view_features = config.get("use_view_features", True)
        
        # Feature extraction with residual connections and layer normalization
        input_dim = 3
        if self.use_view_features:
            input_dim += 3  # Add view direction features
            
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Scale prediction with bounded output
        self.scale_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.LayerNorm(self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 3),  # 3D scales
            nn.Sigmoid()  # Bound outputs between 0 and 1
        )
        
        # Rotation prediction with quaternion normalization
        self.rotation_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.LayerNorm(self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 4)  # Quaternion
        )
        
        # Optional confidence prediction
        self.confidence_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.LayerNorm(self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        points: torch.Tensor,
        view_directions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Input validation
        if points.dim() != 2 or points.size(-1) != 3:
            raise ValueError("Expected points tensor of shape (N, 3)")
            
        if self.use_view_features:
            if view_directions is None:
                raise ValueError("View directions required when use_view_features=True")
            if view_directions.shape != points.shape:
                raise ValueError("View directions must match points shape")
                
        # Prepare input features
        if self.use_view_features:
            inputs = torch.cat([points, view_directions], dim=-1)
        else:
            inputs = points
            
        # Extract features with gradient checkpointing for memory efficiency
        features = self.feature_extractor(inputs)
        
        # Predict scales with bounds
        scales = self.scale_net(features)
        scales = self.min_scale + (self.max_scale - self.min_scale) * scales
        
        # Predict rotations with stable normalization
        rotations = self.rotation_net(features)
        rotations_norm = torch.norm(rotations, dim=-1, keepdim=True).clamp(min=1e-8)
        rotations = rotations / rotations_norm
        
        # Predict confidence scores
        confidence = self.confidence_net(features)
        
        # Compute covariance matrices
        covariances = self._compute_covariance(scales, rotations)
        
        # Scale covariances by confidence
        covariances = covariances * confidence.unsqueeze(-1).unsqueeze(-1)
        
        return {
            "scales": scales,
            "rotations": rotations,
            "covariances": covariances,
            "features": features,
            "confidence": confidence
        }