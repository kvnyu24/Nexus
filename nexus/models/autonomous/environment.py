import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule
from ..cv.rcnn import FPNBackbone

class EnvironmentModelingModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Following CityReconstructionModel pattern
        self.hidden_dim = config.get("hidden_dim", 256)
        
        # Reuse FPN backbone (similar to SceneUnderstandingModule)
        self.backbone = FPNBackbone(config)
        
        # Environment feature processing (similar to MotionPredictionModule)
        self.env_processor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Prediction heads (following BehaviorPredictionModule pattern)
        self.occupancy_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, config.get("grid_size", 100) ** 2)
        )
        
        self.risk_assessment_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim // 2, config.get("num_risk_levels", 5))
        )
        
    def forward(
        self,
        sensor_features: torch.Tensor,
        scene_context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Process environmental features
        combined_features = torch.cat([sensor_features, scene_context], dim=-1)
        env_features = self.env_processor(combined_features)
        
        # Generate predictions
        occupancy_grid = self.occupancy_head(env_features)
        risk_assessment = self.risk_assessment_head(env_features)
        
        return {
            "occupancy_grid": occupancy_grid.view(-1, self.grid_size, self.grid_size),
            "risk_assessment": risk_assessment,
            "env_features": env_features
        }
