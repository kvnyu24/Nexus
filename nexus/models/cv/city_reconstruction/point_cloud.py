import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ....core.base import NexusModule

class PointCloudProcessor(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = config.get("hidden_dim", 256)
        
        self.feature_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, 3, 1)  # XYZ coordinates
        )
        
        self.confidence_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim // 2, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        camera_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        # Process multi-scale features
        point_clouds = []
        confidences = []
        
        for level, feat in features.items():
            points = self.feature_conv(feat)
            conf = self.confidence_head(feat)
            
            if camera_params is not None:
                points = self._project_to_world(points, camera_params)
                
            point_clouds.append(points)
            confidences.append(conf)
            
        # Combine multi-scale point clouds
        point_cloud = torch.cat(point_clouds, dim=1)
        confidence = torch.cat(confidences, dim=1)
        
        return point_cloud * confidence 