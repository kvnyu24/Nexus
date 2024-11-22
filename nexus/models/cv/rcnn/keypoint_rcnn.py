import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from ....core.base import NexusModule
from .base_rcnn import BaseRCNN
from ....components.attention import SpatialAttention

class KeypointHead(nn.Module):
    def __init__(self, in_channels: int, num_keypoints: int, hidden_dim: int = 256):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.spatial_attention = SpatialAttention()
        self.keypoint_pred = nn.Conv2d(hidden_dim, num_keypoints, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.spatial_attention(x)
        return self.keypoint_pred(x)

class KeypointRCNN(BaseRCNN):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Keypoint-specific configuration
        self.num_keypoints = config.get("num_keypoints", 17)  # Default: COCO keypoints
        self.keypoint_threshold = config.get("keypoint_threshold", 0.2)
        
        # Initialize keypoint head
        self.keypoint_head = KeypointHead(
            in_channels=256,  # Match FPN output channels
            num_keypoints=self.num_keypoints,
            hidden_dim=config.get("keypoint_hidden_dim", 256)
        )
        
    def forward(
        self,
        images: torch.Tensor,
        image_shapes: List[tuple],
        targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        # Get base RCNN features and predictions
        base_outputs = super().forward(images, image_shapes, targets)
        
        # Extract keypoints for detected instances
        roi_features = base_outputs["roi_features"]
        keypoint_features = self.keypoint_head(roi_features)
        
        outputs = {
            **base_outputs,
            "keypoint_features": keypoint_features
        }
        
        return outputs 