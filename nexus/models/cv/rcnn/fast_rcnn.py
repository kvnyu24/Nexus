from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ....core.base import NexusModule
from ....components.blocks import ResidualBlock

class RoIHead(NexusModule):
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_classes = config.get("num_classes", 80)
        self.pool_size = config.get("roi_pool_size", 7)
        
        # RoI feature extractor
        self.roi_layers = nn.Sequential(
            ResidualBlock(self.hidden_dim, self.hidden_dim // 4),
            ResidualBlock(self.hidden_dim, self.hidden_dim // 4),
            ResidualBlock(self.hidden_dim, self.hidden_dim // 4),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_classes + 1)  # +1 for background
        )
        
        # Box regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 4 * self.num_classes)  # 4 coords per class
        )
        
    def forward(
        self,
        roi_features: torch.Tensor,
        proposals: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Extract RoI features
        x = self.roi_layers(roi_features)
        x = x.flatten(1)
        
        # Generate predictions
        cls_scores = self.cls_head(x)
        bbox_deltas = self.bbox_head(x)
        
        return {
            "cls_scores": cls_scores,
            "bbox_deltas": bbox_deltas,
            "roi_features": roi_features
        }

class FastRCNNPredictor(NexusModule):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        
        return self.cls_score(x), self.bbox_pred(x) 