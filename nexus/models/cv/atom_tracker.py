import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin
from ...components.attention import SpatialAttention

class TargetEncoder(NexusModule):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        return self.spatial_attention(x)

class SearchRegionEncoder(NexusModule):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1, stride=2),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)

class ATOMTracker(WeightInitMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Model dimensions
        self.hidden_dim = config.get("hidden_dim", 256)
        self.in_channels = config.get("in_channels", 3)

        # Target and search region encoders
        self.target_encoder = TargetEncoder(self.in_channels, self.hidden_dim)
        self.search_encoder = SearchRegionEncoder(self.in_channels, self.hidden_dim)

        # Cross-correlation prediction head
        self.prediction_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, 4, 1)  # 4 for bbox coordinates
        )

        # Initialize weights
        self.init_weights_vision()
            
    def forward(
        self,
        target_image: torch.Tensor,
        search_image: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Encode target template
        target_feat = self.target_encoder(target_image)
        
        # Encode search region
        search_feat = self.search_encoder(search_image)
        
        # Combine features
        batch_size = search_feat.size(0)
        combined_feat = torch.cat([
            target_feat.expand(batch_size, -1, -1, -1),
            search_feat
        ], dim=1)
        
        # Predict bounding box
        pred_bbox = self.prediction_head(combined_feat)
        
        return {
            "pred_bbox": pred_bbox,
            "target_features": target_feat,
            "search_features": search_feat
        }
