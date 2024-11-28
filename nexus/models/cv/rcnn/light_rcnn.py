import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from ....core.base import NexusModule
from .base_rcnn import BaseRCNN
from ....components.blocks import DepthwiseSeparableConv

class LightFPNBackbone(NexusModule):
    def __init__(self, in_channels: int, hidden_dim: int = 256):
        super().__init__()
        
        self.conv1 = DepthwiseSeparableConv(in_channels, hidden_dim, 3, stride=2)
        self.conv2 = DepthwiseSeparableConv(hidden_dim, hidden_dim * 2, 3, stride=2)
        self.conv3 = DepthwiseSeparableConv(hidden_dim * 2, hidden_dim * 4, 3, stride=2)
        
        # Lightweight FPN layers
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim * mult, hidden_dim, 1)
            for mult in [4, 2, 1]
        ])
        
        self.output_convs = nn.ModuleList([
            DepthwiseSeparableConv(hidden_dim, hidden_dim, 3)
            for _ in range(3)
        ])
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Bottom-up pathway
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        
        # Top-down pathway
        features = {}
        prev_features = self.lateral_convs[0](c3)
        features["p3"] = self.output_convs[0](prev_features)
        
        for idx, (lateral_conv, output_conv, feature) in enumerate(
            zip(self.lateral_convs[1:], self.output_convs[1:], [c2, c1])
        ):
            top_down_features = nn.functional.interpolate(
                prev_features, scale_factor=2, mode="nearest"
            )
            lateral_features = lateral_conv(feature)
            prev_features = top_down_features + lateral_features
            features[f"p{idx+4}"] = output_conv(prev_features)
            
        return features

class LightRCNN(BaseRCNN):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Replace standard backbone with lightweight version
        self.backbone = LightFPNBackbone(
            in_channels=config.get("in_channels", 3),
            hidden_dim=config.get("hidden_dim", 256)
        )
        
        # Optimize RoI head for speed
        self.box_head = nn.Sequential(
            nn.Linear(256 * self.roi_pool_size ** 2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True)
        ) 