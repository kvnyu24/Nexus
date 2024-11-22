import torch
import torch.nn as nn
from typing import Dict, Any
from ...core.base import NexusModule
import math

class MBConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        stride: int,
        kernel_size: int,
        se_ratio: float = 0.25
    ):
        super().__init__()
        self.skip_connection = stride == 1 and in_channels == out_channels
        
        # Expansion
        expanded_channels = in_channels * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        ) if expand_ratio != 1 else nn.Identity()
        
        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size,
                     stride=stride, padding=kernel_size//2, groups=expanded_channels,
                     bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        )
        
        # Squeeze-and-Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, se_channels, 1),
            nn.SiLU(),
            nn.Conv2d(se_channels, expanded_channels, 1),
            nn.Sigmoid()
        )
        
        # Project
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        x = self.expand(x)
        x = self.depthwise(x)
        x = x * self.se(x)
        x = self.project(x)
        
        if self.skip_connection:
            x = x + identity
            
        return x

class EfficientNet(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.width_multiplier = config.get("width_multiplier", 1.0)
        self.depth_multiplier = config.get("depth_multiplier", 1.0)
        self.num_classes = config.get("num_classes", 1000)
        
        # Initial conv
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, self._scale_channels(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self._scale_channels(32)),
            nn.SiLU()
        )
        
        # Build stages
        self.stages = self._build_stages()
        
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(config.get("dropout", 0.2)),
            nn.Linear(self._scale_channels(1280), self.num_classes)
        )
        
    def _scale_channels(self, channels: int) -> int:
        return int(channels * self.width_multiplier)
        
    def _scale_repeats(self, repeats: int) -> int:
        return int(math.ceil(repeats * self.depth_multiplier))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.conv_stem(x)
        features = []
        
        for stage in self.stages:
            x = stage(x)
            features.append(x)
            
        x = self.head(x)
        
        return {
            "logits": x,
            "features": features
        }