import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin
from ...components.layers import DropPath
import math


class MBConvBlock(NexusModule):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        stride: int,
        kernel_size: int,
        se_ratio: float = 0.25,
        drop_rate: float = 0.0,
        activation: nn.Module = nn.SiLU
    ):
        super().__init__()
        self.skip_connection = stride == 1 and in_channels == out_channels
        
        # Expansion
        expanded_channels = in_channels * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            activation(inplace=True)
        ) if expand_ratio != 1 else nn.Identity()
        
        # Depthwise with larger kernel
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size,
                     stride=stride, padding=kernel_size//2, groups=expanded_channels,
                     bias=False),
            nn.BatchNorm2d(expanded_channels),
            activation(inplace=True)
        )
        
        # Enhanced Squeeze-and-Excitation with channel mixing
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, se_channels, 1),
            activation(inplace=True),
            nn.Conv2d(se_channels, expanded_channels, 1),
            nn.Hardsigmoid(inplace=True)
        )
        
        # Project with layer scaling
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Stochastic depth (drop path)
        self.stochastic_depth = DropPath(drop_rate)
        
        # Layer scale parameter
        self.layer_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 1e-5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        x = self.expand(x)
        x = self.depthwise(x)
        x = x * self.se(x)
        x = self.project(x)
        x = x * self.layer_scale
        
        if self.skip_connection:
            x = identity + self.stochastic_depth(x)
            
        return x

class EfficientNet(WeightInitMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.width_multiplier = config.get("width_multiplier", 1.0)
        self.depth_multiplier = config.get("depth_multiplier", 1.0)
        self.num_classes = config.get("num_classes", 1000)

        # Initial conv with larger kernel
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, self._scale_channels(32), 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(self._scale_channels(32)),
            nn.SiLU(inplace=True)
        )

        # Build stages with progressive stochastic depth
        self.stages = self._build_stages()

        # Enhanced head with label smoothing
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(self._scale_channels(1280)),
            nn.Dropout(config.get("dropout", 0.3)),
            nn.Linear(self._scale_channels(1280), self.num_classes)
        )

        # Initialize weights using mixin
        self.init_weights_vision()

    def _scale_channels(self, channels: int) -> int:
        return int(math.ceil(channels * self.width_multiplier))

    def _scale_repeats(self, repeats: int) -> int:
        return int(math.ceil(repeats * self.depth_multiplier))
        
    def forward(self, x: torch.Tensor, return_features: Optional[bool] = False) -> Dict[str, torch.Tensor]:
        x = self.conv_stem(x)
        features = []
        
        for stage in self.stages:
            x = stage(x)
            features.append(x)
            
        x = self.head(x)
        
        outputs = {"logits": x}
        if return_features:
            outputs["features"] = features
            
        return outputs