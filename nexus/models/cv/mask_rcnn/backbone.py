import torch
import torch.nn as nn
from torch.nn import functional as F
from ....core.base import NexusModule
from ....components.blocks import ResidualBlock

class FPNBackbone(NexusModule):
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Base ResNet configuration
        self.in_channels = 64
        self.base_channels = config.get("base_channels", 64)
        self.block_config = config.get("block_config", [3, 4, 6, 3])
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet stages
        self.layer1 = self._make_layer(self.base_channels, self.block_config[0])
        self.layer2 = self._make_layer(self.base_channels * 2, self.block_config[1], stride=2)
        self.layer3 = self._make_layer(self.base_channels * 4, self.block_config[2], stride=2)
        self.layer4 = self._make_layer(self.base_channels * 8, self.block_config[3], stride=2)
        
        # FPN layers
        self.fpn_sizes = [
            self.base_channels * 2,
            self.base_channels * 4,
            self.base_channels * 8,
            self.base_channels * 16
        ]
        
        self.fpn_conv = nn.ModuleDict({
            f'p{i}': nn.Conv2d(size, 256, kernel_size=1)
            for i, size in enumerate(self.fpn_sizes)
        })
        
        self.fpn_upsample = nn.ModuleDict({
            f'p{i}': nn.Conv2d(256, 256, kernel_size=3, padding=1)
            for i in range(len(self.fpn_sizes))
        })
        
    def _make_layer(self, channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        
        # Add downsample if needed
        downsample = None
        if stride != 1 or self.in_channels != channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * 4)
            )
            
        layers.append(ResidualBlock(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * 4
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.in_channels, channels))
            
        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> dict:
        # ResNet forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # FPN forward
        features = {}
        p5 = self.fpn_conv['p3'](c5)
        features['p5'] = self.fpn_upsample['p3'](p5)
        
        for idx, feature in enumerate([c4, c3, c2]):
            p = self.fpn_conv[f'p{idx}'](feature)
            p = p + F.interpolate(features[f'p{idx+2}'], size=p.shape[-2:])
            features[f'p{idx+1}'] = self.fpn_upsample[f'p{idx}'](p)
            
        return features 