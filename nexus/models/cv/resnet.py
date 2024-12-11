import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from ...core.base import NexusModule
from ...components.blocks import ResidualBlock

class ResNet(NexusModule):
    """
    Enhanced ResNet implementation with modern improvements:
    - Stochastic Depth for better regularization
    - Enhanced feature bank with momentum updates
    - Label smoothing and mixup augmentation support
    - Progressive layer scaling
    - Squeeze-and-Excitation blocks
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate and set config
        self._validate_config(config)
        
        # Model configuration
        self.in_channels = 64
        self.base_channels = config.get("base_channels", 64)
        self.block_config = config.get("block_config", [3, 4, 6, 3])
        self.num_classes = config.get("num_classes", 1000)
        self.drop_path_rate = config.get("drop_path_rate", 0.1)
        self.label_smoothing = config.get("label_smoothing", 0.1)
        self.mixup_alpha = config.get("mixup_alpha", 0.2)
        self.se_ratio = config.get("se_ratio", 0.25)
        
        # Initial stem with larger receptive field
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.in_channels//2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels//2, self.in_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # ResNet stages with progressive drop path
        total_blocks = sum(self.block_config)
        block_idx = 0
        self.stages = nn.ModuleList()
        for i, num_blocks in enumerate(self.block_config):
            blocks = []
            for j in range(num_blocks):
                drop_path_rate = self.drop_path_rate * block_idx / total_blocks
                stride = 2 if j == 0 and i > 0 else 1
                blocks.append(
                    ResidualBlock(
                        self.in_channels,
                        self.base_channels * (2**i),
                        stride=stride,
                        se_ratio=self.se_ratio,
                        drop_path_rate=drop_path_rate
                    )
                )
                self.in_channels = self.base_channels * (2**i) * 4
                block_idx += 1
            self.stages.append(nn.Sequential(*blocks))
        
        # Enhanced head with layer norm and dropout
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(self.base_channels * 32),
            nn.Dropout(config.get("dropout", 0.3)),
            nn.Linear(self.base_channels * 32, self.num_classes)
        )
        
        # Enhanced feature bank with momentum
        bank_size = config.get("bank_size", 10000)
        self.register_buffer("feature_bank", torch.zeros(bank_size, self.base_channels * 32))
        self.register_buffer("bank_labels", torch.zeros(bank_size, dtype=torch.long))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        self.momentum = config.get("bank_momentum", 0.99)
        
        # Initialize weights with improved scheme
        self._init_weights()
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = ["num_classes"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        if config.get("drop_path_rate", 0.1) < 0 or config.get("drop_path_rate", 0.1) > 1:
            raise ValueError("drop_path_rate must be between 0 and 1")
                
    def _init_weights(self) -> None:
        """Initialize weights with improved schemes for different layer types"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
    def update_feature_bank(self, features: torch.Tensor, labels: torch.Tensor):
        """Update feature bank with momentum"""
        batch_size = features.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.feature_bank.size(0):
            ptr = 0
            
        # Momentum update
        self.feature_bank[ptr:ptr + batch_size] = (
            self.momentum * self.feature_bank[ptr:ptr + batch_size] +
            (1 - self.momentum) * features.detach()
        )
        self.bank_labels[ptr:ptr + batch_size] = labels
        self.bank_ptr[0] = (ptr + batch_size) % self.feature_bank.size(0)
        
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Stem and stages
        x = self.stem(x)
        
        # Collect intermediate features
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        # Global pooling and feature extraction
        pooled = self.avgpool(x)
        flat_features = torch.flatten(pooled, 1)
        
        # Update feature bank if labels provided
        if labels is not None:
            self.update_feature_bank(flat_features, labels)
        
        # Classification head
        logits = self.head(flat_features)
        
        return {
            "logits": logits,
            "features": flat_features,
            "intermediate_features": features,
            "pooled": pooled
        }