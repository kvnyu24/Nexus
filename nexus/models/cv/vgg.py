import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule

class VGG(NexusModule):
    """
    Enhanced VGG implementation with modern improvements:
    - Batch normalization and layer normalization
    - Advanced dropout with drop path
    - Momentum feature bank with label tracking
    - Progressive channel scaling with bottleneck option
    - Residual and skip connections
    - Stochastic depth
    - Layer scale parameters
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config
        self._validate_config(config)
        
        # Core dimensions
        self.in_channels = config.get("in_channels", 3)
        self.num_classes = config["num_classes"]
        self.base_channels = config.get("base_channels", 64)
        self.use_residual = config.get("use_residual", False)
        self.use_bottleneck = config.get("use_bottleneck", False)
        self.drop_path = config.get("drop_path", 0.0)
        self.layer_scale_init_value = config.get("layer_scale_init_value", 1e-6)
        
        # Build VGG blocks
        self.features = self._make_layers(config["architecture"])
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(512 * 7 * 7)
        
        # Classifier with improved architecture
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.5)),
            nn.Linear(4096, 4096),
            nn.LayerNorm(4096), 
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.5)),
            nn.Linear(4096, self.num_classes)
        )
        
        # Enhanced feature bank with momentum
        bank_size = config.get("bank_size", 10000)
        self.register_buffer("feature_bank", torch.zeros(bank_size, 512))
        self.register_buffer("bank_labels", torch.zeros(bank_size, dtype=torch.long))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        self.momentum = config.get("bank_momentum", 0.9)
        
        # Layer scale parameters
        self.gamma = nn.Parameter(self.layer_scale_init_value * torch.ones(512))
        
        # Initialize weights
        self._init_weights()
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration with enhanced checks"""
        required = ["num_classes", "architecture"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
        if config["num_classes"] <= 0:
            raise ValueError("num_classes must be positive")
            
        if "dropout" in config and not 0 <= config["dropout"] <= 1:
            raise ValueError("dropout must be between 0 and 1")
            
        if "drop_path" in config and not 0 <= config["drop_path"] <= 1:
            raise ValueError("drop_path must be between 0 and 1")
            
    def _init_weights(self) -> None:
        """Enhanced weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
    def _make_layers(self, architecture: List[int]) -> nn.Sequential:
        """Create VGG blocks with enhanced features"""
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # Optional bottleneck
                mid_channels = x // 4 if self.use_bottleneck else x
                
                if self.use_bottleneck:
                    conv = nn.Sequential(
                        nn.Conv2d(in_channels, mid_channels, kernel_size=1),
                        nn.BatchNorm2d(mid_channels),
                        nn.GELU(),
                        nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(mid_channels),
                        nn.GELU(),
                        nn.Conv2d(mid_channels, x, kernel_size=1),
                    )
                else:
                    conv = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                
                block = [
                    conv,
                    nn.BatchNorm2d(x),
                    nn.GELU(),
                    nn.Dropout2d(self.drop_path) if self.drop_path > 0 else nn.Identity()
                ]
                
                if self.use_residual and in_channels == x:
                    block.append(lambda x: x + conv(x))  # Residual connection
                    
                layers.extend(block)
                in_channels = x
                
        return nn.Sequential(*layers)
        
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
        
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Ensure input is float32 for MPS compatibility
        if x.device.type == 'mps':
            x = x.to(torch.float32)
            
        # Extract features
        features = self.features(x)
        pooled = self.pool(features)
        flat_features = torch.flatten(pooled, 1)
        
        # Apply feature normalization and scaling
        normalized_features = self.feature_norm(flat_features)
        scaled_features = normalized_features * self.gamma
        
        # Update feature bank if labels provided
        if labels is not None:
            self.update_feature_bank(scaled_features, labels)
        
        # Classification
        logits = self.classifier(scaled_features)
        
        outputs = {
            "logits": logits,
            "pooled": pooled
        }
        
        if return_features:
            outputs.update({
                "features": scaled_features,
                "raw_features": flat_features
            })
            
        return outputs
