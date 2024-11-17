import torch
import torch.nn as nn
from typing import Dict, Any
from ...core.base import NexusModule

class CompactCNN(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.num_classes = config.get("num_classes", 10)
        self.dropout_rate = config.get("dropout", 0.2)
        
        # Simple but effective CNN architecture
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(self.dropout_rate),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(self.dropout_rate),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.num_classes)
        )
        
    def forward(self, image: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        features = self.features(image)
        logits = self.classifier(features)
        
        return {
            "logits": logits,
            "embeddings": features.flatten(1)
        } 