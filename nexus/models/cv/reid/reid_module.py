import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ....core.base import NexusModule
from ....components.attention import MultiHeadSelfAttention

class ReIDBackbone(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Base configuration
        self.in_channels = config.get("in_channels", 3)
        self.base_channels = config.get("base_channels", 64)
        
        # Backbone layers
        self.conv1 = nn.Conv2d(self.in_channels, self.base_channels, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Feature extraction blocks
        self.layer1 = self._make_layer(self.base_channels, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

class PedestrianReID(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model configuration
        self.hidden_dim = config.get("hidden_dim", 512)
        self.num_classes = config.get("num_classes", 1000)
        self.feature_dim = config.get("feature_dim", 2048)
        
        # Backbone network
        self.backbone = ReIDBackbone(config)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Feature embedding
        self.embedding = nn.Sequential(
            nn.Linear(512, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        
        # Classification head (for training)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        
        # Attention module for part-based features
        self.attention = MultiHeadSelfAttention(
            hidden_size=self.feature_dim,
            num_heads=8
        )
        
    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.backbone(images)
        
        # Global features
        global_features = self.gap(features)
        global_features = global_features.view(global_features.size(0), -1)
        
        # Embedding
        embeddings = self.embedding(global_features)
        
        # Classification logits
        logits = self.classifier(embeddings)
        
        outputs = {
            "embeddings": embeddings,
            "logits": logits,
            "features": features
        }
        
        if labels is not None:
            # Calculate losses if labels are provided
            cls_loss = nn.CrossEntropyLoss()(logits, labels)
            triplet_loss = self._compute_triplet_loss(embeddings, labels)
            outputs["loss"] = cls_loss + triplet_loss
            
        return outputs
        
    def _compute_triplet_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.3
    ) -> torch.Tensor:
        """Compute triplet loss with hard mining"""
        pairwise_dist = torch.cdist(embeddings, embeddings)
        
        # Get hardest positive and negative pairs
        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_neg = labels.unsqueeze(0) != labels.unsqueeze(1)
        
        hardest_pos = (pairwise_dist * mask_pos.float()).max(dim=1)[0]
        hardest_neg = (pairwise_dist + 1e5 * mask_pos.float()).min(dim=1)[0]
        
        loss = torch.clamp(hardest_pos - hardest_neg + margin, min=0)
        return loss.mean()
