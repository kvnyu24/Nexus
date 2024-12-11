import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from ....core.base import NexusModule
from ....components.attention import MultiHeadSelfAttention
from .temporal_attention import TemporalAttention
from ....components import ResidualBlock

class ReIDBackbone(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Enhanced configuration with validation
        self.in_channels = config.get("in_channels", 3)
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
            
        self.base_channels = config.get("base_channels", 64)
        if self.base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {self.base_channels}")
            
        # Improved backbone with residual connections
        self.conv1 = nn.Conv2d(self.in_channels, self.base_channels, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Enhanced feature extraction with residual blocks
        self.layer1 = self._make_residual_layer(self.base_channels, 64, 2)
        self.layer2 = self._make_residual_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_residual_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_residual_layer(256, 512, 2, stride=2)
        
        # Feature normalization
        self.feature_norm = nn.BatchNorm2d(512)
        
    def _make_residual_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        layers = []
        
        # Downsample layer if needed
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        # First block with potential downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input validation
        if not torch.isfinite(x).all():
            raise ValueError("Input contains inf or nan values")
            
        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Normalize features
        x = self.feature_norm(x)
        
        return x

class PedestrianReID(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Enhanced model configuration
        self.hidden_dim = config.get("hidden_dim", 512)
        self.num_classes = config.get("num_classes", 1000)
        self.feature_dim = config.get("feature_dim", 2048)
        self.dropout_rate = config.get("dropout_rate", 0.5)
        self.num_parts = config.get("num_parts", 6)  # For part-based features
        
        # Backbone network
        self.backbone = ReIDBackbone(config)
        
        # Global and part-based pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.part_pool = nn.AdaptiveAvgPool2d((self.num_parts, 1))
        
        # Enhanced feature embedding with batch norm
        self.embedding = nn.Sequential(
            nn.Linear(512, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate)
        )
        
        # Part-based embeddings
        self.part_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_rate)
            ) for _ in range(self.num_parts)
        ])
        
        # Enhanced classification heads
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        self.part_classifiers = nn.ModuleList([
            nn.Linear(self.feature_dim, self.num_classes) 
            for _ in range(self.num_parts)
        ])
        
        # Multi-head attention for feature refinement
        self.attention = MultiHeadSelfAttention(
            hidden_size=self.feature_dim,
            num_heads=8
        )
        
    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Input validation
        if not torch.isfinite(images).all():
            raise ValueError("Input contains inf or nan values")
            
        # Extract features
        features = self.backbone(images)
        
        # Global features
        global_features = self.gap(features)
        global_features = global_features.view(global_features.size(0), -1)
        
        # Part-based features
        part_features = self.part_pool(features)
        part_features = part_features.view(part_features.size(0), part_features.size(1), -1)
        
        # Embeddings
        global_embeddings = self.embedding(global_features)
        part_embeddings = [embed(part_features[:,:,i]) for i, embed in enumerate(self.part_embeddings)]
        
        # Apply attention to refine features
        refined_embeddings = self.attention(global_embeddings.unsqueeze(1)).squeeze(1)
        
        # Classification
        global_logits = self.classifier(refined_embeddings)
        part_logits = [clf(emb) for clf, emb in zip(self.part_classifiers, part_embeddings)]
        
        outputs = {
            "embeddings": refined_embeddings,
            "part_embeddings": part_embeddings,
            "global_logits": global_logits,
            "part_logits": part_logits,
            "features": features
        }
        
        if labels is not None:
            # Calculate comprehensive losses
            cls_loss = self._compute_classification_losses(global_logits, part_logits, labels)
            triplet_loss = self._compute_triplet_loss(refined_embeddings, labels)
            part_loss = self._compute_part_loss(part_embeddings, labels)
            
            outputs["loss"] = cls_loss + triplet_loss + part_loss
            outputs["cls_loss"] = cls_loss
            outputs["triplet_loss"] = triplet_loss
            outputs["part_loss"] = part_loss
            
        return outputs
        
    def _compute_classification_losses(
        self,
        global_logits: torch.Tensor,
        part_logits: List[torch.Tensor],
        labels: torch.Tensor
    ) -> torch.Tensor:
        criterion = nn.CrossEntropyLoss()
        global_loss = criterion(global_logits, labels)
        part_losses = torch.stack([criterion(logits, labels) for logits in part_logits])
        return global_loss + 0.5 * part_losses.mean()
        
    def _compute_triplet_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.3
    ) -> torch.Tensor:
        """Enhanced triplet loss with distance weighted sampling"""
        pairwise_dist = torch.cdist(embeddings, embeddings)
        
        # Get hardest positive and negative pairs with distance weighting
        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_neg = labels.unsqueeze(0) != labels.unsqueeze(1)
        
        # Weight distances by exp(-dist)
        weights = torch.exp(-pairwise_dist)
        
        hardest_pos = (pairwise_dist * mask_pos.float() * weights).max(dim=1)[0]
        hardest_neg = (pairwise_dist + 1e5 * mask_pos.float()).min(dim=1)[0]
        
        loss = F.relu(hardest_pos - hardest_neg + margin)
        return loss.mean()
        
    def _compute_part_loss(
        self,
        part_embeddings: List[torch.Tensor],
        labels: torch.Tensor,
        margin: float = 0.3
    ) -> torch.Tensor:
        """Compute triplet loss for part-based features"""
        losses = []
        for embeddings in part_embeddings:
            loss = self._compute_triplet_loss(embeddings, labels, margin)
            losses.append(loss)
        return torch.stack(losses).mean()
