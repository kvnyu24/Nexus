import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ....core.base import NexusModule
from .reid_module import ReIDBackbone
from .temporal_attention import TemporalAttention

class EnhancedReID(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core components
        self.backbone = ReIDBackbone(config)
        self.temporal_attention = TemporalAttention(
            hidden_dim=config.get("hidden_dim", 512),
            num_heads=config.get("num_heads", 8)
        )
        
        # Feature bank for efficient retrieval
        self.register_buffer(
            "feature_bank",
            torch.zeros(
                config.get("bank_size", 10000),
                config.get("feature_dim", 2048)
            )
        )
        self.register_buffer("bank_labels", torch.zeros(config.get("bank_size", 10000)))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
        # Additional loss components
        self.center_loss = nn.Parameter(
            torch.zeros(config.get("num_classes", 1000), config.get("feature_dim", 2048))
        )
        
    def update_feature_bank(self, features: torch.Tensor, labels: torch.Tensor):
        batch_size = features.size(0)
        ptr = int(self.bank_ptr)
        
        # Update feature bank with new features
        if ptr + batch_size > self.feature_bank.size(0):
            ptr = 0
        
        self.feature_bank[ptr:ptr + batch_size] = features.detach()
        self.bank_labels[ptr:ptr + batch_size] = labels.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.feature_bank.size(0)
        
    def compute_losses(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Triplet loss
        triplet_loss = self._compute_triplet_loss(embeddings, labels)
        
        # Center loss
        centers = self.center_loss[labels]
        center_loss = torch.mean(torch.sum((embeddings - centers) ** 2, dim=1))
        
        # Cross entropy loss
        cls_loss = nn.CrossEntropyLoss()(features, labels)
        
        return {
            "triplet_loss": triplet_loss,
            "center_loss": center_loss * 0.01,  # Scale center loss
            "cls_loss": cls_loss,
            "total_loss": triplet_loss + cls_loss + center_loss * 0.01
        }
        
    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        is_training: bool = True
    ) -> Dict[str, torch.Tensor]:
        # Extract features using backbone
        features = self.backbone(images)
        
        # Apply temporal attention if sequence data
        if len(images.shape) == 5:  # (batch, sequence, channels, height, width)
            b, s, c, h, w = images.shape
            features = features.view(b, s, -1)
            features, attention_weights = self.temporal_attention(features)
        
        # Generate embeddings
        embeddings = self.embedding(features)
        
        outputs = {
            "embeddings": embeddings,
            "features": features
        }
        
        if is_training and labels is not None:
            # Update feature bank
            self.update_feature_bank(embeddings, labels)
            
            # Compute losses
            losses = self.compute_losses(embeddings, labels, features)
            outputs.update(losses)
            
        return outputs 