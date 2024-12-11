import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from ....core.base import NexusModule
from .reid_module import ReIDBackbone
from .temporal_attention import TemporalAttention

class AdaptiveReIDWithMemory(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config
        self._validate_config(config)
        
        # Enhanced backbone with higher capacity
        backbone_config = config.copy()
        backbone_config["base_channels"] = config.get("backbone_channels", 96)
        self.backbone = ReIDBackbone(backbone_config)
        
        # Multi-scale temporal modeling
        self.temporal_attention = nn.ModuleList([
            TemporalAttention(
                hidden_dim=config.get("hidden_dim", 512),
                num_heads=2**i  # Varying head numbers for multi-scale
            ) for i in range(3)  # 2, 4, 8 heads
        ])
        
        # Adaptive memory bank with priority sampling
        bank_size = config.get("bank_size", 10000)
        feature_dim = config.get("feature_dim", 2048)
        self.register_buffer("feature_bank", torch.zeros(bank_size, feature_dim))
        self.register_buffer("bank_labels", torch.zeros(bank_size))
        self.register_buffer("bank_priorities", torch.zeros(bank_size))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
        # Learnable center embeddings with uncertainty
        num_classes = config.get("num_classes", 1000)
        self.center_embeddings = nn.Parameter(
            torch.randn(num_classes, feature_dim)
        )
        self.center_uncertainty = nn.Parameter(
            torch.ones(num_classes)
        )
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Loss weights
        self.loss_weights = {
            "triplet": config.get("triplet_weight", 1.0),
            "center": config.get("center_weight", 0.01),
            "uncertainty": config.get("uncertainty_weight", 0.1)
        }
        
    def _validate_config(self, config: Dict[str, Any]):
        required = ["hidden_dim", "feature_dim", "num_classes"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
    def update_feature_bank(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        losses: torch.Tensor
    ):
        batch_size = features.size(0)
        ptr = int(self.bank_ptr)
        
        # Priority-based update using sample losses
        priorities = F.softmax(losses, dim=0)
        
        # Update feature bank with priority sampling
        if ptr + batch_size > self.feature_bank.size(0):
            # Find lowest priority samples to replace
            _, indices = torch.topk(
                self.bank_priorities,
                k=batch_size,
                largest=False
            )
            self.feature_bank[indices] = features.detach()
            self.bank_labels[indices] = labels.detach()
            self.bank_priorities[indices] = priorities
        else:
            self.feature_bank[ptr:ptr + batch_size] = features.detach()
            self.bank_labels[ptr:ptr + batch_size] = labels.detach()
            self.bank_priorities[ptr:ptr + batch_size] = priorities
            self.bank_ptr[0] = (ptr + batch_size) % self.feature_bank.size(0)
        
    def compute_losses(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Enhanced triplet loss with uncertainty
        triplet_loss = self._compute_triplet_loss(embeddings, labels)
        
        # Uncertainty-aware center loss
        centers = self.center_embeddings[labels]
        uncertainties = F.softplus(self.center_uncertainty[labels])
        center_loss = torch.mean(
            torch.sum((embeddings - centers) ** 2, dim=1) / uncertainties +
            torch.log(uncertainties)
        )
        
        # Feature consistency loss
        transformed = self.feature_transform(features)
        consistency_loss = F.mse_loss(transformed, embeddings)
        
        weighted_loss = (
            self.loss_weights["triplet"] * triplet_loss +
            self.loss_weights["center"] * center_loss +
            self.loss_weights["uncertainty"] * consistency_loss
        )
        
        return {
            "triplet_loss": triplet_loss,
            "center_loss": center_loss,
            "consistency_loss": consistency_loss,
            "total_loss": weighted_loss
        }
        
    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        is_training: bool = True
    ) -> Dict[str, torch.Tensor]:
        # Input validation
        if not torch.isfinite(images).all():
            raise ValueError("Input contains invalid values")
            
        # Extract backbone features
        features = self.backbone(images)
        
        # Multi-scale temporal attention for sequences
        if len(images.shape) == 5:  # (batch, sequence, channels, height, width)
            b, s, c, h, w = images.shape
            features = features.view(b, s, -1)
            
            # Apply multi-scale attention
            attention_outputs = []
            for attention_layer in self.temporal_attention:
                attended, weights = attention_layer(features)
                attention_outputs.append(attended)
            
            # Combine multi-scale features
            features = torch.stack(attention_outputs).mean(0)
        
        # Generate embeddings with transformation
        embeddings = self.feature_transform(features)
        
        outputs = {
            "embeddings": embeddings,
            "features": features
        }
        
        if is_training and labels is not None:
            # Compute all losses
            losses = self.compute_losses(embeddings, labels, features)
            
            # Update memory bank with loss-based priorities
            self.update_feature_bank(
                embeddings,
                labels,
                losses["total_loss"].detach()
            )
            
            outputs.update(losses)
            
        return outputs