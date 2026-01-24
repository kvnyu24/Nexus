import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from ....core.base import NexusModule
from ....core.mixins import ConfigValidatorMixin, FeatureBankMixin
from .reid_module import ReIDBackbone
from .temporal_attention import TemporalAttention

class AdaptiveReIDWithMemory(ConfigValidatorMixin, FeatureBankMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config using ConfigValidatorMixin
        self.validate_config(config, required_keys=["hidden_dim", "feature_dim", "num_classes"])
        
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
        
        # Adaptive memory bank with priority sampling using FeatureBankMixin
        self.bank_size = config.get("bank_size", 10000)
        self.feature_dim = config.get("feature_dim", 2048)
        self.register_feature_bank("feature", self.bank_size, self.feature_dim)
        self.register_buffer("bank_labels", torch.zeros(self.bank_size))
        self.register_buffer("bank_priorities", torch.zeros(self.bank_size))
        
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
                
    def update_feature_bank_with_priority(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        losses: torch.Tensor
    ):
        """Update feature bank with priority-based sampling"""
        if not torch.isfinite(features).all():
            return

        batch_size = features.size(0)
        ptr = self.feature_ptr
        bank = self.feature_bank

        # Priority-based update using sample losses
        priorities = F.softmax(losses, dim=0)

        # Update feature bank with priority sampling
        if ptr.item() + batch_size > self.bank_size:
            # Find lowest priority samples to replace
            _, indices = torch.topk(
                self.bank_priorities,
                k=batch_size,
                largest=False
            )
            bank[indices] = features.detach()
            self.bank_labels[indices] = labels.detach()
            self.bank_priorities[indices] = priorities
        else:
            end_idx = ptr.item() + batch_size
            bank[ptr.item():end_idx] = features.detach()
            self.bank_labels[ptr.item():end_idx] = labels.detach()
            self.bank_priorities[ptr.item():end_idx] = priorities
            ptr[0] = end_idx % self.bank_size
            if ptr.item() == 0 or self.feature_filled.item():
                self.feature_filled[0] = True
        
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
            self.update_feature_bank_with_priority(
                embeddings,
                labels,
                losses["total_loss"].detach().expand(embeddings.size(0))
            )
            
            outputs.update(losses)
            
        return outputs