from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from ....core.base import NexusModule
from .reid_module import ReIDBackbone
from ....components.attention import MultiHeadSelfAttention

class TemporalAttention(NexusModule):
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_dim,
            num_heads=num_heads
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attention(x)
        return self.norm(x + attended)

class TemporalReIDModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core configuration
        self.hidden_dim = config.get("hidden_dim", 512)
        self.feature_dim = config.get("feature_dim", 2048)
        self.num_classes = config.get("num_classes", 1000)
        self.sequence_length = config.get("sequence_length", 8)
        
        # Backbone network (reuse existing ReIDBackbone)
        self.backbone = ReIDBackbone(config)
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            hidden_dim=self.feature_dim,
            num_heads=config.get("num_heads", 8)
        )
        
        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature embedding
        self.embedding = nn.Sequential(
            nn.Linear(512, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        
        # Classification head
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        
    def forward(
        self,
        image_sequence: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, channels, height, width = image_sequence.shape
        
        # Process each frame through backbone
        features = []
        for t in range(seq_len):
            frame_features = self.backbone(image_sequence[:, t])
            frame_features = self.embedding(frame_features.mean([-2, -1]))
            features.append(frame_features)
            
        # Stack temporal features
        features = torch.stack(features, dim=1)  # [B, T, D]
        
        # Apply temporal attention
        temporal_features = self.temporal_attention(features)
        
        # Temporal pooling
        pooled_features = self.temporal_pool(
            temporal_features.transpose(1, 2)
        ).squeeze(-1)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        outputs = {
            "embeddings": pooled_features,
            "logits": logits,
            "temporal_features": temporal_features
        }
        
        if labels is not None:
            cls_loss = nn.CrossEntropyLoss()(logits, labels)
            triplet_loss = self._compute_triplet_loss(pooled_features, labels)
            outputs["loss"] = cls_loss + triplet_loss
            
        return outputs
        
    def _compute_triplet_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.3
    ) -> torch.Tensor:
        pairwise_dist = torch.cdist(embeddings, embeddings)
        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_neg = labels.unsqueeze(0) != labels.unsqueeze(1)
        
        hardest_pos = (pairwise_dist * mask_pos.float()).max(dim=1)[0]
        hardest_neg = (pairwise_dist + 1e5 * mask_pos.float()).min(dim=1)[0]
        
        loss = torch.clamp(hardest_pos - hardest_neg + margin, min=0)
        return loss.mean() 