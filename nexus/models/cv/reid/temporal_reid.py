from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ....core.base import NexusModule
from .reid_module import ReIDBackbone
from ....components.attention import MultiHeadSelfAttention
from ....components.blocks import ResidualBlock, InvertedResidualBlock

class TemporalAttention(NexusModule):
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_dim,
            num_heads=num_heads
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Enhanced FFN with gated activation
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GLU(dim=-1)
        )
        
        # Add position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Add positional embeddings
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len]
        
        # Multi-head attention with pre-norm
        normed = self.norm1(x)
        attended = self.attention(normed, attention_mask=mask)
        x = x + self.dropout(attended)
        
        # FFN with pre-norm
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)
        
        return x

class TemporalReIDModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Enhanced configuration
        self.hidden_dim = config.get("hidden_dim", 512)
        self.feature_dim = config.get("feature_dim", 2048)
        self.num_classes = config.get("num_classes", 1000)
        self.sequence_length = config.get("sequence_length", 8)
        self.dropout = config.get("dropout", 0.1)
        self.use_motion = config.get("use_motion", True)
        
        # Backbone network with optional pretrained weights
        self.backbone = ReIDBackbone(config)
        
        # Motion feature extractor
        if self.use_motion:
            self.motion_net = nn.Sequential(
                InvertedResidualBlock(6, 64),  # Concatenated consecutive frames
                InvertedResidualBlock(64, 128),
                InvertedResidualBlock(128, 256),
                nn.AdaptiveAvgPool2d(1)
            )
        
        # Multiple temporal attention layers with skip connections
        num_layers = config.get("num_attention_layers", 3)
        self.temporal_attention_layers = nn.ModuleList([
            TemporalAttention(
                hidden_dim=self.feature_dim,
                num_heads=config.get("num_heads", 8),
                dropout=self.dropout
            ) for _ in range(num_layers)
        ])
        
        # Enhanced temporal pooling
        self.temporal_pool_type = config.get("temporal_pool", "attention")
        if self.temporal_pool_type == "attention":
            self.temporal_pool = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 2),
                nn.Tanh(),
                nn.Linear(self.feature_dim // 2, 1),
                nn.Softmax(dim=1)
            )
        else:
            self.temporal_pool = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Dropout(0.1)
            )
            
        # Enhanced feature embedding with residual connections
        self.embedding = nn.Sequential(
            ResidualBlock(512, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            ResidualBlock(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Multi-task heads
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        self.reid_proj = nn.Linear(self.feature_dim, 512)  # ReID specific projection
        self.label_smoothing = config.get("label_smoothing", 0.1)
        
        # Center learning
        self.center_features = nn.Parameter(torch.randn(self.num_classes, self.feature_dim))
        nn.init.xavier_uniform_(self.center_features)
        
    def forward(
        self,
        image_sequence: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Input validation and normalization
        if image_sequence.dim() != 5:
            raise ValueError(f"Expected 5D input (B,T,C,H,W), got shape {image_sequence.shape}")
            
        if not torch.isfinite(image_sequence).all():
            raise ValueError("Input contains inf or nan values")
            
        batch_size, seq_len, channels, height, width = image_sequence.shape
        
        # Process each frame through backbone with motion features
        features = []
        motion_features = []
        
        for t in range(seq_len):
            # Extract frame features
            frame_features = self.backbone(image_sequence[:, t])
            frame_features = self.embedding(frame_features.mean([-2, -1]))
            features.append(frame_features)
            
            # Extract motion features
            if self.use_motion and t > 0:
                motion = self.motion_net(
                    torch.cat([image_sequence[:, t-1], image_sequence[:, t]], dim=2)
                ).squeeze(-1).squeeze(-1)
                motion_features.append(motion)
                
        # Stack features
        features = torch.stack(features, dim=1)  # [B, T, D]
        if motion_features:
            motion_features = torch.stack(motion_features, dim=1)
            features = features + F.pad(motion_features, (0, 0, 1, 0))  # Add motion information
            
        # Apply temporal attention with residual connections
        temporal_features = features
        attention_weights = []
        
        for attention_layer in self.temporal_attention_layers:
            attended = attention_layer(temporal_features, attention_mask)
            attention_weights.append(attended)
            temporal_features = temporal_features + attended
            
        # Enhanced temporal pooling
        if self.temporal_pool_type == "attention":
            weights = self.temporal_pool(temporal_features)
            pooled_features = (temporal_features * weights).sum(dim=1)
        else:
            pooled_features = self.temporal_pool(
                temporal_features.transpose(1, 2)
            ).squeeze(-1)
            
        # Multi-task outputs
        logits = self.classifier(pooled_features)
        reid_features = self.reid_proj(pooled_features)
        
        outputs = {
            "embeddings": pooled_features,
            "reid_features": reid_features,
            "logits": logits,
            "temporal_features": temporal_features,
            "attention_weights": attention_weights
        }
        
        if labels is not None:
            # Enhanced loss computation
            cls_loss = self._compute_smooth_loss(logits, labels)
            triplet_loss = self._compute_triplet_loss(reid_features, labels)
            center_loss = self._compute_center_loss(reid_features, labels)
            
            # Adaptive loss weighting
            total_loss = cls_loss + triplet_loss + 0.1 * center_loss
            
            outputs["loss"] = total_loss
            outputs["loss_components"] = {
                "cls_loss": cls_loss,
                "triplet_loss": triplet_loss,
                "center_loss": center_loss
            }
            
        return outputs
        
    def _compute_smooth_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        # Enhanced cross entropy with label smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(1))
        smooth_loss = -log_probs.mean(dim=-1)
        return (1 - self.label_smoothing) * nll_loss.mean() + self.label_smoothing * smooth_loss.mean()
        
    def _compute_triplet_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.3
    ) -> torch.Tensor:
        # Enhanced triplet loss with distance weighted sampling
        pairwise_dist = torch.cdist(embeddings, embeddings)
        
        # Create masks with temperature scaling
        temp = 0.05
        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_neg = labels.unsqueeze(0) != labels.unsqueeze(1)
        
        # Weighted hard mining
        hardest_pos = (pairwise_dist * mask_pos.float()).max(dim=1)[0]
        
        # Semi-hard negative mining with temperature
        semi_hard_neg = (
            pairwise_dist +
            (mask_pos.float() * 1e5) +
            ((pairwise_dist <= hardest_pos.unsqueeze(1)).float() * 1e5) / temp
        ).min(dim=1)[0]
        
        # Compute loss with adaptive margin
        loss = torch.clamp(hardest_pos - semi_hard_neg + margin, min=0)
        return loss.mean()
        
    def _compute_center_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 0.5
    ) -> torch.Tensor:
        # Enhanced center loss with momentum update
        unique_labels = labels.unique()
        batch_centers = []
        
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                center = embeddings[mask].mean(0)
                batch_centers.append((label, center))
                
        # Update centers with momentum
        with torch.no_grad():
            for label, center in batch_centers:
                self.center_features[label] = (
                    alpha * self.center_features[label] +
                    (1 - alpha) * center
                )
                
        # Compute distances to corresponding centers
        label_centers = self.center_features[labels]
        center_loss = F.mse_loss(embeddings, label_centers)
        
        return center_loss