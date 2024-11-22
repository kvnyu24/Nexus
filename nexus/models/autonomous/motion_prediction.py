import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule

class MotionEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.hidden_dim = config.get("hidden_dim", 256)
        
        # Temporal encoding (similar to TemporalReIDModule)
        self.temporal_encoder = nn.GRU(
            input_size=config.get("motion_dim", 6),
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Attention mechanism (similar to EnhancedRAGModule)
        self.motion_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1)
        )

class MotionPredictionModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core components
        self.motion_encoder = MotionEncoder(config)
        
        # Prediction heads (similar to BaseRCNN pattern)
        self.trajectory_head = nn.Sequential(
            nn.Linear(config.get("hidden_dim", 256), 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, config.get("num_trajectories", 6) * 2)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.get("hidden_dim", 256), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, config.get("num_trajectories", 6))
        )
        
    def forward(
        self,
        motion_history: torch.Tensor,
        scene_context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode motion
        motion_features = self.motion_encoder(motion_history)
        
        # Apply attention with scene context if available
        if scene_context is not None:
            attended_features, attention_weights = self.motion_attention(
                motion_features,
                scene_context,
                scene_context,
                key_padding_mask=attention_mask
            )
        else:
            attended_features = motion_features
            attention_weights = None
        
        # Generate predictions
        trajectories = self.trajectory_head(attended_features)
        uncertainties = self.uncertainty_head(attended_features)
        
        return {
            "predicted_trajectories": trajectories.view(-1, self.num_trajectories, 2),
            "trajectory_uncertainties": uncertainties,
            "attention_weights": attention_weights,
            "motion_features": motion_features
        } 