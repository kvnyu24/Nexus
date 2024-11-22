import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule

class SensorFusionModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Following CityReconstructionModel pattern for feature extraction
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_heads = config.get("num_heads", 8)
        
        # Sensor-specific encoders
        self.lidar_encoder = nn.Sequential(
            nn.Linear(config.get("lidar_dim", 64), self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.radar_encoder = nn.Sequential(
            nn.Linear(config.get("radar_dim", 32), self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Cross-attention mechanism (similar to InteractionModule)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=config.get("dropout", 0.1)
        )
        
        # Feature fusion (following SceneUnderstandingModule pattern)
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
    def forward(
        self,
        camera_features: torch.Tensor,
        lidar_data: torch.Tensor,
        radar_data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode sensor data
        lidar_features = self.lidar_encoder(lidar_data)
        radar_features = self.radar_encoder(radar_data)
        
        # Cross-attention between modalities
        fused_features, attention_weights = self.cross_attention(
            camera_features,
            torch.cat([lidar_features, radar_features], dim=1),
            torch.cat([lidar_features, radar_features], dim=1),
            key_padding_mask=attention_mask
        )
        
        # Combine all features
        combined_features = self.fusion_layer(
            torch.cat([camera_features, lidar_features, radar_features], dim=-1)
        )
        
        return {
            "fused_features": combined_features,
            "attention_weights": attention_weights,
            "modality_features": {
                "camera": camera_features,
                "lidar": lidar_features,
                "radar": radar_features
            }
        }
