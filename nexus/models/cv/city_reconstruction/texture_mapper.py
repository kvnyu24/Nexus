import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from ....core.base import NexusModule

class TextureMapper(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Configuration
        self.hidden_dim = config.get("hidden_dim", 256)
        self.texture_dim = config.get("texture_dim", 64)
        self.num_heads = config.get("num_heads", 8)
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.texture_dim, 3, padding=1),
            nn.BatchNorm2d(self.texture_dim),
            nn.ReLU(),
            nn.Conv2d(self.texture_dim, self.texture_dim, 3, padding=1),
            nn.BatchNorm2d(self.texture_dim),
            nn.ReLU()
        )
        
        # Multi-view attention
        self.view_attention = nn.MultiheadAttention(
            embed_dim=self.texture_dim,
            num_heads=self.num_heads,
            dropout=config.get("dropout", 0.1)
        )
        
        # Texture generation
        self.texture_generator = nn.Sequential(
            nn.Conv2d(self.texture_dim, self.texture_dim * 2, 3, padding=1),
            nn.BatchNorm2d(self.texture_dim * 2),
            nn.ReLU(),
            nn.Conv2d(self.texture_dim * 2, 3, 1),  # RGB texture
            nn.Sigmoid()
        )
        
        # UV coordinate predictor
        self.uv_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 2),  # UV coordinates
            nn.Sigmoid()
        )
        
    def forward(
        self,
        mesh_features: torch.Tensor,
        image_features: Dict[str, torch.Tensor],
        camera_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        # Process image features
        processed_features = {}
        for level, features in image_features.items():
            processed_features[level] = self.feature_processor(features)
            
        # Generate UV coordinates
        uv_coords = self.uv_predictor(mesh_features)
        
        # Combine multi-view features using attention
        feature_list = list(processed_features.values())
        combined_features = feature_list[0]
        for features in feature_list[1:]:
            # Reshape for attention
            q = combined_features.flatten(2).permute(2, 0, 1)
            k = features.flatten(2).permute(2, 0, 1)
            v = k
            
            # Apply attention
            attended_features, _ = self.view_attention(q, k, v)
            attended_features = attended_features.permute(1, 2, 0).view_as(features)
            
            # Update combined features
            combined_features = combined_features + attended_features
            
        # Generate textures
        textures = self.texture_generator(combined_features)
        
        return {
            "textures": textures,
            "uv_coordinates": uv_coords,
            "processed_features": processed_features
        } 