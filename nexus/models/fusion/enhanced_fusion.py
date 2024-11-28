from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from ...core.base import NexusModule

class EnhancedFusionModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_dim = config["hidden_dim"]
        
        # Modality encoders (following EnhancedVAE pattern)
        self.modality_encoders = nn.ModuleDict({
            'text': self._build_encoder(config.get("text_dim", 768)),
            'image': self._build_encoder(config.get("image_dim", 2048)),
            'audio': self._build_encoder(config.get("audio_dim", 512))
        })
        
        # Cross-modal attention (following RAGModule pattern)
        self.cross_attention = nn.ModuleDict({
            f'{m1}_{m2}': nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=config.get("num_heads", 8),
                dropout=config.get("dropout", 0.1)
            )
            for m1 in self.modality_encoders.keys()
            for m2 in self.modality_encoders.keys()
            if m1 != m2
        })
        
        # Fusion MLP (following InteractionModule pattern)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * len(self.modality_encoders), self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
    def _build_encoder(self, input_dim: int) -> NexusModule:
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode each modality
        encoded = {
            name: encoder(inputs[name])
            for name, encoder in self.modality_encoders.items()
            if name in inputs
        }
        
        # Cross-modal attention
        attended = {}
        attention_weights = {}
        for m1 in encoded.keys():
            for m2 in encoded.keys():
                if m1 != m2:
                    key = f'{m1}_{m2}'
                    attended[key], attention_weights[key] = self.cross_attention[key](
                        encoded[m1].unsqueeze(0),
                        encoded[m2].unsqueeze(0),
                        encoded[m2].unsqueeze(0)
                    )
        
        # Fuse all modalities
        fusion_input = torch.cat([
            encoded[name] for name in sorted(encoded.keys())
        ], dim=-1)
        
        fused = self.fusion_mlp(fusion_input)
        
        return {
            "fused_features": fused,
            "modality_features": encoded,
            "attention_weights": attention_weights
        } 