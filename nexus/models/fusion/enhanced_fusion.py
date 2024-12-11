from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule

class FusionModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_dim = config["hidden_dim"]
        self.num_heads = config.get("num_heads", 8)
        self.dropout = config.get("dropout", 0.1)
        
        # Modality encoders with residual connections and layer norm
        self.modality_encoders = nn.ModuleDict({
            'text': self._build_encoder(config.get("text_dim", 768)),
            'image': self._build_encoder(config.get("image_dim", 2048)), 
            'audio': self._build_encoder(config.get("audio_dim", 512))
        })
        
        # Cross-modal attention with gating
        self.cross_attention = nn.ModuleDict({
            f'{m1}_{m2}': nn.ModuleDict({
                'attention': nn.MultiheadAttention(
                    embed_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout
                ),
                'gate': nn.Sequential(
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.Sigmoid()
                )
            })
            for m1 in self.modality_encoders.keys()
            for m2 in self.modality_encoders.keys()
            if m1 != m2
        })
        
        # Modality-specific context layers
        self.context_layers = nn.ModuleDict({
            name: nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=self.dropout
            )
            for name in self.modality_encoders.keys()
        })
        
        # Fusion network with skip connections
        self.fusion_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * len(self.modality_encoders), self.hidden_dim * 2),
                nn.LayerNorm(self.hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            )
            for _ in range(2)  # Multiple fusion layers
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
    def _build_encoder(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        # Input validation
        valid_modalities = [name for name in self.modality_encoders.keys() if name in inputs]
        if len(valid_modalities) < 2:
            raise ValueError("At least 2 modalities required for fusion")
            
        # Encode each modality
        encoded = {
            name: self.modality_encoders[name](inputs[name])
            for name in valid_modalities
        }
        
        # Apply context layers
        contextualized = {
            name: self.context_layers[name](feat.unsqueeze(0)).squeeze(0)
            for name, feat in encoded.items()
        }
        
        # Cross-modal attention with gating
        attended = {}
        attention_weights = {}
        for m1 in valid_modalities:
            for m2 in valid_modalities:
                if m1 != m2:
                    key = f'{m1}_{m2}'
                    # Apply attention
                    att_output, att_weights = self.cross_attention[key]['attention'](
                        contextualized[m1].unsqueeze(0),
                        contextualized[m2].unsqueeze(0),
                        contextualized[m2].unsqueeze(0),
                        key_padding_mask=attention_mask.get(m2) if attention_mask else None
                    )
                    
                    # Compute gate values
                    gate_input = torch.cat([
                        contextualized[m1],
                        att_output.squeeze(0)
                    ], dim=-1)
                    gate = self.cross_attention[key]['gate'](gate_input)
                    
                    # Apply gating
                    attended[key] = gate * att_output.squeeze(0)
                    attention_weights[key] = att_weights
        
        # Progressive fusion with residual connections
        fusion_input = torch.cat([
            contextualized[name] for name in sorted(valid_modalities)
        ], dim=-1)
        
        fused = fusion_input
        for fusion_layer in self.fusion_net:
            fused = fusion_layer(fused) + fused[:, :self.hidden_dim]
            
        # Final projection
        fused = self.output_proj(fused)
        
        return {
            "fused_features": fused,
            "modality_features": contextualized,
            "attention_weights": attention_weights,
            "attended_features": attended
        }