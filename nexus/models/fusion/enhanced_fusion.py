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
        self.use_layer_scale = config.get("use_layer_scale", True)
        self.layer_scale_init = config.get("layer_scale_init", 1e-5)
        self.fusion_layers = config.get("fusion_layers", 3)
        
        # Modality encoders with residual connections and layer norm
        self.modality_encoders = nn.ModuleDict({
            'text': self._build_encoder(config.get("text_dim", 768)),
            'image': self._build_encoder(config.get("image_dim", 2048)), 
            'audio': self._build_encoder(config.get("audio_dim", 512))
        })
        
        # Cross-modal attention with gating and relative position encoding
        self.cross_attention = nn.ModuleDict({
            f'{m1}_{m2}': nn.ModuleDict({
                'attention': nn.MultiheadAttention(
                    embed_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    batch_first=True
                ),
                'gate': nn.Sequential(
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.Sigmoid()
                ),
                'rel_pos_bias': nn.Parameter(torch.zeros(self.num_heads, 32, 32))
            })
            for m1 in self.modality_encoders.keys()
            for m2 in self.modality_encoders.keys()
            if m1 != m2
        })
        
        # Modality-specific context layers with stochastic depth
        self.context_layers = nn.ModuleDict({
            name: nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.hidden_dim * 4,
                    dropout=self.dropout,
                    activation=F.gelu,
                    batch_first=True,
                    norm_first=True
                )
                for _ in range(2)
            ])
            for name in self.modality_encoders.keys()
        })
        
        # Fusion network with adaptive layer scaling
        self.fusion_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * len(self.modality_encoders), self.hidden_dim * 2),
                nn.LayerNorm(self.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            )
            for _ in range(self.fusion_layers)
        ])
        
        if self.use_layer_scale:
            self.layer_scale_params = nn.ParameterList([
                nn.Parameter(torch.ones(1, 1, self.hidden_dim) * self.layer_scale_init)
                for _ in range(self.fusion_layers)
            ])
        
        # Modality-specific confidence estimators
        self.confidence_estimators = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.LayerNorm(self.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            for name in self.modality_encoders.keys()
        })
        
        # Output projection with skip connection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
    def _build_encoder(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        # Input validation and preprocessing
        valid_modalities = [name for name in self.modality_encoders.keys() if name in inputs]
        if len(valid_modalities) < 2:
            raise ValueError("At least 2 modalities required for fusion")
            
        batch_size = next(iter(inputs.values())).size(0)
        device = next(iter(inputs.values())).device
            
        # Encode each modality
        encoded = {
            name: self.modality_encoders[name](inputs[name])
            for name in valid_modalities
        }
        
        # Apply context layers with stochastic depth
        contextualized = {}
        for name, feat in encoded.items():
            x = feat
            for layer in self.context_layers[name]:
                if self.training and torch.rand(1).item() < 0.1:  # Stochastic depth
                    continue
                x = layer(x)
            contextualized[name] = x
        
        # Estimate modality-specific confidence
        confidence_scores = {
            name: self.confidence_estimators[name](feat).squeeze(-1)
            for name, feat in contextualized.items()
        }
        
        # Cross-modal attention with gating and relative position bias
        attended = {}
        attention_weights = {}
        for m1 in valid_modalities:
            for m2 in valid_modalities:
                if m1 != m2:
                    key = f'{m1}_{m2}'
                    
                    # Add relative position bias
                    rel_pos_bias = self.cross_attention[key]['rel_pos_bias']
                    rel_pos_bias = rel_pos_bias[:, :batch_size, :batch_size]
                    
                    # Apply attention
                    att_output, att_weights = self.cross_attention[key]['attention'](
                        contextualized[m1],
                        contextualized[m2],
                        contextualized[m2],
                        key_padding_mask=attention_mask.get(m2) if attention_mask else None,
                        attn_mask=rel_pos_bias
                    )
                    
                    # Compute adaptive gate values
                    gate_input = torch.cat([
                        contextualized[m1],
                        att_output
                    ], dim=-1)
                    gate = self.cross_attention[key]['gate'](gate_input)
                    
                    # Apply gating weighted by confidence scores
                    conf_weight = confidence_scores[m2].unsqueeze(-1)
                    attended[key] = gate * att_output * conf_weight
                    attention_weights[key] = att_weights
        
        # Progressive fusion with adaptive layer scaling and residual connections
        fusion_input = torch.cat([
            contextualized[name] for name in sorted(valid_modalities)
        ], dim=-1)
        
        fused = fusion_input
        for i, fusion_layer in enumerate(self.fusion_net):
            layer_output = fusion_layer(fused)
            if self.use_layer_scale:
                layer_output = layer_output * self.layer_scale_params[i]
            fused = layer_output + fused[:, :self.hidden_dim]
            
        # Final projection with skip connection
        output = self.output_proj(fused) + fused
        
        return {
            "fused_features": output,
            "modality_features": contextualized,
            "attention_weights": attention_weights,
            "attended_features": attended,
            "confidence_scores": confidence_scores
        }