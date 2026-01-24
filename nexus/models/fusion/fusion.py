from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule
from ...core.mixins import FeatureBankMixin
from ...visualization.hierarchical import HierarchicalVisualizer


class FusionModule(FeatureBankMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core dimensions and configuration
        self.hidden_dim = config["hidden_dim"]
        self.num_heads = config.get("num_heads", 8)
        self.dropout = config.get("dropout", 0.1)
        self.use_layer_scale = config.get("use_layer_scale", True)
        self.layer_scale_init = config.get("layer_scale_init", 1e-5)
        self.fusion_layers = config.get("fusion_layers", 3)
        self.track_gradients = config.get("track_gradients", False)
        
        # Modality encoders with enhanced residual connections and normalization
        self.modality_encoders = nn.ModuleDict({
            'text': self._build_encoder(config.get("text_dim", 768)),
            'image': self._build_encoder(config.get("image_dim", 2048)), 
            'audio': self._build_encoder(config.get("audio_dim", 512)),
            'graph': self._build_encoder(config.get("graph_dim", 256))
        })
        
        # Cross-modal attention with enhanced gating and relative position encoding
        self.cross_attention = nn.ModuleDict({
            f'{m1}_{m2}': nn.ModuleDict({
                'attention': nn.MultiheadAttention(
                    embed_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    batch_first=True
                ),
                'gate': nn.Sequential(
                    nn.LayerNorm(self.hidden_dim * 2),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.Sigmoid()
                ),
                'rel_pos_bias': nn.Parameter(torch.zeros(self.num_heads, 32, 32)),
                'uncertainty': nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.num_heads)
                )
            })
            for m1 in self.modality_encoders.keys()
            for m2 in self.modality_encoders.keys()
            if m1 != m2
        })
        
        # Enhanced modality-specific context layers with stochastic depth
        self.context_layers = nn.ModuleDict({
            name: nn.ModuleList([
                nn.Sequential(
                    nn.TransformerEncoderLayer(
                        d_model=self.hidden_dim,
                        nhead=self.num_heads,
                        dim_feedforward=self.hidden_dim * 4,
                        dropout=self.dropout,
                        activation=F.gelu,
                        batch_first=True,
                        norm_first=True
                    ),
                    nn.LayerNorm(self.hidden_dim)
                )
                for _ in range(2)
            ])
            for name in self.modality_encoders.keys()
        })
        
        # Enhanced fusion network with adaptive layer scaling and skip connections
        self.fusion_net = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.hidden_dim * len(self.modality_encoders)),
                nn.Linear(self.hidden_dim * len(self.modality_encoders), self.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim)
            )
            for _ in range(self.fusion_layers)
        ])
        
        if self.use_layer_scale:
            self.layer_scale_params = nn.ParameterList([
                nn.Parameter(torch.ones(1, 1, self.hidden_dim) * self.layer_scale_init)
                for _ in range(self.fusion_layers)
            ])
        
        # Enhanced modality-specific confidence estimators with uncertainty
        self.confidence_estimators = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, self.num_heads + 1),  # +1 for overall confidence
                nn.Sigmoid()
            )
            for name in self.modality_encoders.keys()
        })
        
        # Enhanced output projection with residual connection and normalization
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Memory bank for storing important cross-modal patterns using FeatureBankMixin
        self.memory_size = config.get("memory_size", 1024)
        self.register_feature_bank("feature", self.memory_size, self.hidden_dim)
        
        # Visualization support
        self.visualizer = HierarchicalVisualizer(config)
        
    def _build_encoder(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[Dict[str, torch.Tensor]] = None,
        return_intermediates: bool = False
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Enhanced input validation and preprocessing
        valid_modalities = [name for name in self.modality_encoders.keys() if name in inputs]
        if len(valid_modalities) < 2:
            raise ValueError(f"At least 2 modalities required for fusion, got {len(valid_modalities)}")
            
        batch_size = next(iter(inputs.values())).size(0)
        device = next(iter(inputs.values())).device
        
        intermediates = {} if return_intermediates else None
            
        # Enhanced encoding with gradient tracking
        encoded = {}
        for name in valid_modalities:
            encoded[name] = self.modality_encoders[name](inputs[name])
            if self.track_gradients:
                encoded[name].register_hook(
                    lambda grad, n=name: self.visualizer.log_gradient(f"encoder_{n}", grad)
                )
        
        if return_intermediates:
            intermediates["encoded"] = encoded
        
        # Enhanced contextualization with stochastic depth and residual connections
        contextualized = {}
        for name, feat in encoded.items():
            x = feat
            for i, layer in enumerate(self.context_layers[name]):
                if self.training and torch.rand(1).item() < 0.1:
                    continue
                x = layer(x) + x
                if self.track_gradients:
                    x.register_hook(
                        lambda grad, n=name, l=i: self.visualizer.log_gradient(f"context_{n}_{l}", grad)
                    )
            contextualized[name] = x
            
        if return_intermediates:
            intermediates["contextualized"] = contextualized
        
        # Enhanced confidence estimation with per-head uncertainty
        confidence_scores = {}
        uncertainty_scores = {}
        for name, feat in contextualized.items():
            scores = self.confidence_estimators[name](feat)
            confidence_scores[name] = scores[:, 0]  # Overall confidence
            uncertainty_scores[name] = scores[:, 1:]  # Per-head uncertainty
        
        # Enhanced cross-modal attention with uncertainty-weighted gating
        attended = {}
        attention_weights = {}
        for m1 in valid_modalities:
            for m2 in valid_modalities:
                if m1 != m2:
                    key = f'{m1}_{m2}'
                    
                    # Enhanced relative position bias
                    rel_pos_bias = self.cross_attention[key]['rel_pos_bias']
                    rel_pos_bias = rel_pos_bias[:, :batch_size, :batch_size]
                    
                    # Uncertainty-aware attention
                    att_output, att_weights = self.cross_attention[key]['attention'](
                        contextualized[m1],
                        contextualized[m2],
                        contextualized[m2],
                        key_padding_mask=attention_mask.get(m2) if attention_mask else None,
                        attn_mask=rel_pos_bias
                    )
                    
                    # Enhanced adaptive gating with uncertainty
                    gate_input = torch.cat([contextualized[m1], att_output], dim=-1)
                    gate = self.cross_attention[key]['gate'](gate_input)
                    
                    # Apply uncertainty-weighted gating
                    uncertainty = self.cross_attention[key]['uncertainty'](att_output)
                    conf_weight = confidence_scores[m2].unsqueeze(-1) * (1 - uncertainty.mean(-1, keepdim=True))
                    attended[key] = gate * att_output * conf_weight
                    attention_weights[key] = att_weights
                    
                    if self.track_gradients:
                        attended[key].register_hook(
                            lambda grad, k=key: self.visualizer.log_gradient(f"attention_{k}", grad)
                        )
        
        if return_intermediates:
            intermediates["attended"] = attended
            intermediates["attention_weights"] = attention_weights
        
        # Enhanced progressive fusion with adaptive scaling and residual connections
        fusion_input = torch.cat([
            contextualized[name] for name in sorted(valid_modalities)
        ], dim=-1)
        
        fused = fusion_input
        for i, fusion_layer in enumerate(self.fusion_net):
            layer_output = fusion_layer(fused)
            if self.use_layer_scale:
                layer_output = layer_output * self.layer_scale_params[i]
            fused = layer_output + fused[:, :self.hidden_dim]
            
            if self.track_gradients:
                fused.register_hook(
                    lambda grad, l=i: self.visualizer.log_gradient(f"fusion_{l}", grad)
                )
        
        # Enhanced output projection with memory update
        output = self.output_proj(fused) + fused
        
        # Update memory bank with important patterns using FeatureBankMixin
        if self.training:
            # Handle sequence outputs by taking mean across sequence dimension
            if output.dim() == 3:
                features_to_store = output.mean(dim=1)
            else:
                features_to_store = output
            self.update_feature_bank("feature", features_to_store)
        
        return {
            "fused_features": output,
            "modality_features": contextualized,
            "attention_weights": attention_weights,
            "attended_features": attended,
            "confidence_scores": confidence_scores,
            "uncertainty_scores": uncertainty_scores,
            "intermediates": intermediates if return_intermediates else None
        }