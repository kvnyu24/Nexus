import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule
from .moe_layer import MoELayer

class MoETransformer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.num_heads = config.get("num_heads", 8)
        
        # Transformer layers with MoE
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(
                    self.hidden_size,
                    self.num_heads,
                    dropout=config.get("dropout", 0.1)
                ),
                'moe': MoELayer(config),
                'layer_norm1': nn.LayerNorm(self.hidden_size),
                'layer_norm2': nn.LayerNorm(self.hidden_size)
            }) for _ in range(self.num_layers)
        ])
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        all_hidden_states = []
        all_routing_weights = []
        
        for layer in self.layers:
            # Self-attention
            normed_states = layer['layer_norm1'](hidden_states)
            attention_output, _ = layer['attention'](
                normed_states,
                normed_states,
                normed_states,
                key_padding_mask=attention_mask
            )
            hidden_states = hidden_states + attention_output
            
            # MoE FFN
            moe_outputs = layer['moe'](hidden_states, attention_mask)
            hidden_states = moe_outputs["hidden_states"]
            
            all_hidden_states.append(hidden_states)
            all_routing_weights.append(moe_outputs["routing_weights"])
        
        return {
            "hidden_states": hidden_states,
            "all_hidden_states": all_hidden_states,
            "routing_weights": all_routing_weights
        }
