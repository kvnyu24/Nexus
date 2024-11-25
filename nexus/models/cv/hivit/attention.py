import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ....core.base import NexusModule

class MultiScaleAttention(NexusModule):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__({})
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention for each scale
        self.scale_attention = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dim // (2 ** i),
                num_heads // (2 ** i),
                dropout=dropout
            ) for i in range(3)
        ])
        
        # Cross-scale attention
        self.cross_scale_attention = nn.ModuleDict({
            f'scale_{i}_{j}': nn.Linear(
                hidden_dim // (2 ** i),
                hidden_dim // (2 ** j)
            )
            for i in range(3)
            for j in range(3)
            if i != j
        })
        
        # Output projection for each scale
        self.output_projs = nn.ModuleList([
            nn.Linear(
                hidden_dim // (2 ** i),
                hidden_dim // (2 ** i)
            ) for i in range(3)
        ])
        
    def forward(
        self,
        scale_features: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        attention_weights = []
        output_features = []
        
        # Process each scale
        for scale_idx, (features, attn) in enumerate(
            zip(scale_features, self.scale_attention)
        ):
            # Self attention
            attn_output, attn_weights = attn(
                features,
                features,
                features,
                key_padding_mask=attention_mask
            )
            attention_weights.append(attn_weights)
            
            # Cross-scale interactions
            cross_scale_outputs = [attn_output]
            for other_idx, other_features in enumerate(scale_features):
                if other_idx != scale_idx:
                    # Project other scale to current scale
                    proj_name = f'scale_{other_idx}_{scale_idx}'
                    cross_output = self.cross_scale_attention[proj_name](
                        other_features
                    )
                    cross_scale_outputs.append(cross_output)
            
            # Combine outputs
            combined = sum(cross_scale_outputs) / len(cross_scale_outputs)
            output = self.output_projs[scale_idx](combined)
            output_features.append(output)
        
        return {
            "scale_features": output_features,
            "attention_weights": attention_weights
        } 