import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule
from .router import BalancedTopKRouter

class ExpertTransformerLayer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_size = config["hidden_size"]
        self.num_experts = config["num_experts"]
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1)
        )
        
        # Expert router
        self.router = BalancedTopKRouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            capacity_factor=config.get("capacity_factor", 1.25)
        )
        
        # Expert FFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, 4 * self.hidden_size),
                nn.GELU(),
                nn.Linear(4 * self.hidden_size, self.hidden_size),
                nn.Dropout(config.get("dropout", 0.1))
            ) for _ in range(self.num_experts)
        ])
        
        # Layer norms
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.expert_norm = nn.LayerNorm(self.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Self-attention
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        attention_output, _ = self.self_attention(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=attention_mask
        )
        hidden_states = residual + attention_output
        
        # Expert routing
        residual = hidden_states
        hidden_states = self.expert_norm(hidden_states)
        
        routing_outputs = self.router(hidden_states)
        
        # Process through experts
        expert_output = torch.zeros_like(hidden_states)
        for idx, expert in enumerate(self.experts):
            expert_mask = routing_outputs["expert_indices"] == idx
            if expert_mask.any():
                tokens = hidden_states[expert_mask]
                processed = expert(tokens)
                expert_output[expert_mask] = processed
        
        # Combine expert outputs with routing weights
        combined_output = torch.sum(
            expert_output * routing_outputs["routing_weights"].unsqueeze(-1),
            dim=1
        )
        
        hidden_states = residual + combined_output
        
        return {
            "hidden_states": hidden_states,
            "router_logits": routing_outputs["router_logits"],
            "expert_patterns": routing_outputs["expert_indices"],
            "routing_weights": routing_outputs["routing_weights"]
        } 