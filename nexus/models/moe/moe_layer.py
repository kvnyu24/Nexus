import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule
from .expert import ExpertModule, ExpertRouter

class MoELayer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config
        self._validate_config(config)
        
        self.num_experts = config["num_experts"]
        self.hidden_size = config["hidden_size"]
        self.top_k = config.get("top_k", 2)
        
        # Initialize experts
        self.experts = nn.ModuleList([
            ExpertModule(config) for _ in range(self.num_experts)
        ])
        
        # Initialize router
        self.router = ExpertRouter(config)
        
        # Layer norm for input
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required_keys = ["num_experts", "hidden_size"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Normalize input
        normalized_states = self.layer_norm(hidden_states)
        
        # Get routing weights and expert assignments
        routing_outputs = self.router(
            normalized_states,
            top_k=self.top_k
        )
        
        # Process inputs through experts
        expert_outputs = torch.zeros_like(hidden_states)
        for idx, expert in enumerate(self.experts):
            # Get tokens routed to this expert
            expert_mask = routing_outputs["expert_indices"] == idx
            if expert_mask.any():
                tokens = normalized_states[expert_mask]
                processed = expert(tokens)
                expert_outputs[expert_mask] = processed
        
        # Combine expert outputs with routing weights
        combined_output = torch.sum(
            expert_outputs * routing_outputs["routing_weights"].unsqueeze(-1),
            dim=1
        )
        
        # Residual connection
        output = hidden_states + combined_output
        
        return {
            "hidden_states": output,
            "routing_weights": routing_outputs["routing_weights"],
            "expert_indices": routing_outputs["expert_indices"]
        }
