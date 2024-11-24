import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule

class ExpertModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config.get("intermediate_size", self.hidden_size * 4)
        
        # Expert FFN layers
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size),
            nn.GELU(),
            nn.Linear(self.intermediate_size, self.hidden_size),
            nn.Dropout(config.get("dropout", 0.1))
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class ExpertRouter(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.num_experts = config["num_experts"]
        self.hidden_size = config["hidden_size"]
        self.capacity_factor = config.get("capacity_factor", 1.25)
        
        # Router projection
        self.router = nn.Linear(self.hidden_size, self.num_experts)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k: int = 2
    ) -> Dict[str, torch.Tensor]:
        # Calculate routing probabilities
        router_logits = self.router(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, k=top_k, dim=-1
        )
        
        # Normalize weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return {
            "routing_weights": top_k_weights,
            "expert_indices": top_k_indices,
            "router_logits": router_logits
        }
