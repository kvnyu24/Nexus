import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from ...core.base import NexusModule

class BalancedTopKRouter(NexusModule):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float = 1.25
    ):
        super().__init__({})
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        
        # Router projection
        self.router = nn.Linear(hidden_size, num_experts)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k: int = 2
    ) -> Dict[str, torch.Tensor]:
        # Calculate routing probabilities
        router_logits = self.router(hidden_states)
        
        # Apply load balancing
        routing_weights = F.softmax(router_logits, dim=-1)
        routing_weights = self._balance_experts(routing_weights)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(
            routing_weights,
            k=min(top_k, self.num_experts),
            dim=-1
        )
        
        # Normalize selected weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return {
            "routing_weights": top_k_weights,
            "expert_indices": top_k_indices,
            "router_logits": router_logits
        }
        
    def _balance_experts(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply load balancing to prevent expert collapse"""
        # Calculate expert assignment ratios
        expert_counts = weights.sum(dim=0)
        expert_ratios = expert_counts / expert_counts.sum()
        
        # Adjust weights based on expert utilization
        balanced_weights = weights / (expert_ratios + 1e-6)
        return balanced_weights 