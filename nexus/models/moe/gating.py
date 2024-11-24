import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from ...core.base import NexusModule

class TopKBalancedGate(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.num_experts = config["num_experts"]
        self.hidden_dim = config["hidden_dim"]
        self.top_k = config.get("top_k", 2)
        self.capacity_factor = config.get("capacity_factor", 1.25)
        
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.num_experts)
        )
        
        # Load balancing parameters
        self.balance_alpha = config.get("balance_alpha", 0.01)
        self.register_buffer("expert_counts", torch.zeros(self.num_experts))
        
    def compute_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Compute load balancing auxiliary loss"""
        # Calculate mean probability for each expert
        mean_probs = router_probs.mean(dim=0)
        # Penalize deviation from uniform distribution
        balance_loss = torch.sum(mean_probs * torch.log(mean_probs + 1e-10)) * self.num_experts
        return balance_loss * self.balance_alpha
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        return_auxiliary_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        batch_size = hidden_states.size(0)
        
        # Compute gate logits and probabilities
        gate_logits = self.gate(hidden_states)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(gate_probs, k=self.top_k, dim=-1)
        
        # Normalize probabilities
        top_k_probs_normalized = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        outputs = {
            "gate_logits": gate_logits,
            "expert_weights": top_k_probs_normalized,
            "expert_indices": top_k_indices
        }
        
        if return_auxiliary_loss:
            outputs["balance_loss"] = self.compute_balance_loss(gate_probs)
            
        return outputs
