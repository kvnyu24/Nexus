import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule
from ...core.mixins import ConfigValidatorMixin
from .gating import TopKBalancedGate
from .expert_types import ConditionalExpert

class EnhancedMoELayer(ConfigValidatorMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Validate config
        self.validate_config(config, required_keys=["num_experts", "hidden_dim"])

        # Core components
        self.gate = TopKBalancedGate(config)
        self.experts = nn.ModuleList([
            ConditionalExpert(config) for _ in range(config["num_experts"])
        ])

        # Optional components
        self.dropout = nn.Dropout(config.get("dropout", 0.1))
        self.layer_norm = nn.LayerNorm(config["hidden_dim"])

        # Expert capacity control
        self.capacity_factor = config.get("capacity_factor", 1.25)
        self.drop_tokens = config.get("drop_tokens", True)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        condition_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Gate tokens to experts
        gate_outputs = self.gate(hidden_states, return_auxiliary_loss=True)
        
        # Process tokens through experts
        expert_outputs = torch.zeros_like(hidden_states)
        total_tokens = 0
        
        for idx, expert in enumerate(self.experts):
            # Get tokens routed to this expert
            expert_mask = gate_outputs["expert_indices"] == idx
            if expert_mask.any():
                tokens = hidden_states[expert_mask]
                processed = expert(tokens, condition_ids[expert_mask] if condition_ids is not None else None)
                expert_outputs[expert_mask] = processed
                total_tokens += expert_mask.sum()
        
        # Combine expert outputs
        combined_output = torch.sum(
            expert_outputs * gate_outputs["expert_weights"].unsqueeze(-1),
            dim=1
        )
        
        # Apply dropout and residual connection
        output = hidden_states + self.dropout(combined_output)
        
        return {
            "hidden_states": output,
            "gate_loss": gate_outputs["balance_loss"],
            "expert_counts": total_tokens,
            "routing_weights": gate_outputs["expert_weights"]
        }
