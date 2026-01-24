from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from ...core.base import NexusModule
from ...core.mixins import ConfigValidatorMixin, FeatureBankMixin
from ..moe.expert_types import ConditionalExpert
from .enhanced_stable_diffusion import EnhancedStableDiffusion

class MixtureDiffusionExperts(ConfigValidatorMixin, FeatureBankMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config using ConfigValidatorMixin
        self.validate_config(config, required_keys=["hidden_dim", "num_experts"])
        
        # Core dimensions
        self.hidden_dim = config["hidden_dim"]
        self.num_experts = config["num_experts"]
        
        # Expert diffusion models
        self.experts = nn.ModuleList([
            EnhancedStableDiffusion(config)
            for _ in range(self.num_experts)
        ])
        
        # Expert routing
        self.router = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_experts)
        )
        
        # Feature bank using FeatureBankMixin
        self.bank_size = config.get("bank_size", 10000)
        self.register_feature_bank("feature", self.bank_size, self.hidden_dim)
                
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Route input to experts
        routing_weights = self.router(x).softmax(dim=-1)
        
        # Gather expert outputs
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x, condition, timesteps)
            weighted_out = {
                k: v * routing_weights[:, i:i+1]
                for k, v in expert_out.items()
                if isinstance(v, torch.Tensor)
            }
            expert_outputs.append(weighted_out)
            
        # Combine expert outputs
        combined_output = {
            k: sum(exp[k] for exp in expert_outputs)
            for k in expert_outputs[0].keys()
        }
        
        return {
            **combined_output,
            "routing_weights": routing_weights
        } 