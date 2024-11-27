from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from ...core.base import NexusModule
from ..moe.expert_types import ConditionalExpert
from .enhanced_stable_diffusion import EnhancedStableDiffusion

class MixtureDiffusionExperts(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config
        self._validate_config(config)
        
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
        
        # Feature bank (following EnhancedReID pattern)
        self.register_buffer(
            "feature_bank",
            torch.zeros(config.get("bank_size", 10000), self.hidden_dim)
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = ["hidden_dim", "num_experts"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
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