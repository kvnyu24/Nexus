import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule

class ConditionalExpert(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_dim = config["hidden_dim"]
        self.intermediate_dim = config.get("intermediate_dim", self.hidden_dim * 4)
        self.num_conditions = config.get("num_conditions", 4)
        
        # Conditional layers
        self.condition_embedding = nn.Embedding(self.num_conditions, self.hidden_dim)
        
        # Expert network with conditional modulation
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.ff1 = nn.Linear(self.hidden_dim, self.intermediate_dim)
        self.ff2 = nn.Linear(self.intermediate_dim, self.hidden_dim)
        
        # Conditional scaling factors
        self.scale_generator = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        condition_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        
        # Apply conditional modulation
        if condition_ids is not None:
            condition_emb = self.condition_embedding(condition_ids)
            scale, shift = self.scale_generator(condition_emb).chunk(2, dim=-1)
            hidden_states = hidden_states * scale + shift
        
        # FFN with GELU
        hidden_states = self.ff1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)
        hidden_states = self.ff2(hidden_states)
        
        return residual + hidden_states
