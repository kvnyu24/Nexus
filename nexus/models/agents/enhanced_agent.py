import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule
from .memory import AgentMemoryStream

class EnhancedSocialAgent(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_dim = config["hidden_dim"]
        
        # Core components
        self.state_encoder = nn.Linear(config["state_dim"], self.hidden_dim)
        self.memory_stream = AgentMemoryStream(config)
        
        # Planning components (similar to BehaviorPredictionModule)
        self.plan_generator = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, config["num_actions"])
        )
        
        # Reflection mechanism (following StateTransitionModule pattern)
        self.reflection_module = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
    def forward(
        self,
        state: torch.Tensor,
        social_context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode state
        encoded_state = self.state_encoder(state)
        
        # Process memory
        memory_outputs = self.memory_stream(
            encoded_state,
            context=social_context
        )
        
        # Generate plan
        plan_input = torch.cat([
            encoded_state,
            memory_outputs["memory_encoding"],
            memory_outputs["episodic_memory"].mean(dim=0).expand(encoded_state.size(0), -1)
        ], dim=-1)
        
        action_logits = self.plan_generator(plan_input)
        
        # Reflection
        reflection_state = self.reflection_module(
            torch.cat([
                encoded_state,
                memory_outputs["semantic_memory"].mean(dim=0).expand(encoded_state.size(0), -1)
            ], dim=-1)
        )
        
        return {
            "action_logits": action_logits,
            "reflection_state": reflection_state,
            "memory_outputs": memory_outputs
        } 