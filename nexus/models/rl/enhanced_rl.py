from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from ...core.base import NexusModule

class EnhancedRLModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core components
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)
        
        # State encoder (following EnhancedT5 pattern)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Policy network (similar to BehaviorPredictionModule)
        self.policy_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        
        # Value network (following AlphaFold pattern)
        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Feature bank (following EnhancedReID pattern)
        self.register_buffer(
            "experience_bank",
            torch.zeros(
                config.get("bank_size", 10000),
                self.hidden_dim
            )
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
    def forward(
        self,
        states: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode states
        state_features = self.state_encoder(states)
        
        # Generate policy and value
        action_logits = self.policy_net(state_features)
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))
        
        value = self.value_net(state_features)
        
        return {
            "action_logits": action_logits,
            "value": value,
            "state_features": state_features
        } 