from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ....core.base import NexusModule

class EnhancedRewardModel(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core dimensions
        self.hidden_dim = config["hidden_dim"]
        
        # Feature extractor (following EnhancedRLModule pattern)
        self.feature_extractor = nn.Sequential(
            nn.Linear(config["input_dim"], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Reward head (following ActorCritic value head pattern)
        self.reward_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.LayerNorm(self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 1)
            ) for _ in range(config.get("num_reward_heads", 3))
        ])
        
        # Feature bank (following EnhancedReID pattern)
        self.register_buffer(
            "preference_bank",
            torch.zeros(config.get("bank_size", 10000), self.hidden_dim)
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = ["input_dim", "hidden_dim"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
    def update_preference_bank(self, features: torch.Tensor):
        """Update preference bank following EnhancedReID pattern"""
        batch_size = features.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.preference_bank.size(0):
            ptr = 0
            
        self.preference_bank[ptr:ptr + batch_size] = features.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.preference_bank.size(0)
        
    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.feature_extractor(inputs)
        
        # Get rewards from ensemble
        rewards = torch.cat([head(features) for head in self.reward_head], dim=-1)
        
        # Calculate statistics
        mean_reward = rewards.mean(dim=-1, keepdim=True)
        reward_uncertainty = rewards.std(dim=-1, keepdim=True)
        
        # Update feature bank
        self.update_preference_bank(features)
        
        return {
            "rewards": mean_reward,
            "uncertainty": reward_uncertainty,
            "features": features,
            "raw_rewards": rewards
        } 