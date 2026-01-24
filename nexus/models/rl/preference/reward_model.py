from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ....core.base import NexusModule
from ....core.mixins import ConfigValidatorMixin, FeatureBankMixin

class EnhancedRewardModel(NexusModule, ConfigValidatorMixin, FeatureBankMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Validate config using mixin
        self.validate_config(config, required_keys=["input_dim", "hidden_dim"])

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

        # Feature bank using mixin
        bank_size = config.get("bank_size", 10000)
        self.register_feature_bank("preference", bank_size, self.hidden_dim)
        
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
        
        # Update feature bank using mixin
        self.update_feature_bank("preference", features)
        
        return {
            "rewards": mean_reward,
            "uncertainty": reward_uncertainty,
            "features": features,
            "raw_rewards": rewards
        } 