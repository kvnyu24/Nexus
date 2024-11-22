import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule

class DecisionMakingModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_dim = config.get("hidden_dim", 256)
        
        # Feature fusion (similar to CityReconstructionModel pattern)
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Decision heads (following BaseRCNN pattern)
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, config.get("num_actions", 9))
        )
        
        # Safety assessment (similar to existing safety patterns)
        self.safety_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Feature memory bank (following EnhancedReID pattern)
        self.register_buffer(
            "decision_memory",
            torch.zeros(
                config.get("memory_size", 1000),
                self.hidden_dim
            )
        )
        self.register_buffer("memory_ptr", torch.zeros(1, dtype=torch.long))
    
    def forward(
        self,
        perception_features: torch.Tensor,
        behavior_features: torch.Tensor,
        scene_context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Fuse features
        combined_features = torch.cat([
            perception_features,
            behavior_features,
            scene_context
        ], dim=-1)
        
        fused_features = self.feature_fusion(combined_features)
        
        # Generate decisions
        actions = self.action_head(fused_features)
        safety_score = self.safety_head(fused_features)
        
        return {
            "actions": actions,
            "safety_score": safety_score,
            "decision_features": fused_features
        } 