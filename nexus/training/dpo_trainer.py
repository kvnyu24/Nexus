from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import NexusModule

class DPOTrainer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core parameters (following AdvancedDistillationModule pattern)
        self.beta = config.get("beta", 0.1)
        self.reference_free = config.get("reference_free", False)
        
        # Loss weights (following EnhancedSFTLoss pattern)
        self.kl_weight = config.get("kl_weight", 1.0)
        self.reward_weight = config.get("reward_weight", 0.1)
        
        # Optional components
        self.use_reward_model = config.get("use_reward_model", True)
        if self.use_reward_model:
            self.reward_model = EnhancedRewardModel(config)
            
    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor] = None,
        reference_rejected_logps: Optional[torch.Tensor] = None,
        reward_chosen: Optional[torch.Tensor] = None,
        reward_rejected: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Compute policy losses
        if self.reference_free:
            policy_loss = -F.logsigmoid(policy_chosen_logps - policy_rejected_logps)
        else:
            chosen_diff = policy_chosen_logps - reference_chosen_logps
            rejected_diff = policy_rejected_logps - reference_rejected_logps
            policy_loss = -F.logsigmoid(self.beta * (chosen_diff - rejected_diff))
            
        if attention_mask is not None:
            policy_loss = policy_loss * attention_mask
            
        policy_loss = policy_loss.mean()
        
        losses = {"policy_loss": policy_loss}
        
        # Add reward losses if available
        if reward_chosen is not None and reward_rejected is not None:
            reward_loss = F.mse_loss(
                reward_chosen - reward_rejected,
                torch.ones_like(reward_chosen)
            )
            losses["reward_loss"] = reward_loss
            losses["total_loss"] = (
                self.kl_weight * policy_loss +
                self.reward_weight * reward_loss
            )
        else:
            losses["total_loss"] = policy_loss
            
        return losses