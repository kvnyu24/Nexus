from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ....core.base import NexusModule
from ....core.mixins import ConfigValidatorMixin, FeatureBankMixin

class EnhancedDPO(NexusModule, ConfigValidatorMixin, FeatureBankMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Validate config using mixin
        self.validate_config(config, required_keys=["hidden_dim", "model_dim"])

        # Core parameters
        self.hidden_dim = config["hidden_dim"]
        self.beta = config.get("beta", 0.1)
        self.reference_free = config.get("reference_free", True)

        # Preference head
        self.preference_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1)
        )

        # Feature banks using mixin
        bank_size = config.get("bank_size", 10000)
        self.register_feature_bank("preference", bank_size, self.hidden_dim)
        self.register_feature_bank("label", bank_size, 1)
        
    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor] = None,
        reference_rejected_logps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute DPO loss with optional reference model"""
        if self.reference_free or reference_chosen_logps is None:
            # Reference-free DPO
            chosen_rewards = self.preference_head(policy_chosen_logps)
            rejected_rewards = self.preference_head(policy_rejected_logps)
            
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        else:
            # Reference-based DPO
            chosen_diff = policy_chosen_logps - reference_chosen_logps
            rejected_diff = policy_rejected_logps - reference_rejected_logps
            
            loss = -F.logsigmoid(
                self.beta * (chosen_diff - rejected_diff)
            ).mean()
            
        return loss
        
    def forward(
        self,
        chosen_hidden_states: torch.Tensor,
        rejected_hidden_states: torch.Tensor,
        reference_model: Optional[nn.Module] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Get log probabilities
        policy_chosen_logps = self.preference_head(chosen_hidden_states)
        policy_rejected_logps = self.preference_head(rejected_hidden_states)
        
        # Get reference model outputs if provided
        reference_chosen_logps = None
        reference_rejected_logps = None
        if reference_model is not None:
            with torch.no_grad():
                reference_outputs = reference_model(
                    chosen_hidden_states=chosen_hidden_states,
                    rejected_hidden_states=rejected_hidden_states,
                    attention_mask=attention_mask
                )
                reference_chosen_logps = reference_outputs["chosen_logps"]
                reference_rejected_logps = reference_outputs["rejected_logps"]
        
        # Compute DPO loss
        loss = self.compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )
        
        # Update preference banks using mixin
        combined_features = torch.cat([chosen_hidden_states, rejected_hidden_states])
        combined_labels = torch.cat([
            torch.ones(chosen_hidden_states.size(0), 1),
            torch.zeros(rejected_hidden_states.size(0), 1)
        ]).to(chosen_hidden_states.device)
        self.update_feature_bank("preference", combined_features)
        self.update_feature_bank("label", combined_labels)
        
        return {
            "loss": loss,
            "chosen_logps": policy_chosen_logps,
            "rejected_logps": policy_rejected_logps,
            "reference_chosen_logps": reference_chosen_logps,
            "reference_rejected_logps": reference_rejected_logps
        } 