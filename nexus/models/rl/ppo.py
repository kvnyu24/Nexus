import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List, Union
from ...core.base import NexusModule
from ...core.mixins import ConfigValidatorMixin, FeatureBankMixin
import numpy as np

class ActorCritic(NexusModule, ConfigValidatorMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Validate config using mixin
        self.validate_config(config, required_keys=["state_dim", "action_dim"])

        # Core dimensions
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)

        # Additional validation using mixin
        if self.hidden_dim < 32:
            raise ValueError("hidden_dim must be at least 32")
        if config.get("num_value_heads", 3) < 2:
            raise ValueError("num_value_heads must be at least 2")

        # Enhanced feature extractor with residual connections
        self.features = nn.ModuleDict({
            'input': nn.Linear(self.state_dim, self.hidden_dim),
            'norm1': nn.LayerNorm(self.hidden_dim),
            'hidden': nn.Linear(self.hidden_dim, self.hidden_dim),
            'norm2': nn.LayerNorm(self.hidden_dim),
            'residual': nn.Linear(self.state_dim, self.hidden_dim, bias=False)
        })

        # Advanced policy head with uncertainty estimation
        self.policy = nn.ModuleDict({
            'shared': nn.Linear(self.hidden_dim, self.hidden_dim),
            'mean': nn.Linear(self.hidden_dim, self.action_dim),
            'logvar': nn.Linear(self.hidden_dim, self.action_dim),
            'norm': nn.LayerNorm(self.hidden_dim)
        })

        # Enhanced value head with deeper ensemble
        num_value_heads = config.get("num_value_heads", 3)
        self.value_ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 1)
            ) for _ in range(num_value_heads)
        ])

        # Additional components for enhanced stability
        self.feature_dropout = nn.Dropout(config.get("dropout", 0.1))
        self.action_scaling = config.get("action_scaling", 1.0)
        self.min_std = config.get("min_std", 1e-6)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Feature extraction with residual connection
        x = F.relu(self.features['norm1'](self.features['input'](state)))
        x = self.feature_dropout(x)
        x = F.relu(self.features['norm2'](self.features['hidden'](x)))
        features = x + self.features['residual'](state)
        
        # Policy head with enhanced uncertainty estimation
        policy_hidden = F.relu(self.policy['norm'](self.policy['shared'](features)))
        action_mean = self.policy['mean'](policy_hidden)
        action_logvar = self.policy['logvar'](policy_hidden)
        
        # Bounded standard deviation
        action_std = torch.clamp(
            F.softplus(action_logvar),
            min=self.min_std,
            max=self.action_scaling
        )
        
        # Enhanced value ensemble with uncertainty
        values = torch.cat([head(features) for head in self.value_ensemble], dim=-1)
        value = values.mean(dim=-1, keepdim=True)
        value_uncertainty = values.std(dim=-1, keepdim=True)
        value_max = values.max(dim=-1, keepdim=True)[0]
        value_min = values.min(dim=-1, keepdim=True)[0]
        
        return {
            "action_mean": action_mean,
            "action_std": action_std,
            "value": value,
            "value_uncertainty": value_uncertainty,
            "value_range": (value_min, value_max),
            "features": features,
            "raw_values": values
        }

class PPOAgent(NexusModule, FeatureBankMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Core components with enhanced network
        self.network = ActorCritic(config)

        # Extended training parameters
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_range = config.get("clip_range", 0.2)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)

        # Advanced experience bank with prioritization using mixin
        bank_size = config.get("bank_size", 10000)
        hidden_dim = config.get("hidden_dim", 256)
        self.register_feature_bank("experience", bank_size, hidden_dim)
        self.register_feature_bank("priority", bank_size, 1)

        # Enhanced optimizer with gradient clipping
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            eps=1e-5,
            weight_decay=config.get("weight_decay", 0.01),
            amsgrad=True
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("max_steps", 1000),
            eta_min=self.learning_rate * 0.1
        )

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            # Handle both numpy arrays and tensors
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
            else:
                state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
                
            outputs = self.network(state_tensor)
            
            if deterministic:
                action = outputs["action_mean"]
            else:
                dist = torch.distributions.Normal(
                    outputs["action_mean"],
                    outputs["action_std"]
                )
                action = dist.sample()
                
                # Clip actions to valid range
                action = torch.clamp(action, -self.network.action_scaling, self.network.action_scaling)
            
            # Update experience bank using mixin
            self.update_feature_bank("experience", outputs["features"])
            self.update_feature_bank("priority", outputs["value_uncertainty"])
            
            return action, {
                "value": outputs["value"],
                "action_mean": outputs["action_mean"],
                "action_std": outputs["action_std"],
                "value_uncertainty": outputs["value_uncertainty"],
                "value_range": outputs["value_range"],
                "raw_values": outputs["raw_values"]
            }

    def compute_gae(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute advantages using GAE with masking support
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        # Create default mask if none provided
        if mask is None:
            mask = torch.ones_like(dones)
            
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae * mask[t]
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values
        
        return advantages, returns