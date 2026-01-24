import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule
from ...core.mixins import ConfigValidatorMixin, FeatureBankMixin

class AgentStateEncoder(FeatureBankMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.hidden_dim = config.get("hidden_dim", 256)
        
        # State encoding (similar to SocialAgent pattern)
        self.encoder = nn.Sequential(
            nn.Linear(config.get("state_dim", 64), self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Memory mechanism using FeatureBankMixin
        self.memory_size = config.get("memory_size", 1000)
        self.register_feature_bank("state_memory", self.memory_size, self.hidden_dim)

class BehaviorPredictionModule(ConfigValidatorMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate configuration using ConfigValidatorMixin
        self.validate_config(config, required_keys=["hidden_dim", "state_dim"])
        
        # Core components
        self.state_encoder = AgentStateEncoder(config)
        self.temporal_encoder = nn.GRU(
            input_size=config.get("hidden_dim", 256),
            hidden_size=config.get("hidden_dim", 256),
            num_layers=2,
            batch_first=True
        )
        
        # Multi-head attention (similar to InteractionModule)
        self.behavior_attention = nn.MultiheadAttention(
            embed_dim=config.get("hidden_dim", 256),
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1)
        )
        
        # Prediction heads
        self.intention_head = nn.Linear(
            config.get("hidden_dim", 256),
            config.get("num_intentions", 8)
        )
        self.trajectory_head = nn.Linear(
            config.get("hidden_dim", 256),
            config.get("prediction_horizon", 10) * 2
        )

        # Store prediction_horizon for forward pass
        self.prediction_horizon = config.get("prediction_horizon", 10)

    def forward(
        self,
        agent_states: torch.Tensor,
        scene_context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode agent states
        encoded_states = self.state_encoder(agent_states)
        
        # Temporal encoding
        temporal_features, _ = self.temporal_encoder(encoded_states)
        
        # Apply attention with scene context
        if scene_context is not None:
            attended_features, attention_weights = self.behavior_attention(
                temporal_features,
                scene_context,
                scene_context,
                key_padding_mask=attention_mask
            )
        else:
            attended_features = temporal_features
            attention_weights = None
        
        # Generate predictions
        intentions = self.intention_head(attended_features)
        trajectories = self.trajectory_head(attended_features)
        
        return {
            "predicted_intentions": intentions,
            "predicted_trajectories": trajectories.view(-1, self.prediction_horizon, 2),
            "attention_weights": attention_weights,
            "behavior_features": attended_features
        } 