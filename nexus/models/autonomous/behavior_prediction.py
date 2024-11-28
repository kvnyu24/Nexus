import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule

class AgentStateEncoder(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.hidden_dim = config.get("hidden_dim", 256)
        
        # State encoding (similar to SocialAgent pattern)
        self.encoder = nn.Sequential(
            nn.Linear(config.get("state_dim", 64), self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Memory mechanism (following EnhancedReID pattern)
        self.memory_size = config.get("memory_size", 1000)
        self.register_buffer(
            "state_memory",
            torch.zeros(self.memory_size, self.hidden_dim)
        )
        self.register_buffer("memory_ptr", torch.zeros(1, dtype=torch.long))

class BehaviorPredictionModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate configuration
        self._validate_config(config)
        
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
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Following FasterRCNN validation pattern"""
        required_keys = ["hidden_dim", "state_dim"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
    
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