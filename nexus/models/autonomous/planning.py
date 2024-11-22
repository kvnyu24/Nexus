import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from ...core.base import NexusModule

class TrajectoryEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_layers = config.get("num_layers", 3)
        
        # Temporal encoding (similar to TemporalReIDModule pattern)
        self.temporal_net = nn.GRU(
            input_size=config.get("trajectory_dim", 6),
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        # Attention mechanism (similar to InteractionModule pattern)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1)
        )

class EnhancedPlanningModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core components
        self.trajectory_encoder = TrajectoryEncoder(config)
        self.route_planner = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config.get("hidden_dim", 256),
                nhead=config.get("num_heads", 8)
            ),
            num_layers=config.get("num_layers", 6)
        )
        
        # Prediction heads
        self.waypoint_head = nn.Linear(
            config.get("hidden_dim", 256),
            config.get("num_waypoints", 10) * 2
        )
        self.uncertainty_head = nn.Linear(
            config.get("hidden_dim", 256),
            config.get("num_waypoints", 10)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        route_info: Optional[Dict[str, torch.Tensor]] = None,
        traffic_info: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode trajectory features
        encoded_trajectory = self.trajectory_encoder(features)
        
        # Plan route considering traffic
        if traffic_info is not None:
            route_features = self.route_planner(
                encoded_trajectory,
                src_key_padding_mask=traffic_info.get("attention_mask")
            )
        else:
            route_features = self.route_planner(encoded_trajectory)
        
        # Generate predictions with uncertainty
        waypoints = self.waypoint_head(route_features)
        uncertainties = self.uncertainty_head(route_features)
        
        return {
            "waypoints": waypoints.view(-1, self.num_waypoints, 2),
            "uncertainties": uncertainties,
            "route_features": route_features
        }
