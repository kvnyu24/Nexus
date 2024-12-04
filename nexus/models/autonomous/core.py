from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from ...core.base import NexusModule
from ..cv.rcnn import FPNBackbone
from ..agents.interaction import InteractionModule
from .perception import EnhancedPerceptionModule
from .planning import EnhancedPlanningModule
from .scene_understanding import SceneUnderstandingModule
from .motion_prediction import MotionPredictionModule
from .behavior_prediction import BehaviorPredictionModule
from .decision import DecisionMakingModule
from ..reasoning.mcts_reasoning import MCTSReasoning

class AutonomousDrivingSystem(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate configuration
        self._validate_config(config)

        # Vision backbone (FPN)
        self.fpn_backbone = FPNBackbone(config)

        
        # Perception stack
        self.scene_understanding = SceneUnderstandingModule(config)
        self.perception = EnhancedPerceptionModule(config)

        # Agent interaction
        self.interaction_module = InteractionModule(config)

        
        # Prediction stack
        self.motion_prediction = MotionPredictionModule(config)
        self.behavior_prediction = BehaviorPredictionModule(config)
        
        # Planning and decision stack
        self.planning = EnhancedPlanningModule(config)
        self.decision_making = DecisionMakingModule(config)
        
        # Multi-scale feature fusion (following SceneUnderstandingModule pattern)
        self.feature_fusion = nn.ModuleDict({
            'perception': nn.Linear(256 * 2, 256),
            'prediction': nn.Linear(256 * 2, 256),
            'planning': nn.Linear(256 * 2, 256)
        })
        
        # Global context integration (similar to InteractionModule)
        self.context_attention = nn.MultiheadAttention(
            embed_dim=config.get("hidden_dim", 256),
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1)
        )
        
        # Safety monitoring (from DecisionMakingModule pattern)
        self.safety_monitor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # MCTS-based decision making
        self.mcts_reasoning = MCTSReasoning(config)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Following BehaviorPredictionModule validation pattern"""
        required_keys = ["hidden_dim", "num_heads", "dropout"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
    
    def forward(
        self,
        images: torch.Tensor,
        agent_states: torch.Tensor,
        other_agents: Optional[torch.Tensor] = None,
        motion_history: Optional[torch.Tensor] = None,
        route_info: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Extract FPN features
        fpn_features = self.fpn_backbone(images)

        # Scene understanding and perception
        scene_outputs = self.scene_understanding(images, attention_mask)
        perception_outputs = self.perception(images)
        
        # Fuse perception features
        perception_features = self.feature_fusion['perception'](
            torch.cat([
                scene_outputs["features"]["p5"],
                perception_outputs["features"]["p5"]
            ], dim=-1)
        )

        # Agent interaction processing
        if other_agents is not None:
            interaction_outputs = self.interaction_module(
                agent_states,
                other_agents,
                attention_mask=attention_mask
            )
            interaction_features = interaction_outputs["interaction_features"]
        else:
            interaction_features = torch.zeros_like(perception_features)
            interaction_outputs = {}

        
        # Motion and behavior prediction
        motion_outputs = self.motion_prediction(
            motion_history,
            scene_context=perception_features,
            interaction_context=interaction_features,
            attention_mask=attention_mask
        )
        
        behavior_outputs = self.behavior_prediction(
            agent_states,
            scene_context=perception_features,
            interaction_context=interaction_features,
            attention_mask=attention_mask
        )
        
        # Fuse prediction features
        prediction_features = self.feature_fusion['prediction'](
            torch.cat([
                motion_outputs["motion_features"],
                behavior_outputs["behavior_features"]
            ], dim=-1)
        )
        
        # Planning and decision making
        planning_outputs = self.planning(
            prediction_features,
            route_info=route_info,
            traffic_info={
                "attention_mask": attention_mask,
                "scene_context": perception_features,
                "interaction_context": interaction_features
            }
        )
        
        mcts_outputs = self.mcts_reasoning(
            agent_states,
            num_simulations=self.config.get("num_simulations", 50),
            temperature=self.config.get("temperature", 1.0)
        )


        
        # Global context integration
        global_context, attention_weights = self.context_attention(
            prediction_features,
            perception_features,
            perception_features,
            key_padding_mask=attention_mask
        )
        
        # Final decision making
        decision_outputs = self.decision_making(
            perception_features,
            prediction_features,
            global_context,
            interaction_features,
            attention_mask=attention_mask
        )
        
        # Safety assessment
        safety_score = self.safety_monitor(global_context)
        
        return {
            **scene_outputs,
            **perception_outputs,
            **interaction_outputs,
            **motion_outputs,
            **behavior_outputs,
            **planning_outputs,
            **decision_outputs,
            **mcts_outputs,
            "fpn_features": fpn_features,
            "safety_score": safety_score,
            "attention_weights": attention_weights,
            "global_context": global_context
        }
