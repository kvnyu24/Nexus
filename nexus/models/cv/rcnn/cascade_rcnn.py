from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
from ....core.base import NexusModule
from .base_rcnn import BaseRCNN
from .fast_rcnn import RoIHead

class CascadeRoIHead(RoIHead):
    def __init__(self, config: dict, stage: int):
        super().__init__(config)
        self.stage = stage
        
        # Adjust thresholds based on stage
        self.iou_threshold = 0.5 + stage * 0.1
        
        # Additional stage-specific layers
        self.stage_specific = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
    def forward(
        self,
        roi_features: torch.Tensor,
        proposals: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Get base RoI features
        base_outputs = super().forward(roi_features, proposals)
        
        # Add stage-specific processing
        refined_features = self.stage_specific(base_outputs["roi_features"].flatten(1))
        
        return {
            **base_outputs,
            "stage_features": refined_features,
            "stage": self.stage
        }

class CascadeRCNN(BaseRCNN):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Cascade stages configuration
        self.num_stages = config.get("num_stages", 3)
        
        # Create cascade of RoI heads
        self.cascade_roi_heads = nn.ModuleList([
            CascadeRoIHead(config, stage)
            for stage in range(self.num_stages)
        ])
        
    def forward(
        self,
        images: torch.Tensor,
        image_shapes: List[Tuple[int, int]],
        targets: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Extract features using backbone
        features = self.backbone(images)
        
        # Generate initial proposals
        rpn_output = self.rpn(features)
        proposals = rpn_output["objectness_scores"]
        
        # Cascade refinement
        cascade_outputs = []
        current_proposals = proposals
        
        for roi_head in self.cascade_roi_heads:
            # ROI Align for current stage
            roi_features = self._roi_align(features, current_proposals, image_shapes)
            
            # Process through current stage
            stage_output = roi_head(roi_features, current_proposals)
            cascade_outputs.append(stage_output)
            
            # Update proposals for next stage
            current_proposals = self._refine_proposals(
                current_proposals,
                stage_output["bbox_deltas"]
            )
            
        return {
            "features": features,
            "rpn_scores": rpn_output["objectness_scores"],
            "rpn_bbox": rpn_output["bbox_preds"],
            "cascade_outputs": cascade_outputs,
            "final_proposals": current_proposals
        }
        
    def _refine_proposals(
        self,
        proposals: torch.Tensor,
        bbox_deltas: torch.Tensor
    ) -> torch.Tensor:
        # Apply bbox deltas to refine proposals
        refined = proposals.clone()
        
        # Convert deltas to absolute coordinates
        refined[:, 0] += bbox_deltas[:, 0] * proposals[:, 2]
        refined[:, 1] += bbox_deltas[:, 1] * proposals[:, 3]
        refined[:, 2] *= torch.exp(bbox_deltas[:, 2])
        refined[:, 3] *= torch.exp(bbox_deltas[:, 3])
        
        return refined
        