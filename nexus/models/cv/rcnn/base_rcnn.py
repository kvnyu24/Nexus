import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from ....core.base import NexusModule
from ....core.initialization import WeightInitMixin
from ....core.mixins import ConfigValidatorMixin
from .backbone import FPNBackbone
from .rpn import RegionProposalNetwork

class BaseRCNN(ConfigValidatorMixin, WeightInitMixin, NexusModule):
    """
    Base R-CNN implementation that serves as a foundation for specialized variants.
    Follows Nexus module patterns and interfaces.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Validate configuration
        self.validate_config(config, required_keys=["in_channels", "num_classes"])
        self.validate_positive(config.get("num_classes", 0), "num_classes")

        # Core components
        self.backbone = FPNBackbone(config)
        self.rpn = RegionProposalNetwork(config)

        # Configuration
        self.num_classes = config.get("num_classes", 80)
        self.roi_pool_size = config.get("roi_pool_size", 7)
        self.representation_size = config.get("representation_size", 1024)

        # RoI pooling and box head
        self.box_head = nn.Sequential(
            nn.Linear(256 * self.roi_pool_size ** 2, self.representation_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.representation_size, self.representation_size),
            nn.ReLU(inplace=True)
        )

        # Classification and regression heads
        self.cls_head = nn.Linear(self.representation_size, self.num_classes + 1)  # +1 for background
        self.bbox_head = nn.Linear(self.representation_size, self.num_classes * 4)

        # Initialize weights
        self.init_weights_vision()
                        
    def _roi_align(
        self,
        features: Dict[str, torch.Tensor],
        proposals: torch.Tensor,
        image_shapes: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Perform ROI Align operation on feature maps."""
        roi_features = []
        
        for level, feature_map in features.items():
            level_roi = torch.ops.torchvision.roi_align(
                feature_map,
                proposals,
                self.roi_pool_size,
                spatial_scale=1.0 / (2 ** int(level[1])),
                sampling_ratio=-1
            )
            roi_features.append(level_roi)
            
        return torch.cat(roi_features, dim=0)
        
    def forward(
        self,
        images: torch.Tensor,
        image_shapes: List[Tuple[int, int]],
        targets: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass implementation."""
        # Extract features using FPN backbone
        features = self.backbone(images)
        
        # Generate region proposals
        rpn_output = self.rpn(features)
        proposals = rpn_output["objectness_scores"]
        
        # ROI Align
        roi_features = self._roi_align(features, proposals, image_shapes)
        
        # Box head processing
        box_features = self.box_head(roi_features.flatten(1))
        
        # Generate predictions
        class_logits = self.cls_head(box_features)
        box_regression = self.bbox_head(box_features)
        
        outputs = {
            "class_logits": class_logits,
            "box_regression": box_regression,
            "proposals": proposals,
            "features": features
        }
        
        if targets is not None:
            # Add training-specific outputs
            outputs.update({
                "rpn_objectness": rpn_output["objectness_scores"],
                "rpn_bbox_preds": rpn_output["bbox_preds"]
            })
            
        return outputs 