import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from ....core.base import NexusModule
from .backbone import FPNBackbone
from .rpn import RegionProposalNetwork
from .fast_rcnn import RoIHead, FastRCNNPredictor

class FasterRCNN(NexusModule):
    """
    Faster R-CNN implementation following Nexus module patterns.
    Integrates FPN backbone, RPN, and Fast R-CNN head components.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate configuration
        self._validate_config(config)
        
        # Initialize components
        self.backbone = FPNBackbone(config)
        self.rpn = RegionProposalNetwork(config)
        self.roi_head = RoIHead(config)
        
        # Fast R-CNN specific components
        self.fast_rcnn_predictor = FastRCNNPredictor(
            in_channels=config.get("in_channels", 256),
            num_classes=config.get("num_classes", 80)
        )
        
        # ROI pooling configuration
        self.roi_pool_size = config.get("roi_pool_size", 7)
        
        # Initialize weights
        self._init_weights()
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        required_keys = ["in_channels", "num_classes"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
                
        if config.get("num_classes", 0) <= 0:
            raise ValueError("num_classes must be positive")
            
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in [self.rpn, self.roi_head, self.fast_rcnn_predictor]:
            if hasattr(module, 'apply'):
                module.apply(self._init_layer_weights)
                
    def _init_layer_weights(self, module: NexusModule) -> None:
        """Initialize layer weights following Nexus patterns."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            
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
        targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass implementation."""
        # Extract features using FPN backbone
        features = self.backbone(images)
        
        # Generate region proposals
        rpn_output = self.rpn(features)
        proposals = rpn_output["objectness_scores"]
        
        # ROI Align
        roi_features = self._roi_align(features, proposals, image_shapes)
        
        # Process ROI features
        roi_output = self.roi_head(roi_features)
        
        # Generate final predictions
        cls_scores, bbox_preds = self.fast_rcnn_predictor(roi_features)
        
        outputs = {
            "proposals": proposals,
            "roi_features": roi_features,
            "cls_scores": cls_scores,
            "bbox_preds": bbox_preds,
            "features": features
        }
        
        if targets is not None:
            # Add training-specific outputs
            outputs.update({
                "rpn_objectness": rpn_output["objectness_scores"],
                "rpn_bbox_preds": rpn_output["bbox_preds"],
                "roi_cls_scores": roi_output["cls_scores"],
                "roi_bbox_deltas": roi_output["bbox_deltas"]
            })
            
        return outputs