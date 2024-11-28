import torch
import torch.nn as nn
from typing import Dict, Any
from ....core.base import NexusModule
from .backbone import FPNBackbone
from .rpn import RegionProposalNetwork
from .fast_rcnn import RoIHead, FastRCNNPredictor
from ....components.attention import SpatialAttention

class MaskHead(NexusModule):
    def __init__(self, in_channels: int, hidden_dim: int = 256):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.spatial_attention = SpatialAttention()
        self.mask_pred = nn.Conv2d(hidden_dim, 1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.spatial_attention(x)
        return self.mask_pred(x)

class EnhancedMaskRCNN(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize components
        self.backbone = FPNBackbone(config)
        self.rpn = RegionProposalNetwork(config)
        self.roi_head = RoIHead(config)
        self.mask_head = MaskHead(config.get("in_channels", 256))
        
        # Fast R-CNN specific components
        self.fast_rcnn_predictor = FastRCNNPredictor(
            in_channels=config.get("in_channels", 256),
            num_classes=config.get("num_classes", 80)
        )
        
        # ROI pooling configuration
        self.roi_pool_size = config.get("roi_pool_size", 7)
        
    def _roi_align(self, features: Dict[str, torch.Tensor], proposals: torch.Tensor) -> torch.Tensor:
        """Perform ROI Align operation on feature maps"""
        roi_features = []
        for level, feature in features.items():
            roi_features.append(
                torch.ops.torchvision.roi_align(
                    feature, proposals,
                    self.roi_pool_size,
                    spatial_scale=1.0 / (2 ** int(level[1])),
                    sampling_ratio=-1
                )
            )
        return torch.cat(roi_features, dim=0)
        
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features using FPN backbone
        features = self.backbone(images)
        
        # Generate region proposals
        rpn_output = self.rpn(features)
        
        # Get ROI features
        roi_features = self._roi_align(features, rpn_output["proposals"])
        
        # Fast R-CNN predictions
        roi_output = self.roi_head(roi_features)
        
        # Generate masks for selected proposals
        masks = self.mask_head(roi_features)
        
        # Fast R-CNN final predictions
        cls_scores, bbox_preds = self.fast_rcnn_predictor(roi_features)
        
        return {
            "features": features,
            "rpn_scores": rpn_output["objectness_scores"],
            "rpn_bbox": rpn_output["bbox_preds"],
            "roi_features": roi_features,
            "cls_scores": cls_scores,
            "bbox_preds": bbox_preds,
            "masks": masks
        } 