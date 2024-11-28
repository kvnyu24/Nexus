import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule
from ..cv.rcnn import FPNBackbone
from torch.nn import functional as F

class SceneUnderstandingHead(NexusModule):
    def __init__(self, in_channels: int, config: Dict[str, Any]):
        super().__init__()
        
        hidden_dim = config.get("hidden_dim", 256)
        
        # Multi-scale feature processing (similar to KeypointHead pattern)
        self.feature_processor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Task-specific heads
        self.semantic_head = nn.Conv2d(hidden_dim, config.get("num_classes", 19), 1)
        self.instance_head = nn.Conv2d(hidden_dim, config.get("num_instances", 1000), 1)
        self.depth_head = nn.Conv2d(hidden_dim, 1, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_processor(x)
        return {
            "semantic": self.semantic_head(features),
            "instance": self.instance_head(features),
            "depth": self.depth_head(features)
        }

class SceneUnderstandingModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Reuse FPN backbone (similar to CityReconstructionModel)
        self.backbone = FPNBackbone(config)
        
        # Multi-scale scene understanding heads
        self.scene_heads = nn.ModuleDict({
            f'p{i}': SceneUnderstandingHead(256, config)
            for i in range(2, 6)  # P2 to P5 features
        })
        
        # Feature fusion (similar to RoIHead pattern)
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(256 * 4, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
    def forward(
        self,
        images: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Extract multi-scale features
        features = self.backbone(images)
        
        # Process each scale
        multi_scale_outputs = {
            task: [] for task in ['semantic', 'instance', 'depth']
        }
        
        for level, feature in features.items():
            outputs = self.scene_heads[level](feature)
            for task, output in outputs.items():
                # Upsample to match input resolution
                output = F.interpolate(
                    output,
                    size=images.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                multi_scale_outputs[task].append(output)
        
        # Fuse multi-scale predictions
        fused_outputs = {}
        for task, outputs in multi_scale_outputs.items():
            fused = self.fusion_layer(torch.cat(outputs, dim=1))
            fused_outputs[f"{task}_logits"] = fused
        
        return {
            **fused_outputs,
            "features": features
        } 