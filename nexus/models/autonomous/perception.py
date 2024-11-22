import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule
from ..cv.rcnn import FPNBackbone
from torch.nn import functional as F

class MultiTaskPerceptionHead(nn.Module):
    def __init__(self, in_channels: int, config: Dict[str, Any]):
        super().__init__()
        
        hidden_dim = config.get("hidden_dim", 256)
        
        # Segmentation head
        self.segmentation = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, config.get("num_seg_classes", 19), 1)
        )
        
        # Depth estimation head
        self.depth = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1)
        )
        
        # Object detection head (reusing FasterRCNN patterns)
        self.detection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 4 + config.get("num_classes", 80), 1)
        )

class EnhancedPerceptionModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Reuse FPN backbone (similar to MaskRCNN pattern)
        self.backbone = FPNBackbone(config)
        
        # Multi-task heads
        self.perception_heads = nn.ModuleDict({
            level: MultiTaskPerceptionHead(256, config)
            for level in ['p2', 'p3', 'p4', 'p5']
        })
        
        # Feature fusion
        self.fusion = nn.ModuleDict({
            'segmentation': nn.Conv2d(256 * 4, 256, 1),
            'depth': nn.Conv2d(256 * 4, 256, 1),
            'detection': nn.Conv2d(256 * 4, 256, 1)
        })
        
    def forward(
        self,
        images: torch.Tensor,
        camera_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        # Extract multi-scale features
        features = self.backbone(images)
        
        # Process each scale
        multi_scale_outputs = {
            task: [] for task in ['segmentation', 'depth', 'detection']
        }
        
        for level, feature in features.items():
            head_outputs = self.perception_heads[level](feature)
            for task, output in head_outputs.items():
                multi_scale_outputs[task].append(
                    F.interpolate(output, size=images.shape[-2:])
                )
        
        # Fuse multi-scale predictions
        outputs = {}
        for task in ['segmentation', 'depth', 'detection']:
            fused = self.fusion[task](torch.cat(multi_scale_outputs[task], dim=1))
            outputs[f"{task}_logits"] = fused
            
        return outputs
