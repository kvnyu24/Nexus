import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule
from ..cv.rcnn import FPNBackbone
from torch.nn import functional as F
from nexus.models.nvlm.base import NVLMMixin
from nexus.models.nvlm.processor import NVLMProcessor

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

class EnhancedPerceptionModule(NexusModule, NVLMMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Original perception initialization
        self.backbone = FPNBackbone(config)
        self.perception_heads = nn.ModuleDict({
            level: MultiTaskPerceptionHead(256, config)
            for level in ['p2', 'p3', 'p4', 'p5']
        })
        
        # Initialize NVLM if configured
        if config.get("use_nvlm", False):
            self.init_nvlm(config.get("nvlm_config", {}))
    
    def forward(
        self,
        images: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        outputs = {}
        
        # Original perception forward pass
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
        for task in ['segmentation', 'depth', 'detection']:
            fused = self.fusion[task](torch.cat(multi_scale_outputs[task], dim=1))
            outputs[f"{task}_logits"] = fused
            
        # Add NVLM processing if enabled
        if hasattr(self, "vision_encoder"):
            visual_features = NVLMProcessor.process_visual_features(
                images, self.vision_encoder, self.downsample,
                self.tile_embeddings, self.max_tiles
            )
            
            if self.arch_type == "cross":
                for layer in self.cross_attention:
                    visual_features = layer(text_features, visual_features)
                outputs["visual_features"] = visual_features
            else:
                visual_features = self.projector(visual_features)
                outputs["visual_embeddings"] = visual_features
                
        return outputs
