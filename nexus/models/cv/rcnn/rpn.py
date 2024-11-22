import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from ....core.base import NexusModule
from torch.nn import functional as F

class RegionProposalNetwork(NexusModule):
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.in_channels = config.get("in_channels", 256)
        self.anchor_scales = config.get("anchor_scales", [32, 64, 128, 256, 512])
        self.anchor_ratios = config.get("anchor_ratios", [0.5, 1.0, 2.0])
        
        # RPN head
        self.conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1)
        self.objectness = nn.Conv2d(self.in_channels, len(self.anchor_scales) * len(self.anchor_ratios) * 2, kernel_size=1)
        self.bbox_pred = nn.Conv2d(self.in_channels, len(self.anchor_scales) * len(self.anchor_ratios) * 4, kernel_size=1)
        
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rpn_features = {}
        objectness_scores = {}
        bbox_preds = {}
        
        for level, feature in features.items():
            x = F.relu(self.conv(feature))
            objectness_scores[level] = self.objectness(x)
            bbox_preds[level] = self.bbox_pred(x)
            
        return {
            "objectness_scores": objectness_scores,
            "bbox_preds": bbox_preds
        } 