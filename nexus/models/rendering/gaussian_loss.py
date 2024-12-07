import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule

class GaussianRenderingLoss(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate and set loss weights
        self.rgb_weight = max(0.0, float(config.get("rgb_weight", 1.0)))
        self.depth_weight = max(0.0, float(config.get("depth_weight", 0.1))) 
        self.coverage_weight = max(0.0, float(config.get("coverage_weight", 0.01)))
        
        # Feature matching loss
        self.feature_loss = nn.MSELoss(reduction='mean')
        
        # Temperature parameter for loss scaling
        self.register_buffer("temperature", torch.ones(1))
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        # Input validation
        if "color" not in predictions or "color" not in targets:
            raise ValueError("Color values missing from predictions or targets")
            
        if predictions["color"].shape != targets["color"].shape:
            raise ValueError("Color shape mismatch between predictions and targets")
            
        # RGB loss with error checking
        rgb_diff = predictions["color"] - targets["color"]
        rgb_loss = torch.mean(rgb_diff.pow(2))
        
        # Depth loss with robust masking
        depth_loss = torch.tensor(0.0, device=rgb_loss.device)
        if "depth" in targets and "depth" in predictions:
            depth_diff = predictions["depth"] - targets["depth"]
            if masks is not None and "depth" in masks:
                valid_mask = masks["depth"].bool()
                if valid_mask.any():
                    depth_loss = torch.mean(depth_diff.pow(2)[valid_mask])
            else:
                depth_loss = torch.mean(depth_diff.pow(2))
                
        # Coverage regularization with numerical stability
        weights_sum = predictions["weights"].sum(dim=1).clamp(min=1e-12)
        coverage_loss = -torch.mean(torch.log(weights_sum))
        
        # Total loss with gradient scaling
        total_loss = (
            self.rgb_weight * rgb_loss +
            self.depth_weight * depth_loss +
            self.coverage_weight * coverage_loss
        ) / self.temperature
        
        return {
            "rgb_loss": rgb_loss.detach(),
            "depth_loss": depth_loss.detach(),
            "coverage_loss": coverage_loss.detach(),
            "total_loss": total_loss
        }