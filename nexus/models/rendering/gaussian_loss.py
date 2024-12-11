import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule

class GaussianRenderingLoss(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate and set loss weights with defaults
        self.rgb_weight = max(0.0, float(config.get("rgb_weight", 1.0)))
        self.depth_weight = max(0.0, float(config.get("depth_weight", 0.1)))
        self.coverage_weight = max(0.0, float(config.get("coverage_weight", 0.01)))
        self.feature_weight = max(0.0, float(config.get("feature_weight", 0.1)))
        self.smoothness_weight = max(0.0, float(config.get("smoothness_weight", 0.01)))
        
        # Loss functions
        self.feature_loss = nn.MSELoss(reduction='mean')
        self.huber_loss = nn.HuberLoss(reduction='mean', delta=0.1)
        
        # Temperature parameter for loss scaling with better initialization
        self.register_buffer("temperature", torch.ones(1) * config.get("init_temperature", 1.0))
        
        # Adaptive weighting
        self.use_adaptive_weights = config.get("use_adaptive_weights", False)
        if self.use_adaptive_weights:
            self.register_buffer("loss_history", torch.zeros(4))  # Track last N losses
            self.register_buffer("weight_momentum", torch.ones(4) * 0.9)
        
    def _compute_adaptive_weights(self, losses: torch.Tensor) -> torch.Tensor:
        """Compute adaptive weights based on loss history"""
        if not self.use_adaptive_weights:
            return torch.ones_like(losses)
            
        # Update history with exponential moving average
        self.loss_history = (
            self.weight_momentum * self.loss_history + 
            (1 - self.weight_momentum) * losses.detach()
        )
        
        # Compute inverse variance weights
        weights = 1.0 / (self.loss_history + 1e-8)
        return weights / weights.sum()
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        
        # Validate inputs
        for key in ["color", "depth", "weights"]:
            if key in predictions and key in targets:
                if predictions[key].shape != targets[key].shape:
                    raise ValueError(f"{key} shape mismatch between predictions and targets")
        
        # RGB loss with perceptual weighting
        if "color" in predictions and "color" in targets:
            rgb_diff = predictions["color"] - targets["color"]
            # Apply perceptual weighting (emphasize errors in green channel)
            rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=rgb_diff.device)
            rgb_loss = torch.mean((rgb_diff.pow(2) * rgb_weights[None, :]).sum(-1))
            losses["rgb_loss"] = rgb_loss
        
        # Depth loss with robust Huber loss and masking
        if "depth" in predictions and "depth" in targets:
            depth_diff = predictions["depth"] - targets["depth"]
            if masks is not None and "depth" in masks:
                valid_mask = masks["depth"].bool()
                if valid_mask.any():
                    depth_loss = self.huber_loss(
                        predictions["depth"][valid_mask],
                        targets["depth"][valid_mask]
                    )
                else:
                    depth_loss = torch.tensor(0.0, device=depth_diff.device)
            else:
                depth_loss = self.huber_loss(predictions["depth"], targets["depth"])
            losses["depth_loss"] = depth_loss
                
        # Coverage regularization with numerical stability
        if "weights" in predictions:
            weights_sum = predictions["weights"].sum(dim=1).clamp(min=1e-12)
            coverage_loss = -torch.mean(torch.log(weights_sum))
            losses["coverage_loss"] = coverage_loss
            
        # Feature matching loss if available
        if "features" in predictions and "features" in targets:
            feature_loss = self.feature_loss(predictions["features"], targets["features"])
            losses["feature_loss"] = feature_loss
            
        # Smoothness regularization on gaussian parameters
        if "covariances" in predictions:
            smoothness_loss = torch.mean(torch.abs(
                predictions["covariances"][..., 1:] - 
                predictions["covariances"][..., :-1]
            ))
            losses["smoothness_loss"] = smoothness_loss
            
        # Compute adaptive weights
        loss_values = torch.stack([
            losses.get("rgb_loss", torch.tensor(0.0, device=self.temperature.device)),
            losses.get("depth_loss", torch.tensor(0.0, device=self.temperature.device)),
            losses.get("coverage_loss", torch.tensor(0.0, device=self.temperature.device)),
            losses.get("feature_loss", torch.tensor(0.0, device=self.temperature.device))
        ])
        adaptive_weights = self._compute_adaptive_weights(loss_values)
        
        # Total loss with temperature scaling and adaptive weighting
        weights = torch.tensor([
            self.rgb_weight, self.depth_weight,
            self.coverage_weight, self.feature_weight
        ], device=self.temperature.device)
        
        if self.use_adaptive_weights:
            weights = weights * adaptive_weights
            
        total_loss = (weights * loss_values).sum() / self.temperature
        losses["total_loss"] = total_loss
        
        # Detach all losses except total_loss for return
        return {k: (v if k == "total_loss" else v.detach()) for k, v in losses.items()}