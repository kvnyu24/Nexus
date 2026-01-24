import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple
from .base import NexusModule
import torch.nn.functional as F

class ModelTrimmer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.threshold = config.get("pruning_threshold", 0.1)
        self.min_channels = config.get("min_channels", 4)
        self.structured_pruning = config.get("structured_pruning", True)
        
    def analyze_importance(self, module: NexusModule) -> Dict[str, torch.Tensor]:
        """Analyze parameter importance using L1-norm"""
        importance_scores = {}
        
        for name, param in module.named_parameters():
            if param.dim() > 1:  # Only analyze weights, not biases
                if self.structured_pruning:
                    # Channel-wise L1-norm for structured pruning
                    scores = torch.norm(param.view(param.size(0), -1), p=1, dim=1)
                else:
                    # Element-wise L1-norm for unstructured pruning
                    scores = torch.abs(param)
                importance_scores[name] = scores
                
        return importance_scores
        
    def trim_module(
        self,
        module: NexusModule,
        importance_scores: Dict[str, torch.Tensor]
    ) -> NexusModule:
        """Trim model based on importance scores"""
        for name, param in module.named_parameters():
            if name in importance_scores:
                scores = importance_scores[name]
                mask = scores > (scores.max() * self.threshold)
                
                if self.structured_pruning:
                    # Ensure minimum channels are retained
                    if mask.sum() < self.min_channels:
                        _, top_idx = torch.topk(scores, self.min_channels)
                        mask = torch.zeros_like(mask, dtype=torch.bool)
                        mask[top_idx] = True
                        
                # Apply mask
                param.data = param.data * mask.to(param.device).view(*mask.shape, *([1] * (param.dim() - 1)))
                
        return module

    def forward(self, module: NexusModule) -> NexusModule:
        """Analyze and trim the model"""
        importance_scores = self.analyze_importance(module)
        return self.trim_module(module, importance_scores)

class AdvancedModelTrimmer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Trimming configuration
        self.sparsity_target = config.get("sparsity_target", 0.5)
        self.min_channels = config.get("min_channels", 4)
        self.importance_metric = config.get("importance_metric", "l1_norm")
        self.granularity = config.get("granularity", "channel")  # channel, block, or layer
        self.preserve_outputs = config.get("preserve_outputs", True)
        
    def compute_importance(
        self,
        param: torch.Tensor,
        method: str = "l1_norm"
    ) -> torch.Tensor:
        if method == "l1_norm":
            return torch.norm(param.view(param.size(0), -1), p=1, dim=1)
        elif method == "l2_norm":
            return torch.norm(param.view(param.size(0), -1), p=2, dim=1)
        elif method == "fisher":
            # Approximate Fisher Information
            grad = param.grad
            if grad is not None:
                return (grad ** 2).sum(dim=list(range(1, len(param.shape))))
            return torch.ones(param.size(0), device=param.device)
            
    def get_pruning_mask(
        self,
        importance: torch.Tensor,
        sparsity: float
    ) -> torch.Tensor:
        threshold = torch.quantile(importance, sparsity)
        mask = importance > threshold
        
        # Ensure minimum channels are preserved
        if mask.sum() < self.min_channels:
            _, top_idx = torch.topk(importance, self.min_channels)
            mask = torch.zeros_like(mask, dtype=torch.bool)
            mask[top_idx] = True
            
        return mask
        
    def trim_layer(
        self,
        layer: NexusModule,
        sparsity: float
    ) -> Tuple[NexusModule, Dict[str, torch.Tensor]]:
        importance_scores = {}
        masks = {}
        
        for name, param in layer.named_parameters():
            if param.dim() > 1:  # Only process weights
                importance = self.compute_importance(param, self.importance_metric)
                mask = self.get_pruning_mask(importance, sparsity)
                
                if self.granularity == "channel":
                    param.data = param.data * mask.to(param.device).view(-1, *([1] * (param.dim() - 1)))
                
                importance_scores[name] = importance
                masks[name] = mask
                
        return layer, {"importance": importance_scores, "masks": masks}

    def forward(
        self,
        model: NexusModule,
        example_input: Optional[torch.Tensor] = None
    ) -> Tuple[NexusModule, Dict[str, Any]]:
        metrics = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module, layer_metrics = self.trim_layer(module, self.sparsity_target)
                metrics[name] = layer_metrics
                
        if self.preserve_outputs and example_input is not None:
            with torch.no_grad():
                original_output = model(example_input)
                pruned_output = model(example_input)
                metrics["output_difference"] = {
                    "mse": F.mse_loss(original_output, pruned_output),
                    "cosine": F.cosine_similarity(
                        original_output.view(-1),
                        pruned_output.view(-1),
                        dim=0
                    )
                }
                
        return model, metrics