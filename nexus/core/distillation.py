import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from .base import NexusModule

class KnowledgeDistiller(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.temperature = config.get("temperature", 2.0)
        self.alpha = config.get("distillation_alpha", 0.5)
        self.feature_loss_weight = config.get("feature_loss_weight", 0.1)
        
        # Feature transformation layers
        if config.get("enable_feature_distillation", True):
            self.feature_transforms = nn.ModuleDict()
            for name, dims in config.get("feature_dimensions", {}).items():
                self.feature_transforms[name] = nn.Linear(dims[0], dims[1])
                
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute distillation loss with optional hard labels"""
        # Soft targets loss
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='none'
        )
        
        if mask is not None:
            soft_loss = soft_loss * mask.unsqueeze(-1)
            
        soft_loss = soft_loss.mean()
        
        outputs = {"soft_loss": soft_loss * (self.temperature ** 2)}
        
        # Hard targets loss if labels provided
        if labels is not None:
            hard_loss = F.cross_entropy(student_logits, labels)
            outputs["hard_loss"] = hard_loss
            outputs["total_loss"] = self.alpha * hard_loss + (1 - self.alpha) * outputs["soft_loss"]
        
        return outputs
        
    def compute_feature_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute feature-level distillation loss"""
        feature_loss = 0.0
        
        for name, student_feat in student_features.items():
            if name in teacher_features and name in self.feature_transforms:
                transformed_student = self.feature_transforms[name](student_feat)
                feature_loss += F.mse_loss(
                    transformed_student,
                    teacher_features[name].detach()
                )
                
        return feature_loss * self.feature_loss_weight

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute all distillation losses"""
        losses = self.compute_distillation_loss(
            student_outputs["logits"],
            teacher_outputs["logits"],
            labels,
            attention_mask
        )
        
        if self.feature_transforms:
            feature_loss = self.compute_feature_loss(
                student_outputs.get("features", {}),
                teacher_outputs.get("features", {})
            )
            losses["feature_loss"] = feature_loss
            if "total_loss" in losses:
                losses["total_loss"] += feature_loss
                
        return losses


class AdvancedDistillationModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Distillation parameters
        self.temperature = config.get("temperature", 2.0)
        self.alpha = config.get("distillation_alpha", 0.5)
        self.feature_weights = config.get("feature_weights", {})
        self.attention_distill = config.get("attention_distill", True)
        self.layer_mapping = config.get("layer_mapping", {})
        
        # Feature transformation layers
        self.feature_transforms = nn.ModuleDict()
        for name, dims in config.get("feature_dimensions", {}).items():
            self.feature_transforms[name] = nn.Sequential(
                nn.Linear(dims[0], dims[1]),
                nn.ReLU(),
                nn.Linear(dims[1], dims[1])
            )
            
        # Optional contrastive loss
        self.use_contrastive = config.get("use_contrastive", False)
        if self.use_contrastive:
            self.temperature_contrastive = config.get("temperature_contrastive", 0.07)
            
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Knowledge distillation loss
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='none'
        )
        
        if mask is not None:
            soft_loss = soft_loss * mask.unsqueeze(-1)
            
        soft_loss = soft_loss.mean() * (self.temperature ** 2)
        
        losses = {"soft_loss": soft_loss}
        
        # Hard label loss if provided
        if labels is not None:
            hard_loss = F.cross_entropy(student_logits, labels)
            losses["hard_loss"] = hard_loss
            losses["total_loss"] = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
            
        return losses
        
    def compute_feature_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        feature_losses = {}
        
        for name, student_feat in student_features.items():
            if name in teacher_features and name in self.feature_transforms:
                transformed_student = self.feature_transforms[name](student_feat)
                teacher_feat = teacher_features[name].detach()
                
                # Compute multiple similarity metrics
                mse_loss = F.mse_loss(transformed_student, teacher_feat)
                cosine_loss = 1 - F.cosine_similarity(
                    transformed_student.view(-1),
                    teacher_feat.view(-1),
                    dim=0
                )
                
                weight = self.feature_weights.get(name, 1.0)
                feature_losses[f"{name}_mse"] = mse_loss * weight
                feature_losses[f"{name}_cosine"] = cosine_loss * weight
                
        return feature_losses

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        
        # Logit distillation
        distill_losses = self.compute_distillation_loss(
            student_outputs["logits"],
            teacher_outputs["logits"],
            labels,
            attention_mask
        )
        losses.update(distill_losses)
        
        # Feature distillation
        if "features" in student_outputs and "features" in teacher_outputs:
            feature_losses = self.compute_feature_loss(
                student_outputs["features"],
                teacher_outputs["features"]
            )
            losses.update(feature_losses)
            
        # Attention distillation
        if self.attention_distill and "attention_maps" in student_outputs:
            attention_loss = self.compute_attention_loss(
                student_outputs["attention_maps"],
                teacher_outputs["attention_maps"]
            )
            losses["attention_loss"] = attention_loss
            
        return losses