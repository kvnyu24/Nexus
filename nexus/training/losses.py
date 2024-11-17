import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union

class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        
    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        distances = F.pairwise_distance(embeddings1, embeddings2)
        losses = labels.float() * distances.pow(2) + \
                (1 - labels).float() * F.relu(self.margin - distances).pow(2)
        return losses.mean() 

class CircleLoss(nn.Module):
    def __init__(
        self,
        m: float = 0.25,
        gamma: float = 256,
        reduction: str = 'mean'
    ):
        """
        Circle Loss for deep metric learning
        
        Args:
            m: Margin parameter
            gamma: Scale factor
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        # Calculate pairwise distances
        dist_mat = torch.cdist(embeddings, embeddings)
        
        # Get positive and negative mask
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        neg_mask = (labels != labels.T).float()
        
        # Calculate positive and negative scores
        pos_scores = -dist_mat * pos_mask
        neg_scores = dist_mat * neg_mask
        
        # Apply margin
        pos_scores = pos_scores + self.m
        neg_scores = neg_scores - self.m
        
        # Get positive and negative weights
        pos_weights = torch.exp(self.gamma * pos_scores) * pos_mask
        neg_weights = torch.exp(-self.gamma * neg_scores) * neg_mask
        
        # Calculate loss
        loss = torch.log(1 + torch.sum(pos_weights) * torch.sum(neg_weights))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss 

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        Triplet loss for metric learning
        
        Args:
            margin: Margin between positive and negative pairs
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        """
        Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
        Used in contrastive learning frameworks like SimCLR
        
        Args:
            temperature: Temperature parameter for scaling
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        # Normalize embeddings
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        # Gather all embeddings if using distributed training
        N = z1_norm.size(0)
        z_all = torch.cat([z1_norm, z2_norm], dim=0)
        
        # Compute similarity matrix
        sim = torch.mm(z_all, z_all.t()) / self.temperature
        
        # Mask out self-similarity
        sim_i_j = torch.diag(sim, N)
        sim_j_i = torch.diag(sim, -N)
        
        # Create positive pairs
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0)
        
        # Create negative mask
        negative_mask = torch.ones_like(sim)
        negative_mask[range(N), range(N)] = 0
        negative_mask[range(N, 2*N), range(N, 2*N)] = 0
        
        # Compute loss
        numerator = torch.exp(positive_samples)
        denominator = negative_mask * torch.exp(sim)
        
        loss = -torch.log(numerator / denominator.sum(dim=1))
        return loss.mean()

class WingLoss(nn.Module):
    def __init__(self, omega: float = 10.0, epsilon: float = 2.0):
        """
        Wing Loss for robust regression, especially useful for facial landmark detection
        
        Args:
            omega: Sets the range for nonlinear optimization
            epsilon: Controls the curvature of the nonlinear part
        """
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = (target - pred).abs()
        c = self.omega * (1.0 - torch.log(1.0 + self.omega/self.epsilon))
        
        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1.0 + delta/self.epsilon),
            delta - c
        )
        return loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, square: bool = False):
        """
        Dice Loss for image segmentation tasks
        
        Args:
            smooth: Smoothing factor to prevent division by zero
            square: Whether to square the terms in numerator and denominator
        """
        super().__init__()
        self.smooth = smooth
        self.square = square
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        if self.square:
            intersection = (pred * target).sum(dim=(2,3))
            union = (pred * pred).sum(dim=(2,3)) + (target * target).sum(dim=(2,3))
        else:
            intersection = (pred * target).sum(dim=(2,3))
            union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
            
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()