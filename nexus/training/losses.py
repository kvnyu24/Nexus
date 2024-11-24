import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Dict, Any
import math

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
    
class SSIMLoss(nn.Module):
    def __init__(
        self,
        window_size: int = 11,
        size_average: bool = True,
        channel: int = 3,
        sigma: float = 1.5
    ):
        """
        SSIM Loss for comparing structural similarity between images
        
        Args:
            window_size: Size of the gaussian window
            size_average: Whether to average over batch
            channel: Number of channels in the image
            sigma: Standard deviation for Gaussian window
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.sigma = sigma
        
        # Create Gaussian window
        self.window = self._create_window()
        
    def _create_window(self) -> torch.Tensor:
        """Creates a 2D Gaussian window"""
        coords = torch.arange(self.window_size, dtype=torch.float32)
        coords -= (self.window_size - 1) / 2

        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g = g / g.sum()
        
        window = g.unsqueeze(0) * g.unsqueeze(1)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.expand(self.channel, 1, self.window_size, self.window_size)
        
        return window
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Calculate SSIM loss between prediction and target
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            reduction: Reduction method ('mean', 'sum', or 'none')
            
        Returns:
            SSIM loss
        """
        # Move window to same device as input
        window = self.window.to(pred.device)
        
        # Constants for stability
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Calculate means
        mu1 = F.conv2d(pred, window, groups=self.channel, padding=self.window_size//2)
        mu2 = F.conv2d(target, window, groups=self.channel, padding=self.window_size//2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = F.conv2d(pred * pred, window, groups=self.channel, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, groups=self.channel, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, groups=self.channel, padding=self.window_size//2) - mu1_mu2
        
        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Convert to loss (1 - SSIM)
        loss = 1 - ssim_map
        
        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss
    
class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        """
        InfoNCE Loss for contrastive learning
        
        Args:
            temperature: Temperature parameter for scaling logits
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        # Normalize embeddings
        anchors = F.normalize(anchors, dim=1)
        positives = F.normalize(positives, dim=1)
        negatives = F.normalize(negatives, dim=1)
        
        # Positive logits
        pos_logits = torch.sum(anchors * positives, dim=1) / self.temperature
        
        # Negative logits
        neg_logits = torch.matmul(anchors, negatives.t()) / self.temperature
        
        # Combined logits and labels
        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)

class AdaCosLoss(nn.Module):
    def __init__(self, num_classes: int, embedding_size: int):
        """
        Adaptive Cosine Loss for deep metric learning
        
        Args:
            num_classes: Number of classes
            embedding_size: Size of feature embeddings
        """
        super().__init__()
        self.num_classes = num_classes
        self.scale = nn.Parameter(torch.ones(1) * 30.0)
        self.W = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.W)
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize weights
        W_norm = F.normalize(self.W, dim=1)
        
        # Normalize features
        features_norm = F.normalize(features, dim=1)
        
        # Calculate logits
        logits = F.linear(features_norm, W_norm)
        
        # Apply adaptive scaling
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.scale * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / features.size(0)
            theta_med = torch.median(theta[one_hot == 1])
            self.scale.data = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        
        output = self.scale * logits
        return F.cross_entropy(output, labels)

class PolyLoss(nn.Module):
    def __init__(self, epsilon: float = 1.0, power: float = 2.0):
        """
        Polynomial Loss for robust classification
        
        Args:
            epsilon: Balancing factor
            power: Power for polynomial scaling
        """
        super().__init__()
        self.epsilon = epsilon
        self.power = power
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        
        poly_term = self.epsilon * (1 - pt).pow(self.power + 1)
        loss = ce + poly_term
        
        return loss.mean()

class WeightedFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Union[float, List[float]],
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Weighted Focal Loss with class-specific weights
        
        Args:
            alpha: Class weights (float or list of class-specific weights)
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.alpha = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Get class-specific weights
        if self.alpha.dim() > 0:
            alpha_t = self.alpha[targets]
        else:
            alpha_t = self.alpha
            
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    
class SFTLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.2):
        super().__init__()
        self.alpha = alpha  # Weight for verification loss
        self.beta = beta   # Weight for quality loss
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        verification_scores: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Calculate main supervised loss
        supervised_loss = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Calculate verification-based loss
        verification_loss = -torch.log(verification_scores + 1e-10).mean()
        
        losses = {
            "supervised_loss": supervised_loss,
            "verification_loss": verification_loss * self.alpha,
            "total_loss": supervised_loss + (verification_loss * self.alpha)
        }
        
        # Add quality loss if provided
        if quality_scores is not None:
            quality_loss = -torch.log(quality_scores + 1e-10).mean()
            losses["quality_loss"] = quality_loss * self.beta
            losses["total_loss"] = losses["total_loss"] + (quality_loss * self.beta)
            
        return losses
    

class EnhancedSFTLoss(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Loss weights following AdvancedDistillationModule pattern
        self.alpha = config.get("response_loss_weight", 1.0)
        self.beta = config.get("quality_loss_weight", 0.1)
        self.gamma = config.get("hallucination_loss_weight", 0.3)
        
        # Temperature scaling following StableDiffusion pattern
        self.register_buffer("temperature", torch.ones(1))
        
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        quality_scores: torch.Tensor,
        hallucination_scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Response generation loss
        response_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction='none'
        )
        
        if attention_mask is not None:
            response_loss = response_loss * attention_mask.view(-1)
            
        response_loss = response_loss.mean()
        
        # Quality assessment loss following AlphaFold pattern
        quality_loss = -torch.log(quality_scores + 1e-10).mean()
        
        # Hallucination reduction loss
        hallucination_loss = -torch.log(hallucination_scores + 1e-10).mean()
        
        # Combine losses
        total_loss = (
            self.alpha * response_loss +
            self.beta * quality_loss +
            self.gamma * hallucination_loss
        )
        
        return {
            "response_loss": response_loss,
            "quality_loss": quality_loss,
            "hallucination_loss": hallucination_loss,
            "total_loss": total_loss
        }