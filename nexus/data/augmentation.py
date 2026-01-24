import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import List, Optional, Tuple, Union
import random
import numpy as np
from nexus.utils.logging import Logger
from PIL import Image

class AugmentationPipeline:
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]],
        augmentation_strength: float = 1.0,
        include_random_crop: bool = True,
        include_color_jitter: bool = True,
        include_random_flip: bool = True
    ):
        """Initialize augmentation pipeline.
        
        Args:
            image_size: Target image size as int or (height, width) tuple
            augmentation_strength: Multiplier for augmentation intensity (0.0 to 1.0)
            include_random_crop: Whether to include random cropping
            include_color_jitter: Whether to include color jittering
            include_random_flip: Whether to include random flipping
        """
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.augmentation_strength = max(0.0, min(1.0, augmentation_strength))
        self.logger = Logger(self.__class__.__name__)
        
        self.transforms = []
        
        if include_random_crop:
            self.transforms.extend([
                T.RandomResizedCrop(
                    self.image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.333),
                    interpolation=T.InterpolationMode.BILINEAR
                )
            ])
            
        if include_color_jitter and self.augmentation_strength > 0:
            self.transforms.extend([
                T.ColorJitter(
                    brightness=0.4 * self.augmentation_strength,
                    contrast=0.4 * self.augmentation_strength,
                    saturation=0.4 * self.augmentation_strength,
                    hue=0.1 * self.augmentation_strength
                )
            ])
            
        if include_random_flip:
            self.transforms.extend([
                T.RandomHorizontalFlip(p=0.5)
            ])
            
        # Always include ToTensor and normalization
        self.transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
            
        self.transforms = T.Compose(self.transforms)
        
    def __call__(self, image: Union[torch.Tensor, np.ndarray, Image.Image]) -> torch.Tensor:
        """Apply augmentation pipeline to an image.
        
        Args:
            image: Input image as tensor, numpy array or PIL Image
            
        Returns:
            Augmented image as tensor
        """
        try:
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    image = image.permute(1, 2, 0)
                image = Image.fromarray((image.numpy() * 255).astype(np.uint8))
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
                
            return self.transforms(image)
            
        except Exception as e:
            self.logger.error(f"Augmentation failed: {str(e)}")
            raise

class Mixup:
    def __init__(self, alpha: float = 1.0):
        """Initialize Mixup augmentation.
        
        Args:
            alpha: Parameter for beta distribution. Higher means more mixing.
        """
        self.alpha = max(0.0, alpha)
        self.logger = Logger(self.__class__.__name__)
        
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply Mixup augmentation to a batch of images.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
            
        Returns:
            Tuple of (mixed images, labels_a, labels_b, mix ratio lambda)
        """
        try:
            if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
                raise TypeError("Images and labels must be torch tensors")
                
            if images.dim() != 4:
                raise ValueError("Images must have shape (B, C, H, W)")
                
            batch_size = images.size(0)
            if batch_size < 2:
                return images, labels, labels, 1.0
                
            if self.alpha > 0:
                lam = float(np.random.beta(self.alpha, self.alpha))
            else:
                lam = 1.0
                
            # Ensure lambda is between 0 and 1
            lam = max(0.0, min(1.0, lam))
                
            index = torch.randperm(batch_size, device=images.device)
            
            mixed_images = lam * images + (1 - lam) * images[index]
            labels_a, labels_b = labels, labels[index]
            
            return mixed_images, labels_a, labels_b, lam
            
        except Exception as e:
            self.logger.error(f"Mixup augmentation failed: {str(e)}")
            raise