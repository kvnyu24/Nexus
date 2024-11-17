import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import List, Optional, Tuple
import random

class AugmentationPipeline:
    def __init__(
        self,
        image_size: Tuple[int, int],
        augmentation_strength: float = 1.0,
        include_random_crop: bool = True,
        include_color_jitter: bool = True,
        include_random_flip: bool = True
    ):
        self.image_size = image_size
        self.augmentation_strength = augmentation_strength
        
        self.transforms = []
        
        if include_random_crop:
            self.transforms.extend([
                T.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.333)
                )
            ])
            
        if include_color_jitter:
            self.transforms.extend([
                T.ColorJitter(
                    brightness=0.4 * augmentation_strength,
                    contrast=0.4 * augmentation_strength,
                    saturation=0.4 * augmentation_strength,
                    hue=0.1 * augmentation_strength
                )
            ])
            
        if include_random_flip:
            self.transforms.extend([
                T.RandomHorizontalFlip(p=0.5)
            ])
            
        self.transforms = T.Compose(self.transforms)
        
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.transforms(image)

class MixupAugmentation:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Applies Mixup augmentation to a batch of images."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam 