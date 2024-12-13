import torch
import torchvision.transforms as T
from typing import List, Union, Tuple, Optional, Any
import numpy as np
from PIL import Image
import logging

class Transform:
    """Base transform class that all transforms should inherit from."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def __call__(self, x: Any) -> Any:
        """Apply the transform to input x."""
        raise NotImplementedError
        
    def __repr__(self) -> str:
        return self.__class__.__name__

class Compose(Transform):
    """Composes multiple transforms together."""
    def __init__(self, transforms: List[Transform]):
        super().__init__()
        if not transforms:
            raise ValueError("transforms list cannot be empty")
        self.transforms = transforms
        
    def __call__(self, x: Any) -> Any:
        try:
            for transform in self.transforms:
                x = transform(x)
            return x
        except Exception as e:
            self.logger.error(f"Error in transform pipeline: {str(e)}")
            raise
            
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string

class Resize(Transform):
    """Resize the input image to given size."""
    def __init__(self, size: Union[int, Tuple[int, int]], interpolation: int = Image.BILINEAR):
        super().__init__()
        self.size = size if isinstance(size, tuple) else (size, size)
        self.interpolation = interpolation
        self.transform = T.Resize(self.size, interpolation=interpolation)
        
    def __call__(self, x: Union[Image.Image, torch.Tensor, np.ndarray]) -> Union[Image.Image, torch.Tensor]:
        try:
            if isinstance(x, np.ndarray):
                x = Image.fromarray(x)
            return self.transform(x)
        except Exception as e:
            self.logger.error(f"Failed to resize input: {str(e)}")
            raise

class RandomCrop(Transform):
    """Randomly crop the input image."""
    def __init__(self, size: Union[int, Tuple[int, int]], padding: Optional[int] = None, 
                 padding_mode: str = 'constant', fill: int = 0):
        super().__init__()
        self.size = size if isinstance(size, tuple) else (size, size)
        self.padding = padding
        self.padding_mode = padding_mode
        self.fill = fill
        self.transform = T.RandomCrop(self.size, padding=padding, 
                                    padding_mode=padding_mode, fill=fill)
        
    def __call__(self, x: Union[Image.Image, torch.Tensor, np.ndarray]) -> Union[Image.Image, torch.Tensor]:
        try:
            if isinstance(x, np.ndarray):
                x = Image.fromarray(x)
            return self.transform(x)
        except Exception as e:
            self.logger.error(f"Failed to crop input: {str(e)}")
            raise

class RandomHorizontalFlip(Transform):
    """Randomly flip the input image horizontally."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError("Probability must be between 0 and 1")
        self.p = p
        self.transform = T.RandomHorizontalFlip(p)
        
    def __call__(self, x: Union[Image.Image, torch.Tensor, np.ndarray]) -> Union[Image.Image, torch.Tensor]:
        try:
            if isinstance(x, np.ndarray):
                x = Image.fromarray(x)
            return self.transform(x)
        except Exception as e:
            self.logger.error(f"Failed to flip input: {str(e)}")
            raise

class RandomVerticalFlip(Transform):
    """Randomly flip the input image vertically."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError("Probability must be between 0 and 1")
        self.p = p
        self.transform = T.RandomVerticalFlip(p)
        
    def __call__(self, x: Union[Image.Image, torch.Tensor, np.ndarray]) -> Union[Image.Image, torch.Tensor]:
        try:
            if isinstance(x, np.ndarray):
                x = Image.fromarray(x)
            return self.transform(x)
        except Exception as e:
            self.logger.error(f"Failed to flip input: {str(e)}")
            raise

class RandomRotation(Transform):
    """Randomly rotate the input image."""
    def __init__(self, degrees: Union[float, Tuple[float, float]], 
                 interpolation: int = Image.BILINEAR, expand: bool = False):
        super().__init__()
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.interpolation = interpolation
        self.expand = expand
        self.transform = T.RandomRotation(self.degrees, interpolation=interpolation,
                                        expand=expand)
        
    def __call__(self, x: Union[Image.Image, torch.Tensor, np.ndarray]) -> Union[Image.Image, torch.Tensor]:
        try:
            if isinstance(x, np.ndarray):
                x = Image.fromarray(x)
            return self.transform(x)
        except Exception as e:
            self.logger.error(f"Failed to rotate input: {str(e)}")
            raise

class Normalize(Transform):
    """Normalize a tensor image with mean and standard deviation."""
    def __init__(self, mean: Union[float, List[float]], std: Union[float, List[float]]):
        super().__init__()
        self.mean = mean if isinstance(mean, (list, tuple)) else [mean]
        self.std = std if isinstance(std, (list, tuple)) else [std]
        if any(s <= 0 for s in self.std):
            raise ValueError("Standard deviation must be positive")
        self.transform = T.Normalize(self.mean, self.std)
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        try:
            if not isinstance(x, torch.Tensor):
                raise TypeError("Input must be a torch tensor")
            return self.transform(x)
        except Exception as e:
            self.logger.error(f"Failed to normalize input: {str(e)}")
            raise

class ToTensor(Transform):
    """Convert a PIL Image or numpy.ndarray to tensor."""
    def __init__(self):
        super().__init__()
        self.transform = T.ToTensor()
        
    def __call__(self, x: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        try:
            if isinstance(x, torch.Tensor):
                return x
            return self.transform(x)
        except Exception as e:
            self.logger.error(f"Failed to convert input to tensor: {str(e)}")
            raise