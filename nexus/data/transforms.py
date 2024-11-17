import torch
import torchvision.transforms as T
from typing import List, Union, Tuple, Optional
import numpy as np
from PIL import Image

class Transform:
    def __call__(self, x):
        raise NotImplementedError

class Compose:
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms
        
    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

class Resize(Transform):
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.transform = T.Resize(self.size)
        
    def __call__(self, x: Image.Image) -> Image.Image:
        return self.transform(x)

class RandomCrop(Transform):
    def __init__(self, size: Union[int, Tuple[int, int]], padding: Optional[int] = None):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.padding = padding
        self.transform = T.RandomCrop(self.size, padding=padding)
        
    def __call__(self, x: Image.Image) -> Image.Image:
        return self.transform(x)

class RandomHorizontalFlip(Transform):
    def __init__(self, p: float = 0.5):
        self.transform = T.RandomHorizontalFlip(p)
        
    def __call__(self, x: Image.Image) -> Image.Image:
        return self.transform(x)

class RandomVerticalFlip(Transform):
    def __init__(self, p: float = 0.5):
        self.transform = T.RandomVerticalFlip(p)
        
    def __call__(self, x: Image.Image) -> Image.Image:
        return self.transform(x)

class RandomRotation(Transform):
    def __init__(self, degrees: Union[float, Tuple[float, float]]):
        self.transform = T.RandomRotation(degrees)
        
    def __call__(self, x: Image.Image) -> Image.Image:
        return self.transform(x)

class Normalize(Transform):
    def __init__(self, mean: List[float], std: List[float]):
        self.transform = T.Normalize(mean, std)
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

class ToTensor(Transform):
    def __init__(self):
        self.transform = T.ToTensor()
        
    def __call__(self, x: Image.Image) -> torch.Tensor:
        return self.transform(x) 