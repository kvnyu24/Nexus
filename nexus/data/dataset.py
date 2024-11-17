import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import os
from PIL import Image

class Dataset(TorchDataset):
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        # Get all image files and their corresponding labels
        self.samples = []
        self.targets = []
        
        for class_folder in sorted(os.listdir(self.data_dir)):
            class_path = self.data_dir / class_folder
            if not class_path.is_dir():
                continue
                
            class_idx = len(self.targets)
            for img_file in class_path.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append(img_file)
                    self.targets.append(class_idx)
                    
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.samples[idx]
        target = self.targets[idx]
        
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return {
            "image": image,
            "label": target
        } 