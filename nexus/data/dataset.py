import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Optional, Callable, Dict, Any, List, Union
from pathlib import Path
import os
from PIL import Image
from nexus.utils.logging import Logger

from .validation import DataValidation

class Dataset(TorchDataset):
    """Dataset class for loading image data from a directory structure.
    
    Expects data organized in class folders:
    data_dir/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img3.jpg
            ...
    """
    
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_images: bool = False
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Root directory containing class folders
            transform: Optional transform for images
            target_transform: Optional transform for labels/targets
            cache_images: If True, cache images in memory after first load
        """
        self.data_dir = Path(data_dir)
        DataValidation.validate_directory_exists(self.data_dir)
            
        self.transform = transform
        self.target_transform = target_transform
        self.cache_images = cache_images
        self.logger = Logger(self.__class__.__name__)
        
        # Setup class mapping
        self.class_to_idx = {}
        self.classes = []
        
        # Get all image files and their corresponding labels
        self.samples: List[Path] = []
        self.targets: List[int] = []
        self.image_cache: Dict[Path, Image.Image] = {}
        
        self._load_dataset()
        
    def _load_dataset(self) -> None:
        """Load dataset structure and build class mapping."""
        dir_contents = list(self.data_dir.iterdir())
        DataValidation.validate_not_empty(dir_contents, "data_dir contents")
            
        # Build class mapping
        for class_folder in sorted(x for x in self.data_dir.iterdir() if x.is_dir()):
            class_name = class_folder.name
            class_idx = len(self.classes)
            self.class_to_idx[class_name] = class_idx
            self.classes.append(class_name)
            
            # Get all valid images
            for img_path in class_folder.glob("*"):
                if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append(img_path)
                    self.targets.append(class_idx)
                    
        DataValidation.validate_not_empty(self.samples, "samples")
            
        self.logger.info(f"Loaded dataset with {len(self.samples)} images across {len(self.classes)} classes")
        
    def _load_image(self, img_path: Path) -> Image.Image:
        """Load and convert image, with caching if enabled."""
        if self.cache_images and img_path in self.image_cache:
            return self.image_cache[img_path]
            
        try:
            image = Image.open(img_path).convert('RGB')
            if self.cache_images:
                self.image_cache[img_path] = image
            return image
        except Exception as e:
            self.logger.error(f"Error loading image {img_path}: {str(e)}")
            raise
            
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index.
        
        Returns:
            Dict containing 'image' and 'label' keys
        """
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
            
        img_path = self.samples[idx]
        target = self.targets[idx]
        
        # Load and transform image
        image = self._load_image(img_path)
        if self.transform is not None:
            image = self.transform(image)
            
        # Transform target if needed
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return {
            "image": image,
            "label": target,
            "path": str(img_path)
        }