from typing import Dict, Any, Union, List, Optional, Tuple
import torch
import numpy as np
from PIL import Image
import json
from pathlib import Path
from dataclasses import dataclass
from nexus.utils.logging import Logger
from torchvision import transforms as T

@dataclass
class InputConfig:
    """Configuration for input processing."""
    input_type: str  # One of: 'text', 'image', 'audio', 'multimodal'
    max_length: Optional[int] = None
    image_size: Optional[Tuple[int, int]] = None
    normalize: bool = True
    augment: bool = False

    def __post_init__(self):
        valid_types = {'text', 'image', 'audio', 'multimodal'}
        if self.input_type not in valid_types:
            raise ValueError(f"input_type must be one of {valid_types}")
        
        if self.image_size and not (
            isinstance(self.image_size, tuple) and 
            len(self.image_size) == 2 and
            all(isinstance(x, int) and x > 0 for x in self.image_size)
        ):
            raise ValueError("image_size must be tuple of two positive integers")

class InputProcessor:
    """Processes various types of inputs into tensor format."""
    
    def __init__(self, config: InputConfig):
        self.config = config
        self.logger = Logger(self.__class__.__name__)
        self.processors = {
            'text': self.process_text,
            'image': self.process_image,
            'audio': self.process_audio,
            'multimodal': self.process_multimodal
        }
        
        # Setup image transforms
        self.image_transforms = []
        if self.config.image_size:
            self.image_transforms.append(T.Resize(self.config.image_size))
        self.image_transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if self.config.normalize else T.Lambda(lambda x: x)
        ])
        self.image_transform = T.Compose(self.image_transforms)
        
    def process(self, input_data: Any) -> Dict[str, torch.Tensor]:
        """Process input data based on configured type."""
        try:
            processor = self.processors.get(self.config.input_type)
            if processor is None:
                raise ValueError(f"Unsupported input type: {self.config.input_type}")
            return processor(input_data)
        except Exception as e:
            self.logger.error(f"Error processing {self.config.input_type} input: {str(e)}")
            raise
        
    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text input."""
        # Placeholder for text processing logic
        raise NotImplementedError("Text processing not yet implemented")
        
    def process_image(self, image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process image input from various formats."""
        try:
            # Handle different input types
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(np.uint8(image)).convert('RGB')
            elif isinstance(image, torch.Tensor):
                if image.ndim == 4:
                    image = image.squeeze(0)
                if image.ndim != 3:
                    raise ValueError(f"Expected 3D tensor, got shape {image.shape}")
                image = T.ToPILImage()(image)
            elif not isinstance(image, Image.Image):
                raise TypeError(f"Unsupported image type: {type(image)}")
                
            # Apply transforms
            tensor = self.image_transform(image)
            
            return {"image": tensor}
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise
            
    def process_audio(self, audio: Any) -> Dict[str, torch.Tensor]:
        """Process audio input."""
        # Placeholder for audio processing logic
        raise NotImplementedError("Audio processing not yet implemented")
        
    def process_multimodal(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process multimodal input."""
        # Placeholder for multimodal processing logic
        raise NotImplementedError("Multimodal processing not yet implemented")