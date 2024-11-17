from typing import Dict, Any, Union, List, Optional
import torch
import numpy as np
from PIL import Image
import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class InputConfig:
    input_type: str
    max_length: Optional[int] = None
    image_size: Optional[tuple] = None
    normalize: bool = True
    augment: bool = False

class InputProcessor:
    def __init__(self, config: InputConfig):
        self.config = config
        self.processors = {
            'text': self.process_text,
            'image': self.process_image,
            'audio': self.process_audio,
            'multimodal': self.process_multimodal
        }
        
    def process(self, input_data: Any) -> Dict[str, torch.Tensor]:
        processor = self.processors.get(self.config.input_type)
        if processor is None:
            raise ValueError(f"Unsupported input type: {self.config.input_type}")
        return processor(input_data)
        
    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        # Implement text processing logic
        pass
        
    def process_image(self, image: Union[str, Image.Image]) -> Dict[str, torch.Tensor]:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            
        if self.config.image_size:
            image = image.resize(self.config.image_size)
            
        # Convert to tensor
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        
        if self.config.normalize:
            tensor = tensor / 255.0
            tensor = self.normalize_image(tensor)
            
        return {"image": tensor}
        
    @staticmethod
    def normalize_image(tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        return (tensor - mean) / std 