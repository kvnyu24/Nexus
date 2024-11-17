import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import json
import os

class NexusModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._is_training = True
        self._device = "cpu"
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError
        
    def get_config(self) -> Dict[str, Any]:
        return self.config
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NexusModule':
        return cls(config)
        
    def save(self, path: str) -> None:
        """Save model weights and config to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config
        }, path)
        
    @classmethod
    def load(cls, path: str) -> 'NexusModule':
        """Load model from disk"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
        
    def to_device(self, device: Union[str, torch.device]) -> 'NexusModule':
        """Move model to specified device"""
        self._device = device
        return self.to(device)
        
    def train(self, mode: bool = True) -> 'NexusModule':
        """Set training mode"""
        super().train(mode)
        self._is_training = mode
        return self
        
    def get_parameter_count(self) -> Dict[str, int]:
        """Get number of parameters in model"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable
        }