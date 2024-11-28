import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import json
import os
import datetime
from nexus.core.base import NexusModule


class NexusModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._is_training = True
        self._device = "cpu"
        self._frozen = False
        self._gradient_checkpointing_enabled = False
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError
        
    def get_config(self) -> Dict[str, Any]:
        return self.config
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NexusModule':
        return cls(config)
        
    def save(self, path: str, include_optimizer: bool = False) -> None:
        """Enhanced save with optimizer state and metadata"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'state_dict': self.state_dict(),
            'config': self.config,
            'model_type': self.__class__.__name__,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        if include_optimizer and hasattr(self, 'optimizer'):
            save_dict['optimizer_state'] = self.optimizer.state_dict()
            
        torch.save(save_dict, path)
        
    @classmethod
    def load(cls, path: str, map_location: Optional[Union[str, torch.device]] = None) -> 'NexusModule':
        """Enhanced load with device mapping and validation"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        checkpoint = torch.load(path, map_location=map_location)
        
        # Validate model type
        if checkpoint.get('model_type') != cls.__name__:
            raise ValueError(f"Model type mismatch. Expected {cls.__name__}, got {checkpoint.get('model_type')}")
            
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        
        if 'optimizer_state' in checkpoint and hasattr(model, 'optimizer'):
            model.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
        return model
        
    def to_device(self, device: Union[str, torch.device]) -> 'NexusModule':
        """Move model to specified device with validation"""
        if isinstance(device, str) and device not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Invalid device: {device}")
        self._device = device
        return self.to(device)
        
    def train(self, mode: bool = True) -> 'NexusModule':
        """Enhanced training mode setter"""
        if mode != self._is_training:
            super().train(mode)
            self._is_training = mode
        return self
        
    def freeze(self) -> 'NexusModule':
        """Freeze model parameters"""
        for param in self.parameters():
            param.requires_grad = False
        self._frozen = True
        return self
        
    def unfreeze(self) -> 'NexusModule':
        """Unfreeze model parameters"""
        for param in self.parameters():
            param.requires_grad = True
        self._frozen = False
        return self
        
    def enable_gradient_checkpointing(self) -> 'NexusModule':
        """Enable gradient checkpointing for memory efficiency"""
        self._gradient_checkpointing_enabled = True
        self.gradient_checkpointing_enable()
        return self
        
    def get_parameter_count(self) -> Dict[str, int]:
        """Get detailed parameter counts"""
        total = trainable = frozen = 0
        for param in self.parameters():
            count = param.numel()
            total += count
            if param.requires_grad:
                trainable += count
            else:
                frozen += count
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen
        }
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get model memory usage in MB"""
        mem_params = sum([param.nelement() * param.element_size() for param in self.parameters()])
        mem_buffers = sum([buf.nelement() * buf.element_size() for buf in self.buffers()])
        return {
            'parameters_mb': mem_params / 1024**2,
            'buffers_mb': mem_buffers / 1024**2,
            'total_mb': (mem_params + mem_buffers) / 1024**2
        }