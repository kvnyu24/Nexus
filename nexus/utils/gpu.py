import torch
import numpy as np
from typing import Union, List, Optional
import pynvml
import os

class GPUManager:
    def __init__(self):
        self.initialized = False
        self.device_type = self._detect_device_type()
        
    def _detect_device_type(self) -> str:
        # Check for Apple Silicon
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        # Check for NVIDIA
        try:
            pynvml.nvmlInit()
            self.initialized = True
            self.device_count = pynvml.nvmlDeviceGetCount()
            return 'cuda'
        except:
            # Check for ROCm (AMD)
            if torch.version.hip is not None:
                return 'rocm'
            return 'cpu'

    def get_gpu_memory_info(self) -> List[dict]:
        if not self.initialized:
            return []
            
        memory_info = []
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_info.append({
                "device": i,
                "total": info.total / 1024**2,  # MB
                "free": info.free / 1024**2,    # MB
                "used": info.used / 1024**2     # MB
            })
        return memory_info
        
    def get_optimal_device(self) -> torch.device:
        if self.device_type == 'mps':
            return torch.device('mps')
        elif self.device_type == 'cuda' and torch.cuda.is_available():
            # Your existing NVIDIA GPU selection logic
            memory_info = self.get_gpu_memory_info()
            if not memory_info:
                return torch.device('cuda:0')
            free_memory = [info['free'] for info in memory_info]
            optimal_device = np.argmax(free_memory)
            return torch.device(f'cuda:{optimal_device}')
        elif self.device_type == 'rocm':
            return torch.device('cuda:0')  # ROCm uses CUDA device naming
        else:
            return torch.device('cpu')

class AutoDevice:
    def __init__(self, tensor_or_module: Union[torch.Tensor, torch.nn.Module]):
        self.gpu_manager = GPUManager()
        self.device = self.gpu_manager.get_optimal_device()
        self.tensor_or_module = tensor_or_module
        
    def __enter__(self):
        if isinstance(self.tensor_or_module, torch.Tensor):
            return self.tensor_or_module.to(self.device)
        else:
            return self.tensor_or_module.to(self.device)
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass 