import torch
import numpy as np
from typing import Union, List, Optional
import pynvml
import os
from .logging import Logger

class GPUManager:
    def __init__(self):
        self.initialized = False
        self.logger = Logger("GPUManager")
        self.device_type = self._detect_device_type()
        
    def _detect_device_type(self) -> str:
        # Check for Apple Silicon
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Verify MPS backend is actually working
            try:
                test_tensor = torch.zeros(1).to('mps')
                self.logger.info("Apple Silicon (MPS) device verified and working")
                return 'mps'
            except Exception as e:
                self.logger.warning(f"MPS device detected but not working properly: {e}")
                self.logger.info("Falling back to CPU")
                return 'cpu'
        # Check for NVIDIA
        try:
            pynvml.nvmlInit()
            self.initialized = True
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.logger.info(f"NVIDIA CUDA device(s) detected: {self.device_count}")
            return 'cuda'
        except:
            # Check for ROCm (AMD)
            if torch.version.hip is not None:
                self.logger.info("AMD ROCm device detected")
                return 'rocm'
            self.logger.info("No GPU detected, using CPU")
            return 'cpu'

    def get_gpu_memory_info(self) -> List[dict]:
        if not self.initialized:
            self.logger.warning("GPU manager not initialized, no memory info available")
            return []
            
        memory_info = []
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            device_info = {
                "device": i,
                "total": info.total / 1024**2,  # MB
                "free": info.free / 1024**2,    # MB
                "used": info.used / 1024**2     # MB
            }
            memory_info.append(device_info)
            self.logger.info(
                f"GPU {i}: {device_info['used']:.0f}MB used / "
                f"{device_info['free']:.0f}MB free / "
                f"{device_info['total']:.0f}MB total"
            )
        return memory_info
        
    def get_optimal_device(self) -> torch.device:
        if self.device_type == 'mps':
            # Set default dtype for MPS
            torch.set_default_dtype(torch.float32)
            return torch.device('mps')
        elif self.device_type == 'cuda' and torch.cuda.is_available():
            memory_info = self.get_gpu_memory_info()
            if not memory_info:
                self.logger.info("No GPU memory info available, using cuda:0")
                return torch.device('cuda:0')
            free_memory = [info['free'] for info in memory_info]
            optimal_device = np.argmax(free_memory)
            self.logger.info(f"Selected optimal GPU device: cuda:{optimal_device}")
            return torch.device(f'cuda:{optimal_device}')
        elif self.device_type == 'rocm':
            self.logger.info("Using ROCm device")
            return torch.device('cuda:0')  # ROCm uses CUDA device naming
        else:
            self.logger.info("Using CPU device")
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