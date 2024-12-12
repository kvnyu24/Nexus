import torch
import numpy as np
from typing import Union, List, Optional, Dict, Any
import pynvml
import os
import psutil
import time
from .logging import Logger
from .apple_gpu import AppleGPUManager
from nexus.core.base import NexusModule

class GPUManager:
    def __init__(self, min_memory_threshold: float = 0.1):
        """
        Initialize GPU Manager
        Args:
            min_memory_threshold: Minimum free memory ratio required to consider a GPU usable
        """
        self.initialized = False
        self.logger = Logger("GPUManager")
        self.min_memory_threshold = min_memory_threshold
        self.device_type = self._detect_device_type()
        self.apple_gpu = AppleGPUManager() if self.device_type == 'mps' else None
        self._cached_memory_info: Optional[List[Dict[str, float]]] = None
        self._last_memory_check = 0
        
    def _detect_device_type(self) -> str:
        # Check for Apple Silicon
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Verify MPS backend is actually working with more thorough tests
            try:
                test_tensor = torch.zeros(1).to('mps')
                test_tensor = test_tensor + 1
                test_tensor = test_tensor * 2
                test_tensor = torch.nn.functional.relu(test_tensor)
                _ = test_tensor.cpu()  # Test transfer back
                self.logger.info("Apple Silicon (MPS) device verified and working")
                return 'mps'
            except Exception as e:
                self.logger.warning(f"MPS device detected but failed verification: {e}")
                self.logger.info("Falling back to CPU")
                return 'cpu'
                
        # Check for NVIDIA
        try:
            pynvml.nvmlInit()
            self.initialized = True
            self.device_count = pynvml.nvmlDeviceGetCount()
            if self.device_count > 0:
                self.logger.info(f"NVIDIA CUDA device(s) detected: {self.device_count}")
                return 'cuda'
            raise RuntimeError("No NVIDIA devices found")
        except:
            # Check for ROCm (AMD)
            if torch.version.hip is not None and torch.cuda.is_available():
                try:
                    test_tensor = torch.zeros(1).cuda()
                    self.device_count = torch.cuda.device_count()
                    self.logger.info(f"AMD ROCm device(s) detected: {self.device_count}")
                    return 'rocm'
                except Exception as e:
                    self.logger.warning(f"ROCm initialization failed: {e}")
            
            self.logger.info("No GPU detected, using CPU")
            return 'cpu'

    def get_gpu_memory_info(self, force_refresh: bool = False) -> List[dict]:
        """Get GPU memory information with caching"""
        current_time = time.time()
        if (not force_refresh and 
            self._cached_memory_info is not None and 
            current_time - self._last_memory_check < 1.0):  # Cache for 1 second
            return self._cached_memory_info
            
        if not self.initialized:
            if self.device_type == 'mps':
                return [self.apple_gpu.get_memory_info()]
            self.logger.warning("GPU manager not initialized, no memory info available")
            return []
            
        memory_info = []
        try:
            for i in range(self.device_count):
                if self.device_type == 'cuda':
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    device_info = {
                        "device": i,
                        "total": info.total / 1024**2,  # MB
                        "free": info.free / 1024**2,    # MB
                        "used": info.used / 1024**2,    # MB
                        "gpu_util": utilization.gpu,    # %
                        "memory_util": utilization.memory,  # %
                        "temperature": temperature      # °C
                    }
                else:  # ROCm
                    torch.cuda.set_device(i)
                    device_info = {
                        "device": i,
                        "total": torch.cuda.get_device_properties(i).total_memory / 1024**2,
                        "free": torch.cuda.memory_allocated(i) / 1024**2,
                        "used": torch.cuda.memory_reserved(i) / 1024**2
                    }
                    
                memory_info.append(device_info)
                self.logger.info(
                    f"GPU {i}: {device_info['used']:.0f}MB used / "
                    f"{device_info['free']:.0f}MB free / "
                    f"{device_info['total']:.0f}MB total"
                )
                
            self._cached_memory_info = memory_info
            self._last_memory_check = current_time
            return memory_info
            
        except Exception as e:
            self.logger.error(f"Error getting GPU memory info: {e}")
            return []
        
    def get_optimal_device(self, required_memory: Optional[float] = None) -> torch.device:
        """
        Get optimal device based on memory availability and requirements
        Args:
            required_memory: Required GPU memory in MB
        """
        if self.device_type == 'mps':
            # Use Apple Silicon GPU
            torch.set_default_dtype(torch.float32)  # Ensure proper dtype for MPS
            return torch.device('mps')
            
        elif self.device_type in ['cuda', 'rocm'] and torch.cuda.is_available():
            memory_info = self.get_gpu_memory_info()
            if not memory_info:
                self.logger.info("No GPU memory info available, using device 0")
                return torch.device('cuda:0')
                
            # Filter devices based on memory requirements
            available_devices = []
            for idx, info in enumerate(memory_info):
                free_ratio = info['free'] / info['total']
                if free_ratio >= self.min_memory_threshold:
                    if required_memory is None or info['free'] >= required_memory:
                        available_devices.append((idx, info['free']))
                        
            if not available_devices:
                self.logger.warning("No GPU meets memory requirements, using CPU")
                return torch.device('cpu')
                
            # Select device with most free memory
            optimal_device = max(available_devices, key=lambda x: x[1])[0]
            device_name = 'cuda' if self.device_type == 'cuda' else 'rocm'
            self.logger.info(f"Selected optimal GPU device: {device_name}:{optimal_device}")
            return torch.device(f'cuda:{optimal_device}')  # Both CUDA and ROCm use cuda: prefix
            
        else:
            self.logger.info("Using CPU device")
            return torch.device('cpu')

class AutoDevice:
    def __init__(self, 
                 tensor_or_module: Union[torch.Tensor, NexusModule],
                 required_memory: Optional[float] = None,
                 fallback_to_cpu: bool = True):
        self.gpu_manager = GPUManager()
        self.device = self.gpu_manager.get_optimal_device(required_memory)
        self.tensor_or_module = tensor_or_module
        self.fallback_to_cpu = fallback_to_cpu
        self.original_device = None
        
    def __enter__(self):
        try:
            if isinstance(self.tensor_or_module, torch.Tensor):
                self.original_device = self.tensor_or_module.device
                return self.tensor_or_module.to(self.device)
            else:
                self.original_device = next(self.tensor_or_module.parameters()).device
                return self.tensor_or_module.to(self.device)
        except RuntimeError as e:
            if self.fallback_to_cpu:
                self.logger.warning(f"Failed to move to {self.device}, falling back to CPU: {e}")
                return self.tensor_or_module.to('cpu')
            raise
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_device is not None and self.fallback_to_cpu:
            try:
                self.tensor_or_module.to(self.original_device)
            except Exception as e:
                self.logger.warning(f"Failed to move back to original device: {e}")