import torch
from nexus.utils.logging import Logger
from typing import Dict, Type
import os
from nexus.core.base import NexusModule

class AppleGPUManager:
    def __init__(self):
        self.logger = Logger("AppleGPUManager")
        self.device_type = self._detect_mps_device()
        self.initialized = False
        
    def _detect_mps_device(self) -> str:
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            self.logger.warning("MPS not available on this system")
            return 'cpu'
            
        try:
            # Test MPS backend with small tensor
            test_tensor = torch.zeros(1).to('mps')
            # Test basic operations
            test_result = test_tensor + 1
            test_result = test_result.to('cpu')  # Test transfer back to CPU
            
            self.initialized = True
            self.logger.info("Apple Silicon MPS device initialized successfully")
            return 'mps'
        except Exception as e:
            self.logger.error(f"MPS initialization failed: {e}")
            return 'cpu'

    def get_memory_info(self) -> Dict[str, float]:
        """
        Get memory information for Apple Silicon GPU.
        Note: Apple's MPS doesn't provide direct memory query methods
        """
        if not self.initialized:
            return {"available": 0, "total": 0, "used": 0}
            
        try:
            # Use platform-specific command to get GPU memory info
            import subprocess
            result = subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], 
                                 capture_output=True, text=True)
            used_memory = int(result.stdout.strip()) * 1024  # Convert KB to bytes
            
            return {
                "used": used_memory,
                "total": -1,  # Not directly accessible on MPS
                "available": -1  # Not directly accessible on MPS
            }
        except Exception as e:
            self.logger.warning(f"Failed to get memory info: {e}")
            return {"available": 0, "total": 0, "used": 0}
            
    def optimize_for_metal(self, model: NexusModule) -> NexusModule:
        """Apply Metal-specific optimizations to the model"""
        if not self.initialized:
            return model
            
        # Set appropriate dtype for Metal
        model = model.to(torch.float32)  # MPS works best with float32
        
        # Enable graph mode for better performance
        torch._C._jit_set_bailout_depth(20)
        
        return model

    def create_optimizer(self, model: NexusModule, 
                        optimizer_class: Type[torch.optim.Optimizer],
                        **kwargs) -> torch.optim.Optimizer:
        """Create an optimizer optimized for Metal"""
        if 'lr' in kwargs:
            # Adjust learning rate for MPS (typically needs to be lower)
            kwargs['lr'] *= 0.1
            
        return optimizer_class(model.parameters(), **kwargs)