import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Type
from ..core.base import NexusModule
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from .logging import get_logger

logger = get_logger(__name__)

class TPUManager:
    """Manages TPU device allocation and optimization for Nexus modules."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initialized = False
        self._setup_tpu()
        
    def _setup_tpu(self) -> None:
        """Initialize TPU device and verify availability."""
        try:
            self.device = xm.xla_device()
            self.initialized = True
            logger.info(f"TPU device initialized: {self.device}")
        except Exception as e:
            logger.warning(f"TPU initialization failed: {e}")
            self.device = torch.device("cpu")
            
    def optimize_for_tpu(self, model: NexusModule) -> NexusModule:
        """Apply TPU-specific optimizations to the model."""
        if not self.initialized:
            return model
            
        # Convert model to TPU format
        model = model.to(self.device)
        
        # Enable TPU-specific optimizations
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
            
        # Mark model for TPU compilation
        xm.mark_step()
        
        return model
        
    def create_tpu_optimizer(
        self,
        model: NexusModule,
        optimizer_class: Type[torch.optim.Optimizer],
        **kwargs
    ) -> torch.optim.Optimizer:
        """Create TPU-optimized optimizer."""
        # Adjust learning rate for TPU (typically needs to be scaled)
        if "lr" in kwargs:
            kwargs["lr"] *= xm.xrt_world_size()
            
        optimizer = optimizer_class(model.parameters(), **kwargs)
        return optimizer
        
    def create_tpu_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
        device_iterations: int = 8
    ) -> pl.MpDeviceLoader:
        """Create TPU-optimized dataloader."""
        if not self.initialized:
            return dataloader
            
        return pl.MpDeviceLoader(
            dataloader,
            self.device,
            device_iterations=device_iterations
        )
        
    def get_tpu_metrics(self) -> Dict[str, Any]:
        """Get TPU device metrics and statistics."""
        if not self.initialized:
            return {}
            
        metrics = {
            "device_type": str(self.device),
            "memory_allocated": xm.get_memory_info(self.device)["kb_allocated"],
            "world_size": xm.xrt_world_size(),
            "device_ordinal": xm.get_ordinal(),
            "is_master": xm.is_master_ordinal()
        }
        
        return metrics

class TPUModule(NexusModule):
    """Mixin class for TPU-specific functionality in Nexus modules."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tpu_manager = TPUManager(config)
        
    def to_tpu(self) -> 'TPUModule':
        """Move module to TPU device with optimizations."""
        return self.tpu_manager.optimize_for_tpu(self)
        
    def create_tpu_optimizer(
        self,
        optimizer_class: Type[torch.optim.Optimizer],
        **kwargs
    ) -> torch.optim.Optimizer:
        """Create TPU-optimized optimizer for this module."""
        return self.tpu_manager.create_tpu_optimizer(
            self,
            optimizer_class,
            **kwargs
        )
        
    def sync_tpu(self) -> None:
        """Synchronize TPU device operations."""
        if self.tpu_manager.initialized:
            xm.mark_step() 