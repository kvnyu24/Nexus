"""Unified device manager interface for cross-platform GPU/TPU support."""
from abc import ABC, abstractmethod
from typing import Dict, Type, TYPE_CHECKING
import torch
from enum import Enum

if TYPE_CHECKING:
    from nexus.core.base import NexusModule


class DeviceType(str, Enum):
    """Enumeration of supported device types."""
    CUDA = 'cuda'
    ROCM = 'rocm'
    MPS = 'mps'
    TPU = 'tpu'
    CPU = 'cpu'


class DeviceManager(ABC):
    """Abstract base class for device managers.

    Provides a unified interface for managing different compute devices
    (CUDA, ROCm, MPS, TPU, CPU) with consistent memory info, model
    optimization, and optimizer creation APIs.
    """

    @property
    @abstractmethod
    def device_type(self) -> DeviceType:
        """Get the type of device this manager handles."""
        pass

    @abstractmethod
    def get_device(self) -> torch.device:
        """Get the primary device managed by this manager."""
        pass

    @abstractmethod
    def get_memory_info(self) -> Dict[str, float]:
        """Get memory information for the device.

        Returns:
            Dict with keys:
                - 'total': Total memory in bytes (-1 if unavailable)
                - 'used': Used memory in bytes
                - 'free': Free memory in bytes (-1 if unavailable)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the device is available and initialized."""
        pass

    @abstractmethod
    def optimize_model(self, model: 'NexusModule') -> 'NexusModule':
        """Apply device-specific optimizations to a model.

        Args:
            model: The model to optimize

        Returns:
            The optimized model (may be the same instance)
        """
        pass

    @abstractmethod
    def create_optimizer(
        self,
        model: 'NexusModule',
        optimizer_class: Type[torch.optim.Optimizer],
        **kwargs
    ) -> torch.optim.Optimizer:
        """Create an optimizer with device-specific settings.

        Args:
            model: The model whose parameters to optimize
            optimizer_class: The optimizer class to instantiate
            **kwargs: Additional arguments for the optimizer

        Returns:
            Configured optimizer instance
        """
        pass
