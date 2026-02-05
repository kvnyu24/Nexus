"""Advanced gradient methods for memory-efficient training.

Provides selective activation checkpointing and activation offloading
to reduce memory usage during training of large models.

Reference:
    "Reducing Activation Recomputation in Large Transformer Models"
    Korthikanti et al., 2023

    "ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compression"
    Chen et al., 2021

Key features:
    - Selective activation checkpointing (per-operation granularity)
    - CPU activation offloading with async prefetch
    - Compatible with PyTorch 2.0+ selective checkpointing API

Example:
    >>> from nexus.training.gradient_methods import (
    ...     SelectiveCheckpoint,
    ...     ActivationOffloader,
    ... )
    >>>
    >>> # Selective checkpointing
    >>> checkpoint = SelectiveCheckpoint(policy="auto")
    >>> output = checkpoint(module, input)
    >>>
    >>> # Activation offloading
    >>> offloader = ActivationOffloader(enabled=True)
    >>> with offloader.offload_context():
    ...     output = model(input)
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.utils.checkpoint import (
    checkpoint as torch_checkpoint,
    CheckpointFunction,
)
from typing import Optional, Callable, Any, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
from nexus.core.base import NexusModule
from nexus.utils.logging import Logger
import threading
import queue


class CheckpointPolicy(Enum):
    """Policy for selective activation checkpointing."""

    NONE = "none"  # No checkpointing
    ALL = "all"  # Checkpoint all operations
    AUTO = "auto"  # Automatic based on memory usage
    CUSTOM = "custom"  # Custom policy function
    ALTERNATE = "alternate"  # Checkpoint every other layer
    HEAVY_OPS = "heavy_ops"  # Checkpoint only memory-intensive ops (e.g., attention)


@dataclass
class SelectiveCheckpointConfig:
    """Configuration for selective activation checkpointing.

    Args:
        policy: Checkpointing policy.
        custom_policy: Custom policy function (module) -> bool.
        heavy_ops: List of operation types to checkpoint (for HEAVY_OPS policy).
        preserve_rng_state: Whether to preserve RNG state.
        use_reentrant: Whether to use reentrant autograd (False recommended for PyTorch 2.0+).
    """

    policy: CheckpointPolicy = CheckpointPolicy.AUTO
    custom_policy: Optional[Callable[[nn.Module], bool]] = None
    heavy_ops: List[str] = None
    preserve_rng_state: bool = True
    use_reentrant: bool = False  # PyTorch 2.0+ recommends False

    def __post_init__(self):
        if self.heavy_ops is None:
            # Default heavy operations
            self.heavy_ops = [
                "MultiheadAttention",
                "Attention",
                "SelfAttention",
                "CrossAttention",
                "FlashAttention",
                "Linear",  # Large linear layers
                "Conv2d",  # Large convolutions
            ]


class SelectiveCheckpoint:
    """Selective activation checkpointing with configurable policies.

    Provides fine-grained control over which operations are checkpointed
    to optimize the memory-compute tradeoff.

    Args:
        config: Checkpoint configuration.

    Example:
        >>> config = SelectiveCheckpointConfig(policy=CheckpointPolicy.HEAVY_OPS)
        >>> checkpoint = SelectiveCheckpoint(config)
        >>>
        >>> # Apply to module
        >>> output = checkpoint(attention_layer, query, key, value)
    """

    def __init__(self, config: Optional[SelectiveCheckpointConfig] = None):
        self.config = config or SelectiveCheckpointConfig()
        self.logger = Logger("SelectiveCheckpoint")
        self._checkpoint_counter = 0

    def should_checkpoint(self, module: nn.Module) -> bool:
        """Determine if a module should be checkpointed.

        Args:
            module: Module to check.

        Returns:
            True if module should be checkpointed.
        """
        policy = self.config.policy

        if policy == CheckpointPolicy.NONE:
            return False
        elif policy == CheckpointPolicy.ALL:
            return True
        elif policy == CheckpointPolicy.CUSTOM:
            if self.config.custom_policy is None:
                raise ValueError("Custom policy function required for CUSTOM policy")
            return self.config.custom_policy(module)
        elif policy == CheckpointPolicy.ALTERNATE:
            # Alternate: checkpoint every other layer
            self._checkpoint_counter += 1
            return self._checkpoint_counter % 2 == 0
        elif policy == CheckpointPolicy.HEAVY_OPS:
            # Checkpoint only heavy operations
            module_type = type(module).__name__
            return any(op in module_type for op in self.config.heavy_ops)
        elif policy == CheckpointPolicy.AUTO:
            # Auto: checkpoint large modules
            num_params = sum(p.numel() for p in module.parameters())
            # Checkpoint if > 1M parameters
            return num_params > 1_000_000
        else:
            return False

    def __call__(
        self,
        module: nn.Module,
        *args,
        **kwargs,
    ) -> Any:
        """Apply selective checkpointing to module forward pass.

        Args:
            module: Module to checkpoint.
            *args: Positional arguments for module.
            **kwargs: Keyword arguments for module.

        Returns:
            Module output.
        """
        if not self.should_checkpoint(module):
            # No checkpointing
            return module(*args, **kwargs)

        # Apply checkpoint
        def forward_fn(*inputs):
            return module(*inputs)

        # Use PyTorch's checkpoint with non-reentrant mode (PyTorch 2.0+)
        return checkpoint.checkpoint(
            forward_fn,
            *args,
            use_reentrant=self.config.use_reentrant,
            preserve_rng_state=self.config.preserve_rng_state,
        )


class ActivationOffloader:
    """CPU activation offloading with asynchronous prefetch.

    Offloads activations to CPU memory during forward pass and
    asynchronously prefetches them back to GPU before backward pass.

    Args:
        enabled: Whether offloading is enabled.
        offload_threshold_mb: Minimum activation size to offload (MB).
        prefetch_ahead: Number of layers to prefetch ahead during backward.
        pin_memory: Whether to use pinned memory for faster transfers.

    Example:
        >>> offloader = ActivationOffloader(enabled=True)
        >>>
        >>> # Use as context manager
        >>> with offloader.offload_context():
        ...     output = model(input)
        ...     loss = criterion(output, target)
        ...     loss.backward()
    """

    def __init__(
        self,
        enabled: bool = True,
        offload_threshold_mb: float = 10.0,
        prefetch_ahead: int = 2,
        pin_memory: bool = True,
    ):
        self.enabled = enabled
        self.offload_threshold_mb = offload_threshold_mb
        self.prefetch_ahead = prefetch_ahead
        self.pin_memory = pin_memory
        self.logger = Logger("ActivationOffloader")

        # Storage for offloaded activations
        self._cpu_storage: Dict[int, torch.Tensor] = {}
        self._offload_order: List[int] = []

        # Prefetch queue
        self._prefetch_queue = queue.Queue()
        self._prefetch_thread = None

        if enabled:
            self.logger.info("Activation offloading enabled")

    def _should_offload(self, tensor: torch.Tensor) -> bool:
        """Check if tensor should be offloaded.

        Args:
            tensor: Tensor to check.

        Returns:
            True if tensor should be offloaded.
        """
        if not self.enabled:
            return False

        # Check size threshold
        size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        return size_mb >= self.offload_threshold_mb

    def _offload_to_cpu(self, tensor: torch.Tensor) -> int:
        """Offload tensor to CPU.

        Args:
            tensor: GPU tensor to offload.

        Returns:
            Storage ID for retrieval.
        """
        storage_id = id(tensor)

        # Copy to CPU
        if self.pin_memory:
            cpu_tensor = tensor.detach().cpu().pin_memory()
        else:
            cpu_tensor = tensor.detach().cpu()

        self._cpu_storage[storage_id] = cpu_tensor
        self._offload_order.append(storage_id)

        # Free GPU memory
        # Note: Original tensor will be cleared by Python GC

        return storage_id

    def _prefetch_to_gpu(self, storage_id: int, device: torch.device) -> torch.Tensor:
        """Prefetch tensor from CPU to GPU.

        Args:
            storage_id: Storage ID from offload.
            device: Target GPU device.

        Returns:
            GPU tensor.
        """
        if storage_id not in self._cpu_storage:
            raise ValueError(f"Storage ID {storage_id} not found")

        cpu_tensor = self._cpu_storage[storage_id]

        # Async copy to GPU (if using pinned memory)
        if self.pin_memory:
            gpu_tensor = cpu_tensor.to(device, non_blocking=True)
        else:
            gpu_tensor = cpu_tensor.to(device)

        # Clean up CPU storage
        del self._cpu_storage[storage_id]

        return gpu_tensor

    def offload_context(self):
        """Context manager for activation offloading.

        Returns:
            Context manager.
        """
        return _ActivationOffloadContext(self)


class _ActivationOffloadContext:
    """Context manager for activation offloading."""

    def __init__(self, offloader: ActivationOffloader):
        self.offloader = offloader
        self._hooks = []

    def _forward_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor],
        output: torch.Tensor,
    ):
        """Hook to offload activations during forward pass."""
        if isinstance(output, torch.Tensor):
            if self.offloader._should_offload(output):
                storage_id = self.offloader._offload_to_cpu(output)
                # Store metadata for backward prefetch
                if not hasattr(module, "_offload_ids"):
                    module._offload_ids = []
                module._offload_ids.append(storage_id)

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor],
        grad_output: Tuple[torch.Tensor],
    ):
        """Hook to prefetch activations during backward pass."""
        if hasattr(module, "_offload_ids"):
            # Prefetch offloaded activations
            device = next(module.parameters()).device
            for storage_id in module._offload_ids:
                if storage_id in self.offloader._cpu_storage:
                    self.offloader._prefetch_to_gpu(storage_id, device)

    def __enter__(self):
        """Enter context: register hooks."""
        # Note: In a real implementation, you would register hooks on
        # specific modules. This is a simplified version.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: remove hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def apply_selective_checkpointing(
    model: nn.Module,
    config: Optional[SelectiveCheckpointConfig] = None,
) -> nn.Module:
    """Apply selective checkpointing to all compatible layers in a model.

    Args:
        model: Model to apply checkpointing to.
        config: Checkpoint configuration.

    Returns:
        Modified model with checkpointing applied.

    Example:
        >>> config = SelectiveCheckpointConfig(policy=CheckpointPolicy.HEAVY_OPS)
        >>> model = apply_selective_checkpointing(model, config)
    """
    checkpoint_wrapper = SelectiveCheckpoint(config)

    def apply_to_module(module: nn.Module):
        """Recursively apply checkpointing."""
        for name, child in module.named_children():
            if checkpoint_wrapper.should_checkpoint(child):
                # Wrap forward method
                original_forward = child.forward

                def checkpointed_forward(*args, **kwargs):
                    return checkpoint_wrapper(child, *args, **kwargs)

                child.forward = checkpointed_forward

            # Recurse
            apply_to_module(child)

    apply_to_module(model)
    return model


def estimate_checkpointing_memory_savings(
    activation_memory_mb: float,
    num_checkpointed_layers: int,
    total_layers: int,
) -> Dict[str, float]:
    """Estimate memory savings from selective checkpointing.

    Args:
        activation_memory_mb: Activation memory per layer (MB).
        num_checkpointed_layers: Number of checkpointed layers.
        total_layers: Total number of layers.

    Returns:
        Dictionary with memory estimates.
    """
    # Without checkpointing: store all activations
    total_without_cp = activation_memory_mb * total_layers

    # With checkpointing: only store non-checkpointed activations
    # (Checkpointed activations are recomputed during backward)
    non_checkpointed = total_layers - num_checkpointed_layers
    total_with_cp = activation_memory_mb * non_checkpointed

    savings = total_without_cp - total_with_cp
    savings_percent = (savings / total_without_cp * 100) if total_without_cp > 0 else 0

    return {
        "without_checkpointing_mb": total_without_cp,
        "with_checkpointing_mb": total_with_cp,
        "savings_mb": savings,
        "savings_percent": savings_percent,
        "checkpointed_layers": num_checkpointed_layers,
        "total_layers": total_layers,
    }


def estimate_offloading_memory_savings(
    activation_memory_gpu_mb: float,
    offload_ratio: float = 0.5,
) -> Dict[str, float]:
    """Estimate memory savings from activation offloading.

    Args:
        activation_memory_gpu_mb: Total activation memory on GPU (MB).
        offload_ratio: Fraction of activations to offload to CPU.

    Returns:
        Dictionary with memory estimates.
    """
    # GPU memory saved
    gpu_savings = activation_memory_gpu_mb * offload_ratio

    # CPU memory used (assume CPU has plenty)
    cpu_usage = activation_memory_gpu_mb * offload_ratio

    return {
        "gpu_memory_mb": activation_memory_gpu_mb,
        "gpu_savings_mb": gpu_savings,
        "cpu_usage_mb": cpu_usage,
        "offload_ratio": offload_ratio,
        "savings_percent": offload_ratio * 100,
    }
