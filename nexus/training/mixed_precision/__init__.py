"""Mixed precision training utilities for the Nexus library.

This module provides comprehensive support for mixed precision training,
including FP16, BF16, and FP8 formats. It enables memory-efficient training
while maintaining model quality through careful scaling and precision management.

Key Components:
    - MixedPrecisionConfig: Configuration dataclass for precision settings
    - FP8Linear: FP8 linear layer for memory-efficient training
    - FP8ScalingManager: Dynamic scaling for FP8 training
    - GradScaler: Enhanced gradient scaler with overflow detection
    - AdaptiveGradScaler: Gradient scaler with auto-tuning

Example:
    >>> from nexus.training.mixed_precision import (
    ...     MixedPrecisionConfig,
    ...     FP8Linear,
    ...     GradScaler
    ... )
    >>>
    >>> # Configure mixed precision training
    >>> config = MixedPrecisionConfig.bf16()
    >>>
    >>> # Use FP8 for linear layers
    >>> layer = FP8Linear(768, 3072)
    >>>
    >>> # Create gradient scaler
    >>> scaler = GradScaler(init_scale=65536.0)

Reference:
    - DeepSeek V3 uses FP8 training for efficient large-scale training
    - NVIDIA Transformer Engine for FP8 support on Hopper GPUs
"""

from .config import (
    MixedPrecisionConfig,
    FP8Format,
)

from .fp8 import (
    FP8Linear,
    FP8ScalingManager,
    FP8Tensor,
    FP8LayerNorm,
    FP8LinearFunction,
    convert_to_fp8,
    get_fp8_memory_savings,
    FP8_HARDWARE_AVAILABLE,
    FP8_HARDWARE_MESSAGE,
)

from .grad_scaler import (
    GradScaler,
    AdaptiveGradScaler,
    ScalerState,
    create_grad_scaler,
)


__all__ = [
    # Configuration
    "MixedPrecisionConfig",
    "FP8Format",

    # FP8 components
    "FP8Linear",
    "FP8ScalingManager",
    "FP8Tensor",
    "FP8LayerNorm",
    "FP8LinearFunction",
    "convert_to_fp8",
    "get_fp8_memory_savings",
    "FP8_HARDWARE_AVAILABLE",
    "FP8_HARDWARE_MESSAGE",

    # Gradient scaling
    "GradScaler",
    "AdaptiveGradScaler",
    "ScalerState",
    "create_grad_scaler",
]


def is_fp8_available() -> bool:
    """Check if FP8 hardware support is available.

    Returns:
        True if FP8 hardware support is detected.
    """
    return FP8_HARDWARE_AVAILABLE


def get_recommended_config(device: str = "auto") -> MixedPrecisionConfig:
    """Get recommended mixed precision configuration for the current hardware.

    Args:
        device: Device to check ('cuda', 'cpu', or 'auto').

    Returns:
        Recommended MixedPrecisionConfig.
    """
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        # CPU: use float32
        return MixedPrecisionConfig(
            compute_dtype=torch.float32,
            param_dtype=torch.float32,
            enabled=False
        )

    if device == "cuda":
        # Check GPU capabilities
        if not torch.cuda.is_available():
            return MixedPrecisionConfig(enabled=False)

        capability = torch.cuda.get_device_capability()
        major, minor = capability

        if major >= 9 or (major == 8 and minor >= 9):
            # Hopper or Ada Lovelace: use FP8
            return MixedPrecisionConfig.fp8()
        elif major >= 8:
            # Ampere: use BF16
            return MixedPrecisionConfig.bf16()
        elif major >= 7:
            # Volta/Turing: use FP16
            return MixedPrecisionConfig.fp16()
        else:
            # Older GPUs: limited mixed precision support
            return MixedPrecisionConfig.fp16()

    return MixedPrecisionConfig(enabled=False)
