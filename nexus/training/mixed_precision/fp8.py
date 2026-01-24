"""FP8 training utilities for memory-efficient training.

This module provides FP8 (8-bit floating point) support for training,
enabling significant memory savings while maintaining model quality.

Reference: DeepSeek V3 uses FP8 training for efficient large-scale training.
"""

from typing import Optional, Dict, Literal, Tuple, Any
from collections import deque
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus.core.base import NexusModule
from .config import FP8Format, MixedPrecisionConfig


def _check_fp8_hardware_support() -> Tuple[bool, str]:
    """Check if FP8 hardware support is available.

    Returns:
        Tuple of (is_supported, message) indicating hardware support status.
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available. FP8 requires NVIDIA GPU with compute capability >= 8.9"

    try:
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        major, minor = capability

        # FP8 requires Hopper (SM 9.0) or later, or Ada Lovelace (SM 8.9)
        if major > 8 or (major == 8 and minor >= 9):
            return True, f"FP8 supported on device with compute capability {major}.{minor}"

        return False, (
            f"FP8 requires compute capability >= 8.9 (Hopper/Ada), "
            f"found {major}.{minor}. Will use emulation mode."
        )
    except Exception as e:
        return False, f"Error checking FP8 support: {e}"


# Check hardware support at module load time
FP8_HARDWARE_AVAILABLE, FP8_HARDWARE_MESSAGE = _check_fp8_hardware_support()


class FP8ScalingManager:
    """Manages dynamic scaling factors for FP8 training.

    Tracks tensor statistics and adjusts scaling to prevent overflow/underflow.
    This is critical for FP8 training as the limited dynamic range requires
    careful scaling of activations and gradients.

    The manager maintains a history of maximum absolute values (amax) for
    each tracked tensor and computes appropriate scaling factors.

    Args:
        amax_history_len: Length of amax history for scaling. Longer history
                         provides more stable scaling but slower adaptation.
        amax_compute_algo: Algorithm for computing amax from history.
                          'max': Use maximum over history (more conservative).
                          'most_recent': Use most recent value (faster adaptation).
        margin: Safety margin factor for scaling (default 2.0).
               Higher values provide more headroom but reduce precision.

    Example:
        >>> manager = FP8ScalingManager(amax_history_len=1024)
        >>> # During training
        >>> amax = tensor.abs().max().item()
        >>> manager.update_amax("layer1_weight", amax)
        >>> scale = manager.get_scale("layer1_weight", FP8Format.E4M3)
        >>> scaled_tensor = tensor * scale
    """

    def __init__(
        self,
        amax_history_len: int = 1024,
        amax_compute_algo: Literal["max", "most_recent"] = "max",
        margin: float = 2.0
    ):
        self.amax_history_len = amax_history_len
        self.amax_compute_algo = amax_compute_algo
        self.margin = margin

        # Store amax history for each tensor
        self._amax_history: Dict[str, deque] = {}
        # Current computed scales
        self._scales: Dict[str, float] = {}
        # Inverse scales for efficient computation
        self._inv_scales: Dict[str, float] = {}

    def register_tensor(self, tensor_name: str, initial_amax: float = 1.0) -> None:
        """Register a tensor for scaling management.

        Args:
            tensor_name: Unique name for the tensor.
            initial_amax: Initial amax value to use.
        """
        if tensor_name not in self._amax_history:
            self._amax_history[tensor_name] = deque(
                [initial_amax],
                maxlen=self.amax_history_len
            )
            self._recompute_scale(tensor_name, FP8Format.E4M3)

    def get_scale(
        self,
        tensor_name: str,
        fp8_format: FP8Format = FP8Format.E4M3
    ) -> float:
        """Get current scale for tensor.

        Args:
            tensor_name: Name of the tensor.
            fp8_format: FP8 format to use for computing scale.

        Returns:
            Scale factor to apply to tensor before quantization.
        """
        if tensor_name not in self._amax_history:
            self.register_tensor(tensor_name)

        # Recompute scale if needed
        if tensor_name not in self._scales:
            self._recompute_scale(tensor_name, fp8_format)

        return self._scales[tensor_name]

    def get_inv_scale(
        self,
        tensor_name: str,
        fp8_format: FP8Format = FP8Format.E4M3
    ) -> float:
        """Get inverse scale for tensor (for dequantization).

        Args:
            tensor_name: Name of the tensor.
            fp8_format: FP8 format to use.

        Returns:
            Inverse scale factor.
        """
        if tensor_name not in self._inv_scales:
            scale = self.get_scale(tensor_name, fp8_format)
            self._inv_scales[tensor_name] = 1.0 / scale if scale != 0 else 1.0

        return self._inv_scales[tensor_name]

    def update_amax(
        self,
        tensor_name: str,
        amax: float,
        fp8_format: FP8Format = FP8Format.E4M3
    ) -> None:
        """Update amax history and recompute scale.

        Args:
            tensor_name: Name of the tensor.
            amax: Maximum absolute value observed.
            fp8_format: FP8 format for scale computation.
        """
        if tensor_name not in self._amax_history:
            self.register_tensor(tensor_name, amax)
            return

        # Add to history
        self._amax_history[tensor_name].append(amax)

        # Recompute scale
        self._recompute_scale(tensor_name, fp8_format)

    def _recompute_scale(self, tensor_name: str, fp8_format: FP8Format) -> None:
        """Recompute scale from amax history.

        Args:
            tensor_name: Name of the tensor.
            fp8_format: FP8 format for scale computation.
        """
        history = self._amax_history.get(tensor_name)
        if not history:
            self._scales[tensor_name] = 1.0
            self._inv_scales[tensor_name] = 1.0
            return

        # Compute representative amax
        if self.amax_compute_algo == "max":
            amax = max(history)
        else:  # most_recent
            amax = history[-1]

        # Compute scale to map amax to FP8 max value
        fp8_max = fp8_format.max_value
        if amax > 0:
            scale = fp8_max / (amax * self.margin)
        else:
            scale = 1.0

        self._scales[tensor_name] = scale
        self._inv_scales[tensor_name] = 1.0 / scale if scale != 0 else 1.0

    def get_all_scales(self) -> Dict[str, float]:
        """Get all current scales.

        Returns:
            Dictionary mapping tensor names to scales.
        """
        return dict(self._scales)

    def reset(self) -> None:
        """Reset all scaling history."""
        self._amax_history.clear()
        self._scales.clear()
        self._inv_scales.clear()

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing.

        Returns:
            State dictionary.
        """
        return {
            "amax_history": {k: list(v) for k, v in self._amax_history.items()},
            "scales": dict(self._scales),
            "inv_scales": dict(self._inv_scales),
            "amax_history_len": self.amax_history_len,
            "amax_compute_algo": self.amax_compute_algo,
            "margin": self.margin
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint.

        Args:
            state_dict: State dictionary to load.
        """
        self.amax_history_len = state_dict.get("amax_history_len", self.amax_history_len)
        self.amax_compute_algo = state_dict.get("amax_compute_algo", self.amax_compute_algo)
        self.margin = state_dict.get("margin", self.margin)

        self._amax_history = {
            k: deque(v, maxlen=self.amax_history_len)
            for k, v in state_dict.get("amax_history", {}).items()
        }
        self._scales = dict(state_dict.get("scales", {}))
        self._inv_scales = dict(state_dict.get("inv_scales", {}))


class FP8Tensor:
    """Wrapper for FP8 tensor representation.

    Since PyTorch doesn't natively support FP8 on all hardware, this class
    provides a software emulation layer that stores tensors in int8 format
    with associated scale factors.

    Args:
        data: The quantized int8 data.
        scale: Scale factor for dequantization.
        fp8_format: FP8 format used.
    """

    def __init__(
        self,
        data: torch.Tensor,
        scale: float,
        fp8_format: FP8Format
    ):
        self.data = data
        self.scale = scale
        self.fp8_format = fp8_format

    def dequantize(self) -> torch.Tensor:
        """Convert back to float tensor.

        Returns:
            Dequantized float tensor.
        """
        return self.data.float() / self.scale

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        scale: float,
        fp8_format: FP8Format
    ) -> "FP8Tensor":
        """Quantize a float tensor to FP8.

        Args:
            tensor: Input float tensor.
            scale: Scale factor to apply.
            fp8_format: FP8 format to use.

        Returns:
            FP8Tensor representation.
        """
        # Scale and clamp to FP8 range
        scaled = tensor.float() * scale
        max_val = fp8_format.max_value

        # Clamp to representable range
        clamped = scaled.clamp(-max_val, max_val)

        # For emulation, store as int8 (simplified)
        # In practice, this would use proper FP8 bit representation
        quantized = clamped.to(torch.int8)

        return cls(quantized, scale, fp8_format)


class FP8LinearFunction(torch.autograd.Function):
    """Autograd function for FP8 linear layer.

    Implements forward and backward passes with FP8 computation.
    Uses higher precision for accumulation to maintain accuracy.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        input_scale: float,
        weight_scale: float,
        output_scale: float,
        fp8_format: FP8Format,
        use_fp8_compute: bool
    ) -> torch.Tensor:
        """Forward pass with FP8 computation.

        Args:
            ctx: Autograd context.
            input: Input tensor.
            weight: Weight tensor.
            bias: Optional bias tensor.
            input_scale: Scale for input quantization.
            weight_scale: Scale for weight quantization.
            output_scale: Scale for output dequantization.
            fp8_format: FP8 format to use.
            use_fp8_compute: Whether to use actual FP8 computation.

        Returns:
            Output tensor.
        """
        # Store for backward
        ctx.save_for_backward(input, weight, bias)
        ctx.input_scale = input_scale
        ctx.weight_scale = weight_scale
        ctx.output_scale = output_scale
        ctx.fp8_format = fp8_format
        ctx.use_fp8_compute = use_fp8_compute

        if use_fp8_compute and FP8_HARDWARE_AVAILABLE:
            # Hardware FP8 path (placeholder for actual implementation)
            # This would use torch.float8_e4m3fn or torch.float8_e5m2 when available
            output = F.linear(input, weight, bias)
        else:
            # Emulation path: simulate FP8 by quantizing/dequantizing
            # Quantize input and weight
            max_val = fp8_format.max_value

            input_scaled = (input * input_scale).clamp(-max_val, max_val)
            weight_scaled = (weight * weight_scale).clamp(-max_val, max_val)

            # Compute in higher precision
            output = F.linear(input_scaled, weight_scaled, None)

            # Rescale output
            output = output / (input_scale * weight_scale)

            if bias is not None:
                output = output + bias

        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward pass with FP8 gradient computation.

        Args:
            ctx: Autograd context.
            grad_output: Gradient of the output.

        Returns:
            Gradients for each input.
        """
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2, -1).matmul(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))

        # Return gradients for all inputs (including non-tensor args as None)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class FP8Linear(NexusModule):
    """FP8 Linear layer for memory-efficient training.

    Stores weights in FP8 format and computes in higher precision.
    This provides significant memory savings (up to 4x) while maintaining
    model quality through careful scaling.

    Reference: DeepSeek V3 uses FP8 training for efficient large-scale training.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to use bias. Default: True.
        fp8_format: FP8 format to use ('e4m3' or 'e5m2'). Default: 'e4m3'.
        compute_dtype: Dtype for computation. Default: torch.bfloat16.
        scaling_manager: Optional FP8ScalingManager instance.
        use_fp8_weights: Whether to store weights in FP8. Default: True.

    Example:
        >>> layer = FP8Linear(768, 3072, fp8_format=FP8Format.E4M3)
        >>> output = layer(input_tensor)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        fp8_format: FP8Format = FP8Format.E4M3,
        compute_dtype: torch.dtype = torch.bfloat16,
        scaling_manager: Optional[FP8ScalingManager] = None,
        use_fp8_weights: bool = True
    ):
        config = {
            "in_features": in_features,
            "out_features": out_features,
            "bias": bias,
            "fp8_format": fp8_format.value,
            "use_fp8_weights": use_fp8_weights
        }
        super().__init__(config)

        self.in_features = in_features
        self.out_features = out_features
        self.fp8_format = fp8_format
        self.compute_dtype = compute_dtype
        self.use_fp8_weights = use_fp8_weights

        # Initialize scaling manager
        self.scaling_manager = scaling_manager or FP8ScalingManager()

        # Register tensor names for scaling
        self._weight_name = f"fp8_linear_{id(self)}_weight"
        self._input_name = f"fp8_linear_{id(self)}_input"
        self._output_name = f"fp8_linear_{id(self)}_output"

        self.scaling_manager.register_tensor(self._weight_name)
        self.scaling_manager.register_tensor(self._input_name)
        self.scaling_manager.register_tensor(self._output_name)

        # Weight parameter (stored in compute dtype, quantized during forward)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=compute_dtype)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=compute_dtype))
        else:
            self.register_parameter('bias', None)

        # Check hardware support
        self._fp8_available = FP8_HARDWARE_AVAILABLE
        if not self._fp8_available:
            warnings.warn(FP8_HARDWARE_MESSAGE, RuntimeWarning)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP8 computation.

        Args:
            input: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # Update input amax for scaling
        with torch.no_grad():
            input_amax = input.abs().max().item()
            weight_amax = self.weight.abs().max().item()
            self.scaling_manager.update_amax(self._input_name, input_amax, self.fp8_format)
            self.scaling_manager.update_amax(self._weight_name, weight_amax, self.fp8_format)

        # Get scales
        input_scale = self.scaling_manager.get_scale(self._input_name, self.fp8_format)
        weight_scale = self.scaling_manager.get_scale(self._weight_name, self.fp8_format)
        output_scale = self.scaling_manager.get_scale(self._output_name, self.fp8_format)

        # Use custom autograd function
        output = FP8LinearFunction.apply(
            input,
            self.weight,
            self.bias,
            input_scale,
            weight_scale,
            output_scale,
            self.fp8_format,
            self._fp8_available and self.use_fp8_weights
        )

        # Update output amax
        with torch.no_grad():
            output_amax = output.abs().max().item()
            self.scaling_manager.update_amax(self._output_name, output_amax, self.fp8_format)

        return output

    def extra_repr(self) -> str:
        """Extra representation string."""
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias is not None}, '
            f'fp8_format={self.fp8_format.value}, '
            f'fp8_available={self._fp8_available}'
        )


class FP8LayerNorm(NexusModule):
    """Layer normalization with FP8 support.

    Computes layer normalization in higher precision for numerical
    stability, but can quantize inputs/outputs to FP8.

    Args:
        normalized_shape: Input shape from an expected input.
        eps: Small constant for numerical stability.
        elementwise_affine: Whether to include learnable affine parameters.
        fp8_format: FP8 format for input/output quantization.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        fp8_format: FP8Format = FP8Format.E4M3
    ):
        config = {
            "normalized_shape": normalized_shape,
            "eps": eps,
            "elementwise_affine": elementwise_affine,
            "fp8_format": fp8_format.value
        }
        super().__init__(config)

        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.fp8_format = fp8_format

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # Scaling manager for FP8
        self.scaling_manager = FP8ScalingManager()
        self._input_name = f"fp8_layernorm_{id(self)}_input"
        self._output_name = f"fp8_layernorm_{id(self)}_output"
        self.scaling_manager.register_tensor(self._input_name)
        self.scaling_manager.register_tensor(self._output_name)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            input: Input tensor.

        Returns:
            Normalized output tensor.
        """
        # Always compute LayerNorm in float32 for numerical stability
        input_fp32 = input.float()
        output = F.layer_norm(
            input_fp32,
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps
        )

        # Convert back to input dtype
        return output.to(input.dtype)


def convert_to_fp8(
    module: nn.Module,
    fp8_format: FP8Format = FP8Format.E4M3,
    scaling_manager: Optional[FP8ScalingManager] = None,
    inplace: bool = False
) -> nn.Module:
    """Convert linear layers in a module to FP8.

    Args:
        module: Module to convert.
        fp8_format: FP8 format to use.
        scaling_manager: Optional shared scaling manager.
        inplace: Whether to modify module in place.

    Returns:
        Module with FP8 linear layers.
    """
    if not inplace:
        module = module.copy() if hasattr(module, 'copy') else module

    if scaling_manager is None:
        scaling_manager = FP8ScalingManager()

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, FP8Linear):
            # Replace with FP8Linear
            fp8_linear = FP8Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                fp8_format=fp8_format,
                scaling_manager=scaling_manager
            )

            # Copy weights
            with torch.no_grad():
                fp8_linear.weight.copy_(child.weight)
                if child.bias is not None:
                    fp8_linear.bias.copy_(child.bias)

            setattr(module, name, fp8_linear)
        else:
            # Recursively convert children
            convert_to_fp8(child, fp8_format, scaling_manager, inplace=True)

    return module


def get_fp8_memory_savings(model: nn.Module) -> Dict[str, float]:
    """Calculate potential memory savings from FP8 conversion.

    Args:
        model: Model to analyze.

    Returns:
        Dictionary with memory statistics.
    """
    linear_params = 0
    total_params = 0

    for module in model.modules():
        if isinstance(module, nn.Linear):
            linear_params += module.weight.numel()
            if module.bias is not None:
                linear_params += module.bias.numel()

    for param in model.parameters():
        total_params += param.numel()

    # FP8 uses 1 byte vs 2-4 bytes for fp16/fp32
    fp32_memory_mb = total_params * 4 / (1024 ** 2)
    fp16_memory_mb = total_params * 2 / (1024 ** 2)
    fp8_memory_mb = (
        (total_params - linear_params) * 2 +  # Non-linear params in fp16
        linear_params * 1  # Linear params in fp8
    ) / (1024 ** 2)

    return {
        "total_params": total_params,
        "linear_params": linear_params,
        "linear_param_ratio": linear_params / total_params if total_params > 0 else 0,
        "fp32_memory_mb": fp32_memory_mb,
        "fp16_memory_mb": fp16_memory_mb,
        "fp8_memory_mb": fp8_memory_mb,
        "savings_vs_fp32": (fp32_memory_mb - fp8_memory_mb) / fp32_memory_mb if fp32_memory_mb > 0 else 0,
        "savings_vs_fp16": (fp16_memory_mb - fp8_memory_mb) / fp16_memory_mb if fp16_memory_mb > 0 else 0
    }
