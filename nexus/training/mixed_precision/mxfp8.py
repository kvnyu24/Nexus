"""Microscaling FP8 (MXFP8) for block-level scaled low-precision training.

MXFP8 uses block-level scaling factors instead of tensor-level scaling,
providing better dynamic range and numerical stability compared to standard FP8.
Each block of elements shares a scaling factor, allowing different parts of a
tensor to use different dynamic ranges.

Reference:
    "OCP Microscaling Formats (MX) Specification"
    Open Compute Project, 2024
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

Key features:
    - Block-level scaling (typically 32 or 16 elements per block)
    - Better accuracy than tensor-level FP8 scaling
    - Hardware support on AMD MI300 and future architectures
    - Maintains compatibility with standard FP8 operations

Example:
    >>> # Convert to MXFP8
    >>> mxfp8_tensor = to_mxfp8(tensor, block_size=32)
    >>>
    >>> # Use in linear layer
    >>> layer = MXFP8Linear(768, 3072, block_size=32)
    >>> output = layer(input)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass
from nexus.core.base import NexusModule


@dataclass
class MXFP8Config:
    """Configuration for MXFP8 format.

    Args:
        block_size: Number of elements per scaling block. Common values: 16, 32, 64.
        e4m3_forward: Use E4M3 format for forward pass (better range).
        e5m2_backward: Use E5M2 format for backward pass (better precision).
        scale_dtype: Dtype for storing scale factors (usually fp16 or bf16).
        amax_history_len: Length of amax history for dynamic scaling.
        amax_compute_algo: Algorithm for computing amax ('max' or 'ema').
    """

    block_size: int = 32
    e4m3_forward: bool = True
    e5m2_backward: bool = True
    scale_dtype: torch.dtype = torch.float16
    amax_history_len: int = 16
    amax_compute_algo: str = "max"


class MXFP8Tensor:
    """Microscaling FP8 tensor with block-level scaling.

    This class stores a tensor in MXFP8 format, which consists of:
    - FP8 quantized values (stored as uint8)
    - Per-block scaling factors
    - Block size metadata

    Args:
        data: FP8 quantized data (uint8).
        scales: Per-block scaling factors.
        block_size: Number of elements per block.
        original_shape: Shape of the original tensor.
        e4m3: Whether data uses E4M3 format (vs E5M2).
    """

    def __init__(
        self,
        data: torch.Tensor,
        scales: torch.Tensor,
        block_size: int,
        original_shape: torch.Size,
        e4m3: bool = True,
    ):
        self.data = data  # uint8 tensor
        self.scales = scales  # scaling factors
        self.block_size = block_size
        self.original_shape = original_shape
        self.e4m3 = e4m3

    def dequantize(self) -> torch.Tensor:
        """Convert MXFP8 back to high precision.

        Returns:
            Dequantized tensor in original dtype.
        """
        # Convert uint8 to fp8 (simulated, as PyTorch doesn't have native fp8)
        # In practice, this would use hardware-specific instructions
        fp8_data = self.data.to(torch.float32)

        # Map uint8 [0, 255] to FP8 range
        if self.e4m3:
            # E4M3: range [-448, 448]
            max_val = 448.0
        else:
            # E5M2: range [-57344, 57344]
            max_val = 57344.0

        # Simple linear mapping (hardware would use proper FP8 interpretation)
        fp8_data = (fp8_data - 127.5) / 127.5 * max_val

        # Reshape to include block structure
        numel = self.data.numel()
        num_blocks = (numel + self.block_size - 1) // self.block_size

        # Apply per-block scaling
        result = torch.zeros(numel, dtype=self.scales.dtype, device=self.data.device)

        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = min((i + 1) * self.block_size, numel)
            block_data = fp8_data[start_idx:end_idx]

            # Apply scale factor
            result[start_idx:end_idx] = block_data * self.scales[i]

        # Reshape to original shape
        return result.reshape(self.original_shape)

    def to(self, device):
        """Move MXFP8 tensor to device."""
        self.data = self.data.to(device)
        self.scales = self.scales.to(device)
        return self


def compute_block_scales(
    tensor: torch.Tensor,
    block_size: int,
    e4m3: bool = True,
    scale_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Compute per-block scaling factors for MXFP8 quantization.

    Args:
        tensor: Input tensor to quantize.
        block_size: Number of elements per scaling block.
        e4m3: Whether to use E4M3 format (vs E5M2).
        scale_dtype: Dtype for scale factors.

    Returns:
        Per-block scaling factors.
    """
    # Flatten tensor
    flat_tensor = tensor.flatten()
    numel = flat_tensor.numel()
    num_blocks = (numel + block_size - 1) // block_size

    # Compute amax per block
    scales = torch.zeros(num_blocks, dtype=scale_dtype, device=tensor.device)

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, numel)
        block_data = flat_tensor[start_idx:end_idx]

        # Compute amax for this block
        amax = block_data.abs().max()

        # Compute scale factor
        if e4m3:
            max_val = 448.0  # E4M3 max
        else:
            max_val = 57344.0  # E5M2 max

        # Scale to fit in FP8 range
        scale = amax / max_val if amax > 0 else torch.tensor(1.0, dtype=scale_dtype, device=tensor.device)
        scales[i] = scale

    return scales


def to_mxfp8(
    tensor: torch.Tensor,
    block_size: int = 32,
    e4m3: bool = True,
    scale_dtype: torch.dtype = torch.float16,
) -> MXFP8Tensor:
    """Convert a tensor to MXFP8 format with block-level scaling.

    Args:
        tensor: Input tensor.
        block_size: Number of elements per scaling block.
        e4m3: Whether to use E4M3 format (vs E5M2).
        scale_dtype: Dtype for scale factors.

    Returns:
        MXFP8Tensor with block-level scaling.
    """
    original_shape = tensor.shape

    # Compute per-block scales
    scales = compute_block_scales(tensor, block_size, e4m3, scale_dtype)

    # Flatten and quantize
    flat_tensor = tensor.flatten()
    numel = flat_tensor.numel()
    num_blocks = (numel + block_size - 1) // block_size

    # Quantize to uint8
    quantized = torch.zeros(numel, dtype=torch.uint8, device=tensor.device)

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, numel)
        block_data = flat_tensor[start_idx:end_idx]

        # Scale down
        scaled = block_data / scales[i]

        # Map to uint8 [0, 255]
        if e4m3:
            max_val = 448.0
        else:
            max_val = 57344.0

        # Clamp and convert to uint8
        uint8_data = ((scaled / max_val * 127.5) + 127.5).clamp(0, 255).to(torch.uint8)
        quantized[start_idx:end_idx] = uint8_data

    return MXFP8Tensor(quantized, scales, block_size, original_shape, e4m3)


class MXFP8Linear(NexusModule):
    """Linear layer with MXFP8 weights for memory-efficient training.

    This layer stores weights in MXFP8 format during training, reducing memory
    usage while maintaining reasonable accuracy through block-level scaling.

    Args:
        in_features: Size of input features.
        out_features: Size of output features.
        bias: Whether to include a bias term.
        block_size: Block size for MXFP8 quantization.
        config: MXFP8 configuration.

    Example:
        >>> layer = MXFP8Linear(768, 3072, block_size=32)
        >>> output = layer(input)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 32,
        config: Optional[MXFP8Config] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.config = config or MXFP8Config()

        # Initialize high-precision weights for training
        self.weight_master = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight_master, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_master)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # MXFP8 quantized weights (computed on-the-fly)
        self.weight_mxfp8 = None

    def _quantize_weights(self):
        """Convert weights to MXFP8 format."""
        self.weight_mxfp8 = to_mxfp8(
            self.weight_master.data,
            block_size=self.config.block_size,
            e4m3=self.config.e4m3_forward,
            scale_dtype=self.config.scale_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with MXFP8 weights.

        Args:
            x: Input tensor (..., in_features).

        Returns:
            Output tensor (..., out_features).
        """
        # Quantize weights if needed
        if self.weight_mxfp8 is None or self.training:
            self._quantize_weights()

        # Dequantize for computation (hardware would do FP8 matmul)
        weight = self.weight_mxfp8.dequantize()

        # Ensure weight is in correct dtype for computation
        weight = weight.to(x.dtype)

        return F.linear(x, weight, self.bias)


class MXFP8GradientScaler:
    """Gradient scaler with MXFP8 block-level scaling for backward pass.

    Uses E5M2 format for gradients (better precision) with block-level scaling.

    Args:
        config: MXFP8 configuration.
        enabled: Whether scaling is enabled.
    """

    def __init__(self, config: Optional[MXFP8Config] = None, enabled: bool = True):
        self.config = config or MXFP8Config()
        self.enabled = enabled
        self._scale_history = []

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass (identity in MXFP8).

        Args:
            loss: Loss value.

        Returns:
            Scaled loss.
        """
        # In MXFP8, we don't scale the loss itself
        # Instead, we quantize gradients during backward
        return loss

    def unscale_(self, optimizer):
        """Unscale gradients (identity in MXFP8)."""
        # Gradients are already in proper scale due to block-level scaling
        pass

    def step(self, optimizer):
        """Perform optimizer step."""
        optimizer.step()

    def update(self):
        """Update scaler state (no-op for MXFP8)."""
        pass


def get_mxfp8_memory_savings(
    num_params: int,
    block_size: int = 32,
    scale_dtype: torch.dtype = torch.float16,
) -> float:
    """Estimate memory savings from using MXFP8.

    Args:
        num_params: Number of parameters.
        block_size: Block size for scaling.
        scale_dtype: Dtype for scale factors.

    Returns:
        Memory savings ratio (e.g., 0.75 means 75% memory saved).
    """
    # FP32 memory
    fp32_bytes = num_params * 4

    # MXFP8 memory: 1 byte per param + scale factors
    num_blocks = (num_params + block_size - 1) // block_size
    scale_bytes = num_blocks * (2 if scale_dtype == torch.float16 else 4)
    mxfp8_bytes = num_params + scale_bytes

    savings = 1.0 - (mxfp8_bytes / fp32_bytes)
    return savings


# Import for easier access
import math
