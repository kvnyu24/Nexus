"""FP4 and MXFP4 (Microscaling FP4) for 4-bit training.

FP4 is an extremely low-precision floating-point format that enables
training with 4 bits per parameter. MXFP4 extends this with block-level
scaling for better accuracy. This is useful for extreme memory reduction
on very large models.

Reference:
    "FP8 Formats for Deep Learning" (adapted to FP4)
    Micikevicius et al., 2022

    "OCP Microscaling Formats (MX) Specification"
    Open Compute Project, 2024

Key features:
    - 4-bit floating point representation
    - Block-level scaling for MXFP4 variant
    - 8x memory reduction vs FP32 (for weights)
    - Suitable for very large models where memory is critical

FP4 format (E2M1):
    - 1 sign bit
    - 2 exponent bits
    - 1 mantissa bit
    - Range: ~[-6, 6] with limited precision

Example:
    >>> # Convert to FP4
    >>> fp4_tensor = to_fp4(tensor)
    >>>
    >>> # Use in linear layer
    >>> layer = FP4Linear(768, 3072, use_microscaling=True)
    >>> output = layer(input)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass
from nexus.core.base import NexusModule
import math


@dataclass
class FP4Config:
    """Configuration for FP4/MXFP4 training.

    Args:
        use_microscaling: Whether to use block-level scaling (MXFP4).
        block_size: Block size for microscaling. Typical: 16, 32.
        scale_dtype: Dtype for scale factors (fp16 or bf16).
        stochastic_rounding: Use stochastic rounding for better accuracy.
        clip_value: Maximum absolute value for clipping before quantization.
    """

    use_microscaling: bool = True
    block_size: int = 32
    scale_dtype: torch.dtype = torch.float16
    stochastic_rounding: bool = True
    clip_value: float = 6.0  # FP4 E2M1 range


# FP4 E2M1 quantization table (8 values for positive half)
# Sign bit is handled separately
FP4_E2M1_VALUES = torch.tensor(
    [
        0.0,      # 0b000
        0.5,      # 0b001
        1.0,      # 0b010
        1.5,      # 0b011
        2.0,      # 0b100
        3.0,      # 0b101
        4.0,      # 0b110
        6.0,      # 0b111
    ],
    dtype=torch.float32,
)


def quantize_to_fp4(
    tensor: torch.Tensor,
    stochastic: bool = False,
) -> torch.Tensor:
    """Quantize tensor to FP4 E2M1 format (stored as uint8).

    Args:
        tensor: Input tensor (should be in range [-6, 6]).
        stochastic: Use stochastic rounding.

    Returns:
        Quantized tensor stored as uint8 (0-15, using 4 bits).
    """
    # Separate sign
    sign = (tensor < 0).to(torch.uint8)
    abs_tensor = tensor.abs()

    # Clamp to valid range
    abs_tensor = abs_tensor.clamp(0, 6.0)

    # Find nearest FP4 value
    fp4_values = FP4_E2M1_VALUES.to(tensor.device)

    # Compute distances to all FP4 values
    distances = (abs_tensor.unsqueeze(-1) - fp4_values).abs()

    if stochastic:
        # Stochastic rounding: probabilistically choose between two nearest values
        nearest_idx = distances.argmin(dim=-1)
        nearest_val = fp4_values[nearest_idx]

        # Find next nearest (for values between two levels)
        next_idx = torch.clamp(nearest_idx + 1, max=len(fp4_values) - 1)
        next_val = fp4_values[next_idx]

        # Compute interpolation probability
        diff = next_val - nearest_val
        prob = torch.where(
            diff > 0,
            (abs_tensor - nearest_val) / diff,
            torch.zeros_like(abs_tensor),
        )

        # Stochastically choose
        use_next = torch.rand_like(prob) < prob
        idx = torch.where(use_next, next_idx, nearest_idx)
    else:
        # Deterministic rounding: choose nearest
        idx = distances.argmin(dim=-1)

    # Combine sign (1 bit) and magnitude (3 bits)
    # Format: [sign][3-bit index]
    quantized = (sign << 3) | idx.to(torch.uint8)

    return quantized


def dequantize_from_fp4(quantized: torch.Tensor) -> torch.Tensor:
    """Dequantize FP4 E2M1 values back to FP32.

    Args:
        quantized: Quantized tensor (uint8, using 4 bits).

    Returns:
        Dequantized FP32 tensor.
    """
    # Extract sign and magnitude
    sign = (quantized >> 3) & 0x1
    magnitude_idx = (quantized & 0x7).long()  # Convert to long for indexing

    # Get FP4 values
    fp4_values = FP4_E2M1_VALUES.to(quantized.device)

    # Look up magnitude
    magnitude = fp4_values[magnitude_idx]

    # Apply sign
    result = torch.where(sign.bool(), -magnitude, magnitude)

    return result


class FP4Tensor:
    """FP4 tensor representation.

    Args:
        data: Quantized data (uint8, 4 bits per value).
        scales: Per-block scales (for MXFP4) or global scale.
        block_size: Block size (None for global scaling).
        original_shape: Original tensor shape.
    """

    def __init__(
        self,
        data: torch.Tensor,
        scales: torch.Tensor,
        block_size: Optional[int],
        original_shape: torch.Size,
    ):
        self.data = data  # uint8
        self.scales = scales
        self.block_size = block_size
        self.original_shape = original_shape

    def dequantize(self) -> torch.Tensor:
        """Dequantize to high precision.

        Returns:
            Dequantized tensor.
        """
        # Dequantize from FP4
        dequant = dequantize_from_fp4(self.data)

        # Apply scaling
        if self.block_size is not None:
            # Block-level scaling (MXFP4)
            numel = self.data.numel()
            num_blocks = (numel + self.block_size - 1) // self.block_size

            for i in range(num_blocks):
                start_idx = i * self.block_size
                end_idx = min((i + 1) * self.block_size, numel)
                dequant[start_idx:end_idx] *= self.scales[i]
        else:
            # Global scaling
            dequant *= self.scales

        return dequant.reshape(self.original_shape)

    def to(self, device):
        """Move to device."""
        self.data = self.data.to(device)
        self.scales = self.scales.to(device)
        return self


def to_fp4(
    tensor: torch.Tensor,
    config: Optional[FP4Config] = None,
) -> FP4Tensor:
    """Convert tensor to FP4 or MXFP4 format.

    Args:
        tensor: Input tensor.
        config: FP4 configuration.

    Returns:
        FP4Tensor.
    """
    if config is None:
        config = FP4Config()

    original_shape = tensor.shape
    flat_tensor = tensor.flatten()
    numel = flat_tensor.numel()

    # Clip to valid range
    clipped = flat_tensor.clamp(-config.clip_value, config.clip_value)

    if config.use_microscaling:
        # MXFP4: block-level scaling
        num_blocks = (numel + config.block_size - 1) // config.block_size
        scales = torch.zeros(num_blocks, dtype=config.scale_dtype, device=tensor.device)
        quantized = torch.zeros(numel, dtype=torch.uint8, device=tensor.device)

        for i in range(num_blocks):
            start_idx = i * config.block_size
            end_idx = min((i + 1) * config.block_size, numel)
            block = clipped[start_idx:end_idx]

            # Compute scale for this block
            amax = block.abs().max()
            scale = amax / config.clip_value if amax > 0 else torch.tensor(1.0, dtype=config.scale_dtype)
            scales[i] = scale

            # Normalize and quantize block
            normalized = block / scale
            quantized[start_idx:end_idx] = quantize_to_fp4(
                normalized, stochastic=config.stochastic_rounding
            )

        return FP4Tensor(quantized, scales, config.block_size, original_shape)

    else:
        # Standard FP4: global scaling
        amax = clipped.abs().max()
        scale = amax / config.clip_value if amax > 0 else torch.tensor(1.0, dtype=config.scale_dtype)

        # Normalize and quantize
        normalized = clipped / scale
        quantized = quantize_to_fp4(normalized, stochastic=config.stochastic_rounding)

        return FP4Tensor(quantized, scale, None, original_shape)


class FP4Linear(NexusModule):
    """Linear layer with FP4/MXFP4 weights.

    Stores weights in 4-bit precision for extreme memory reduction.
    Uses block-level scaling (MXFP4) for better accuracy.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include bias.
        config: FP4 configuration.

    Example:
        >>> layer = FP4Linear(768, 3072, config=FP4Config(use_microscaling=True))
        >>> output = layer(input)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[FP4Config] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or FP4Config()

        # Master weights for training (high precision)
        self.weight_master = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize
        nn.init.kaiming_uniform_(self.weight_master, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_master)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # FP4 cache
        self.weight_fp4 = None

    def _quantize_weights(self):
        """Quantize weights to FP4."""
        self.weight_fp4 = to_fp4(self.weight_master.data, self.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP4 weights.

        Args:
            x: Input tensor (..., in_features).

        Returns:
            Output tensor (..., out_features).
        """
        # Quantize weights
        if self.weight_fp4 is None or self.training:
            self._quantize_weights()

        # Dequantize for computation
        weight = self.weight_fp4.dequantize()

        # Ensure weight is in correct dtype for computation
        weight = weight.to(x.dtype)

        return F.linear(x, weight, self.bias)


def get_fp4_memory_savings(
    num_params: int,
    use_microscaling: bool = True,
    block_size: int = 32,
) -> float:
    """Estimate memory savings from FP4.

    Args:
        num_params: Number of parameters.
        use_microscaling: Whether using MXFP4.
        block_size: Block size for MXFP4.

    Returns:
        Memory savings ratio.
    """
    # FP32: 4 bytes per param
    fp32_bytes = num_params * 4

    # FP4: 0.5 bytes per param (4 bits)
    fp4_bytes = num_params * 0.5

    if use_microscaling:
        # Add scale factors (2 bytes per block)
        num_blocks = (num_params + block_size - 1) // block_size
        fp4_bytes += num_blocks * 2

    savings = 1.0 - (fp4_bytes / fp32_bytes)
    return savings


class FP4GradientScaler:
    """Gradient scaler for FP4 training.

    Handles loss scaling to prevent underflow in FP4 gradients.

    Args:
        init_scale: Initial loss scale.
        scale_factor: Growth factor for scale.
        scale_window: Steps between scale updates.
    """

    def __init__(
        self,
        init_scale: float = 2**10,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
    ):
        self._scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self._growth_tracker = 0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss before backward."""
        return loss * self._scale

    def unscale_(self, optimizer):
        """Unscale gradients."""
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.div_(self._scale)

    def step(self, optimizer):
        """Perform optimizer step with gradient clipping."""
        # Check for inf/nan
        has_inf_or_nan = False
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        has_inf_or_nan = True
                        break

        if has_inf_or_nan:
            # Reduce scale
            self._scale /= self.scale_factor
            self._growth_tracker = 0
        else:
            # Successful step
            optimizer.step()
            self._growth_tracker += 1

            # Increase scale
            if self._growth_tracker >= self.scale_window:
                self._scale *= self.scale_factor
                self._growth_tracker = 0

    def update(self):
        """Update scaler state."""
        pass
