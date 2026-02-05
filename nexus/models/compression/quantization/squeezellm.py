"""SqueezeLLM: Dense-and-Sparse Hybrid Quantization for Large Language Models.

Reference:
    Kim, S., et al. "SqueezeLLM: Dense-and-Sparse Quantization."
    ICML 2024. https://arxiv.org/abs/2306.07629

SqueezeLLM introduces a novel hybrid quantization approach that combines:
1. Dense low-bit quantization for most weights (e.g., 3-bit or 4-bit)
2. Sparse high-precision storage for outlier weights that are sensitive

This approach handles the challenge that certain "outlier" weights in LLMs
are critical for model quality and require higher precision. By identifying
and preserving these outliers while aggressively quantizing the rest,
SqueezeLLM achieves excellent compression with minimal accuracy loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import math

from nexus.core.base import NexusModule


@dataclass
class SqueezeLLMConfig:
    """Configuration for SqueezeLLM quantization.

    Attributes:
        bits: Number of bits for dense quantization (typically 3 or 4).
        outlier_threshold: Percentile threshold for identifying outliers.
            For example, 99.5 means weights above 99.5th percentile are outliers.
        group_size: Group size for blockwise quantization.
        use_sparse_format: Whether to use sparse tensor format for outliers.
        calibration_samples: Number of samples to use for outlier detection.
    """
    bits: int = 4
    outlier_threshold: float = 99.5
    group_size: int = 128
    use_sparse_format: bool = True
    calibration_samples: int = 128


class SqueezeLLMLinear(NexusModule):
    """Linear layer with SqueezeLLM dense-and-sparse quantization.

    Stores most weights in low-bit format with a sparse set of high-precision
    outlier weights for optimal compression and accuracy trade-off.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        config: SqueezeLLM configuration.
        bias: If True, includes bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: SqueezeLLMConfig,
        bias: bool = True,
    ):
        super().__init__(config.__dict__)

        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Dense quantized weights (low-bit)
        num_groups = (in_features + config.group_size - 1) // config.group_size
        self.register_buffer(
            'dense_weight',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )

        # Scales for dequantization (one per group)
        self.register_buffer(
            'weight_scales',
            torch.ones(out_features, num_groups, dtype=torch.float32)
        )

        # Zeros for dequantization (one per group)
        self.register_buffer(
            'weight_zeros',
            torch.zeros(out_features, num_groups, dtype=torch.float32)
        )

        # Sparse outlier weights (high-precision)
        # Stored as COO format: (indices, values)
        self.register_buffer('outlier_indices', torch.zeros(0, 2, dtype=torch.long))
        self.register_buffer('outlier_values', torch.zeros(0, dtype=torch.float32))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def identify_outliers(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Identify outlier weights that need high precision.

        Args:
            weight: Weight tensor of shape (out_features, in_features).

        Returns:
            Tuple of (outlier_mask, outlier_indices, outlier_values).
        """
        # Compute magnitude-based outlier threshold
        abs_weight = torch.abs(weight)
        threshold_value = torch.quantile(abs_weight.flatten(), self.config.outlier_threshold / 100.0)

        # Create mask for outliers
        outlier_mask = abs_weight > threshold_value

        # Get indices and values of outliers
        outlier_indices = torch.nonzero(outlier_mask)
        outlier_values = weight[outlier_mask]

        return outlier_mask, outlier_indices, outlier_values

    def quantize_weight(self, weight: torch.Tensor):
        """Quantize a weight matrix using SqueezeLLM.

        Args:
            weight: Weight tensor of shape (out_features, in_features).
        """
        out_features, in_features = weight.shape

        # Step 1: Identify outliers
        outlier_mask, outlier_indices, outlier_values = self.identify_outliers(weight)

        # Store outliers in sparse format
        self.outlier_indices = outlier_indices
        self.outlier_values = outlier_values

        # Step 2: Create weight with outliers zeroed out for dense quantization
        weight_without_outliers = weight.clone()
        weight_without_outliers[outlier_mask] = 0.0

        # Step 3: Dense quantization of non-outlier weights
        num_groups = (in_features + self.config.group_size - 1) // self.config.group_size
        max_val = 2 ** self.config.bits - 1

        for i in range(out_features):
            for j in range(num_groups):
                start_idx = j * self.config.group_size
                end_idx = min(start_idx + self.config.group_size, in_features)
                block = weight_without_outliers[i, start_idx:end_idx]

                # Compute scale and zero point
                min_val = block.min()
                max_val_block = block.max()
                scale = (max_val_block - min_val) / max_val if max_val_block != min_val else 1.0
                zero = min_val

                # Store scale and zero
                self.weight_scales[i, j] = scale
                self.weight_zeros[i, j] = zero

                # Quantize
                quantized = torch.round((block - zero) / (scale + 1e-8)).clamp(0, max_val)
                self.dense_weight[i, start_idx:end_idx] = quantized.to(torch.int8)

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantize the weight matrix.

        Returns:
            Dequantized weight tensor of shape (out_features, in_features).
        """
        out_features = self.dense_weight.shape[0]
        in_features = self.dense_weight.shape[1]
        num_groups = (in_features + self.config.group_size - 1) // self.config.group_size

        # Step 1: Dequantize dense weights
        weight = torch.zeros(out_features, in_features, device=self.dense_weight.device, dtype=torch.float32)

        for i in range(out_features):
            for j in range(num_groups):
                start_idx = j * self.config.group_size
                end_idx = min(start_idx + self.config.group_size, in_features)

                # Get quantized values
                quantized = self.dense_weight[i, start_idx:end_idx].float()

                # Denormalize
                scale = self.weight_scales[i, j]
                zero = self.weight_zeros[i, j]
                block = quantized * scale + zero

                weight[i, start_idx:end_idx] = block

        # Step 2: Add back outlier weights
        if len(self.outlier_indices) > 0:
            rows = self.outlier_indices[:, 0]
            cols = self.outlier_indices[:, 1]
            weight[rows, cols] = self.outlier_values

        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantized weights.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        weight = self.dequantize_weight()
        return F.linear(x, weight, self.bias)

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics.

        Returns:
            Dictionary with compression metrics.
        """
        total_weights = self.out_features * self.in_features
        outlier_count = len(self.outlier_values)
        outlier_ratio = outlier_count / total_weights * 100

        # Memory usage
        dense_bits = total_weights * self.config.bits
        outlier_bits = outlier_count * 32  # FP32 for outliers
        total_bits = dense_bits + outlier_bits

        # Baseline (FP32)
        baseline_bits = total_weights * 32

        compression_ratio = baseline_bits / total_bits

        return {
            'total_weights': total_weights,
            'outlier_count': outlier_count,
            'outlier_ratio': outlier_ratio,
            'compression_ratio': compression_ratio,
            'dense_bits_per_weight': self.config.bits,
            'effective_bits_per_weight': total_bits / total_weights,
        }


class SqueezeLLMQuantizer(NexusModule):
    """Quantizer for applying SqueezeLLM to a full model.

    Args:
        config: SqueezeLLM configuration.
    """

    def __init__(self, config: SqueezeLLMConfig):
        super().__init__(config.__dict__)
        self.config = config

    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> nn.Module:
        """Quantize all linear layers in a model using SqueezeLLM.

        Args:
            model: Model to quantize.
            calibration_data: Optional calibration data for outlier detection.
            verbose: If True, print compression statistics.

        Returns:
            Quantized model.
        """
        total_outliers = 0
        total_weights = 0
        modules_quantized = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Create quantized layer
                squeezed_layer = SqueezeLLMLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    config=self.config,
                    bias=module.bias is not None,
                )

                # Quantize weights
                squeezed_layer.quantize_weight(module.weight.data)

                # Copy bias if present
                if module.bias is not None:
                    squeezed_layer.bias.data = module.bias.data.clone()

                # Replace module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                setattr(parent, child_name, squeezed_layer)

                # Track statistics
                stats = squeezed_layer.get_compression_stats()
                total_outliers += stats['outlier_count']
                total_weights += stats['total_weights']
                modules_quantized += 1

        if verbose:
            print(f"SqueezeLLM Quantization Complete:")
            print(f"  Modules quantized: {modules_quantized}")
            print(f"  Total weights: {total_weights:,}")
            print(f"  Outlier weights: {total_outliers:,} ({total_outliers/total_weights*100:.2f}%)")
            print(f"  Dense quantization: {self.config.bits}-bit")
            print(f"  Average compression ratio: {32 / (self.config.bits + 32 * total_outliers / total_weights):.2f}x")

        return model


def apply_squeezellm(
    model: nn.Module,
    config: Optional[SqueezeLLMConfig] = None,
    calibration_data: Optional[torch.Tensor] = None,
    verbose: bool = True
) -> nn.Module:
    """Apply SqueezeLLM quantization to a model.

    Args:
        model: Model to quantize.
        config: SqueezeLLM configuration (uses defaults if None).
        calibration_data: Optional calibration data.
        verbose: If True, print statistics.

    Returns:
        Quantized model.
    """
    config = config or SqueezeLLMConfig()
    quantizer = SqueezeLLMQuantizer(config)
    return quantizer.quantize_model(model, calibration_data, verbose)
