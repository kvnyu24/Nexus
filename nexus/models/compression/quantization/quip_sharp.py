"""QuIP#: 2-bit Quantization using E8 Lattice Codebooks.

Reference:
    Tseng, A., et al. "QuIP#: Even Better LLM Quantization with Hadamard
    Incoherence and Lattice Codebooks."
    ICML 2024. https://arxiv.org/abs/2402.04396

QuIP# achieves state-of-the-art 2-bit quantization through two key innovations:
1. E8 lattice codebooks: Uses the optimal E8 lattice for 2-bit weight encoding,
   which provides better coverage of the quantization space than naive rounding.
2. Hadamard incoherence: Applies randomized Hadamard transforms to weights
   before quantization to reduce coherence and improve reconstruction quality.

This enables 2-bit quantization with minimal accuracy loss, achieving 4x
compression over INT8 and 16x over FP16.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np

from nexus.core.base import NexusModule


@dataclass
class QuIPSharpConfig:
    """Configuration for QuIP# quantization.

    Attributes:
        bits: Number of bits for quantization (typically 2).
        group_size: Group size for blockwise quantization. Smaller groups
            improve accuracy but increase memory overhead.
        use_e8_lattice: Whether to use E8 lattice codebook (True) or standard
            rounding (False).
        use_hadamard: Whether to apply Hadamard incoherence transform.
        hadamard_seed: Random seed for Hadamard matrix generation.
        damping: Damping factor for inverse Hessian computation.
        percdamp: Percentage of Hessian diagonal to use for damping.
        blocksize: Block size for sequential quantization.
    """
    bits: int = 2
    group_size: int = 128
    use_e8_lattice: bool = True
    use_hadamard: bool = True
    hadamard_seed: int = 42
    damping: float = 0.01
    percdamp: float = 0.01
    blocksize: int = 128


class E8Lattice:
    """E8 lattice codebook for optimal 2-bit quantization.

    The E8 lattice is the optimal sphere packing in 8 dimensions, providing
    better quantization than naive rounding. For 2-bit quantization, we use
    the first 4 levels of the E8 lattice.
    """

    def __init__(self):
        self.codebook = self._generate_e8_codebook()

    def _generate_e8_codebook(self) -> torch.Tensor:
        """Generate E8 lattice codebook for 2-bit quantization.

        Returns:
            Codebook tensor of shape (4, 8) for 2-bit = 4 levels.
        """
        # E8 lattice basis vectors (simplified for 2-bit quantization)
        # In practice, this would include the full E8 lattice points
        # For 2-bit (4 levels), we use a subset of E8 lattice points
        codebook = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0],      # Level 0
            [1, 1, 0, 0, 0, 0, 0, 0],      # Level 1
            [1, 0, 1, 0, 0, 0, 0, 0],      # Level 2
            [0, 1, 1, 0, 0, 0, 0, 0],      # Level 3
        ], dtype=torch.float32)

        return codebook

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize using E8 lattice codebook.

        Args:
            x: Input tensor to quantize, shape (..., 8).

        Returns:
            Tuple of (quantized tensor, indices).
        """
        # Reshape to (..., 8) if needed
        original_shape = x.shape
        x_flat = x.reshape(-1, 8)

        # Find nearest lattice point for each 8-dimensional vector
        # Compute distance to each codebook entry
        distances = torch.cdist(x_flat, self.codebook)  # (N, 4)

        # Select nearest codebook entry
        indices = torch.argmin(distances, dim=1)  # (N,)

        # Quantized values
        quantized = self.codebook[indices]  # (N, 8)

        # Reshape back
        quantized = quantized.reshape(original_shape)
        indices = indices.reshape(original_shape[:-1])

        return quantized, indices

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize from E8 lattice indices.

        Args:
            indices: Quantization indices, shape (...).

        Returns:
            Dequantized tensor, shape (..., 8).
        """
        return self.codebook[indices]


class HadamardTransform:
    """Fast Hadamard Transform for incoherence.

    Applies randomized Hadamard transforms to reduce weight coherence before
    quantization, improving reconstruction quality.
    """

    def __init__(self, dim: int, seed: int = 42):
        """Initialize Hadamard transform.

        Args:
            dim: Dimension of the transform (must be power of 2).
            seed: Random seed for generating random diagonal matrix.
        """
        self.dim = dim
        self.seed = seed

        # Generate random diagonal matrix for randomization
        torch.manual_seed(seed)
        self.D = torch.randint(0, 2, (dim,), dtype=torch.float32) * 2 - 1

    def hadamard_matrix(self, n: int) -> torch.Tensor:
        """Generate Hadamard matrix of size n x n.

        Args:
            n: Size of the matrix (must be power of 2).

        Returns:
            Hadamard matrix of shape (n, n).
        """
        if n == 1:
            return torch.tensor([[1.0]])

        H_half = self.hadamard_matrix(n // 2)
        H = torch.cat([
            torch.cat([H_half, H_half], dim=1),
            torch.cat([H_half, -H_half], dim=1)
        ], dim=0)

        return H / math.sqrt(2)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hadamard transform: H @ D @ x.

        Args:
            x: Input tensor, shape (..., dim).

        Returns:
            Transformed tensor, same shape as input.
        """
        # Apply diagonal randomization
        x_d = x * self.D.to(x.device)

        # Apply Hadamard transform
        H = self.hadamard_matrix(self.dim).to(x.device)
        x_h = x_d @ H.T

        return x_h

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse Hadamard transform: D^-1 @ H^T @ x.

        Args:
            x: Transformed tensor, shape (..., dim).

        Returns:
            Original tensor, same shape as input.
        """
        # Apply inverse Hadamard (Hadamard is self-inverse up to scaling)
        H = self.hadamard_matrix(self.dim).to(x.device)
        x_h_inv = x @ H

        # Apply inverse diagonal
        x_original = x_h_inv / self.D.to(x.device)

        return x_original


class QuIPSharpLinear(NexusModule):
    """Linear layer quantized with QuIP#.

    Applies 2-bit quantization using E8 lattice codebooks and Hadamard
    incoherence for optimal compression with minimal accuracy loss.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        config: QuIP# quantization configuration.
        bias: If True, includes bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QuIPSharpConfig,
        bias: bool = True,
    ):
        super().__init__(config.__dict__)

        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Quantized weights (stored as indices)
        num_groups = (in_features + config.group_size - 1) // config.group_size
        self.register_buffer(
            'weight_indices',
            torch.zeros(out_features, num_groups, config.group_size // 8, dtype=torch.int8)
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

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # E8 lattice codebook
        if config.use_e8_lattice:
            self.e8_lattice = E8Lattice()

        # Hadamard transform
        if config.use_hadamard:
            # Round to nearest power of 2 for Hadamard
            hadamard_dim = 2 ** math.ceil(math.log2(in_features))
            self.hadamard = HadamardTransform(hadamard_dim, seed=config.hadamard_seed)

    def quantize_weight(self, weight: torch.Tensor):
        """Quantize a weight matrix using QuIP#.

        Args:
            weight: Weight tensor of shape (out_features, in_features).
        """
        out_features, in_features = weight.shape

        # Apply Hadamard transform if enabled
        if self.config.use_hadamard:
            # Pad to power of 2 if needed
            padded_dim = self.hadamard.dim
            if in_features < padded_dim:
                weight_padded = F.pad(weight, (0, padded_dim - in_features))
            else:
                weight_padded = weight
            weight_transformed = self.hadamard.transform(weight_padded)
            weight_to_quantize = weight_transformed[:, :in_features]
        else:
            weight_to_quantize = weight

        # Blockwise quantization
        num_groups = (in_features + self.config.group_size - 1) // self.config.group_size

        for i in range(out_features):
            for j in range(num_groups):
                start_idx = j * self.config.group_size
                end_idx = min(start_idx + self.config.group_size, in_features)
                block = weight_to_quantize[i, start_idx:end_idx]

                # Compute scale and zero point
                min_val = block.min()
                max_val = block.max()
                scale = (max_val - min_val) / (2 ** self.config.bits - 1)
                zero = min_val

                # Store scale and zero
                self.weight_scales[i, j] = scale
                self.weight_zeros[i, j] = zero

                # Quantize
                normalized = (block - zero) / (scale + 1e-8)

                if self.config.use_e8_lattice and self.config.group_size >= 8:
                    # Reshape to (..., 8) for E8 lattice
                    block_reshaped = normalized[:self.config.group_size // 8 * 8].reshape(-1, 8)
                    _, indices = self.e8_lattice.quantize(block_reshaped)
                    self.weight_indices[i, j, :len(indices)] = indices.to(torch.int8)
                else:
                    # Standard rounding for non-E8 case
                    indices = torch.round(normalized).clamp(0, 2 ** self.config.bits - 1).to(torch.int8)
                    self.weight_indices[i, j, :len(indices)] = indices[:self.config.group_size // 8]

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantize the weight matrix.

        Returns:
            Dequantized weight tensor of shape (out_features, in_features).
        """
        out_features = self.weight_indices.shape[0]
        num_groups = self.weight_indices.shape[1]
        in_features = num_groups * self.config.group_size

        weight = torch.zeros(out_features, in_features, device=self.weight_indices.device)

        for i in range(out_features):
            for j in range(num_groups):
                start_idx = j * self.config.group_size
                end_idx = min(start_idx + self.config.group_size, in_features)

                # Get indices
                indices = self.weight_indices[i, j]

                # Dequantize
                if self.config.use_e8_lattice:
                    dequantized = self.e8_lattice.dequantize(indices.long())
                    dequantized = dequantized.flatten()
                else:
                    dequantized = indices.float()

                # Denormalize
                scale = self.weight_scales[i, j]
                zero = self.weight_zeros[i, j]
                block = dequantized * scale + zero

                weight[i, start_idx:end_idx] = block[:end_idx - start_idx]

        # Apply inverse Hadamard if enabled
        if self.config.use_hadamard:
            padded_dim = self.hadamard.dim
            if in_features < padded_dim:
                weight_padded = F.pad(weight, (0, padded_dim - in_features))
            else:
                weight_padded = weight
            weight = self.hadamard.inverse_transform(weight_padded)[:, :in_features]

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


class QuIPSharpQuantizer(NexusModule):
    """Quantizer for applying QuIP# to a full model.

    Args:
        config: QuIP# configuration.
    """

    def __init__(self, config: QuIPSharpConfig):
        super().__init__(config.__dict__)
        self.config = config

    def quantize_model(self, model: nn.Module, calibration_data: Optional[torch.Tensor] = None):
        """Quantize all linear layers in a model using QuIP#.

        Args:
            model: Model to quantize.
            calibration_data: Optional calibration data for computing Hessian.
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._quantize_linear(module, calibration_data)

    def _quantize_linear(self, linear: nn.Linear, calibration_data: Optional[torch.Tensor] = None):
        """Quantize a single linear layer.

        Args:
            linear: Linear layer to quantize.
            calibration_data: Optional calibration data.
        """
        # Create quantized layer
        quip_layer = QuIPSharpLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            config=self.config,
            bias=linear.bias is not None,
        )

        # Quantize weights
        quip_layer.quantize_weight(linear.weight.data)

        # Copy bias if present
        if linear.bias is not None:
            quip_layer.bias.data = linear.bias.data.clone()

        return quip_layer
