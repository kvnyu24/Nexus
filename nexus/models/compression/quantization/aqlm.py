"""AQLM: Additive Quantization of Language Models with Multi-Codebook.

Reference:
    Egiazarian, V., et al. "Extreme Compression of Large Language Models via
    Additive Quantization."
    ICML 2024. https://arxiv.org/abs/2401.06118

AQLM uses additive quantization with multiple codebooks to achieve extreme
compression (2-bit per weight on average) while maintaining model quality.
Instead of quantizing each weight to a single codebook entry, AQLM represents
each weight as a sum of multiple codebook entries, allowing for much finer
granularity and better reconstruction.

Key innovation: Multi-codebook additive quantization enables 2-bit compression
with accuracy close to 4-bit methods by using sum of multiple low-bit codes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import math

from nexus.core.base import NexusModule


@dataclass
class AQLMConfig:
    """Configuration for AQLM quantization.

    Attributes:
        num_codebooks: Number of codebooks to use (typically 8-16).
        codebook_size: Size of each codebook (typically 256 for 8-bit indices).
        out_group_size: Number of output features to group together.
        in_group_size: Number of input features to group together.
        nbits_per_codebook: Number of bits per codebook (typically 8).
        use_beam_search: Whether to use beam search for quantization.
        beam_size: Beam size for beam search quantization.
    """
    num_codebooks: int = 8
    codebook_size: int = 256
    out_group_size: int = 8
    in_group_size: int = 8
    nbits_per_codebook: int = 8
    use_beam_search: bool = True
    beam_size: int = 4


class MultiCodebook(NexusModule):
    """Multi-codebook for additive quantization.

    Maintains multiple codebooks where each weight is represented as a sum
    of entries from different codebooks.

    Args:
        num_codebooks: Number of codebooks.
        codebook_size: Number of entries per codebook.
        codebook_dim: Dimension of each codebook entry.
    """

    def __init__(self, num_codebooks: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        # Initialize codebooks: (num_codebooks, codebook_size, codebook_dim)
        self.codebooks = nn.Parameter(
            torch.randn(num_codebooks, codebook_size, codebook_dim) * 0.01
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices to weights using additive quantization.

        Args:
            indices: Tensor of shape (..., num_codebooks) with codebook indices.

        Returns:
            Decoded tensor of shape (..., codebook_dim).
        """
        # Gather from each codebook and sum
        result = torch.zeros(
            *indices.shape[:-1],
            self.codebook_dim,
            device=indices.device,
            dtype=self.codebooks.dtype
        )

        for i in range(self.num_codebooks):
            # Get indices for this codebook
            cb_indices = indices[..., i]  # (...,)

            # Gather from this codebook
            cb_values = self.codebooks[i][cb_indices]  # (..., codebook_dim)

            # Accumulate
            result = result + cb_values

        return result

    def quantize_greedy(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize using greedy sequential codebook selection.

        Args:
            x: Tensor to quantize, shape (..., codebook_dim).

        Returns:
            Indices tensor of shape (..., num_codebooks).
        """
        residual = x.clone()
        indices = torch.zeros(
            *x.shape[:-1],
            self.num_codebooks,
            device=x.device,
            dtype=torch.long
        )

        for i in range(self.num_codebooks):
            # Find nearest codebook entry
            # residual: (..., codebook_dim)
            # codebooks[i]: (codebook_size, codebook_dim)
            distances = torch.cdist(
                residual.reshape(-1, self.codebook_dim),
                self.codebooks[i]
            )  # (N, codebook_size)

            # Select nearest
            cb_indices = torch.argmin(distances, dim=-1)  # (N,)
            cb_indices = cb_indices.reshape(*x.shape[:-1])

            # Store indices
            indices[..., i] = cb_indices

            # Update residual
            selected = self.codebooks[i][cb_indices]
            residual = residual - selected

        return indices

    def quantize_beam_search(self, x: torch.Tensor, beam_size: int = 4) -> torch.Tensor:
        """Quantize using beam search for better quality.

        Args:
            x: Tensor to quantize, shape (..., codebook_dim).
            beam_size: Size of the beam.

        Returns:
            Indices tensor of shape (..., num_codebooks).
        """
        # For simplicity, fall back to greedy for now
        # Full beam search implementation would be more complex
        return self.quantize_greedy(x)


class AQLMLinear(NexusModule):
    """Linear layer quantized with AQLM multi-codebook approach.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        config: AQLM configuration.
        bias: If True, includes bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: AQLMConfig,
        bias: bool = True,
    ):
        super().__init__(config.__dict__)

        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Compute number of groups
        self.num_out_groups = (out_features + config.out_group_size - 1) // config.out_group_size
        self.num_in_groups = (in_features + config.in_group_size - 1) // config.in_group_size

        # Total dimension for codebooks
        codebook_dim = config.out_group_size * config.in_group_size

        # Multi-codebook for weight quantization
        self.codebooks = MultiCodebook(
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
            codebook_dim=codebook_dim
        )

        # Quantized weight indices
        # Shape: (num_out_groups, num_in_groups, num_codebooks)
        self.register_buffer(
            'weight_indices',
            torch.zeros(
                self.num_out_groups,
                self.num_in_groups,
                config.num_codebooks,
                dtype=torch.long
            )
        )

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def quantize_weight(self, weight: torch.Tensor):
        """Quantize a weight matrix using AQLM.

        Args:
            weight: Weight tensor of shape (out_features, in_features).
        """
        out_features, in_features = weight.shape

        # Reshape weight into groups
        # Pad if necessary
        padded_out = self.num_out_groups * self.config.out_group_size
        padded_in = self.num_in_groups * self.config.in_group_size

        if out_features < padded_out or in_features < padded_in:
            weight_padded = F.pad(
                weight,
                (0, padded_in - in_features, 0, padded_out - out_features)
            )
        else:
            weight_padded = weight

        # Reshape to groups
        weight_grouped = weight_padded.reshape(
            self.num_out_groups,
            self.config.out_group_size,
            self.num_in_groups,
            self.config.in_group_size
        )
        # Rearrange to (num_out_groups, num_in_groups, out_group_size * in_group_size)
        weight_grouped = weight_grouped.permute(0, 2, 1, 3).reshape(
            self.num_out_groups,
            self.num_in_groups,
            self.config.out_group_size * self.config.in_group_size
        )

        # Quantize each group using multi-codebook
        for i in range(self.num_out_groups):
            for j in range(self.num_in_groups):
                group = weight_grouped[i, j]  # (codebook_dim,)

                # Quantize
                if self.config.use_beam_search:
                    indices = self.codebooks.quantize_beam_search(
                        group.unsqueeze(0),
                        beam_size=self.config.beam_size
                    ).squeeze(0)
                else:
                    indices = self.codebooks.quantize_greedy(group.unsqueeze(0)).squeeze(0)

                self.weight_indices[i, j] = indices

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantize the weight matrix.

        Returns:
            Dequantized weight tensor of shape (out_features, in_features).
        """
        # Decode each group
        weight_grouped = self.codebooks(self.weight_indices)
        # weight_grouped: (num_out_groups, num_in_groups, codebook_dim)

        # Reshape back to weight matrix
        weight_grouped = weight_grouped.reshape(
            self.num_out_groups,
            self.num_in_groups,
            self.config.out_group_size,
            self.config.in_group_size
        )
        weight_grouped = weight_grouped.permute(0, 2, 1, 3).reshape(
            self.num_out_groups * self.config.out_group_size,
            self.num_in_groups * self.config.in_group_size
        )

        # Crop to original size
        weight = weight_grouped[:self.out_features, :self.in_features]

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

        # Memory for indices
        indices_bits = (
            self.num_out_groups *
            self.num_in_groups *
            self.config.num_codebooks *
            self.config.nbits_per_codebook
        )

        # Memory for codebooks
        codebook_params = (
            self.config.num_codebooks *
            self.config.codebook_size *
            self.config.out_group_size *
            self.config.in_group_size
        )
        codebook_bits = codebook_params * 32  # FP32 for codebooks

        # Total memory
        total_bits = indices_bits + codebook_bits

        # Baseline (FP32)
        baseline_bits = total_weights * 32

        compression_ratio = baseline_bits / total_bits

        return {
            'total_weights': total_weights,
            'bits_per_weight': total_bits / total_weights,
            'compression_ratio': compression_ratio,
            'num_codebooks': self.config.num_codebooks,
            'codebook_size': self.config.codebook_size,
        }


class AQLMQuantizer(NexusModule):
    """Quantizer for applying AQLM to a full model.

    Args:
        config: AQLM configuration.
    """

    def __init__(self, config: AQLMConfig):
        super().__init__(config.__dict__)
        self.config = config

    def quantize_model(
        self,
        model: nn.Module,
        verbose: bool = True
    ) -> nn.Module:
        """Quantize all linear layers in a model using AQLM.

        Args:
            model: Model to quantize.
            verbose: If True, print compression statistics.

        Returns:
            Quantized model.
        """
        modules_quantized = 0
        total_compression = 0.0

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Create quantized layer
                aqlm_layer = AQLMLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    config=self.config,
                    bias=module.bias is not None,
                )

                # Quantize weights
                aqlm_layer.quantize_weight(module.weight.data)

                # Copy bias if present
                if module.bias is not None:
                    aqlm_layer.bias.data = module.bias.data.clone()

                # Replace module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                setattr(parent, child_name, aqlm_layer)

                # Track statistics
                stats = aqlm_layer.get_compression_stats()
                total_compression += stats['compression_ratio']
                modules_quantized += 1

        if verbose:
            avg_compression = total_compression / modules_quantized if modules_quantized > 0 else 0
            print(f"AQLM Quantization Complete:")
            print(f"  Modules quantized: {modules_quantized}")
            print(f"  Number of codebooks: {self.config.num_codebooks}")
            print(f"  Codebook size: {self.config.codebook_size}")
            print(f"  Average compression ratio: {avg_compression:.2f}x")

        return model


def apply_aqlm(
    model: nn.Module,
    config: Optional[AQLMConfig] = None,
    verbose: bool = True
) -> nn.Module:
    """Apply AQLM quantization to a model.

    Args:
        model: Model to quantize.
        config: AQLM configuration (uses defaults if None).
        verbose: If True, print statistics.

    Returns:
        Quantized model.
    """
    config = config or AQLMConfig()
    quantizer = AQLMQuantizer(config)
    return quantizer.quantize_model(model, verbose)
