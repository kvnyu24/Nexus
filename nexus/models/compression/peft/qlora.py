"""QLoRA: Quantized Low-Rank Adaptation for efficient fine-tuning.

Reference:
    Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized LLMs."
    NeurIPS 2023. https://arxiv.org/abs/2305.14314

QLoRA combines 4-bit NormalFloat (NF4) quantization of the base model
weights with low-rank adapters trained in higher precision (fp16/bf16).
Double quantization further compresses the quantization constants
themselves, reducing the memory footprint to approximately 4 bits per
parameter for the frozen base while keeping adapter updates in full
precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union
import math

from nexus.core.base import NexusModule


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA adaptation.

    Attributes:
        rank: Rank of the LoRA low-rank decomposition.
        alpha: LoRA scaling factor. Effective scaling is alpha / rank.
        bits: Number of bits for base weight quantization (typically 4).
        double_quant: Whether to apply double quantization to the
            quantization constants, further reducing memory.
        quant_type: Quantization type. "nf4" for NormalFloat4 or "fp4"
            for standard 4-bit floating point.
        compute_dtype: Data type for LoRA computations. Use torch.float16
            or torch.bfloat16 for mixed-precision training.
        blocksize: Number of elements per quantization block. Larger blocks
            reduce overhead but may decrease accuracy.
        dropout: Dropout probability on the LoRA adapter input path.
        target_modules: Module name patterns to apply QLoRA to.
    """
    rank: int = 16
    alpha: float = 32.0
    bits: int = 4
    double_quant: bool = True
    quant_type: str = "nf4"
    compute_dtype: torch.dtype = torch.float16
    blocksize: int = 64
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'v_proj', 'k_proj', 'o_proj'])


class NF4Quantize(NexusModule):
    """NormalFloat4 quantization module.

    NF4 is an information-theoretically optimal data type for normally
    distributed weights. It maps 16 quantization bins to values that
    minimize the expected quantization error under a standard normal
    distribution.

    The quantization process:
        1. Divide the weight tensor into blocks of size `blocksize`.
        2. For each block, compute the absmax scaling constant.
        3. Normalize weights to [-1, 1] using the scaling constant.
        4. Map each normalized value to the nearest NF4 quantile.
        5. Store indices (4-bit) and per-block scaling constants.

    Args:
        blocksize: Number of elements per quantization block.
    """

    # Pre-computed NF4 quantile values for a standard normal distribution
    NF4_QUANT_TABLE: torch.Tensor = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367,
        -0.39491748809814453, -0.28444138169288635,
        -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725,
        0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0,
    ], dtype=torch.float32)

    def __init__(self, blocksize: int = 64):
        super().__init__({"blocksize": blocksize})
        self.blocksize = blocksize
        self.register_buffer("quant_table", self.NF4_QUANT_TABLE.clone())

    def quantize(
        self, weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        """Quantize a weight tensor to NF4 format.

        Args:
            weight: The floating-point weight tensor to quantize.

        Returns:
            Tuple of (quantized_indices, absmax_scales, original_shape).
            quantized_indices: uint8 tensor of packed 4-bit indices.
            absmax_scales: Per-block scaling constants in float32.
            original_shape: Original tensor shape for dequantization.
        """
        original_shape = weight.shape
        weight_flat = weight.reshape(-1).float()

        # Pad to multiple of blocksize
        n = weight_flat.numel()
        pad_len = (self.blocksize - n % self.blocksize) % self.blocksize
        if pad_len > 0:
            weight_flat = F.pad(weight_flat, (0, pad_len))

        # Reshape into blocks
        blocks = weight_flat.reshape(-1, self.blocksize)
        absmax = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)

        # Normalize to [-1, 1]
        normalized = blocks / absmax

        # Find nearest NF4 quantile for each element
        quant_table = self.quant_table.to(normalized.device)
        distances = (normalized.unsqueeze(-1) - quant_table.unsqueeze(0).unsqueeze(0))
        indices = distances.abs().argmin(dim=-1)

        # Pack two 4-bit indices into one uint8
        indices_flat = indices.reshape(-1)
        if indices_flat.numel() % 2 != 0:
            indices_flat = F.pad(indices_flat, (0, 1))
        packed = (indices_flat[0::2] << 4) | indices_flat[1::2]
        packed = packed.to(torch.uint8)

        absmax_flat = absmax.squeeze(1)

        return packed, absmax_flat, original_shape

    def dequantize(
        self,
        packed: torch.Tensor,
        absmax: torch.Tensor,
        original_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Dequantize NF4 data back to floating-point.

        Args:
            packed: Packed uint8 tensor of 4-bit quantized indices.
            absmax: Per-block absmax scaling constants.
            original_shape: Original weight tensor shape.

        Returns:
            Dequantized floating-point weight tensor.
        """
        quant_table = self.quant_table.to(packed.device)

        # Unpack 4-bit indices
        high = (packed >> 4).to(torch.long)
        low = (packed & 0x0F).to(torch.long)
        indices = torch.stack([high, low], dim=-1).reshape(-1)

        # Lookup quantized values
        dequant_flat = quant_table[indices]

        # Reshape into blocks and rescale
        n_elements = math.prod(original_shape)
        pad_len = (self.blocksize - n_elements % self.blocksize) % self.blocksize
        total = n_elements + pad_len
        dequant_flat = dequant_flat[:total]
        blocks = dequant_flat.reshape(-1, self.blocksize)
        blocks = blocks * absmax.unsqueeze(1).to(blocks.device)

        # Remove padding and reshape
        result = blocks.reshape(-1)[:n_elements]
        return result.reshape(original_shape)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize and immediately dequantize for simulated quantization.

        Args:
            weight: Input weight tensor.

        Returns:
            Reconstructed weight tensor after NF4 round-trip.
        """
        packed, absmax, shape = self.quantize(weight)
        return self.dequantize(packed, absmax, shape)


class DoubleQuantization(NexusModule):
    """Double quantization: quantize the quantization scaling constants.

    In standard block-wise quantization, each block of weights has an
    associated fp32 scaling constant. Double quantization further
    compresses these constants by quantizing them to 8-bit, reducing the
    per-parameter overhead from 0.5 bits to approximately 0.127 bits.

    Args:
        blocksize: Block size for the second-level quantization of
            the scaling constants.
    """

    def __init__(self, blocksize: int = 256):
        super().__init__({"blocksize": blocksize})
        self.blocksize = blocksize

    def quantize_constants(
        self, absmax: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize the per-block absmax scaling constants to 8-bit.

        Args:
            absmax: Float32 tensor of per-block scaling constants.

        Returns:
            Tuple of (quantized_absmax_uint8, second_level_scales,
            second_level_zeros).
        """
        absmax_flat = absmax.reshape(-1).float()
        n = absmax_flat.numel()
        pad_len = (self.blocksize - n % self.blocksize) % self.blocksize
        if pad_len > 0:
            absmax_flat = F.pad(absmax_flat, (0, pad_len))

        blocks = absmax_flat.reshape(-1, self.blocksize)

        # Compute per-superblock min and max
        block_min = blocks.min(dim=1, keepdim=True).values
        block_max = blocks.max(dim=1, keepdim=True).values
        scale = (block_max - block_min).clamp(min=1e-12) / 255.0

        # Quantize to uint8
        quantized = ((blocks - block_min) / scale).round().clamp(0, 255).to(torch.uint8)

        return quantized, scale.squeeze(1), block_min.squeeze(1)

    def dequantize_constants(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        original_numel: int,
    ) -> torch.Tensor:
        """Dequantize the scaling constants back to float32.

        Args:
            quantized: uint8 quantized scaling constants.
            scale: Second-level scale factors.
            zero: Second-level zero points.
            original_numel: Number of original scaling constants.

        Returns:
            Reconstructed float32 scaling constants.
        """
        blocks = quantized.float() * scale.unsqueeze(1) + zero.unsqueeze(1)
        return blocks.reshape(-1)[:original_numel]

    def forward(
        self, absmax: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the double quantization forward pass.

        Args:
            absmax: Per-block scaling constants to double-quantize.

        Returns:
            Tuple of (quantized, scale, zero) for later dequantization.
        """
        return self.quantize_constants(absmax)


class QLoRALinear(NexusModule):
    """Linear layer with 4-bit quantized base weights and LoRA adapters.

    The base weight W is stored in NF4 format (approximately 4 bits per
    parameter). During the forward pass, W is dequantized on-the-fly to
    compute_dtype, then the standard linear + LoRA computation proceeds:

        output = dequant(W_nf4) @ x + scaling * B @ A @ x

    This achieves near-fp16 quality with a fraction of the memory.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: LoRA adapter rank.
        alpha: LoRA scaling factor.
        bits: Quantization bit width for base weights.
        double_quant: Whether to double-quantize the scaling constants.
        compute_dtype: Precision for dequantized computation.
        blocksize: Block size for NF4 quantization.
        dropout: LoRA dropout probability.
        bias: Whether the layer includes a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        bits: int = 4,
        double_quant: bool = True,
        compute_dtype: torch.dtype = torch.float16,
        blocksize: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        config = {
            "in_features": in_features,
            "out_features": out_features,
            "rank": rank,
            "alpha": alpha,
            "bits": bits,
            "double_quant": double_quant,
            "blocksize": blocksize,
            "dropout": dropout,
            "bias": bias,
        }
        super().__init__(config)

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.bits = bits
        self.compute_dtype = compute_dtype
        self.blocksize = blocksize
        self.use_double_quant = double_quant

        # NF4 quantizer
        self.nf4 = NF4Quantize(blocksize=blocksize)

        # Double quantization module
        self.double_quant = DoubleQuantization() if double_quant else None

        # Quantized weight storage (populated during quantize_weights)
        self.register_buffer("weight_packed", None)
        self.register_buffer("weight_absmax", None)
        self._weight_shape: Optional[Tuple[int, ...]] = None
        self._is_quantized = False

        # Double-quantized constant storage
        self.register_buffer("dq_absmax", None)
        self.register_buffer("dq_scale", None)
        self.register_buffer("dq_zero", None)
        self._dq_original_numel: int = 0

        # Optional bias (stored in compute_dtype)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=compute_dtype))
        else:
            self.bias = None

        # LoRA adapters in higher precision
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, dtype=compute_dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=compute_dtype)
        )

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self._init_lora_weights()

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights: Kaiming for A, zeros for B."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def quantize_weights(self, weight: torch.Tensor) -> None:
        """Quantize a floating-point weight matrix into NF4 format.

        This should be called once during model preparation, before
        training begins.

        Args:
            weight: The original floating-point weight tensor of shape
                (out_features, in_features).
        """
        self._weight_shape = weight.shape
        packed, absmax, _ = self.nf4.quantize(weight)
        self.weight_packed = packed
        self._is_quantized = True

        if self.use_double_quant and self.double_quant is not None:
            self._dq_original_numel = absmax.numel()
            dq_quant, dq_scale, dq_zero = self.double_quant.quantize_constants(absmax)
            self.dq_absmax = dq_quant
            self.dq_scale = dq_scale
            self.dq_zero = dq_zero
        else:
            self.weight_absmax = absmax

    def _dequantize_weight(self) -> torch.Tensor:
        """Dequantize the stored NF4 weight to compute_dtype.

        Returns:
            Dequantized weight tensor of shape (out_features, in_features).
        """
        if self.use_double_quant and self.dq_absmax is not None:
            absmax = self.double_quant.dequantize_constants(
                self.dq_absmax, self.dq_scale, self.dq_zero,
                self._dq_original_numel,
            )
        else:
            absmax = self.weight_absmax

        weight = self.nf4.dequantize(self.weight_packed, absmax, self._weight_shape)
        return weight.to(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization and LoRA.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        x = x.to(self.compute_dtype)

        if self._is_quantized:
            weight = self._dequantize_weight()
        else:
            # Fallback: no quantized weights yet (pre-quantization)
            weight = torch.zeros(
                self.out_features, self.in_features,
                dtype=self.compute_dtype, device=x.device,
            )

        # Base linear computation
        base_output = F.linear(x, weight, self.bias)

        # LoRA adapter path
        lora_input = self.lora_dropout(x)
        lora_output = F.linear(F.linear(lora_input, self.lora_A), self.lora_B)

        return base_output + lora_output * self.scaling

    def get_parameter_count(self) -> Dict[str, int]:
        """Return parameter counts for the QLoRA layer."""
        trainable = self.lora_A.numel() + self.lora_B.numel()
        if self.bias is not None:
            trainable += self.bias.numel()
        frozen = self.in_features * self.out_features
        return {
            "total": frozen + trainable,
            "trainable": trainable,
            "frozen": frozen,
            "memory_bits_base": frozen * self.bits,
        }

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        bits: int = 4,
        double_quant: bool = True,
        compute_dtype: torch.dtype = torch.float16,
        blocksize: int = 64,
        dropout: float = 0.0,
    ) -> "QLoRALinear":
        """Create a QLoRALinear from an existing nn.Linear layer.

        The original weight is quantized to NF4 format and the bias
        (if present) is preserved.

        Args:
            linear: The original nn.Linear layer.
            rank: LoRA rank.
            alpha: LoRA scaling.
            bits: Quantization bits.
            double_quant: Whether to double-quantize scaling constants.
            compute_dtype: Computation precision.
            blocksize: NF4 block size.
            dropout: LoRA dropout.

        Returns:
            A new QLoRALinear with quantized base weights and LoRA adapters.
        """
        has_bias = linear.bias is not None
        qlora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            bits=bits,
            double_quant=double_quant,
            compute_dtype=compute_dtype,
            blocksize=blocksize,
            dropout=dropout,
            bias=has_bias,
        )
        qlora.quantize_weights(linear.weight.data)
        if has_bias:
            qlora.bias.data.copy_(linear.bias.data.to(compute_dtype))
        return qlora


def apply_qlora(
    model: nn.Module,
    config: Optional[QLoRAConfig] = None,
    rank: int = 16,
    alpha: float = 32.0,
    bits: int = 4,
    double_quant: bool = True,
    compute_dtype: torch.dtype = torch.float16,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Apply QLoRA to all matching linear layers in a model.

    This function traverses the model, quantizes target linear layers
    to NF4, and attaches LoRA adapters.

    Args:
        model: The model to adapt.
        config: Optional QLoRAConfig; overrides individual arguments.
        rank: LoRA rank.
        alpha: LoRA scaling.
        bits: Quantization bits.
        double_quant: Whether to double-quantize.
        compute_dtype: Computation precision.
        target_modules: Module name patterns to target.

    Returns:
        The model with QLoRA layers injected (modified in-place).
    """
    import re

    if config is not None:
        rank = config.rank
        alpha = config.alpha
        bits = config.bits
        double_quant = config.double_quant
        compute_dtype = config.compute_dtype
        target_modules = config.target_modules

    target_modules = target_modules or ['q_proj', 'v_proj', 'k_proj', 'o_proj']
    patterns = [re.compile(p) for p in target_modules]

    def _matches(name: str) -> bool:
        return any(pattern.search(name) for pattern in patterns)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace matching layers
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _matches(name):
            qlora_layer = QLoRALinear.from_linear(
                module,
                rank=rank,
                alpha=alpha,
                bits=bits,
                double_quant=double_quant,
                compute_dtype=compute_dtype,
            )
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], qlora_layer)

    return model
