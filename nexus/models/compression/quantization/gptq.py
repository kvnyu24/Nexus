"""GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.

Reference:
    Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2023).
    "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained
    Transformers." In International Conference on Learning Representations
    (ICLR 2023). https://arxiv.org/abs/2210.17323

GPTQ applies layer-wise quantization using an approximate second-order
(Hessian-based) framework derived from Optimal Brain Surgeon (OBS).
It processes columns of the weight matrix one at a time (or in blocks),
quantizes each column, and compensates the remaining unquantized columns
to minimize the overall layer output error. This achieves significantly
better quality than naive round-to-nearest quantization, enabling
4-bit and even 3-bit quantization of large language models with
minimal accuracy loss.

Algorithm overview:
    1. Collect calibration data and compute the Hessian H = 2 * X * X^T,
       where X is the matrix of layer inputs from calibration samples.
    2. Compute the Cholesky decomposition of the inverse Hessian.
    3. For each block of columns:
        a. Quantize each column to the target bit width.
        b. Compute the quantization error for that column.
        c. Compensate all remaining (unquantized) columns using the OBS
           update formula: W_remaining -= error * H_inv_row / H_inv_diag.
    4. Optionally reorder columns by activation magnitude (act_order)
       so that the most salient weights are quantized first with the
       least accumulated error.

Key features:
    - Layer-wise quantization (no full-model retraining required)
    - Column-by-column OBS-style error compensation
    - Support for arbitrary bit widths (2, 3, 4, 8) and group sizes
    - Activation-order (act_order) permutation for improved accuracy
    - Efficient bit-packing for INT storage of quantized weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import math

from nexus.core.base import NexusModule


@dataclass
class GPTQConfig:
    """Configuration for GPTQ quantization.

    Attributes:
        bits: Number of bits per weight element. Typically 2, 3, 4, or 8.
            Lower bit widths yield higher compression but may reduce accuracy.
        group_size: Number of consecutive columns sharing the same quantization
            scale and zero-point. Smaller groups improve accuracy at the cost
            of additional storage for scale/zero metadata. Use -1 for
            per-channel (entire row shares one scale).
        act_order: If True, reorder columns by decreasing activation magnitude
            before quantization (also called "desc_act" in some implementations).
            This ensures the most salient weights are quantized first, reducing
            accumulated error on important channels.
        sym: If True, use symmetric quantization centered at zero (no zero-point
            offset). Symmetric quantization is faster at inference since it
            avoids the zero-point subtraction, but may sacrifice accuracy for
            weight distributions that are not centered around zero.
        damp_percent: Dampening factor added to the Hessian diagonal for
            numerical stability, expressed as a fraction of the mean diagonal
            value. Prevents singularity when inverting the Hessian.
        block_size: Number of columns to quantize simultaneously in each OBS
            block. Larger blocks amortize the cost of Hessian operations but
            require more memory for intermediate results.
        static_groups: If True, compute group quantization parameters (scales
            and zeros) once before OBS compensation, rather than recomputing
            them as weights are updated. Faster but slightly less accurate.
    """
    bits: int = 4
    group_size: int = 128
    act_order: bool = True
    sym: bool = True
    damp_percent: float = 0.01
    block_size: int = 128
    static_groups: bool = False


def pack_weights(int_weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer weight values into 32-bit integers for compact storage.

    Multiple low-bit weight values are packed into each int32 element.
    For example, with 4-bit quantization, 8 weight values fit in one int32.

    The packing order is column-major within each int32: the first weight
    occupies the lowest bits, the second weight occupies the next-lowest
    bits, and so on.

    Args:
        int_weight: Integer weight tensor of shape (out_features, in_features)
            with values in [0, 2^bits - 1].
        bits: Number of bits per weight element (2, 3, 4, or 8).

    Returns:
        Packed weight tensor of shape (out_features, in_features * bits / 32),
        with dtype torch.int32.
    """
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"Unsupported bit width for packing: {bits}. Must be 2, 3, 4, or 8.")

    int_weight = int_weight.to(torch.int32)
    out_features, in_features = int_weight.shape

    if bits == 8:
        # 8-bit: no packing needed beyond dtype cast
        return int_weight

    # Number of weight values packed per int32
    vals_per_int32 = 32 // bits
    # Handle cases where in_features is not divisible by vals_per_int32
    packed_cols = math.ceil(in_features / vals_per_int32)

    # Pad if necessary
    if in_features % vals_per_int32 != 0:
        pad_size = vals_per_int32 - (in_features % vals_per_int32)
        int_weight = F.pad(int_weight, (0, pad_size), value=0)

    packed = torch.zeros(out_features, packed_cols, dtype=torch.int32, device=int_weight.device)

    for i in range(vals_per_int32):
        packed |= int_weight[:, i::vals_per_int32] << (bits * i)

    return packed


def unpack_weights(packed: torch.Tensor, bits: int, out_features: int, in_features: int) -> torch.Tensor:
    """Unpack 32-bit packed integers back to individual weight values.

    Reverses the operation performed by pack_weights(). Extracts each
    low-bit weight value from the packed int32 representation.

    Args:
        packed: Packed weight tensor of shape (out_features, packed_cols)
            with dtype torch.int32.
        bits: Number of bits per weight element (2, 3, 4, or 8).
        out_features: Number of output features (rows).
        in_features: Number of input features (columns) in the original
            unpacked weight matrix.

    Returns:
        Unpacked integer weight tensor of shape (out_features, in_features)
        with values in [0, 2^bits - 1].
    """
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"Unsupported bit width for unpacking: {bits}. Must be 2, 3, 4, or 8.")

    if bits == 8:
        return packed[:, :in_features]

    vals_per_int32 = 32 // bits
    mask = (1 << bits) - 1

    unpacked_cols = packed.shape[1] * vals_per_int32
    unpacked = torch.zeros(out_features, unpacked_cols, dtype=torch.int32, device=packed.device)

    for i in range(vals_per_int32):
        unpacked[:, i::vals_per_int32] = (packed >> (bits * i)) & mask

    return unpacked[:, :in_features]


class QuantizedLinear(nn.Module):
    """Efficient integer-quantized linear layer for inference.

    Stores weights in packed integer format (2/3/4/8 bits per weight) with
    per-group scale and zero-point parameters. During the forward pass,
    weights are dequantized on-the-fly and multiplied with the input.

    This layer is the deployment-ready replacement for nn.Linear after
    GPTQ quantization. It reduces memory footprint by storing weights
    at low precision while performing computation in the original dtype.

    The dequantization formula is:
        W_float = (W_int - zeros) * scales

    where W_int is the integer weight, zeros is the zero-point, and
    scales is the per-group scale factor.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bits: Number of bits per weight (2, 3, 4, or 8).
        group_size: Number of columns per quantization group. Use -1 for
            per-channel quantization.
        bias: If True, includes a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size if group_size > 0 else in_features

        # Number of quantization groups
        self.n_groups = math.ceil(in_features / self.group_size)

        # Packed quantized weights: multiple values per int32
        if bits == 8:
            packed_cols = in_features
        else:
            vals_per_int32 = 32 // bits
            packed_cols = math.ceil(in_features / vals_per_int32)

        self.register_buffer(
            "qweight",
            torch.zeros(out_features, packed_cols, dtype=torch.int32),
        )

        # Per-group scale factors: shape (n_groups, out_features)
        self.register_buffer(
            "scales",
            torch.zeros(self.n_groups, out_features, dtype=torch.float16),
        )

        # Per-group zero points (packed if bits < 8)
        # Store zeros as float for simplicity; can be packed for further savings
        self.register_buffer(
            "zeros",
            torch.zeros(self.n_groups, out_features, dtype=torch.float16),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def _dequantize(self) -> torch.Tensor:
        """Dequantize the packed weight matrix to floating point.

        Unpacks integer weights, then applies per-group scale and zero-point
        to reconstruct the approximate floating-point weight matrix.

        Returns:
            Dequantized weight tensor of shape (out_features, in_features).
        """
        int_weight = unpack_weights(
            self.qweight, self.bits, self.out_features, self.in_features
        ).to(torch.float16)

        weight = torch.zeros(
            self.out_features, self.in_features,
            dtype=torch.float16, device=self.qweight.device,
        )

        for g in range(self.n_groups):
            g_start = g * self.group_size
            g_end = min(g_start + self.group_size, self.in_features)

            # scales[g] shape: (out_features,), zeros[g] shape: (out_features,)
            s = self.scales[g].unsqueeze(1)   # (out_features, 1)
            z = self.zeros[g].unsqueeze(1)    # (out_features, 1)

            weight[:, g_start:g_end] = (int_weight[:, g_start:g_end] - z) * s

        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: dequantize weights and compute linear transformation.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        weight = self._dequantize().to(x.dtype)
        output = F.linear(x, weight)

        if self.bias is not None:
            output = output + self.bias.to(x.dtype)

        return output

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        scales: Optional[torch.Tensor] = None,
        zeros: Optional[torch.Tensor] = None,
        quantized_int_weight: Optional[torch.Tensor] = None,
    ) -> "QuantizedLinear":
        """Create a QuantizedLinear from an nn.Linear and quantization parameters.

        This is typically called after GPTQ quantization has determined the
        optimal integer weights, scales, and zero points.

        Args:
            linear: The original nn.Linear layer.
            bits: Number of bits per weight.
            group_size: Quantization group size.
            scales: Per-group scale factors of shape (n_groups, out_features).
            zeros: Per-group zero points of shape (n_groups, out_features).
            quantized_int_weight: Integer weight tensor of shape
                (out_features, in_features) with values in [0, 2^bits - 1].

        Returns:
            A new QuantizedLinear instance with packed weights.
        """
        has_bias = linear.bias is not None
        ql = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bits=bits,
            group_size=group_size,
            bias=has_bias,
        )

        if quantized_int_weight is not None and scales is not None and zeros is not None:
            ql.qweight.copy_(pack_weights(quantized_int_weight, bits))
            ql.scales.copy_(scales.to(torch.float16))
            ql.zeros.copy_(zeros.to(torch.float16))

        if has_bias:
            ql.bias.copy_(linear.bias.data.to(torch.float16))

        return ql

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, "
            f"bias={self.bias is not None}"
        )


class GPTQQuantizer(NexusModule):
    """GPTQ quantizer: layer-wise post-training quantization using approximate
    second-order information.

    Quantizes the weights of a model layer by layer using calibration data to
    approximate the Hessian (H = 2 * X * X^T). Applies column-by-column OBS
    compensation to minimize the overall output reconstruction error. The
    algorithm processes columns in blocks for efficiency, and optionally
    reorders columns by activation magnitude for improved accuracy.

    The key insight from GPTQ is that quantizing weights one column at a time
    and compensating the error in remaining columns using the inverse Hessian
    yields significantly better results than independent rounding. This is
    derived from the Optimal Brain Surgeon framework but made practical for
    large models through lazy batch updates and Cholesky-based computation.

    Usage::

        gptq_config = GPTQConfig(bits=4, group_size=128)
        quantizer = GPTQQuantizer(gptq_config=gptq_config)

        # Collect calibration data (layer inputs)
        for batch in calibration_loader:
            quantizer.add_calibration_data(batch_activations)

        # Quantize a single layer
        quantized_w, scales, zeros, perm = quantizer.quantize_layer(layer.weight)

        # Or quantize an entire model
        model, metrics = quantizer.quantize_model(model, calibration_data)

    Args:
        config: Optional dict-based config with keys: bits, group_size,
            act_order, sym, damp_percent, block_size, static_groups.
        gptq_config: Optional GPTQConfig dataclass. If provided, takes
            precedence over the dict config.

    Reference:
        Frantar et al., "GPTQ: Accurate Post-Training Quantization for
        Generative Pre-Trained Transformers", ICLR 2023.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        gptq_config: Optional[GPTQConfig] = None,
    ):
        if gptq_config is not None:
            cfg = {
                "bits": gptq_config.bits,
                "group_size": gptq_config.group_size,
                "act_order": gptq_config.act_order,
                "sym": gptq_config.sym,
                "damp_percent": gptq_config.damp_percent,
                "block_size": gptq_config.block_size,
                "static_groups": gptq_config.static_groups,
            }
        else:
            cfg = config or {}

        super().__init__(cfg)

        self.bits = cfg.get("bits", 4)
        self.group_size = cfg.get("group_size", 128)
        self.act_order = cfg.get("act_order", True)
        self.sym = cfg.get("sym", True)
        self.damp_percent = cfg.get("damp_percent", 0.01)
        self.block_size = cfg.get("block_size", 128)
        self.static_groups = cfg.get("static_groups", False)

        # Calibration data accumulator for Hessian computation
        self._hessian: Optional[torch.Tensor] = None
        self._n_samples: int = 0

    @property
    def max_int(self) -> int:
        """Maximum unsigned integer representable with the configured bit width."""
        return 2 ** self.bits - 1

    def _compute_quant_range(self) -> Tuple[float, float]:
        """Compute the quantization grid bounds [qmin, qmax].

        For symmetric quantization, the range is centered at zero:
            [-2^(b-1), 2^(b-1) - 1]
        For asymmetric quantization, the range starts at zero:
            [0, 2^b - 1]

        Returns:
            Tuple of (qmin, qmax) as floats.
        """
        if self.sym:
            qmin = -(2 ** (self.bits - 1))
            qmax = 2 ** (self.bits - 1) - 1
        else:
            qmin = 0
            qmax = self.max_int
        return float(qmin), float(qmax)

    def add_calibration_data(self, inputs: torch.Tensor) -> None:
        """Accumulate the Hessian approximation from calibration inputs.

        The Hessian is approximated as H = (2/N) * X^T * X, where X is the
        matrix of calibration inputs and N is the total number of samples.
        Multiple batches can be added incrementally; the Hessian is normalized
        when _prepare_hessian() is called.

        Args:
            inputs: Calibration input tensor of shape (batch_size, in_features)
                or (batch_size, seq_len, in_features). If 3D, the sequence and
                batch dimensions are flattened.
        """
        if inputs.dim() == 3:
            inputs = inputs.reshape(-1, inputs.shape[-1])

        inputs = inputs.float()
        n = inputs.shape[0]

        # H += X^T @ X (accumulated, normalized later)
        hessian_batch = inputs.T @ inputs

        if self._hessian is None:
            self._hessian = hessian_batch
        else:
            self._hessian += hessian_batch

        self._n_samples += n

    def _compute_hessian(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the Hessian matrix H = 2 * X^T * X from calibration inputs.

        This is a convenience method that computes the Hessian directly from
        a single batch of inputs without using the incremental accumulator.
        The factor of 2 comes from the second-order Taylor expansion of the
        squared error loss.

        Args:
            inputs: Input tensor of shape (n_samples, in_features) or
                (n_samples, seq_len, in_features).

        Returns:
            Hessian matrix of shape (in_features, in_features).
        """
        if inputs.dim() == 3:
            inputs = inputs.reshape(-1, inputs.shape[-1])

        inputs = inputs.float()
        n = inputs.shape[0]
        H = 2.0 * (inputs.T @ inputs) / n

        # Dampening for numerical stability
        damp = self.damp_percent * H.diag().mean()
        H.diagonal().add_(damp)

        return H

    def _prepare_hessian(self) -> torch.Tensor:
        """Normalize and dampen the incrementally accumulated Hessian.

        Divides the accumulated X^T @ X by the number of samples and adds
        a dampening term to the diagonal for numerical stability when
        computing the inverse.

        Returns:
            The processed Hessian matrix of shape (in_features, in_features),
            ready for Cholesky decomposition.

        Raises:
            RuntimeError: If no calibration data has been added.
        """
        if self._hessian is None:
            raise RuntimeError(
                "No calibration data added. Call add_calibration_data() first."
            )

        H = self._hessian / self._n_samples

        # Add dampening: damp = damp_percent * mean(diag(H))
        damp = self.damp_percent * H.diag().mean()
        H.diagonal().add_(damp)

        return H

    def _quantize_column_group(
        self,
        weight_col: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a group of weight values to the target bit width.

        Computes per-group scale and zero-point from the weight range, then
        applies round-to-nearest quantization followed by dequantization.

        Args:
            weight_col: Weight values of shape (out_features, group_cols).

        Returns:
            Tuple of (dequantized_weight, scale, zero_point), where:
                - dequantized_weight has the same shape as input
                - scale has shape (group_cols,) or scalar
                - zero_point has shape (group_cols,) or scalar
        """
        qmin, qmax = self._compute_quant_range()

        w_min = weight_col.min(dim=0, keepdim=True).values
        w_max = weight_col.max(dim=0, keepdim=True).values

        if self.sym:
            w_absmax = torch.max(w_min.abs(), w_max.abs())
            scale = w_absmax / ((qmax - qmin) / 2)
            scale = scale.clamp(min=1e-10)
            zero = torch.zeros_like(scale)
        else:
            scale = (w_max - w_min) / (qmax - qmin)
            scale = scale.clamp(min=1e-10)
            zero = torch.round(-w_min / scale).clamp(qmin, qmax)

        quantized = torch.clamp(
            torch.round(weight_col / scale) + zero, qmin, qmax
        )
        dequantized = (quantized - zero) * scale

        return dequantized, scale.squeeze(0), zero.squeeze(0)

    def _quantize_weight(
        self, W: torch.Tensor, H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply the core GPTQ algorithm: column-by-column quantization with
        OBS error compensation.

        This implements the main loop of GPTQ:
            1. Optionally permute columns by activation magnitude.
            2. Compute the Cholesky decomposition of the inverse Hessian.
            3. For each block of columns:
                a. Quantize each column using group-wise scale/zero.
                b. Compute error = (w_original - w_quantized) / H_inv_diag.
                c. Update remaining columns: W -= error * H_inv_row.

        Args:
            W: Weight matrix of shape (out_features, in_features).
            H: Hessian matrix of shape (in_features, in_features).

        Returns:
            Tuple of:
                - quantized_W: Dequantized weight (float, same shape as W).
                - final_scales: Per-group scales of shape (out_features, n_groups).
                - final_zeros: Per-group zeros of shape (out_features, n_groups).
                - perm: Column permutation tensor if act_order=True, else None.
        """
        out_features, in_features = W.shape
        device = W.device

        W = W.clone().float()
        H = H.clone().float()

        # Activation-order permutation: sort columns by decreasing Hessian diagonal
        perm: Optional[torch.Tensor] = None
        if self.act_order:
            perm = torch.argsort(H.diag(), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        # Compute Cholesky of H^{-1} (upper triangular)
        try:
            H_inv = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(H_inv)
            H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)
        except RuntimeError:
            # Fallback: add stronger dampening and retry
            extra_damp = 0.1 * H.diag().mean()
            H.diagonal().add_(extra_damp)
            H_inv = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(H_inv)
            H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)

        quantized_W = torch.zeros_like(W)
        group_size = self.group_size if self.group_size > 0 else in_features

        # Pre-compute group quantization parameters if static_groups
        if self.static_groups:
            n_groups = math.ceil(in_features / group_size)
            static_scales = []
            static_zeros = []
            for g in range(n_groups):
                g_start = g * group_size
                g_end = min(g_start + group_size, in_features)
                _, s, z = self._quantize_column_group(W[:, g_start:g_end])
                static_scales.append(s)
                static_zeros.append(z)

        # Process columns in blocks
        for block_start in range(0, in_features, self.block_size):
            block_end = min(block_start + self.block_size, in_features)
            block_sz = block_end - block_start

            W_block = W[:, block_start:block_end].clone()
            Q_block = torch.zeros_like(W_block)
            Err_block = torch.zeros_like(W_block)

            H_inv_block = H_inv_chol[block_start:block_end, block_start:block_end]

            for j in range(block_sz):
                col_idx = block_start + j
                w_col = W_block[:, j]
                h_inv_jj = H_inv_block[j, j]

                # Determine quantization parameters for this column's group
                qmin, qmax = self._compute_quant_range()
                group_idx = col_idx // group_size
                g_start = group_idx * group_size
                g_end = min(g_start + group_size, in_features)

                if self.static_groups:
                    # Use pre-computed parameters
                    s_tensor = static_scales[group_idx]
                    z_tensor = static_zeros[group_idx]
                    if s_tensor.dim() == 0:
                        s = s_tensor.item()
                        z = z_tensor.item()
                    else:
                        s = s_tensor.mean().item()
                        z = z_tensor.mean().item()
                else:
                    # Compute from current (possibly updated) weights
                    group_w = W[:, g_start:g_end]
                    w_min = group_w.min()
                    w_max = group_w.max()

                    if self.sym:
                        w_absmax = max(abs(w_min.item()), abs(w_max.item()))
                        s = w_absmax / ((qmax - qmin) / 2)
                        s = max(s, 1e-10)
                        z = 0.0
                    else:
                        s = (w_max.item() - w_min.item()) / (qmax - qmin)
                        s = max(s, 1e-10)
                        z = round(-w_min.item() / s)
                        z = max(qmin, min(qmax, z))

                # Quantize and dequantize this column
                q_col = torch.clamp(torch.round(w_col / s) + z, qmin, qmax)
                dq_col = (q_col - z) * s

                Q_block[:, j] = dq_col

                # Compute OBS error: error = (w - q) / H_inv_jj
                err = (w_col - dq_col) / max(h_inv_jj.item(), 1e-10)
                Err_block[:, j] = err

                # Compensate remaining columns within this block
                if j + 1 < block_sz:
                    W_block[:, j + 1:] -= (
                        err.unsqueeze(1) * H_inv_block[j, j + 1:].unsqueeze(0)
                    )

            quantized_W[:, block_start:block_end] = Q_block

            # Compensate remaining blocks using accumulated error
            if block_end < in_features:
                W[:, block_end:] -= Err_block @ H_inv_chol[block_start:block_end, block_end:]

        # Compute final per-group scales and zeros for the quantized weight
        n_groups = math.ceil(in_features / group_size) if group_size > 0 else 1
        final_scales = torch.zeros(out_features, n_groups, device=device)
        final_zeros = torch.zeros(out_features, n_groups, device=device)

        # Use the quantized weight to compute accurate group parameters
        target_W = quantized_W
        if perm is not None:
            inv_perm = torch.argsort(perm)
            target_W = target_W[:, inv_perm]

        for g in range(n_groups):
            g_s = g * group_size
            g_e = min(g_s + group_size, in_features)
            _, s, z = self._quantize_column_group(target_W[:, g_s:g_e])
            if s.dim() == 0:
                final_scales[:, g] = s.expand(out_features)
                final_zeros[:, g] = z.expand(out_features)
            else:
                final_scales[:, g] = s.mean() if s.numel() > 1 else s
                final_zeros[:, g] = z.mean() if z.numel() > 1 else z

        # Undo permutation for the quantized weight matrix
        if perm is not None:
            quantized_W = quantized_W[:, inv_perm]

        return quantized_W, final_scales, final_zeros, perm

    def quantize_layer(
        self, layer: nn.Linear, inputs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Quantize a single linear layer using the GPTQ algorithm.

        If inputs are provided, they are used to compute the Hessian directly.
        Otherwise, the method uses the incrementally accumulated Hessian from
        prior calls to add_calibration_data().

        Args:
            layer: The nn.Linear layer to quantize, or a weight tensor.
            inputs: Optional calibration inputs of shape
                (n_samples, in_features) or (n_samples, seq_len, in_features).
                If provided, overrides any accumulated Hessian.

        Returns:
            Tuple of:
                - quantized_weight: Dequantized weight tensor (float).
                - scales: Per-group scale factors.
                - zeros: Per-group zero points.
                - perm: Column permutation (if act_order=True), else None.
        """
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data
        else:
            weight = layer

        if inputs is not None:
            H = self._compute_hessian(inputs)
        else:
            H = self._prepare_hessian()

        return self._quantize_weight(weight, H)

    def quantize_model(
        self,
        model: nn.Module,
        dataloader: List[torch.Tensor],
        nsamples: int = 128,
        target_layers: Optional[List[str]] = None,
        replace_with_quantized: bool = False,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Quantize all (or targeted) linear layers in a model using GPTQ.

        For each target layer:
            1. Register a forward hook to capture layer inputs during
               calibration.
            2. Run calibration data through the model to collect activations.
            3. Build the Hessian from collected activations.
            4. Quantize the layer weight using the GPTQ algorithm.
            5. Replace the weight with the quantized version (or optionally
               replace the entire layer with a QuantizedLinear).

        Args:
            model: The model to quantize. Modified in-place.
            dataloader: List of input tensors (or dicts) for calibration.
                At most nsamples batches are used.
            nsamples: Maximum number of calibration batches to use.
            target_layers: Optional list of layer name substrings to target.
                If None, all nn.Linear layers are quantized.
            replace_with_quantized: If True, replace nn.Linear layers with
                QuantizedLinear for memory-efficient inference.

        Returns:
            Tuple of (quantized model, metrics dict). The metrics dict maps
            layer names to their quantization statistics (MSE, bits, etc.).
        """
        metrics: Dict[str, Any] = {}
        was_training = model.training
        model.eval()

        # Limit calibration samples
        calibration_data = dataloader[:nsamples]

        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if target_layers is not None:
                if not any(t in name for t in target_layers):
                    continue

            # Reset calibration state
            self._hessian = None
            self._n_samples = 0

            # Gather calibration activations via forward hook
            activations: List[torch.Tensor] = []

            def _make_hook(acts_list):
                def _hook(mod, inp, out):
                    if isinstance(inp, tuple):
                        acts_list.append(inp[0].detach())
                    else:
                        acts_list.append(inp.detach())
                return _hook

            handle = module.register_forward_hook(_make_hook(activations))

            with torch.no_grad():
                for data in calibration_data:
                    try:
                        if isinstance(data, dict):
                            model(**data)
                        else:
                            model(data)
                    except Exception:
                        pass

            handle.remove()

            # Accumulate Hessian from collected activations
            for act in activations:
                self.add_calibration_data(act)

            if self._hessian is None:
                continue

            # Quantize the layer
            quantized_w, scales, zeros, perm_result = self.quantize_layer(module)

            # Compute reconstruction error
            error = (module.weight.data.float() - quantized_w).pow(2).mean().item()
            metrics[name] = {
                "mse": error,
                "bits": self.bits,
                "group_size": self.group_size,
            }

            if replace_with_quantized:
                # Convert to integer representation for packing
                qmin, qmax = self._compute_quant_range()
                group_size = self.group_size if self.group_size > 0 else module.in_features
                n_groups = math.ceil(module.in_features / group_size)

                # Compute integer weights from the dequantized result
                int_weight = torch.zeros_like(quantized_w, dtype=torch.int32)
                for g in range(n_groups):
                    g_start = g * group_size
                    g_end = min(g_start + group_size, module.in_features)
                    s = scales[:, g].unsqueeze(1)
                    z = zeros[:, g].unsqueeze(1)
                    int_weight[:, g_start:g_end] = torch.clamp(
                        torch.round(quantized_w[:, g_start:g_end] / s.clamp(min=1e-10)) + z,
                        qmin, qmax,
                    ).to(torch.int32)

                # Create QuantizedLinear replacement
                ql = QuantizedLinear.from_linear(
                    module,
                    bits=self.bits,
                    group_size=self.group_size,
                    scales=scales.T,  # (n_groups, out_features)
                    zeros=zeros.T,
                    quantized_int_weight=int_weight,
                )

                # Replace in model
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], ql)
            else:
                # Replace weight with dequantized (float) quantized version
                module.weight.data.copy_(quantized_w.to(module.weight.dtype))

        model.train(was_training)
        return model, metrics

    def reset_calibration(self) -> None:
        """Clear accumulated calibration data and Hessian."""
        self._hessian = None
        self._n_samples = 0

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize a weight tensor using accumulated calibration data.

        This is a convenience method that allows using the quantizer as a
        callable module. Requires prior calibration via add_calibration_data().

        Args:
            weight: Weight tensor of shape (out_features, in_features).

        Returns:
            Dequantized weight tensor after GPTQ quantization.
        """
        quantized_w, _, _, _ = self.quantize_layer(weight)
        return quantized_w
