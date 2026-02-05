"""AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration.

Reference:
    Lin, J., et al. "AWQ: Activation-Aware Weight Quantization for LLM
    Compression and Acceleration." MLSys 2024.
    https://arxiv.org/abs/2306.00978

AWQ observes that not all weight channels are equally important for
model quality. A small fraction of salient weight channels (those
corresponding to large activation magnitudes) disproportionately affect
output quality. Rather than mixed-precision quantization (which is
hardware-unfriendly), AWQ scales salient channels up before quantization
to protect them from quantization error, then adjusts the corresponding
activations to maintain mathematical equivalence.

Key insight: For a weight column w and activation channel x,
    Q(w) * x approx= Q(s * w) * (x / s)
By choosing s proportional to the activation magnitude, salient channels
are effectively quantized at higher precision within a uniform bit width.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import math

from nexus.core.base import NexusModule


@dataclass
class AWQConfig:
    """Configuration for AWQ quantization.

    Attributes:
        bits: Number of bits per weight (typically 4).
        group_size: Number of columns sharing quantization parameters.
            Use -1 for per-channel quantization.
        zero_point: Whether to use asymmetric quantization with a
            zero-point offset. True gives better accuracy for
            asymmetric weight distributions.
        scale_search: Whether to perform grid search for optimal
            scaling factors. Setting False uses activation-magnitude-
            based scaling directly (faster but slightly less accurate).
        n_grid: Number of grid points for scale factor search.
        max_seq_len: Maximum sequence length for calibration.
        duo_scaling: Whether to apply dual scaling (both weight and
            activation scaling) for better accuracy.
    """
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    scale_search: bool = True
    n_grid: int = 20
    max_seq_len: int = 512
    duo_scaling: bool = True


class AWQQuantizer(NexusModule):
    """Activation-Aware Weight Quantizer.

    Implements the AWQ algorithm:
        1. Collect activation statistics from calibration data.
        2. Identify salient weight channels by activation magnitude.
        3. Compute per-channel scaling factors to protect salient channels.
        4. Optionally search for optimal scaling via grid search.
        5. Apply scaling, quantize, and de-scale for the final weights.

    Usage:
        config = AWQConfig(bits=4, group_size=128)
        quantizer = AWQQuantizer(awq_config=config)

        # Collect calibration statistics
        for batch in calibration_loader:
            quantizer.collect_activation_stats(layer_name, activations)

        # Quantize the layer
        result = quantizer.quantize_layer(layer.weight, layer_name)

    Args:
        config: Optional dict-based config.
        awq_config: Optional AWQConfig with quantization parameters.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        awq_config: Optional[AWQConfig] = None,
    ):
        if awq_config is not None:
            cfg = {
                "bits": awq_config.bits,
                "group_size": awq_config.group_size,
                "zero_point": awq_config.zero_point,
                "scale_search": awq_config.scale_search,
                "n_grid": awq_config.n_grid,
                "max_seq_len": awq_config.max_seq_len,
                "duo_scaling": awq_config.duo_scaling,
            }
        else:
            cfg = config or {}

        super().__init__(cfg)

        self.bits = cfg.get("bits", 4)
        self.group_size = cfg.get("group_size", 128)
        self.zero_point = cfg.get("zero_point", True)
        self.scale_search = cfg.get("scale_search", True)
        self.n_grid = cfg.get("n_grid", 20)
        self.max_seq_len = cfg.get("max_seq_len", 512)
        self.duo_scaling = cfg.get("duo_scaling", True)

        # Per-layer activation statistics
        self._activation_stats: Dict[str, torch.Tensor] = {}

    @property
    def max_int(self) -> int:
        """Maximum integer representable with the configured bit width."""
        return 2 ** self.bits - 1

    def _compute_quant_range(self) -> Tuple[float, float]:
        """Compute quantization range based on bit width and zero_point.

        Returns:
            Tuple of (qmin, qmax).
        """
        if self.zero_point:
            qmin = 0
            qmax = self.max_int
        else:
            qmin = -(2 ** (self.bits - 1))
            qmax = 2 ** (self.bits - 1) - 1
        return float(qmin), float(qmax)

    def collect_activation_stats(
        self,
        layer_name: str,
        activations: torch.Tensor,
    ) -> None:
        """Collect per-channel activation magnitude statistics.

        Computes the mean absolute activation value per input channel,
        accumulated across batches via running mean.

        Args:
            layer_name: Identifier for the layer.
            activations: Activation tensor of shape (..., in_features).
        """
        if activations.dim() == 3:
            activations = activations.reshape(-1, activations.shape[-1])

        # Per-channel mean absolute activation
        channel_magnitudes = activations.abs().mean(dim=0).float()

        if layer_name in self._activation_stats:
            self._activation_stats[layer_name] = (
                self._activation_stats[layer_name] + channel_magnitudes
            ) / 2.0
        else:
            self._activation_stats[layer_name] = channel_magnitudes

    def _pseudo_quantize(
        self,
        weight: torch.Tensor,
        group_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate quantization: quantize then immediately dequantize.

        Args:
            weight: Weight tensor of shape (out_features, in_features).
            group_size: Columns per quantization group.

        Returns:
            Tuple of (dequantized_weight, scales, zeros).
        """
        qmin, qmax = self._compute_quant_range()
        out_features, in_features = weight.shape

        if group_size <= 0:
            group_size = in_features

        n_groups = math.ceil(in_features / group_size)
        scales = torch.zeros(out_features, n_groups, device=weight.device)
        zeros = torch.zeros(out_features, n_groups, device=weight.device)
        dequantized = torch.zeros_like(weight)

        for g in range(n_groups):
            g_start = g * group_size
            g_end = min(g_start + group_size, in_features)
            w_group = weight[:, g_start:g_end]

            w_min = w_group.min(dim=1, keepdim=True).values
            w_max = w_group.max(dim=1, keepdim=True).values

            if self.zero_point:
                scale = (w_max - w_min) / (qmax - qmin)
                scale = scale.clamp(min=1e-10)
                zero = torch.round(-w_min / scale).clamp(qmin, qmax)
            else:
                w_absmax = torch.max(w_min.abs(), w_max.abs())
                scale = w_absmax / ((qmax - qmin) / 2)
                scale = scale.clamp(min=1e-10)
                zero = torch.zeros_like(scale)

            quantized = torch.clamp(
                torch.round(w_group / scale) + zero, qmin, qmax
            )
            dequantized[:, g_start:g_end] = (quantized - zero) * scale

            scales[:, g] = scale.squeeze(1)
            zeros[:, g] = zero.squeeze(1)

        return dequantized, scales, zeros

    def _compute_scale_factors(
        self,
        weight: torch.Tensor,
        activation_magnitudes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-channel scaling factors from activation magnitudes.

        The scaling factor for each input channel is proportional to the
        activation magnitude, which protects salient channels.

        Args:
            weight: Weight tensor of shape (out_features, in_features).
            activation_magnitudes: Per-channel activation magnitudes of
                shape (in_features,).

        Returns:
            Scaling factors of shape (in_features,).
        """
        # Normalize activation magnitudes to [0, 1]
        act_mag = activation_magnitudes.float()
        act_mag = act_mag / act_mag.max().clamp(min=1e-10)

        # Base scale: activation magnitude raised to a power
        # Higher power gives more protection to salient channels
        scale = act_mag.clamp(min=1e-5)

        return scale

    def _search_optimal_scale(
        self,
        weight: torch.Tensor,
        activation_magnitudes: torch.Tensor,
        calibration_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Grid search for optimal per-channel scaling factors.

        For each candidate alpha in [0, 1], compute:
            s = activation_mag ^ alpha
        Quantize the scaled weight and measure the reconstruction error.
        Return the scaling factors that minimize error.

        Args:
            weight: Weight tensor of shape (out_features, in_features).
            activation_magnitudes: Per-channel activation magnitudes.
            calibration_input: Optional calibration input for error measurement.

        Returns:
            Optimal scaling factors of shape (in_features,).
        """
        act_mag = activation_magnitudes.float()
        act_mag = act_mag / act_mag.max().clamp(min=1e-10)

        best_error = float('inf')
        best_scale = torch.ones_like(act_mag)

        group_size = self.group_size if self.group_size > 0 else weight.shape[1]

        for i in range(self.n_grid):
            alpha = i / max(self.n_grid - 1, 1)
            # Candidate scaling
            scale = act_mag.pow(alpha).clamp(min=1e-5)

            # Scale weight columns
            scaled_weight = weight * scale.unsqueeze(0)

            # Pseudo-quantize
            dq_weight, _, _ = self._pseudo_quantize(scaled_weight, group_size)

            # De-scale back
            dq_weight = dq_weight / scale.unsqueeze(0)

            # Compute error
            if calibration_input is not None:
                if calibration_input.dim() == 3:
                    cal_flat = calibration_input.reshape(-1, calibration_input.shape[-1])
                else:
                    cal_flat = calibration_input

                original_out = F.linear(cal_flat.float(), weight.float())
                quantized_out = F.linear(cal_flat.float(), dq_weight.float())
                error = (original_out - quantized_out).pow(2).mean().item()
            else:
                error = (weight.float() - dq_weight.float()).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_scale = scale.clone()

        return best_scale

    def quantize_layer(
        self,
        weight: torch.Tensor,
        layer_name: str,
        calibration_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Quantize a weight tensor using AWQ.

        Steps:
            1. Retrieve activation statistics for the layer.
            2. Compute (or search for) optimal per-channel scaling factors.
            3. Scale the weight columns.
            4. Quantize the scaled weight.
            5. De-scale to produce the final quantized weight.

        Args:
            weight: Weight tensor of shape (out_features, in_features).
            layer_name: Layer name to look up activation statistics.
            calibration_input: Optional calibration input tensor.

        Returns:
            Dictionary containing:
                - "quantized_weight": Dequantized weight after AWQ.
                - "scales": Per-group quantization scales.
                - "zeros": Per-group zero points.
                - "channel_scales": Per-channel AWQ scaling factors.
                - "mse": Mean squared error of the quantized weight.
        """
        weight = weight.float()
        out_features, in_features = weight.shape
        group_size = self.group_size if self.group_size > 0 else in_features

        # Get activation statistics
        if layer_name in self._activation_stats:
            act_mag = self._activation_stats[layer_name].to(weight.device)
        else:
            # Fallback: uniform activation magnitudes
            act_mag = torch.ones(in_features, device=weight.device)

        # Compute scaling factors
        if self.scale_search:
            channel_scales = self._search_optimal_scale(
                weight, act_mag, calibration_input
            )
        else:
            channel_scales = self._compute_scale_factors(weight, act_mag)

        # Apply scaling to weight columns
        scaled_weight = weight * channel_scales.unsqueeze(0)

        # Quantize
        dq_weight, scales, zeros = self._pseudo_quantize(scaled_weight, group_size)

        # De-scale
        dq_weight = dq_weight / channel_scales.unsqueeze(0)

        # Compute reconstruction error
        mse = (weight - dq_weight).pow(2).mean()

        return {
            "quantized_weight": dq_weight,
            "scales": scales,
            "zeros": zeros,
            "channel_scales": channel_scales,
            "mse": mse,
        }

    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: List[torch.Tensor],
        target_layers: Optional[List[str]] = None,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Quantize all (or targeted) linear layers in a model using AWQ.

        For each target layer:
            1. Collect activation statistics from calibration data.
            2. Quantize using activation-aware scaling.
            3. Replace the weight with the quantized version.

        Args:
            model: The model to quantize.
            calibration_data: List of input tensors for calibration.
            target_layers: Optional list of layer name patterns. If None,
                all nn.Linear layers are quantized.

        Returns:
            Tuple of (quantized model, metrics dict with per-layer MSE).
        """
        metrics: Dict[str, Any] = {}
        model_mode = model.training
        model.eval()

        # Step 1: Collect activation statistics via hooks
        hooks = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if target_layers is not None:
                if not any(t in name for t in target_layers):
                    continue

            def _make_hook(layer_name):
                def _hook(mod, inp, out):
                    if isinstance(inp, tuple):
                        self.collect_activation_stats(layer_name, inp[0].detach())
                    else:
                        self.collect_activation_stats(layer_name, inp.detach())
                return _hook

            h = module.register_forward_hook(_make_hook(name))
            hooks.append(h)

        # Run calibration data
        with torch.no_grad():
            for data in calibration_data:
                try:
                    if isinstance(data, dict):
                        model(**data)
                    else:
                        model(data)
                except Exception:
                    pass

        # Remove hooks
        for h in hooks:
            h.remove()

        # Step 2: Quantize each layer
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if target_layers is not None:
                if not any(t in name for t in target_layers):
                    continue

            result = self.quantize_layer(
                weight=module.weight.data,
                layer_name=name,
            )

            metrics[name] = {
                "mse": result["mse"].item(),
                "bits": self.bits,
                "group_size": self.group_size,
            }

            # Replace weight
            module.weight.data.copy_(
                result["quantized_weight"].to(module.weight.dtype)
            )

        model.train(model_mode)
        return model, metrics

    def reset_stats(self) -> None:
        """Clear all collected activation statistics."""
        self._activation_stats.clear()

    def forward(self, weight: torch.Tensor, layer_name: str = "default") -> torch.Tensor:
        """Quantize a weight tensor using AWQ (convenience forward).

        Args:
            weight: Weight tensor to quantize.
            layer_name: Layer name for activation statistics lookup.

        Returns:
            Dequantized weight tensor.
        """
        result = self.quantize_layer(weight, layer_name)
        return result["quantized_weight"]
