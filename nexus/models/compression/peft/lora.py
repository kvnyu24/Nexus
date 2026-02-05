"""Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.

Reference:
    Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models."
    ICLR 2022. https://arxiv.org/abs/2106.09685

LoRA freezes the pretrained model weights and injects trainable low-rank
decomposition matrices (B @ A) into each target layer. This reduces the
number of trainable parameters by several orders of magnitude while
maintaining or improving model quality on downstream tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set, Tuple, Union
import math
import re

from nexus.core.base import NexusModule


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation.

    Attributes:
        rank: Rank of the low-rank decomposition. Lower values use fewer
            parameters but may reduce expressiveness.
        alpha: Scaling factor for the LoRA update. The actual scaling applied
            is alpha / rank.
        dropout: Dropout probability applied to the input of the LoRA path.
        target_modules: List of module name patterns (regex supported) to
            apply LoRA to. Common choices include projection layers in
            attention blocks.
        fan_in_fan_out: Set True if the original layer stores weight as
            (fan_in, fan_out), e.g., GPT-2 Conv1D layers.
        bias: Bias handling strategy. "none" trains no bias, "all" trains
            all biases, "lora_only" trains only biases in LoRA layers.
        modules_to_save: Additional modules to mark as trainable without
            LoRA adaptation (e.g., classification heads).
    """
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'v_proj'])
    fan_in_fan_out: bool = False
    bias: str = "none"
    modules_to_save: List[str] = field(default_factory=list)


class LoRALinear(NexusModule):
    """Linear layer augmented with a LoRA low-rank adapter.

    The forward pass computes:
        output = W_frozen @ x + (scaling) * B @ A @ x

    where W_frozen is the original (frozen) weight matrix, A is the
    down-projection (in_features -> rank), and B is the up-projection
    (rank -> out_features). Only A and B are trainable.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        rank: Rank of the low-rank matrices A and B.
        alpha: Scaling factor. The effective scaling is alpha / rank.
        dropout: Dropout probability on the LoRA input path.
        bias: If True, the frozen linear layer includes a bias term.
        fan_in_fan_out: If True, transposes the weight matrix for
            compatibility with Conv1D-style layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
        fan_in_fan_out: bool = False,
    ):
        config = {
            "in_features": in_features,
            "out_features": out_features,
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "bias": bias,
            "fan_in_fan_out": fan_in_fan_out,
        }
        super().__init__(config)

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.fan_in_fan_out = fan_in_fan_out
        self.merged = False

        # Frozen pretrained weight
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout on the LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self._init_lora_weights()

    def _init_lora_weights(self) -> None:
        """Initialize A with Kaiming uniform and B with zeros.

        This ensures the LoRA contribution is zero at initialization,
        preserving the pretrained model's behavior before any training.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining frozen linear output with LoRA adapter.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        base_output = self.linear(x)

        if self.merged:
            return base_output

        # LoRA path: x -> dropout -> A -> B -> scale
        lora_input = self.lora_dropout(x)
        lora_output = F.linear(F.linear(lora_input, self.lora_A), self.lora_B)

        return base_output + lora_output * self.scaling

    def merge_weights(self) -> None:
        """Merge LoRA weights into the frozen linear layer for inference.

        After merging, the forward pass uses only the linear layer with
        no additional computation. Call unmerge_weights() to reverse.
        """
        if self.merged:
            return

        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            if self.fan_in_fan_out:
                delta_w = delta_w.T
            self.linear.weight.data += delta_w

        self.merged = True

    def unmerge_weights(self) -> None:
        """Reverse a previous merge, restoring the LoRA adapter path.

        This subtracts the LoRA contribution from the linear weight so
        that training can resume with separate adapter parameters.
        """
        if not self.merged:
            return

        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            if self.fan_in_fan_out:
                delta_w = delta_w.T
            self.linear.weight.data -= delta_w

        self.merged = False

    def get_parameter_count(self) -> Dict[str, int]:
        """Return counts of total, trainable, and frozen parameters."""
        total = self.linear.weight.numel()
        if self.linear.bias is not None:
            total += self.linear.bias.numel()
        trainable = self.lora_A.numel() + self.lora_B.numel()
        return {
            "total": total + trainable,
            "trainable": trainable,
            "frozen": total,
        }

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        """Create a LoRALinear by wrapping an existing nn.Linear layer.

        The original weight and bias are copied into the frozen linear
        component, preserving the pretrained values.

        Args:
            linear: The nn.Linear layer to wrap.
            rank: LoRA rank.
            alpha: LoRA scaling factor.
            dropout: LoRA dropout.

        Returns:
            A new LoRALinear instance with the original weights frozen.
        """
        has_bias = linear.bias is not None
        lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=has_bias,
        )
        lora_linear.linear.weight.data.copy_(linear.weight.data)
        if has_bias:
            lora_linear.linear.bias.data.copy_(linear.bias.data)
        return lora_linear


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    config: Optional[LoRAConfig] = None,
) -> nn.Module:
    """Inject LoRA adapters into target linear layers of an existing model.

    This function traverses the model, identifies linear layers whose names
    match any pattern in target_modules, and replaces them with LoRALinear
    layers. All original weights are frozen; only LoRA parameters are
    trainable.

    Args:
        model: The model to augment with LoRA.
        rank: LoRA rank (ignored if config is provided).
        alpha: LoRA alpha scaling (ignored if config is provided).
        dropout: LoRA dropout (ignored if config is provided).
        target_modules: Module name patterns to target (ignored if config
            is provided). Defaults to ['q_proj', 'v_proj'].
        config: Optional LoRAConfig; overrides individual arguments.

    Returns:
        The model with LoRA adapters injected (modified in-place).
    """
    if config is not None:
        rank = config.rank
        alpha = config.alpha
        dropout = config.dropout
        target_modules = config.target_modules
        bias_mode = config.bias
        modules_to_save = config.modules_to_save
    else:
        target_modules = target_modules or ['q_proj', 'v_proj']
        bias_mode = "none"
        modules_to_save = []

    # Compile target module patterns
    patterns = [re.compile(p) for p in target_modules]

    def _matches(name: str) -> bool:
        return any(pattern.search(name) for pattern in patterns)

    # Freeze all model parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Replace matching linear layers with LoRALinear
    replaced: Set[str] = set()
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _matches(name):
            lora_layer = LoRALinear.from_linear(
                module, rank=rank, alpha=alpha, dropout=dropout
            )
            # Navigate to parent and replace the child
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], lora_layer)
            replaced.add(name)

    # Handle bias training mode
    if bias_mode == "all":
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
    elif bias_mode == "lora_only":
        for name, param in model.named_parameters():
            if "bias" in name and any(r in name for r in replaced):
                param.requires_grad = True

    # Mark additional modules as trainable
    for mod_name in modules_to_save:
        for name, module in model.named_modules():
            if mod_name in name:
                for param in module.parameters():
                    param.requires_grad = True

    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge all LoRA adapters in a model into their base weights.

    After merging, the model behaves as a standard model with no LoRA
    overhead. This is useful for deployment or inference optimization.

    Args:
        model: The model containing LoRALinear layers.

    Returns:
        The model with merged weights (modified in-place).
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
    return model


def unmerge_lora(model: nn.Module) -> nn.Module:
    """Unmerge all LoRA adapters in a model, restoring adapter paths.

    This reverses a previous merge_lora call so that training can resume.

    Args:
        model: The model containing merged LoRALinear layers.

    Returns:
        The model with unmerged weights (modified in-place).
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge_weights()
    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Collect all trainable LoRA parameters from a model.

    Useful for creating optimizer parameter groups with LoRA-specific
    learning rates or weight decay settings.

    Args:
        model: The model containing LoRALinear layers.

    Returns:
        List of trainable LoRA parameters (lora_A, lora_B).
    """
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only the LoRA adapter state dict from a model.

    This produces a minimal checkpoint containing only the trainable
    adapter weights, which is much smaller than a full model checkpoint.

    Args:
        model: The model containing LoRALinear layers.

    Returns:
        Dictionary mapping parameter names to their tensor values for
        LoRA parameters only.
    """
    state_dict = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            state_dict[name] = param.data.clone()
    return state_dict
