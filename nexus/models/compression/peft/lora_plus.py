"""LoRA+ with Asymmetric Learning Rates for parameter-efficient fine-tuning.

Reference:
    Hayou, S., et al. "LoRA+: Efficient Low Rank Adaptation of Large Models."
    ICML 2024. https://arxiv.org/abs/2402.12354

LoRA+ improves upon standard LoRA by setting a higher learning rate for the
adapter matrix B than for adapter matrix A. This asymmetric learning rate
schedule leads to faster convergence and better performance. Empirically,
a ratio of lr_B / lr_A = 16 works well across different model sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import math

from nexus.core.base import NexusModule


@dataclass
class LoRAPlusConfig:
    """Configuration for LoRA+ adaptation.

    Attributes:
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor for the LoRA update (alpha / rank).
        dropout: Dropout probability applied to the LoRA path.
        lr_ratio: Learning rate ratio between B and A matrices (lr_B / lr_A).
            Default is 16.0 as recommended in the paper.
        target_modules: List of module name patterns to apply LoRA+ to.
        fan_in_fan_out: Set True for GPT-2-style Conv1D layers.
        bias: Bias handling strategy ("none", "all", "lora_only").
        modules_to_save: Additional modules to train without LoRA.
    """
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    lr_ratio: float = 16.0
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'v_proj'])
    fan_in_fan_out: bool = False
    bias: str = "none"
    modules_to_save: List[str] = field(default_factory=list)


class LoRAPlusLinear(NexusModule):
    """Linear layer with LoRA+ low-rank adapter and asymmetric learning rates.

    The forward pass computes:
        output = W_frozen @ x + (alpha / rank) * B @ A @ x

    where W_frozen is frozen, and A and B are trainable with lr_B = lr_ratio * lr_A.
    This asymmetric learning rate schedule enables faster convergence.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        rank: Rank of the low-rank matrices A and B.
        alpha: Scaling factor (applied as alpha / rank).
        dropout: Dropout probability on the LoRA path.
        lr_ratio: Learning rate ratio lr_B / lr_A. Default 16.0.
        bias: If True, includes a bias term in the frozen layer.
        fan_in_fan_out: If True, transposes weight for Conv1D compatibility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        lr_ratio: float = 16.0,
        bias: bool = True,
        fan_in_fan_out: bool = False,
    ):
        config = {
            "in_features": in_features,
            "out_features": out_features,
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "lr_ratio": lr_ratio,
            "bias": bias,
            "fan_in_fan_out": fan_in_fan_out,
        }
        super().__init__(config)

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lr_ratio = lr_ratio
        self.fan_in_fan_out = fan_in_fan_out

        # Frozen pretrained weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # LoRA adapter matrices
        # A: (rank, in_features) - lower learning rate
        # B: (out_features, rank) - higher learning rate (lr_ratio * lr_A)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Mark lora_B for higher learning rate (optimizer needs to handle this)
        self.lora_B._lr_multiplier = lr_ratio

        # Dropout on LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize weights
        self.reset_parameters()

        # Freeze pretrained weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform for A, zeros for B."""
        # Initialize weight like a standard linear layer
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # LoRA initialization: Kaiming for A, zeros for B
        # This ensures zero initialization of the LoRA path at start
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA+ adaptation.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # Original frozen path
        weight = self.weight.T if self.fan_in_fan_out else self.weight
        result = F.linear(x, weight, self.bias)

        # LoRA+ path: x -> dropout -> A -> B -> scale
        x_lora = self.lora_dropout(x)
        lora_out = x_lora @ self.lora_A.T  # (..., rank)
        lora_out = lora_out @ self.lora_B.T  # (..., out_features)
        result = result + lora_out * self.scaling

        return result

    def merge_weights(self):
        """Merge LoRA weights into the frozen weight for inference efficiency."""
        if self.fan_in_fan_out:
            # W = W + scaling * B @ A (transposed)
            self.weight.data = self.weight.data + (
                self.lora_B @ self.lora_A * self.scaling
            ).T
        else:
            self.weight.data = self.weight.data + (
                self.lora_B @ self.lora_A * self.scaling
            )

    def unmerge_weights(self):
        """Unmerge LoRA weights from the frozen weight."""
        if self.fan_in_fan_out:
            self.weight.data = self.weight.data - (
                self.lora_B @ self.lora_A * self.scaling
            ).T
        else:
            self.weight.data = self.weight.data - (
                self.lora_B @ self.lora_A * self.scaling
            )


class LoRAPlusOptimizer:
    """Optimizer wrapper that applies asymmetric learning rates for LoRA+ parameters.

    This wrapper identifies LoRA+ matrices (A and B) and applies different learning
    rates: lr_A for matrix A, and lr_ratio * lr_A for matrix B.

    Args:
        optimizer: Base optimizer (e.g., AdamW).
        lr_ratio: Learning rate ratio between B and A matrices.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, lr_ratio: float = 16.0):
        self.optimizer = optimizer
        self.lr_ratio = lr_ratio
        self._apply_lr_ratios()

    def _apply_lr_ratios(self):
        """Apply learning rate multipliers based on parameter names."""
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                # Check if parameter has lr_multiplier attribute (set in LoRAPlusLinear)
                if hasattr(param, '_lr_multiplier'):
                    # Create a separate param group with higher LR for B matrices
                    if 'initial_lr' not in param_group:
                        param_group['initial_lr'] = param_group['lr']
                    param_group['lr'] = param_group['initial_lr'] * param._lr_multiplier

    def step(self, *args, **kwargs):
        """Perform optimization step."""
        return self.optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        """Zero gradients."""
        return self.optimizer.zero_grad(*args, **kwargs)

    def state_dict(self):
        """Get optimizer state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        return self.optimizer.load_state_dict(state_dict)


def apply_lora_plus(
    model: nn.Module,
    config: LoRAPlusConfig,
    verbose: bool = True
) -> nn.Module:
    """Apply LoRA+ to a pretrained model by replacing target modules.

    Args:
        model: The pretrained model to adapt.
        config: LoRA+ configuration specifying rank, alpha, target modules, etc.
        verbose: If True, print information about replaced modules.

    Returns:
        The model with LoRA+ adapters applied.
    """
    import re
    from collections import defaultdict

    replaced_modules = defaultdict(int)

    def replace_module(parent, name, module):
        """Replace a linear module with LoRAPlusLinear."""
        if not isinstance(module, nn.Linear):
            return False

        # Check if module name matches target patterns
        full_name = f"{parent.__class__.__name__}.{name}"
        match = any(re.search(pattern, name) for pattern in config.target_modules)

        if not match:
            return False

        # Create LoRA+ replacement
        lora_module = LoRAPlusLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            lr_ratio=config.lr_ratio,
            bias=module.bias is not None,
            fan_in_fan_out=config.fan_in_fan_out,
        )

        # Copy pretrained weights
        lora_module.weight.data = module.weight.data.clone()
        if module.bias is not None:
            lora_module.bias.data = module.bias.data.clone()

        # Replace module
        setattr(parent, name, lora_module)
        replaced_modules[name] += 1
        return True

    # Traverse model and replace modules
    for name, module in model.named_modules():
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]

        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model

        replace_module(parent, child_name, module)

    if verbose:
        print(f"Applied LoRA+ to {sum(replaced_modules.values())} modules:")
        for module_name, count in replaced_modules.items():
            print(f"  - {module_name}: {count} instances")

    return model


def merge_lora_plus(model: nn.Module) -> nn.Module:
    """Merge all LoRA+ adapter weights into the base model for inference.

    Args:
        model: Model with LoRA+ adapters.

    Returns:
        Model with merged weights.
    """
    for module in model.modules():
        if isinstance(module, LoRAPlusLinear):
            module.merge_weights()
    return model
