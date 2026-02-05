"""rsLoRA: Rank-Stabilized LoRA for improved scaling and stability.

Reference:
    Kalajdzievski, D. "A Rank Stabilization Scaling Factor for Fine-Tuning
    with LoRA."
    2023. https://arxiv.org/abs/2312.03732

rsLoRA improves upon standard LoRA by modifying the scaling factor to be
rank-dependent in a stabilizing way. Instead of scaling by alpha/r, rsLoRA
scales by alpha/sqrt(r). This stabilization allows for better performance
across different rank choices and more stable training dynamics, especially
at higher ranks.

Key insight: The standard alpha/r scaling causes instability at high ranks.
Using alpha/sqrt(r) provides more consistent performance regardless of rank.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import math

from nexus.core.base import NexusModule


@dataclass
class rsLoRAConfig:
    """Configuration for rsLoRA adaptation.

    Attributes:
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor for the LoRA update. With rsLoRA, the effective
            scaling is alpha / sqrt(rank) instead of alpha / rank.
        dropout: Dropout probability applied to the LoRA path.
        target_modules: List of module name patterns to apply rsLoRA to.
        fan_in_fan_out: Set True for GPT-2-style Conv1D layers.
        bias: Bias handling strategy ("none", "all", "lora_only").
        modules_to_save: Additional modules to train without LoRA.
        use_rslora_scaling: If True, use sqrt(rank) scaling; if False, revert
            to standard LoRA scaling (rank). Default True.
    """
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'v_proj'])
    fan_in_fan_out: bool = False
    bias: str = "none"
    modules_to_save: List[str] = field(default_factory=list)
    use_rslora_scaling: bool = True


class rsLoRALinear(NexusModule):
    """Linear layer with rank-stabilized LoRA adapter.

    The forward pass computes:
        output = W_frozen @ x + (alpha / sqrt(rank)) * B @ A @ x

    where W_frozen is the frozen pretrained weight, and A and B are trainable
    low-rank matrices. The sqrt(rank) scaling provides better stability across
    different rank choices compared to standard LoRA's rank scaling.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        rank: Rank of the low-rank matrices A and B.
        alpha: Scaling factor (applied as alpha / sqrt(rank)).
        dropout: Dropout probability on the LoRA path.
        bias: If True, includes a bias term in the frozen layer.
        fan_in_fan_out: If True, transposes weight for Conv1D compatibility.
        use_rslora_scaling: If True, use sqrt(rank); if False, use rank.
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
        use_rslora_scaling: bool = True,
    ):
        config = {
            "in_features": in_features,
            "out_features": out_features,
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "bias": bias,
            "fan_in_fan_out": fan_in_fan_out,
            "use_rslora_scaling": use_rslora_scaling,
        }
        super().__init__(config)

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.use_rslora_scaling = use_rslora_scaling

        # Compute scaling factor
        if use_rslora_scaling:
            self.scaling = alpha / math.sqrt(rank)
        else:
            self.scaling = alpha / rank

        self.fan_in_fan_out = fan_in_fan_out

        # Frozen pretrained weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # LoRA adapter matrices
        # A: (rank, in_features) - down-projection
        # B: (out_features, rank) - up-projection
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

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
        """Forward pass with rsLoRA adaptation.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # Original frozen path
        weight = self.weight.T if self.fan_in_fan_out else self.weight
        result = F.linear(x, weight, self.bias)

        # rsLoRA path: x -> dropout -> A -> B -> scale
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

    def get_effective_rank(self) -> float:
        """Compute the effective rank of the LoRA adapter.

        Returns:
            Effective rank computed via singular value distribution.
        """
        with torch.no_grad():
            # Compute the combined weight matrix
            W_lora = self.lora_B @ self.lora_A

            # Compute singular values
            _, S, _ = torch.svd(W_lora)

            # Compute effective rank using entropy
            # Normalize singular values
            S_normalized = S / S.sum()

            # Compute Shannon entropy
            entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()

            # Effective rank is exp(entropy)
            effective_rank = torch.exp(entropy)

            return effective_rank.item()


def apply_rslora(
    model: nn.Module,
    config: rsLoRAConfig,
    verbose: bool = True
) -> nn.Module:
    """Apply rsLoRA to a pretrained model by replacing target modules.

    Args:
        model: The pretrained model to adapt.
        config: rsLoRA configuration specifying rank, alpha, target modules, etc.
        verbose: If True, print information about replaced modules.

    Returns:
        The model with rsLoRA adapters applied.
    """
    import re
    from collections import defaultdict

    replaced_modules = defaultdict(int)

    def replace_module(parent, name, module):
        """Replace a linear module with rsLoRALinear."""
        if not isinstance(module, nn.Linear):
            return False

        # Check if module name matches target patterns
        full_name = f"{parent.__class__.__name__}.{name}"
        match = any(re.search(pattern, name) for pattern in config.target_modules)

        if not match:
            return False

        # Create rsLoRA replacement
        lora_module = rsLoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            bias=module.bias is not None,
            fan_in_fan_out=config.fan_in_fan_out,
            use_rslora_scaling=config.use_rslora_scaling,
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
        print(f"Applied rsLoRA to {sum(replaced_modules.values())} modules:")
        for module_name, count in replaced_modules.items():
            print(f"  - {module_name}: {count} instances")

        # Print scaling info
        scaling_type = "sqrt(rank)" if config.use_rslora_scaling else "rank"
        print(f"Using {scaling_type} scaling with alpha={config.alpha}, rank={config.rank}")
        example_scaling = config.alpha / math.sqrt(config.rank) if config.use_rslora_scaling else config.alpha / config.rank
        print(f"Effective scaling factor: {example_scaling:.4f}")

    return model


def merge_rslora(model: nn.Module) -> nn.Module:
    """Merge all rsLoRA adapter weights into the base model for inference.

    Args:
        model: Model with rsLoRA adapters.

    Returns:
        Model with merged weights.
    """
    for module in model.modules():
        if isinstance(module, rsLoRALinear):
            module.merge_weights()
    return model


def analyze_rslora_ranks(model: nn.Module) -> Dict[str, float]:
    """Analyze effective ranks of all rsLoRA adapters in a model.

    Args:
        model: Model with rsLoRA adapters.

    Returns:
        Dictionary mapping module name to effective rank.
    """
    rank_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, rsLoRALinear):
            rank_stats[name] = module.get_effective_rank()
    return rank_stats
