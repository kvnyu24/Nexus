"""DoRA: Weight-Decomposed Low-Rank Adaptation.

Reference:
    Liu, S., et al. "DoRA: Weight-Decomposed Low-Rank Adaptation."
    ICML 2024. https://arxiv.org/abs/2402.09353

DoRA decomposes the pretrained weight W into a magnitude component m
and a directional component V/||V||, then applies LoRA only to the
directional component. This decomposition more closely mirrors the
learning pattern of full fine-tuning, yielding better performance
than standard LoRA with the same number of trainable parameters.

The forward computation is:
    W' = m * (V + delta_V) / ||V + delta_V||
where delta_V = B @ A is the LoRA update to the directional component,
and m is a trainable magnitude vector (one scalar per output neuron).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import math
import re

from nexus.core.base import NexusModule


@dataclass
class DoRAConfig:
    """Configuration for DoRA adaptation.

    Attributes:
        rank: Rank of the low-rank decomposition for the directional
            component update.
        alpha: Scaling factor for the LoRA update. Effective scaling
            is alpha / rank.
        dropout: Dropout probability on the adapter input path.
        target_modules: Module name patterns (regex supported) to apply
            DoRA to.
        magnitude_trainable: Whether the magnitude vector m is trainable.
            Defaults to True following the original paper.
    """
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'v_proj'])
    magnitude_trainable: bool = True


class DoRALinear(NexusModule):
    """Linear layer with DoRA weight-decomposed low-rank adaptation.

    DoRA decomposes the weight matrix W (out_features x in_features) as:
        W = m * V_hat, where V_hat = V / ||V||_column

    Here m is a per-output-neuron magnitude vector and V_hat is the
    column-normalized directional matrix. A LoRA update (B @ A) is added
    to V before re-normalization, and m is optionally made trainable.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: Rank of the LoRA adapter on the directional component.
        alpha: LoRA scaling factor.
        dropout: Dropout probability on the adapter path.
        bias: Whether to include a bias term.
        magnitude_trainable: Whether to train the magnitude vector.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
        magnitude_trainable: bool = True,
    ):
        config = {
            "in_features": in_features,
            "out_features": out_features,
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "bias": bias,
            "magnitude_trainable": magnitude_trainable,
        }
        super().__init__(config)

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.magnitude_trainable = magnitude_trainable

        # Frozen pretrained weight
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False

        # Magnitude vector m (one per output neuron)
        # Initialized from ||W_row|| for each output dimension
        self.magnitude = nn.Parameter(
            torch.ones(out_features), requires_grad=magnitude_trainable
        )

        # LoRA matrices for directional update
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize LoRA weights and magnitude from the pretrained weight.

        The magnitude is set to the column-wise norm of the pretrained
        weight. LoRA matrices are initialized so the initial delta is zero.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def _initialize_magnitude_from_weight(self) -> None:
        """Compute the initial magnitude from the current frozen weight.

        Call this after loading pretrained weights into self.linear to
        ensure the magnitude vector m correctly reflects ||V_row||.
        """
        with torch.no_grad():
            weight = self.linear.weight.data
            # Column-norm: norm along the in_features dimension for each output
            col_norms = torch.norm(weight, dim=1)
            self.magnitude.data.copy_(col_norms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with DoRA weight decomposition.

        Computes:
            V_updated = V + scaling * B @ A
            output = m * (V_updated / ||V_updated||_col) @ x + bias

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        weight = self.linear.weight  # (out_features, in_features)

        # Compute LoRA directional update
        lora_input = self.lora_dropout(x)
        delta_v = self.lora_B @ self.lora_A  # (out_features, in_features)

        # Updated directional component
        v_updated = weight + delta_v * self.scaling

        # Column-normalize: normalize each row (output neuron direction)
        v_norm = torch.norm(v_updated, dim=1, keepdim=True).clamp(min=1e-8)
        v_hat = v_updated / v_norm

        # Apply magnitude scaling
        # m is (out_features,), reshape to (out_features, 1) for broadcasting
        weight_final = self.magnitude.unsqueeze(1) * v_hat

        # Compute output
        output = F.linear(x, weight_final, self.linear.bias)

        return output

    def get_parameter_count(self) -> Dict[str, int]:
        """Return counts of total, trainable, and frozen parameters."""
        frozen = self.linear.weight.numel()
        if self.linear.bias is not None:
            frozen += self.linear.bias.numel()

        trainable = self.lora_A.numel() + self.lora_B.numel()
        if self.magnitude_trainable:
            trainable += self.magnitude.numel()
        else:
            frozen += self.magnitude.numel()

        return {
            "total": frozen + trainable,
            "trainable": trainable,
            "frozen": frozen,
        }

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        magnitude_trainable: bool = True,
    ) -> "DoRALinear":
        """Create a DoRALinear by wrapping an existing nn.Linear layer.

        The pretrained weight is copied and frozen. The magnitude vector
        is initialized from the row norms of the pretrained weight.

        Args:
            linear: The nn.Linear layer to wrap.
            rank: LoRA rank for directional update.
            alpha: LoRA scaling factor.
            dropout: LoRA dropout probability.
            magnitude_trainable: Whether to train the magnitude vector.

        Returns:
            A new DoRALinear instance.
        """
        has_bias = linear.bias is not None
        dora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=has_bias,
            magnitude_trainable=magnitude_trainable,
        )
        dora.linear.weight.data.copy_(linear.weight.data)
        if has_bias:
            dora.linear.bias.data.copy_(linear.bias.data)

        # Initialize magnitude from the pretrained weight norms
        dora._initialize_magnitude_from_weight()

        return dora


def apply_dora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    config: Optional[DoRAConfig] = None,
) -> nn.Module:
    """Inject DoRA adapters into target linear layers of a model.

    This traverses the model, replaces matching nn.Linear layers with
    DoRALinear layers, and freezes all non-DoRA parameters.

    Args:
        model: The model to augment.
        rank: DoRA rank.
        alpha: DoRA scaling factor.
        dropout: Dropout probability.
        target_modules: Module name patterns to target.
        config: Optional DoRAConfig overriding individual args.

    Returns:
        The model with DoRA adapters injected (modified in-place).
    """
    if config is not None:
        rank = config.rank
        alpha = config.alpha
        dropout = config.dropout
        target_modules = config.target_modules
        magnitude_trainable = config.magnitude_trainable
    else:
        target_modules = target_modules or ['q_proj', 'v_proj']
        magnitude_trainable = True

    patterns = [re.compile(p) for p in target_modules]

    def _matches(name: str) -> bool:
        return any(pattern.search(name) for pattern in patterns)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace matching layers
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _matches(name):
            dora_layer = DoRALinear.from_linear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                magnitude_trainable=magnitude_trainable,
            )
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], dora_layer)

    return model
