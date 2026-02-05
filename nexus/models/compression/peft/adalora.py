"""AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.

Reference:
    Zhang, Q., et al. "AdaLoRA: Adaptive Budget Allocation for
    Parameter-Efficient Fine-Tuning." ICLR 2023.
    https://arxiv.org/abs/2303.10512

AdaLoRA parameterizes the incremental weight update in SVD form:
    delta_W = P @ diag(Lambda) @ Q
where P (left singular vectors), Lambda (singular values), and
Q (right singular vectors) are all trainable. An importance score
is computed for each singular value based on a sensitivity measure,
and less important triplets (columns of P, entries of Lambda, rows
of Q) are pruned to reallocate the rank budget across layers.

This allows dynamic rank allocation: layers that need more capacity
retain higher rank, while less important layers are pruned to lower
rank, all under a global parameter budget.
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
class AdaLoRAConfig:
    """Configuration for AdaLoRA adaptation.

    Attributes:
        initial_rank: Initial rank for all SVD-parameterized layers.
            This is the starting point before any pruning.
        target_rank: Target average rank after pruning. The scheduler
            progressively prunes singular values to reach this target.
        beta1: Exponential moving average coefficient for the
            sensitivity-based importance score.
        beta2: Exponential moving average coefficient for the
            uncertainty-based importance score.
        alpha: Scaling factor for the SVD update. Effective scaling
            is alpha / initial_rank.
        dropout: Dropout probability on the adapter input path.
        target_modules: Module name patterns (regex) to apply AdaLoRA to.
        total_step: Total number of training steps for the rank scheduler.
        warmup_step: Number of warmup steps before pruning begins.
        final_warmup_step: Step at which pruning stops and ranks are fixed.
    """
    initial_rank: int = 12
    target_rank: int = 8
    beta1: float = 0.85
    beta2: float = 0.85
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'v_proj'])
    total_step: int = 10000
    warmup_step: int = 500
    final_warmup_step: int = 8000


class AdaLoRALinear(NexusModule):
    """Linear layer with SVD-parameterized adaptive-rank LoRA.

    The incremental update is parameterized as:
        delta_W = P @ diag(Lambda) @ Q

    where:
        P: (out_features, rank) - left singular vectors
        Lambda: (rank,) - singular values
        Q: (rank, in_features) - right singular vectors

    An importance score I_k for each triplet k is maintained via EMA:
        S_k = |Lambda_k| * (||P[:, k]|| + ||Q[k, :]||)
        I_k = beta1 * I_k + (1 - beta1) * S_k

    During training, a scheduler prunes the least important triplets
    to meet the target rank budget.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        initial_rank: Starting rank for the SVD decomposition.
        alpha: Scaling factor.
        dropout: Dropout probability.
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        initial_rank: int = 12,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        config = {
            "in_features": in_features,
            "out_features": out_features,
            "initial_rank": initial_rank,
            "alpha": alpha,
            "dropout": dropout,
            "bias": bias,
        }
        super().__init__(config)

        self.in_features = in_features
        self.out_features = out_features
        self.initial_rank = initial_rank
        self.current_rank = initial_rank
        self.alpha = alpha
        self.scaling = alpha / initial_rank

        # Frozen pretrained weight
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False

        # SVD-parameterized increment: P @ diag(Lambda) @ Q
        self.lora_P = nn.Parameter(torch.empty(out_features, initial_rank))
        self.lora_Lambda = nn.Parameter(torch.ones(initial_rank))
        self.lora_Q = nn.Parameter(torch.empty(initial_rank, in_features))

        # Dropout
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Importance scores (not a parameter; maintained externally)
        self.register_buffer(
            "importance_scores", torch.ones(initial_rank)
        )
        # Mask for active singular value triplets
        self.register_buffer(
            "rank_mask", torch.ones(initial_rank, dtype=torch.bool)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize SVD components.

        P is initialized with orthogonal initialization, Lambda starts
        at zeros (so initial delta_W = 0), and Q uses Kaiming uniform.
        """
        nn.init.orthogonal_(self.lora_P)
        nn.init.zeros_(self.lora_Lambda)
        nn.init.kaiming_uniform_(self.lora_Q, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SVD-parameterized low-rank update.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        base_output = self.linear(x)

        # Apply dropout to LoRA input
        lora_input = self.lora_dropout(x)

        # Masked singular values
        effective_lambda = self.lora_Lambda * self.rank_mask.float()

        # delta_W @ x = P @ diag(Lambda) @ Q @ x
        # Compute step by step for efficiency:
        # 1. Q @ x -> (rank, ...) via batched matmul
        qx = F.linear(lora_input, self.lora_Q)  # (..., rank)
        # 2. Scale by Lambda
        qx = qx * effective_lambda.unsqueeze(0).expand_as(qx)
        # 3. P @ (Lambda * Q @ x)
        lora_output = F.linear(qx, self.lora_P)  # (..., out_features)

        return base_output + lora_output * self.scaling

    def compute_importance(self) -> torch.Tensor:
        """Compute importance scores for each singular value triplet.

        The importance score for triplet k is:
            S_k = |Lambda_k| * (||P[:, k]||_2 + ||Q[k, :]||_2)

        This measures how much each triplet contributes to the overall
        weight update, combining the singular value magnitude with the
        norms of the associated left and right singular vectors.

        Returns:
            Importance scores of shape (initial_rank,).
        """
        with torch.no_grad():
            p_norms = torch.norm(self.lora_P, dim=0)  # (rank,)
            q_norms = torch.norm(self.lora_Q, dim=1)  # (rank,)
            lambda_abs = self.lora_Lambda.abs()
            scores = lambda_abs * (p_norms + q_norms)
        return scores

    def update_importance(self, beta1: float = 0.85) -> None:
        """Update the EMA importance scores.

        Args:
            beta1: Exponential moving average coefficient.
        """
        current_scores = self.compute_importance()
        self.importance_scores.mul_(beta1).add_(current_scores, alpha=1 - beta1)

    def prune_to_rank(self, target_rank: int) -> int:
        """Prune the least important singular value triplets.

        Sets the rank_mask to False for the least important triplets,
        effectively zeroing their contribution in the forward pass.

        Args:
            target_rank: Desired number of active triplets.

        Returns:
            The actual number of active triplets after pruning.
        """
        target_rank = min(target_rank, self.initial_rank)
        if target_rank <= 0:
            self.rank_mask.zero_()
            self.current_rank = 0
            return 0

        # Keep only the top-k most important triplets
        _, top_indices = torch.topk(
            self.importance_scores, k=target_rank, largest=True
        )
        new_mask = torch.zeros_like(self.rank_mask)
        new_mask[top_indices] = True
        self.rank_mask.copy_(new_mask)
        self.current_rank = int(new_mask.sum().item())

        return self.current_rank

    def get_parameter_count(self) -> Dict[str, int]:
        """Return parameter counts for the AdaLoRA layer."""
        frozen = self.linear.weight.numel()
        if self.linear.bias is not None:
            frozen += self.linear.bias.numel()

        # Only count active (unmasked) parameters as trainable
        active = int(self.rank_mask.sum().item())
        trainable = (
            self.out_features * active  # active columns of P
            + active                     # active Lambda entries
            + active * self.in_features  # active rows of Q
        )

        return {
            "total": frozen + self.lora_P.numel() + self.lora_Lambda.numel() + self.lora_Q.numel(),
            "trainable": trainable,
            "frozen": frozen,
            "active_rank": active,
            "initial_rank": self.initial_rank,
        }

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        initial_rank: int = 12,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> "AdaLoRALinear":
        """Create an AdaLoRALinear by wrapping an existing nn.Linear.

        Args:
            linear: The nn.Linear layer to wrap.
            initial_rank: Starting SVD rank.
            alpha: Scaling factor.
            dropout: Dropout probability.

        Returns:
            A new AdaLoRALinear with frozen pretrained weights.
        """
        has_bias = linear.bias is not None
        adalora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            initial_rank=initial_rank,
            alpha=alpha,
            dropout=dropout,
            bias=has_bias,
        )
        adalora.linear.weight.data.copy_(linear.weight.data)
        if has_bias:
            adalora.linear.bias.data.copy_(linear.bias.data)
        return adalora


class AdaLoRAScheduler:
    """Rank scheduling for AdaLoRA: progressively prunes ranks during training.

    The scheduler operates in three phases:
        1. Warmup (steps 0..warmup_step): All ranks are at initial_rank.
           Importance scores accumulate but no pruning occurs.
        2. Pruning (warmup_step..final_warmup_step): Ranks are linearly
           reduced from initial_rank to target_rank based on importance.
        3. Fixed (final_warmup_step..end): Ranks are fixed at target_rank.

    Usage:
        scheduler = AdaLoRAScheduler(model, config)
        for step in range(total_steps):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # updates importance and prunes

    Args:
        model: The model containing AdaLoRALinear layers.
        config: AdaLoRA configuration with scheduling parameters.
    """

    def __init__(self, model: nn.Module, config: AdaLoRAConfig):
        self.model = model
        self.config = config
        self.global_step = 0

        # Collect all AdaLoRA layers
        self.adalora_layers: Dict[str, AdaLoRALinear] = {}
        for name, module in model.named_modules():
            if isinstance(module, AdaLoRALinear):
                self.adalora_layers[name] = module

    def step(self) -> Dict[str, Any]:
        """Perform one scheduler step: update importance and prune if needed.

        Returns:
            Dictionary with scheduling metrics (current step, phase, ranks).
        """
        self.global_step += 1
        metrics: Dict[str, Any] = {
            "global_step": self.global_step,
        }

        # Always update importance scores
        for name, layer in self.adalora_layers.items():
            layer.update_importance(beta1=self.config.beta1)

        # Determine current phase and target rank
        if self.global_step < self.config.warmup_step:
            metrics["phase"] = "warmup"
            # No pruning during warmup
        elif self.global_step < self.config.final_warmup_step:
            metrics["phase"] = "pruning"
            # Linearly interpolate target rank
            progress = (self.global_step - self.config.warmup_step) / max(
                self.config.final_warmup_step - self.config.warmup_step, 1
            )
            current_target = int(
                self.config.initial_rank
                - progress * (self.config.initial_rank - self.config.target_rank)
            )
            current_target = max(current_target, self.config.target_rank)

            # Global pruning: collect all importance scores, find threshold
            all_scores = []
            all_layer_info = []
            for name, layer in self.adalora_layers.items():
                scores = layer.importance_scores.clone()
                all_scores.append(scores)
                all_layer_info.append((name, layer, scores))

            if all_scores:
                all_scores_cat = torch.cat(all_scores)
                total_budget = current_target * len(self.adalora_layers)
                total_budget = min(total_budget, all_scores_cat.numel())

                if total_budget > 0:
                    threshold = torch.topk(
                        all_scores_cat, k=total_budget, largest=True
                    ).values[-1]
                else:
                    threshold = float('inf')

                # Prune each layer based on global threshold
                for name, layer, scores in all_layer_info:
                    active = (scores >= threshold).sum().item()
                    active = max(active, 1)  # at least rank-1
                    layer.prune_to_rank(int(active))
                    metrics[f"rank/{name}"] = layer.current_rank

            metrics["current_target_rank"] = current_target
        else:
            metrics["phase"] = "fixed"
            # Final ranks are already set; no further pruning

        return metrics

    def get_total_rank(self) -> int:
        """Return the sum of active ranks across all AdaLoRA layers."""
        return sum(layer.current_rank for layer in self.adalora_layers.values())


def apply_adalora(
    model: nn.Module,
    config: Optional[AdaLoRAConfig] = None,
    initial_rank: int = 12,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> Tuple[nn.Module, AdaLoRAScheduler]:
    """Inject AdaLoRA adapters into a model and create a rank scheduler.

    This traverses the model, replaces matching nn.Linear layers with
    AdaLoRALinear, freezes all original parameters, and returns both
    the modified model and a rank scheduler for progressive pruning.

    Args:
        model: The model to augment.
        config: Optional AdaLoRAConfig overriding individual arguments.
        initial_rank: Starting rank.
        alpha: Scaling factor.
        dropout: Dropout probability.
        target_modules: Module name patterns to target.

    Returns:
        Tuple of (modified model, AdaLoRAScheduler).
    """
    if config is not None:
        initial_rank = config.initial_rank
        alpha = config.alpha
        dropout = config.dropout
        target_modules = config.target_modules
    else:
        target_modules = target_modules or ['q_proj', 'v_proj']
        config = AdaLoRAConfig(
            initial_rank=initial_rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
        )

    patterns = [re.compile(p) for p in target_modules]

    def _matches(name: str) -> bool:
        return any(pattern.search(name) for pattern in patterns)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace matching layers
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _matches(name):
            adalora_layer = AdaLoRALinear.from_linear(
                module,
                initial_rank=initial_rank,
                alpha=alpha,
                dropout=dropout,
            )
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], adalora_layer)

    scheduler = AdaLoRAScheduler(model, config)

    return model, scheduler
