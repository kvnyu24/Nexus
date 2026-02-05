"""ShortGPT: Pruning via Layer Removal by Block Influence Score.

Reference:
    Men, X., et al. "ShortGPT: Layers in Large Language Models are More
    Redundant Than You Expect."
    2024. https://arxiv.org/abs/2403.03853

ShortGPT prunes transformer models by removing entire layers based on their
influence on the final output. The key insight is that many layers in large
LLMs are redundant and can be removed with minimal impact on model quality.

The method:
1. Computes block influence (BI) scores measuring each layer's contribution
2. Identifies and removes the least influential layers
3. Achieves significant model compression (e.g., 25% layer reduction) with
   minimal accuracy loss

This is simpler and more effective than weight pruning for LLMs.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Callable
import math
from collections import defaultdict

from nexus.core.base import NexusModule


@dataclass
class ShortGPTConfig:
    """Configuration for ShortGPT pruning.

    Attributes:
        pruning_ratio: Fraction of layers to prune (e.g., 0.25 for 25%).
        importance_metric: Metric for computing layer importance:
            - "block_influence": Block Influence score (recommended)
            - "gradient_norm": Gradient magnitude
            - "activation_norm": Activation magnitude
        calibration_samples: Number of samples for computing importance.
        prune_strategy: Strategy for selecting layers to prune:
            - "global": Prune least important layers globally
            - "uniform": Prune uniformly across all layer depths
        preserve_first_k: Number of initial layers to always preserve.
        preserve_last_k: Number of final layers to always preserve.
    """
    pruning_ratio: float = 0.25
    importance_metric: str = "block_influence"
    calibration_samples: int = 128
    prune_strategy: str = "global"
    preserve_first_k: int = 2
    preserve_last_k: int = 2


class BlockInfluenceScorer:
    """Compute Block Influence scores for transformer layers.

    Block Influence measures how much a layer contributes to the final output
    by computing the magnitude of change in activations.
    """

    def __init__(self, model: nn.Module, config: ShortGPTConfig):
        self.model = model
        self.config = config
        self.layer_scores = {}
        self.activation_diffs = defaultdict(list)

    def _get_transformer_layers(self) -> List[Tuple[str, nn.Module]]:
        """Identify transformer layers in the model.

        Returns:
            List of (name, module) tuples for transformer layers.
        """
        layers = []
        for name, module in self.model.named_modules():
            # Common patterns for transformer layers
            if any(pattern in name.lower() for pattern in ['layer', 'block', 'transformer_layer']):
                # Check if it's a top-level layer (not a sub-component)
                if '.' not in name or name.count('.') == 1:
                    layers.append((name, module))
        return layers

    def compute_block_influence(
        self,
        calibration_data: torch.Tensor,
        layer_name: str,
        layer_module: nn.Module
    ) -> float:
        """Compute Block Influence score for a single layer.

        Args:
            calibration_data: Input data for computing influence.
            layer_name: Name of the layer.
            layer_module: The layer module.

        Returns:
            Block Influence score (higher = more important).
        """
        activation_diffs = []

        # Hook to capture input and output
        def hook_fn(module, input, output):
            # Compute norm of change: ||output - input||
            if isinstance(input, tuple):
                input_tensor = input[0]
            else:
                input_tensor = input

            if isinstance(output, tuple):
                output_tensor = output[0]
            else:
                output_tensor = output

            # Compute difference norm
            if input_tensor.shape == output_tensor.shape:
                diff = output_tensor - input_tensor
                diff_norm = torch.norm(diff, p=2).item()
                activation_diffs.append(diff_norm)

        # Register hook
        handle = layer_module.register_forward_hook(hook_fn)

        # Forward pass with calibration data
        with torch.no_grad():
            _ = self.model(calibration_data)

        # Remove hook
        handle.remove()

        # Compute average influence
        if len(activation_diffs) > 0:
            avg_influence = sum(activation_diffs) / len(activation_diffs)
        else:
            avg_influence = 0.0

        return avg_influence

    def compute_gradient_importance(
        self,
        calibration_data: torch.Tensor,
        labels: torch.Tensor,
        layer_module: nn.Module
    ) -> float:
        """Compute gradient-based importance for a layer.

        Args:
            calibration_data: Input data.
            labels: Target labels.
            layer_module: The layer module.

        Returns:
            Gradient importance score.
        """
        gradient_norms = []

        # Forward pass
        outputs = self.model(calibration_data)
        loss = nn.functional.cross_entropy(outputs, labels)

        # Backward pass
        loss.backward()

        # Compute gradient norm for this layer
        total_grad_norm = 0.0
        for param in layer_module.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm(2).item() ** 2

        gradient_norms.append(math.sqrt(total_grad_norm))

        # Zero gradients
        self.model.zero_grad()

        return sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0

    def compute_all_scores(
        self,
        calibration_data: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute importance scores for all layers.

        Args:
            calibration_data: Calibration input data.
            labels: Optional labels for gradient-based metrics.

        Returns:
            Dictionary mapping layer names to importance scores.
        """
        layers = self._get_transformer_layers()

        for layer_name, layer_module in layers:
            if self.config.importance_metric == "block_influence":
                score = self.compute_block_influence(
                    calibration_data,
                    layer_name,
                    layer_module
                )
            elif self.config.importance_metric == "gradient_norm" and labels is not None:
                score = self.compute_gradient_importance(
                    calibration_data,
                    labels,
                    layer_module
                )
            else:
                # Default: use activation norm
                score = 1.0  # Placeholder

            self.layer_scores[layer_name] = score

        return self.layer_scores


class ShortGPTPruner(NexusModule):
    """Pruner for removing transformer layers using ShortGPT.

    Args:
        config: ShortGPT configuration.
    """

    def __init__(self, config: ShortGPTConfig):
        super().__init__(config.__dict__)
        self.config = config

    def identify_layers_to_prune(
        self,
        layer_scores: Dict[str, float]
    ) -> List[str]:
        """Identify which layers to prune based on importance scores.

        Args:
            layer_scores: Dictionary of layer names to importance scores.

        Returns:
            List of layer names to prune.
        """
        # Sort layers by score (ascending - least important first)
        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1])

        num_layers = len(sorted_layers)
        num_to_prune = int(num_layers * self.config.pruning_ratio)

        # Apply preservation constraints
        layers_to_prune = []

        for layer_name, score in sorted_layers:
            if len(layers_to_prune) >= num_to_prune:
                break

            # Extract layer index (assuming naming like "layer.0", "layer.1", etc.)
            # This is a simplified heuristic
            try:
                # Try to extract numeric index
                parts = layer_name.split('.')
                for part in parts:
                    if part.isdigit():
                        layer_idx = int(part)
                        break
                else:
                    layer_idx = -1

                # Check preservation constraints
                if layer_idx < self.config.preserve_first_k:
                    continue
                if layer_idx >= num_layers - self.config.preserve_last_k:
                    continue

            except:
                pass

            layers_to_prune.append(layer_name)

        return layers_to_prune

    def prune_model(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> nn.Module:
        """Prune a model using ShortGPT.

        Args:
            model: Model to prune.
            calibration_data: Calibration data for computing importance.
            labels: Optional labels for gradient-based metrics.
            verbose: If True, print pruning information.

        Returns:
            Pruned model.
        """
        # Step 1: Compute importance scores
        if verbose:
            print("Computing layer importance scores...")

        scorer = BlockInfluenceScorer(model, self.config)
        layer_scores = scorer.compute_all_scores(calibration_data, labels)

        if verbose:
            print(f"Computed scores for {len(layer_scores)} layers")

        # Step 2: Identify layers to prune
        layers_to_prune = self.identify_layers_to_prune(layer_scores)

        if verbose:
            print(f"Pruning {len(layers_to_prune)} layers:")
            for layer_name in layers_to_prune:
                print(f"  - {layer_name} (score: {layer_scores[layer_name]:.4f})")

        # Step 3: Remove layers
        for layer_name in layers_to_prune:
            self._remove_layer(model, layer_name)

        if verbose:
            remaining_layers = len(layer_scores) - len(layers_to_prune)
            print(f"Pruning complete. Remaining layers: {remaining_layers}")

        return model

    def _remove_layer(self, model: nn.Module, layer_name: str):
        """Remove a layer from the model.

        Args:
            model: The model.
            layer_name: Name of the layer to remove.
        """
        # Get parent module
        parts = layer_name.split('.')
        if len(parts) == 1:
            # Top-level module
            if hasattr(model, layer_name):
                setattr(model, layer_name, nn.Identity())
        else:
            parent_name = '.'.join(parts[:-1])
            child_name = parts[-1]
            parent = model.get_submodule(parent_name)

            # Replace with identity
            setattr(parent, child_name, nn.Identity())


def prune_with_shortgpt(
    model: nn.Module,
    calibration_data: torch.Tensor,
    config: Optional[ShortGPTConfig] = None,
    labels: Optional[torch.Tensor] = None,
    verbose: bool = True
) -> nn.Module:
    """Apply ShortGPT pruning to a model.

    Args:
        model: Model to prune.
        calibration_data: Calibration data for computing importance.
        config: ShortGPT configuration (uses defaults if None).
        labels: Optional labels for gradient-based metrics.
        verbose: If True, print information.

    Returns:
        Pruned model.
    """
    config = config or ShortGPTConfig()
    pruner = ShortGPTPruner(config)
    return pruner.prune_model(model, calibration_data, labels, verbose)
