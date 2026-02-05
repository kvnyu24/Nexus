"""
Wanda: A Simple and Effective Pruning Approach for Large Language Models.

Paper: "A Simple and Effective Pruning Approach for Large Language Models"
       Sun et al., ICLR 2024
       https://arxiv.org/abs/2306.11695

Wanda prunes weights based on the product of weight magnitude and input activation
norms: importance = |W| * ||X||_2. This simple metric effectively identifies
unimportant weights without requiring gradient information or fine-tuning.

Key innovations:
- Magnitude-based pruning weighted by activation statistics
- No gradient computation needed (unlike SparseGPT)
- Works well for both unstructured and N:M structured sparsity
- Can achieve 50-60% sparsity with minimal accuracy loss
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from nexus.core.base import NexusModule


class WandaPruner(NexusModule):
    """Wanda pruner for magnitude and activation-based weight pruning.

    Prunes weights based on importance score: |W| * ||X||_2 where W is the
    weight matrix and X is the input activation. This simple heuristic
    effectively identifies weights that have minimal impact on the output.

    Args:
        config: Configuration dictionary with keys:
            - sparsity (float): Target sparsity ratio (0.0-1.0). Default 0.5
            - prune_n (int): N for N:M structured sparsity. Default 0 (unstructured)
            - prune_m (int): M for N:M structured sparsity. Default 0
            - use_variant (bool): Use variant that prunes by output feature. Default False
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.sparsity = self.config.get('sparsity', 0.5)
        self.prune_n = self.config.get('prune_n', 0)
        self.prune_m = self.config.get('prune_m', 0)
        self.use_variant = self.config.get('use_variant', False)

        # Storage for activation norms collected during calibration
        self.activation_norms = {}

    def _compute_importance(
        self,
        weight: torch.Tensor,
        activation_norm: torch.Tensor
    ) -> torch.Tensor:
        """Compute Wanda importance scores.

        Args:
            weight: Weight matrix of shape [out_features, in_features]
            activation_norm: L2 norm of inputs, shape [in_features] or [out_features]

        Returns:
            Importance scores of same shape as weight
        """
        W_abs = weight.abs()

        if self.use_variant:
            # Variant: importance per output feature
            # activation_norm shape: [out_features]
            importance = W_abs * activation_norm.view(-1, 1)
        else:
            # Standard: importance per input feature
            # activation_norm shape: [in_features]
            importance = W_abs * activation_norm.view(1, -1)

        return importance

    def _unstructured_prune(
        self,
        weight: torch.Tensor,
        importance: torch.Tensor,
        sparsity: float
    ) -> torch.Tensor:
        """Apply unstructured pruning based on importance scores.

        Args:
            weight: Weight matrix
            importance: Importance scores
            sparsity: Fraction of weights to prune

        Returns:
            Binary mask (1 = keep, 0 = prune)
        """
        # Flatten importance scores
        importance_flat = importance.view(-1)

        # Compute threshold
        num_prune = int(sparsity * importance_flat.numel())
        threshold = torch.kthvalue(importance_flat, num_prune).values

        # Create mask
        mask = (importance > threshold).float()

        return mask

    def _structured_nm_prune(
        self,
        weight: torch.Tensor,
        importance: torch.Tensor,
        n: int,
        m: int
    ) -> torch.Tensor:
        """Apply N:M structured sparsity pruning.

        For each contiguous group of M weights, keep the N most important
        and prune the rest. Common patterns: 2:4, 4:8.

        Args:
            weight: Weight matrix [out_features, in_features]
            importance: Importance scores
            n: Number of weights to keep per group
            m: Group size

        Returns:
            Binary mask (1 = keep, 0 = prune)
        """
        out_features, in_features = weight.shape

        # Reshape to [out_features, num_groups, m]
        num_groups = in_features // m
        importance_grouped = importance[:, :num_groups * m].reshape(
            out_features, num_groups, m
        )

        # Find top-n indices within each group of m
        _, top_indices = torch.topk(importance_grouped, n, dim=2)

        # Create mask
        mask_grouped = torch.zeros_like(importance_grouped)
        mask_grouped.scatter_(2, top_indices, 1.0)

        # Reshape back to original shape
        mask = mask_grouped.reshape(out_features, num_groups * m)

        # Handle remaining columns if in_features not divisible by m
        if in_features % m != 0:
            remaining = in_features % m
            mask_remaining = torch.ones(
                out_features, remaining,
                device=weight.device, dtype=weight.dtype
            )
            mask = torch.cat([mask, mask_remaining], dim=1)

        return mask

    def prune_layer(
        self,
        layer: nn.Linear,
        activation_norm: torch.Tensor
    ) -> None:
        """Prune a single linear layer in-place.

        Args:
            layer: Linear layer to prune
            activation_norm: Cached activation norm for this layer
        """
        weight = layer.weight.data

        # Compute importance
        importance = self._compute_importance(weight, activation_norm)

        # Apply pruning
        if self.prune_n > 0 and self.prune_m > 0:
            # N:M structured sparsity
            mask = self._structured_nm_prune(
                weight, importance, self.prune_n, self.prune_m
            )
        else:
            # Unstructured sparsity
            mask = self._unstructured_prune(weight, importance, self.sparsity)

        # Apply mask to weights
        layer.weight.data *= mask

    def collect_activations(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 128
    ) -> Dict[str, torch.Tensor]:
        """Collect activation norms for all linear layers.

        Args:
            model: Model to analyze
            dataloader: Calibration data
            num_samples: Number of samples to use

        Returns:
            Dictionary mapping layer names to activation norms
        """
        model.to_evaluation_mode()
        device = next(model.parameters()).device

        # Register hooks to collect activations
        activation_norms = {}
        handles = []

        def get_hook(name):
            def hook(module, input, output):
                # Compute L2 norm of input activations
                inp = input[0].detach()
                # Average over batch and sequence dimensions
                if inp.dim() == 3:  # [batch, seq, hidden]
                    inp = inp.reshape(-1, inp.size(-1))
                elif inp.dim() == 2:  # [batch, hidden]
                    pass
                else:
                    return

                # Compute per-feature L2 norm
                norm = inp.pow(2).sum(dim=0).sqrt()

                if name in activation_norms:
                    activation_norms[name] += norm
                else:
                    activation_norms[name] = norm
            return hook

        # Register hooks on all linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(get_hook(name))
                handles.append(handle)

        # Run forward passes
        num_processed = 0
        with torch.no_grad():
            for batch in dataloader:
                if num_processed >= num_samples:
                    break

                # Move batch to device
                if isinstance(batch, (tuple, list)):
                    batch = [b.to(device) if isinstance(b, torch.Tensor) else b
                            for b in batch]
                    model(*batch)
                elif isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                    model(**batch)
                else:
                    batch = batch.to(device)
                    model(batch)

                num_processed += batch[0].size(0) if isinstance(batch, (tuple, list)) else batch.size(0)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Average norms
        for name in activation_norms:
            activation_norms[name] /= num_processed

        self.activation_norms = activation_norms
        return activation_norms

    def prune_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 128
    ) -> Dict[str, float]:
        """Prune all linear layers in a model.

        Args:
            model: Model to prune
            dataloader: Calibration data
            num_samples: Number of calibration samples

        Returns:
            Dictionary with pruning statistics
        """
        # Collect activation norms
        print("Collecting activation statistics...")
        self.collect_activations(model, dataloader, num_samples)

        # Prune each layer
        print("Pruning layers...")
        total_params = 0
        pruned_params = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if name in self.activation_norms:
                    # Count params before pruning
                    layer_params = module.weight.numel()
                    total_params += layer_params

                    # Prune layer
                    self.prune_layer(module, self.activation_norms[name])

                    # Count params after pruning
                    pruned_params += (module.weight.data == 0).sum().item()

                    print(f"  {name}: {pruned_params/total_params*100:.2f}% sparse")

        actual_sparsity = pruned_params / total_params
        print(f"\nTotal sparsity: {actual_sparsity*100:.2f}%")

        return {
            'total_params': total_params,
            'pruned_params': pruned_params,
            'actual_sparsity': actual_sparsity
        }


# Convenience function
def prune_with_wanda(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    sparsity: float = 0.5,
    prune_n: int = 0,
    prune_m: int = 0,
    num_samples: int = 128
) -> nn.Module:
    """Convenience function to prune a model with Wanda.

    Args:
        model: Model to prune
        dataloader: Calibration data
        sparsity: Target sparsity for unstructured pruning
        prune_n: N for N:M structured sparsity (0 for unstructured)
        prune_m: M for N:M structured sparsity
        num_samples: Number of calibration samples

    Returns:
        Pruned model (modified in-place)
    """
    config = {
        'sparsity': sparsity,
        'prune_n': prune_n,
        'prune_m': prune_m
    }

    pruner = WandaPruner(config)
    pruner.prune_model(model, dataloader, num_samples)

    return model
