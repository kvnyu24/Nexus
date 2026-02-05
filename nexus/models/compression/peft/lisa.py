"""LISA: Layerwise Importance Sampled AdamW for memory-efficient training.

Reference:
    Pan, Y., et al. "LISA: Layerwise Importance Sampling for Memory-Efficient
    Large Language Model Fine-Tuning."
    NeurIPS 2024. https://arxiv.org/abs/2403.17919

LISA reduces memory consumption during fine-tuning by selectively updating only
a subset of layers at each step. It samples layers based on their importance
(computed from loss sensitivity or other metrics), allowing effective training
with limited GPU memory. LISA can train 7B models on 24GB GPUs that would
otherwise require 80GB+.

Key innovation: Instead of updating all layers, sample k layers per step based
on importance scores, significantly reducing optimizer state memory.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import math
import random
from collections import defaultdict

from nexus.core.base import NexusModule


@dataclass
class LISAConfig:
    """Configuration for LISA training.

    Attributes:
        num_layers_to_sample: Number of layers to update per training step.
        sampling_strategy: Strategy for selecting layers:
            - "uniform": Uniform random sampling
            - "importance": Importance-based sampling (requires loss gradients)
            - "cyclic": Cycle through layers deterministically
        importance_warmup_steps: Steps to collect statistics before importance sampling.
        always_update_embeddings: Whether to always update embedding layers.
        always_update_head: Whether to always update the output head.
        layer_pattern: Regex pattern to identify transformer layers.
        update_frequency: Frequency (in steps) to recompute importance scores.
    """
    num_layers_to_sample: int = 2
    sampling_strategy: str = "importance"
    importance_warmup_steps: int = 100
    always_update_embeddings: bool = True
    always_update_head: bool = True
    layer_pattern: str = r"layers\.(\d+)\."
    update_frequency: int = 10


class LISAOptimizer:
    """LISA optimizer wrapper for layerwise importance sampling.

    This optimizer wraps a base optimizer (e.g., AdamW) and implements layerwise
    sampling. At each step, only a subset of layers have their gradients applied,
    significantly reducing memory usage.

    Args:
        model: The model to optimize.
        optimizer: Base optimizer (e.g., AdamW).
        config: LISA configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: LISAConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config

        # Layer tracking
        self.layer_params = self._identify_layers()
        self.num_layers = len(self.layer_params)
        self.current_step = 0

        # Importance tracking
        self.layer_importance = {i: 1.0 for i in range(self.num_layers)}
        self.importance_history = defaultdict(list)

        # Cyclic sampling state
        self.cyclic_index = 0

        # Always-update modules
        self.always_update_params = set()
        if config.always_update_embeddings:
            self.always_update_params.update(self._get_embedding_params())
        if config.always_update_head:
            self.always_update_params.update(self._get_head_params())

    def _identify_layers(self) -> Dict[int, List[nn.Parameter]]:
        """Identify transformer layers and their parameters.

        Returns:
            Dictionary mapping layer index to list of parameters.
        """
        import re
        layer_params = defaultdict(list)

        for name, param in self.model.named_parameters():
            match = re.search(self.config.layer_pattern, name)
            if match:
                layer_idx = int(match.group(1))
                layer_params[layer_idx].append(param)

        return dict(layer_params)

    def _get_embedding_params(self) -> set:
        """Get parameters belonging to embedding layers."""
        embedding_params = set()
        for name, param in self.model.named_parameters():
            if 'embed' in name.lower():
                embedding_params.add(param)
        return embedding_params

    def _get_head_params(self) -> set:
        """Get parameters belonging to the output head."""
        head_params = set()
        for name, param in self.model.named_parameters():
            if 'head' in name.lower() or 'lm_head' in name.lower() or 'output' in name.lower():
                head_params.add(param)
        return head_params

    def _sample_layers_uniform(self) -> List[int]:
        """Sample layers uniformly at random."""
        return random.sample(
            range(self.num_layers),
            k=min(self.config.num_layers_to_sample, self.num_layers)
        )

    def _sample_layers_importance(self) -> List[int]:
        """Sample layers based on importance scores."""
        # Normalize importance scores to probabilities
        total_importance = sum(self.layer_importance.values())
        probs = [
            self.layer_importance[i] / total_importance
            for i in range(self.num_layers)
        ]

        # Sample without replacement
        sampled = torch.multinomial(
            torch.tensor(probs),
            num_samples=min(self.config.num_layers_to_sample, self.num_layers),
            replacement=False
        ).tolist()

        return sampled

    def _sample_layers_cyclic(self) -> List[int]:
        """Sample layers in a cyclic manner."""
        sampled = []
        for _ in range(self.config.num_layers_to_sample):
            sampled.append(self.cyclic_index % self.num_layers)
            self.cyclic_index += 1
        return sampled

    def sample_layers(self) -> List[int]:
        """Sample layers based on the configured strategy.

        Returns:
            List of layer indices to update in this step.
        """
        # Warmup: use uniform sampling
        if self.current_step < self.config.importance_warmup_steps:
            return self._sample_layers_uniform()

        # Use configured strategy
        if self.config.sampling_strategy == "uniform":
            return self._sample_layers_uniform()
        elif self.config.sampling_strategy == "importance":
            return self._sample_layers_importance()
        elif self.config.sampling_strategy == "cyclic":
            return self._sample_layers_cyclic()
        else:
            raise ValueError(f"Unknown sampling strategy: {self.config.sampling_strategy}")

    def compute_layer_importance(self, loss: torch.Tensor):
        """Compute importance scores for each layer based on loss gradients.

        Args:
            loss: The training loss (scalar tensor with grad_fn).
        """
        if self.current_step < self.config.importance_warmup_steps:
            return

        if self.current_step % self.config.update_frequency != 0:
            return

        # Compute gradients w.r.t. each layer's parameters
        for layer_idx, params in self.layer_params.items():
            # Compute gradient norm for this layer
            layer_grad_norm = 0.0
            for param in params:
                if param.grad is not None:
                    layer_grad_norm += param.grad.norm(2).item() ** 2
            layer_grad_norm = math.sqrt(layer_grad_norm)

            # Update importance score (exponential moving average)
            alpha = 0.9
            self.layer_importance[layer_idx] = (
                alpha * self.layer_importance[layer_idx] +
                (1 - alpha) * layer_grad_norm
            )
            self.importance_history[layer_idx].append(layer_grad_norm)

    def step(self, loss: Optional[torch.Tensor] = None, closure=None):
        """Perform optimization step with layerwise sampling.

        Args:
            loss: Optional loss tensor for computing importance.
            closure: Optional closure for optimizer.
        """
        # Sample layers to update
        sampled_layers = self.sample_layers()

        # Build set of parameters to update
        params_to_update = set()

        # Add always-update parameters
        params_to_update.update(self.always_update_params)

        # Add sampled layer parameters
        for layer_idx in sampled_layers:
            if layer_idx in self.layer_params:
                params_to_update.update(self.layer_params[layer_idx])

        # Zero gradients for non-selected parameters
        for param in self.model.parameters():
            if param not in params_to_update and param.grad is not None:
                param.grad = None

        # Perform optimizer step
        if closure is not None:
            result = self.optimizer.step(closure)
        else:
            result = self.optimizer.step()

        # Update importance scores if loss is provided
        if loss is not None:
            self.compute_layer_importance(loss)

        self.current_step += 1
        return result

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Get optimizer state dict."""
        return {
            'optimizer': self.optimizer.state_dict(),
            'layer_importance': self.layer_importance,
            'current_step': self.current_step,
            'cyclic_index': self.cyclic_index,
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.layer_importance = state_dict['layer_importance']
        self.current_step = state_dict['current_step']
        self.cyclic_index = state_dict.get('cyclic_index', 0)

    @property
    def param_groups(self):
        """Access underlying optimizer's parameter groups."""
        return self.optimizer.param_groups


class LISATrainingWrapper(NexusModule):
    """Wrapper for training models with LISA.

    This wrapper facilitates LISA training by managing the optimizer and
    providing convenience methods for training loops.

    Args:
        model: The model to train.
        config: LISA configuration.
        optimizer: Base optimizer (e.g., AdamW).
    """

    def __init__(
        self,
        model: nn.Module,
        config: LISAConfig,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__(config.__dict__)
        self.model = model
        self.lisa_optimizer = LISAOptimizer(model, optimizer, config)

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step with LISA.

        Args:
            batch: Training batch with 'input_ids', 'labels', etc.

        Returns:
            Dictionary with 'loss' and optionally other metrics.
        """
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]

        # Backward pass
        loss.backward()

        # LISA optimization step
        self.lisa_optimizer.step(loss=loss)
        self.lisa_optimizer.zero_grad()

        return {'loss': loss.item()}

    def get_layer_importance_stats(self) -> Dict[int, float]:
        """Get current importance scores for each layer.

        Returns:
            Dictionary mapping layer index to importance score.
        """
        return self.lisa_optimizer.layer_importance.copy()

    def get_importance_history(self) -> Dict[int, List[float]]:
        """Get historical importance scores for analysis.

        Returns:
            Dictionary mapping layer index to list of historical scores.
        """
        return dict(self.lisa_optimizer.importance_history)


def create_lisa_optimizer(
    model: nn.Module,
    base_optimizer_class: type = torch.optim.AdamW,
    base_optimizer_kwargs: Optional[Dict[str, Any]] = None,
    lisa_config: Optional[LISAConfig] = None,
) -> LISAOptimizer:
    """Create a LISA optimizer for the given model.

    Args:
        model: Model to optimize.
        base_optimizer_class: Base optimizer class (default: AdamW).
        base_optimizer_kwargs: Keyword arguments for base optimizer.
        lisa_config: LISA configuration.

    Returns:
        LISAOptimizer instance.
    """
    base_optimizer_kwargs = base_optimizer_kwargs or {}
    lisa_config = lisa_config or LISAConfig()

    # Create base optimizer
    base_optimizer = base_optimizer_class(
        model.parameters(),
        **base_optimizer_kwargs
    )

    # Wrap with LISA
    return LISAOptimizer(model, base_optimizer, lisa_config)
