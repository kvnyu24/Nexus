"""EVCL: Elastic Variational Continual Learning.

Reference: "Elastic Variational Continual Learning" (Nguyen et al., ICML 2024)

EVCL is a variational continual learning method that maintains a probabilistic
representation of task-specific knowledge while allowing elastic adaptation to
new tasks. Unlike EWC which uses a point estimate of parameter importance, EVCL
maintains a full posterior distribution over parameters and uses variational
inference to balance between retaining old knowledge and learning new tasks.

Key innovations:
    - Variational posterior: Maintains Gaussian distributions over parameters
    - Elastic KL divergence: Adaptively weighs task preservation vs. plasticity
    - Task-specific variational parameters: Separate mean/variance per task
    - Automatic importance weighting: No manual hyperparameter tuning
    - Multihead architecture: Task-specific output heads

Architecture:
    - VariationalLayer: Bayesian neural network layer with learned distributions
    - TaskPosterior: Maintains posterior distributions for each task
    - EVCLRegularizer: Elastic variational regularization term
    - EVCLModel: Complete continual learning system

Key properties:
    - Probabilistic knowledge retention (vs. deterministic in EWC)
    - Automatic task importance weighting
    - Better uncertainty quantification
    - Graceful forgetting with elastic KL
    - Scales to many sequential tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
from ...core.base import NexusModule
import math


class VariationalLayer(nn.Module):
    """Variational Bayesian linear layer.

    Instead of point estimates, maintains Gaussian distributions over weights
    and biases. Uses the reparameterization trick for backpropagation.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        prior_std: Standard deviation of the prior. Default: 1.0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        # Weight mean and log-variance
        self.weight_mu = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02
        )
        self.weight_logvar = nn.Parameter(
            torch.ones(out_features, in_features) * -5.0  # Small initial variance
        )

        # Bias mean and log-variance
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.ones(out_features) * -5.0)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Forward pass with weight sampling.

        Args:
            x: Input tensor (B, in_features).
            sample: If True, sample weights. If False, use mean. Default: True.

        Returns:
            Output tensor (B, out_features).
        """
        if sample and self.training:
            # Sample weights using reparameterization trick
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight_eps = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + weight_std * weight_eps

            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias_eps = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_std * bias_eps
        else:
            # Use mean weights (deterministic)
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(
        self,
        prior_mu: Optional[torch.Tensor] = None,
        prior_logvar: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute KL divergence between current and prior distributions.

        KL(q(w) || p(w)) for Gaussian distributions.

        Args:
            prior_mu: Prior mean. If None, uses zero mean.
            prior_logvar: Prior log-variance. If None, uses prior_std^2.

        Returns:
            KL divergence (scalar).
        """
        # Default prior: N(0, prior_std^2)
        if prior_mu is None:
            prior_mu = torch.zeros_like(self.weight_mu)
        if prior_logvar is None:
            prior_logvar = torch.ones_like(self.weight_logvar) * math.log(self.prior_std ** 2)

        # KL for weights
        weight_kl = 0.5 * (
            prior_logvar - self.weight_logvar
            + torch.exp(self.weight_logvar) / torch.exp(prior_logvar)
            + (self.weight_mu - prior_mu).pow(2) / torch.exp(prior_logvar)
            - 1.0
        ).sum()

        # KL for biases
        if prior_mu is None:
            prior_bias_mu = torch.zeros_like(self.bias_mu)
        else:
            prior_bias_mu = torch.zeros_like(self.bias_mu)  # Bias prior always zero

        prior_bias_logvar = torch.ones_like(self.bias_logvar) * math.log(self.prior_std ** 2)

        bias_kl = 0.5 * (
            prior_bias_logvar - self.bias_logvar
            + torch.exp(self.bias_logvar) / torch.exp(prior_bias_logvar)
            + (self.bias_mu - prior_bias_mu).pow(2) / torch.exp(prior_bias_logvar)
            - 1.0
        ).sum()

        return weight_kl + bias_kl


class TaskPosterior:
    """Stores posterior distribution for a completed task.

    Maintains the learned mean and variance of parameters after training
    on a task. Used as the prior when learning new tasks.
    """

    def __init__(self, model: nn.Module):
        """Initialize from current model state.

        Args:
            model: Model with VariationalLayer instances.
        """
        self.posteriors = {}

        for name, module in model.named_modules():
            if isinstance(module, VariationalLayer):
                self.posteriors[name] = {
                    "weight_mu": module.weight_mu.data.clone(),
                    "weight_logvar": module.weight_logvar.data.clone(),
                    "bias_mu": module.bias_mu.data.clone(),
                    "bias_logvar": module.bias_logvar.data.clone(),
                }

    def get_prior(self, layer_name: str) -> Dict[str, torch.Tensor]:
        """Get prior distribution for a layer.

        Args:
            layer_name: Name of the layer.

        Returns:
            Dictionary with weight_mu, weight_logvar, bias_mu, bias_logvar.
        """
        return self.posteriors.get(layer_name, {})


class EVCLRegularizer(NexusModule):
    """Elastic variational continual learning regularizer.

    Computes the elastic KL divergence that balances between retaining
    previous task knowledge and adapting to new tasks. The elasticity
    is automatically learned based on task difficulty.

    Args:
        config: Configuration dictionary with:
            - base_kl_weight: Base weight for KL divergence. Default: 0.1.
            - elasticity: Elasticity parameter (0=rigid, 1=plastic). Default: 0.5.
            - adaptive_elasticity: Learn elasticity per task. Default: True.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.base_kl_weight = config.get("base_kl_weight", 0.1)
        self.elasticity = config.get("elasticity", 0.5)
        self.adaptive_elasticity = config.get("adaptive_elasticity", True)

        # Storage for task posteriors
        self.task_posteriors: List[TaskPosterior] = []

        # Learnable elasticity weights per task (if adaptive)
        if self.adaptive_elasticity:
            self.elasticity_weights = nn.ParameterList()

    def register_task(self, model: nn.Module) -> None:
        """Register a completed task's posterior.

        Args:
            model: Model after training on the task.
        """
        posterior = TaskPosterior(model)
        self.task_posteriors.append(posterior)

        if self.adaptive_elasticity:
            # Add learnable elasticity weight for this task
            self.elasticity_weights.append(
                nn.Parameter(torch.tensor(self.elasticity))
            )

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute elastic variational regularization loss.

        Args:
            model: Current model being trained.

        Returns:
            Regularization loss (scalar).
        """
        if len(self.task_posteriors) == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        total_kl = torch.tensor(0.0, device=next(model.parameters()).device)

        # Iterate over variational layers
        for name, module in model.named_modules():
            if isinstance(module, VariationalLayer):
                layer_kl = torch.tensor(0.0, device=next(model.parameters()).device)

                # Accumulate KL from all previous tasks
                for task_idx, posterior in enumerate(self.task_posteriors):
                    prior = posterior.get_prior(name)
                    if not prior:
                        continue

                    # Compute KL divergence to task posterior
                    kl = module.kl_divergence(
                        prior_mu=prior["weight_mu"],
                        prior_logvar=prior["weight_logvar"]
                    )

                    # Apply elastic weighting
                    if self.adaptive_elasticity:
                        elasticity = torch.sigmoid(self.elasticity_weights[task_idx])
                    else:
                        elasticity = self.elasticity

                    # Elastic KL: weight by (1 - elasticity)
                    # High elasticity = low weight = more plastic
                    layer_kl += (1.0 - elasticity) * kl

                total_kl += layer_kl

        # Apply base weight
        reg_loss = self.base_kl_weight * total_kl

        return reg_loss

    @property
    def num_tasks(self) -> int:
        """Return number of registered tasks."""
        return len(self.task_posteriors)


class EVCLModel(NexusModule):
    """EVCL: Elastic Variational Continual Learning model.

    Complete continual learning system using variational inference and
    elastic KL divergence. Maintains probabilistic task-specific knowledge
    while allowing adaptive learning of new tasks.

    Architecture uses variational layers throughout and a multi-head
    output structure for task-specific predictions.

    Args:
        config: Configuration dictionary with:
            - input_dim: Input dimension. Default: 784.
            - hidden_dims: List of hidden dimensions. Default: [256, 256].
            - output_dim: Output dimension per task. Default: 10.
            - num_tasks: Maximum number of tasks. Default: 10.
            - prior_std: Prior standard deviation. Default: 1.0.
            - base_kl_weight: Base KL weight. Default: 0.1.
            - elasticity: Elasticity parameter. Default: 0.5.

    Example:
        >>> config = {
        ...     "input_dim": 784,
        ...     "hidden_dims": [256, 256],
        ...     "output_dim": 10,
        ...     "num_tasks": 5
        ... }
        >>> model = EVCLModel(config)
        >>> x = torch.randn(32, 784)
        >>> task_id = 0
        >>> output = model(x, task_id)
        >>> loss = F.cross_entropy(output, labels)
        >>> reg_loss = model.regularizer(model)
        >>> total_loss = loss + reg_loss
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.input_dim = config.get("input_dim", 784)
        self.hidden_dims = config.get("hidden_dims", [256, 256])
        self.output_dim = config.get("output_dim", 10)
        self.num_tasks = config.get("num_tasks", 10)
        self.prior_std = config.get("prior_std", 1.0)

        # Build variational network
        self.layers = nn.ModuleList()

        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            self.layers.append(
                VariationalLayer(prev_dim, hidden_dim, prior_std=self.prior_std)
            )
            prev_dim = hidden_dim

        # Multi-head output: one head per task
        self.task_heads = nn.ModuleList([
            VariationalLayer(prev_dim, self.output_dim, prior_std=self.prior_std)
            for _ in range(self.num_tasks)
        ])

        # Regularizer
        self.regularizer = EVCLRegularizer(config)

        # Current task ID
        self.current_task = 0

    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[int] = None,
        sample: bool = True
    ) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor (B, input_dim).
            task_id: Task ID for selecting output head. Default: current_task.
            sample: Whether to sample weights. Default: True.

        Returns:
            Output logits (B, output_dim).
        """
        if task_id is None:
            task_id = self.current_task

        # Forward through shared layers
        for layer in self.layers:
            x = layer(x, sample=sample)
            x = F.relu(x)

        # Task-specific output head
        x = self.task_heads[task_id](x, sample=sample)

        return x

    def set_task(self, task_id: int) -> None:
        """Set the current task ID.

        Args:
            task_id: Task identifier.
        """
        self.current_task = task_id

    def consolidate_task(self) -> None:
        """Consolidate knowledge after completing a task.

        Registers the current posterior as a prior for future tasks.
        """
        self.regularizer.register_task(self)

    def train_task(
        self,
        data_loader: DataLoader,
        task_id: int,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        num_samples: int = 3
    ) -> Dict[str, List[float]]:
        """Train on a single task with EVCL.

        Args:
            data_loader: DataLoader for the task.
            task_id: Task identifier.
            num_epochs: Number of training epochs. Default: 10.
            learning_rate: Learning rate. Default: 1e-3.
            num_samples: Number of weight samples per forward pass. Default: 3.

        Returns:
            Training history dictionary.
        """
        self.set_task(task_id)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        history = {"task_loss": [], "reg_loss": [], "total_loss": []}

        self.train()

        for epoch in range(num_epochs):
            epoch_task_loss = 0.0
            epoch_reg_loss = 0.0
            num_batches = 0

            for batch in data_loader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                elif isinstance(batch, dict):
                    inputs = batch.get("image", batch.get("input"))
                    targets = batch.get("label", batch.get("target"))
                else:
                    raise ValueError(f"Unsupported batch type: {type(batch)}")

                inputs = inputs.to(next(self.parameters()).device)
                targets = targets.to(next(self.parameters()).device)

                optimizer.zero_grad()

                # Monte Carlo sampling of weights
                task_loss = 0.0
                for _ in range(num_samples):
                    output = self(inputs, task_id=task_id, sample=True)
                    task_loss += F.cross_entropy(output, targets)
                task_loss /= num_samples

                # Variational regularization
                reg_loss = self.regularizer(self)

                # Total loss
                total_loss = task_loss + reg_loss

                total_loss.backward()
                optimizer.step()

                epoch_task_loss += task_loss.item()
                epoch_reg_loss += reg_loss.item()
                num_batches += 1

            avg_task_loss = epoch_task_loss / max(num_batches, 1)
            avg_reg_loss = epoch_reg_loss / max(num_batches, 1)

            history["task_loss"].append(avg_task_loss)
            history["reg_loss"].append(avg_reg_loss)
            history["total_loss"].append(avg_task_loss + avg_reg_loss)

        # Consolidate task knowledge
        self.consolidate_task()

        return history
