"""EWC: Elastic Weight Consolidation for Continual Learning.

Reference: "Overcoming catastrophic forgetting in neural networks"
(Kirkpatrick et al., 2017)

Elastic Weight Consolidation (EWC) prevents catastrophic forgetting in
continual learning by adding a regularization term that penalizes changes
to parameters that were important for previous tasks. Importance is
measured using the diagonal of the Fisher information matrix, which
approximates the curvature of the loss landscape.

Architecture:
    - FisherInformation: Computes diagonal Fisher information matrix
    - EWCRegularizer: Computes the EWC penalty term
    - EWCTrainer: Training loop with EWC regularization

Key properties:
    - Protects important parameters from large changes
    - Fisher information measures parameter importance
    - Quadratic penalty: lambda * sum(F_i * (theta_i - theta_star_i)^2)
    - Can accumulate Fisher information across multiple tasks
    - No stored data from previous tasks needed (only Fisher + params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
from typing import Dict, Any, Optional, List, Tuple
from ...core.base import NexusModule


class FisherInformation:
    """Computes the diagonal Fisher information matrix for EWC.

    The Fisher information matrix approximates the importance of each
    parameter for a given task. Parameters with high Fisher information
    are important and should not change much when learning new tasks.

    The diagonal Fisher is computed as:
        F_i = E[grad_i(log p(y|x, theta))^2]

    which is estimated by averaging squared gradients over samples
    from the training data.

    Args:
        model: The neural network model.
        fisher_samples: Number of samples to estimate Fisher. Default: 200.
        empirical: If True, use empirical Fisher (uses true labels).
            If False, use true Fisher (samples from model). Default: True.

    Example:
        >>> fisher_computer = FisherInformation(model, fisher_samples=200)
        >>> fisher_diag, optimal_params = fisher_computer.compute(train_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        fisher_samples: int = 200,
        empirical: bool = True,
    ):
        self.model = model
        self.fisher_samples = fisher_samples
        self.empirical = empirical

    @torch.no_grad()
    def _get_model_params(self) -> Dict[str, torch.Tensor]:
        """Get a copy of current model parameters.

        Returns:
            Dictionary mapping parameter names to their values.
        """
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def compute(
        self,
        data_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute diagonal Fisher information matrix.

        Args:
            data_loader: DataLoader for the current task's training data.
            criterion: Loss function. If None, uses cross-entropy.

        Returns:
            Tuple of (fisher_diagonal, optimal_params).
                fisher_diagonal: Dict mapping param names to Fisher values.
                optimal_params: Dict mapping param names to optimal values.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Store optimal parameters for the current task
        optimal_params = self._get_model_params()

        # Initialize Fisher diagonal
        fisher_diag = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.train()  # Need gradients but in deterministic mode
        num_samples = 0

        for batch_idx, batch in enumerate(data_loader):
            if num_samples >= self.fisher_samples:
                break

            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                inputs = batch.get("image", batch.get("input"))
                targets = batch.get("label", batch.get("target"))
            else:
                raise ValueError(f"Unsupported batch type: {type(batch)}")

            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            targets = targets.to(device)

            batch_size = inputs.shape[0]

            for i in range(min(batch_size, self.fisher_samples - num_samples)):
                self.model.zero_grad()

                # Forward pass for single sample
                output = self.model(inputs[i : i + 1])
                if isinstance(output, dict):
                    output = output.get("logits", output.get("output"))

                if self.empirical:
                    # Empirical Fisher: use true labels
                    loss = criterion(output, targets[i : i + 1])
                else:
                    # True Fisher: sample from model distribution
                    log_probs = F.log_softmax(output, dim=-1)
                    sampled = torch.multinomial(
                        log_probs.exp(), 1
                    ).squeeze(-1)
                    loss = criterion(output, sampled)

                loss.backward()

                # Accumulate squared gradients
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher_diag[name] += param.grad.data.pow(2)

                num_samples += 1

        # Average over samples
        if num_samples > 0:
            for name in fisher_diag:
                fisher_diag[name] /= num_samples

        return fisher_diag, optimal_params


class EWCRegularizer(NexusModule):
    """Elastic Weight Consolidation regularization term.

    Computes the EWC penalty that discourages parameter changes that
    would be detrimental to previously learned tasks:

        L_EWC = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

    where F_i is the Fisher information for parameter i, theta_i is
    the current value, and theta*_i is the optimal value from the
    previous task.

    Supports accumulating Fisher information from multiple tasks
    (online EWC).

    Args:
        config: Configuration dictionary with:
            - ewc_lambda: Strength of the EWC penalty. Default: 1000.0.

    Example:
        >>> regularizer = EWCRegularizer({"ewc_lambda": 1000.0})
        >>> regularizer.register_task(fisher_diag, optimal_params)
        >>> ewc_loss = regularizer(model)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.ewc_lambda = config.get("ewc_lambda", 1000.0)

        # Storage for task-specific Fisher and params
        self._task_fisher: List[Dict[str, torch.Tensor]] = []
        self._task_params: List[Dict[str, torch.Tensor]] = []

    def register_task(
        self,
        fisher_diag: Dict[str, torch.Tensor],
        optimal_params: Dict[str, torch.Tensor],
    ) -> None:
        """Register Fisher information and optimal params for a completed task.

        Args:
            fisher_diag: Diagonal Fisher information for the task.
            optimal_params: Optimal parameters at end of task training.
        """
        self._task_fisher.append(
            {k: v.clone() for k, v in fisher_diag.items()}
        )
        self._task_params.append(
            {k: v.clone() for k, v in optimal_params.items()}
        )

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC regularization loss.

        Args:
            model: Current model whose parameters are being regularized.

        Returns:
            EWC penalty loss (scalar tensor).
        """
        if len(self._task_fisher) == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        ewc_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        for task_fisher, task_params in zip(
            self._task_fisher, self._task_params
        ):
            for name, param in model.named_parameters():
                if name in task_fisher and param.requires_grad:
                    fisher = task_fisher[name].to(param.device)
                    optimal = task_params[name].to(param.device)

                    ewc_loss += (fisher * (param - optimal).pow(2)).sum()

        ewc_loss = (self.ewc_lambda / 2.0) * ewc_loss

        return ewc_loss

    @property
    def num_tasks(self) -> int:
        """Return the number of registered tasks."""
        return len(self._task_fisher)


class EWCTrainer:
    """Trainer with Elastic Weight Consolidation for continual learning.

    Combines standard supervised training with EWC regularization to
    prevent catastrophic forgetting. After training on each task, the
    Fisher information is computed and stored for future regularization.

    Args:
        model: Neural network model.
        fisher_samples: Number of samples for Fisher estimation. Default: 200.
        ewc_lambda: EWC regularization strength. Default: 1000.0.
        learning_rate: Learning rate. Default: 1e-3.
        device: Device for training. Default: auto-detect.

    Example:
        >>> trainer = EWCTrainer(model, ewc_lambda=1000.0)
        >>> trainer.train_task(task1_loader, num_epochs=10)
        >>> trainer.consolidate(task1_loader)
        >>> trainer.train_task(task2_loader, num_epochs=10)
    """

    def __init__(
        self,
        model: nn.Module,
        fisher_samples: int = 200,
        ewc_lambda: float = 1000.0,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.fisher_computer = FisherInformation(
            model, fisher_samples=fisher_samples
        )
        self.regularizer = EWCRegularizer({"ewc_lambda": ewc_lambda})
        self.learning_rate = learning_rate

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate
        )

    def train_task(
        self,
        data_loader: DataLoader,
        num_epochs: int = 10,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, List[float]]:
        """Train on a single task with EWC regularization.

        Args:
            data_loader: Training data for the current task.
            num_epochs: Number of training epochs. Default: 10.
            criterion: Loss function. Default: CrossEntropyLoss.

        Returns:
            Dictionary with training history.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        history = {"task_loss": [], "ewc_loss": [], "total_loss": []}

        self.model.train()

        for epoch in range(num_epochs):
            epoch_task_loss = 0.0
            epoch_ewc_loss = 0.0
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

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(inputs)
                if isinstance(output, dict):
                    output = output.get("logits", output.get("output"))

                # Task-specific loss
                task_loss = criterion(output, targets)

                # EWC regularization loss
                ewc_loss = self.regularizer(self.model)

                # Total loss
                total_loss = task_loss + ewc_loss

                total_loss.backward()
                self.optimizer.step()

                epoch_task_loss += task_loss.item()
                epoch_ewc_loss += ewc_loss.item()
                num_batches += 1

            avg_task_loss = epoch_task_loss / max(num_batches, 1)
            avg_ewc_loss = epoch_ewc_loss / max(num_batches, 1)

            history["task_loss"].append(avg_task_loss)
            history["ewc_loss"].append(avg_ewc_loss)
            history["total_loss"].append(avg_task_loss + avg_ewc_loss)

        return history

    def consolidate(
        self,
        data_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
    ) -> None:
        """Consolidate knowledge after completing a task.

        Computes Fisher information for the current task and registers
        it with the regularizer for future protection.

        Args:
            data_loader: Training data for the completed task.
            criterion: Loss function used during training.
        """
        fisher_diag, optimal_params = self.fisher_computer.compute(
            data_loader, criterion
        )
        self.regularizer.register_task(fisher_diag, optimal_params)
