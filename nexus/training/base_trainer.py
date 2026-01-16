"""Base trainer module providing abstract training interface."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, Tuple, List

import torch
from torch.utils.data import DataLoader

from .checkpointing import CheckpointMixin
from ..utils.logging import Logger
from ..utils.batch_utils import BatchProcessor
from ..core.base import NexusModule


class BaseTrainer(CheckpointMixin, ABC):
    """Abstract base class for all trainers in Nexus.

    Provides common training infrastructure including:
    - Optimizer setup and configuration
    - Batch processing via BatchProcessor
    - Checkpointing support via CheckpointMixin
    - Template train() method for training loops

    Subclasses must implement:
    - training_step(): Single training step logic
    - validation_step(): Single validation step logic
    """

    def __init__(
        self,
        model: NexusModule,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[Logger] = None,
        checkpoint_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the base trainer.

        Args:
            model: The NexusModule model to train.
            optimizer: Optimizer name ('adam', 'sgd', 'adamw').
            learning_rate: Learning rate for the optimizer.
            device: Device to train on ('cuda' or 'cpu').
            logger: Optional logger instance for logging.
            checkpoint_dir: Directory for saving checkpoints.
            config: Optional configuration dictionary.
        """
        self.model = model
        self.device = device
        self.logger = logger or Logger()
        self.checkpoint_dir = checkpoint_dir or "checkpoints"
        self.config = config or {}
        self.scheduler = None

        # Validate configuration
        self._validate_config()

        # Setup optimizer
        self.optimizer = self._setup_optimizer(optimizer, learning_rate)
        self.model.to(device)

    def _validate_config(self) -> None:
        """Validate trainer configuration.

        Override in subclasses to add specific validation logic.
        Raises ValueError if configuration is invalid.
        """
        if self.config.get("learning_rate", 1e-3) <= 0:
            raise ValueError("learning_rate must be positive")

        if self.config.get("batch_size", 32) <= 0:
            raise ValueError("batch_size must be positive")

    def _setup_optimizer(
        self,
        optimizer_name: str,
        lr: float,
        weight_decay: float = 0.0
    ) -> torch.optim.Optimizer:
        """Set up the optimizer for training.

        Args:
            optimizer_name: Name of optimizer ('adam', 'sgd', 'adamw').
            lr: Learning rate.
            weight_decay: Weight decay for regularization.

        Returns:
            Configured optimizer instance.

        Raises:
            ValueError: If optimizer_name is not supported.
        """
        optimizer_name = optimizer_name.lower()

        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=self.config.get("momentum", 0.9)
            )
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay or 0.01
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _prepare_batch(
        self,
        batch: Union[Dict[str, Any], Tuple, List],
        keys: Tuple[str, str] = ('image', 'label')
    ) -> Dict[str, Any]:
        """Prepare a batch for training/validation.

        Uses BatchProcessor to normalize and move batch to device.

        Args:
            batch: Input batch (dict, tuple, or list).
            keys: Keys to use for tuple/list batches.

        Returns:
            Normalized batch dictionary on the correct device.
        """
        return BatchProcessor.prepare_batch(batch, self.device, keys)

    @abstractmethod
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Dictionary containing batch data.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary of metrics (must include 'loss').
        """
        pass

    @abstractmethod
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, float]:
        """Perform a single validation step.

        Args:
            batch: Dictionary containing batch data.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary of metrics.
        """
        pass

    def on_epoch_start(self, epoch: int) -> None:
        """Hook called at the start of each epoch.

        Override in subclasses for custom behavior.

        Args:
            epoch: Current epoch number (0-indexed).
        """
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Hook called at the end of each epoch.

        Override in subclasses for custom behavior.

        Args:
            epoch: Current epoch number (0-indexed).
            metrics: Dictionary of metrics from the epoch.
        """
        pass

    def train(
        self,
        train_dataset,
        val_dataset=None,
        batch_size: int = 32,
        num_epochs: int = 10,
        loss_fn=None,
        scheduler=None,
        checkpoint_frequency: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the model with checkpointing support.

        Template method that orchestrates the training loop. Calls
        training_step() for each batch and validation_step() for
        evaluation.

        Args:
            train_dataset: Training dataset.
            val_dataset: Optional evaluation dataset.
            batch_size: Batch size for data loaders.
            num_epochs: Number of training epochs.
            loss_fn: Optional loss function (if not computed in training_step).
            scheduler: Optional learning rate scheduler.
            checkpoint_frequency: Save checkpoint every N epochs (0 to disable).
            **kwargs: Additional arguments (log_interval, etc.).

        Returns:
            Dictionary containing training history.
        """
        self.scheduler = scheduler
        self.loss_fn = loss_fn

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=kwargs.get('num_workers', 0),
            pin_memory=kwargs.get('pin_memory', False)
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=kwargs.get('num_workers', 0),
                pin_memory=kwargs.get('pin_memory', False)
            )

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        log_interval = kwargs.get('log_interval', 10)

        for epoch in range(num_epochs):
            self.on_epoch_start(epoch)

            # Training phase
            self.model.train()
            total_loss = 0.0
            num_batches = len(train_loader)

            for batch_idx, batch in enumerate(train_loader):
                batch = self._prepare_batch(batch)
                metrics = self.training_step(batch, batch_idx)
                total_loss += metrics['loss']

                # Log batch progress
                if (batch_idx + 1) % log_interval == 0:
                    self.logger.info(
                        f"Epoch [{epoch+1}/{num_epochs}] "
                        f"Batch [{batch_idx+1}/{num_batches}] "
                        f"Loss: {metrics['loss']:.4f}"
                    )

            # Compute epoch metrics
            avg_train_loss = total_loss / num_batches
            current_lr = self.optimizer.param_groups[0]['lr']

            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'learning_rate': current_lr
            }

            history['train_loss'].append(avg_train_loss)
            history['learning_rate'].append(current_lr)

            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} "
                f"Average Loss: {avg_train_loss:.4f} "
                f"LR: {current_lr:.6f}"
            )

            # Validation phase
            if val_loader:
                val_metrics = self._run_validation(val_loader)
                epoch_metrics.update(val_metrics)
                history['val_loss'].append(val_metrics.get('val_loss', 0))
                history['val_accuracy'].append(val_metrics.get('accuracy', 0))

                self.logger.info(
                    f"Val Loss: {val_metrics.get('val_loss', 0):.4f}, "
                    f"Accuracy: {val_metrics.get('accuracy', 0):.2f}%"
                )

            self.on_epoch_end(epoch, epoch_metrics)

            # Save checkpoint
            if checkpoint_frequency > 0 and (epoch + 1) % checkpoint_frequency == 0:
                checkpoint_path = self.save_checkpoint(
                    self.checkpoint_dir,
                    epoch + 1,
                    epoch_metrics
                )
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        return history

    def _run_validation(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation loop.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = self._prepare_batch(batch)
                metrics = self.validation_step(batch, batch_idx)

                total_loss += metrics.get('loss', 0)
                total_correct += metrics.get('correct', 0)
                total_samples += metrics.get('total', batch.get('label', torch.tensor([])).size(0))

        num_batches = len(val_loader)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0

        return {
            'val_loss': avg_loss,
            'accuracy': accuracy
        }
