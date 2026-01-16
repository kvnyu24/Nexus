import torch
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader
from ..utils.logging import Logger
from ..utils.batch_utils import BatchProcessor
from .base_trainer import BaseTrainer
from nexus.core.base import NexusModule


class Trainer(BaseTrainer):
    """Standard trainer for supervised learning tasks.

    Inherits from BaseTrainer and implements training_step and
    validation_step for typical classification/regression tasks.
    """

    def __init__(
        self,
        model: NexusModule,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[Logger] = None,
        checkpoint_dir: Optional[str] = None
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            learning_rate=learning_rate,
            device=device,
            logger=logger,
            checkpoint_dir=checkpoint_dir
        )

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Dictionary containing 'image' and 'label' tensors.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary with 'loss' key.
        """
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(**batch)

        # Calculate loss
        if self.loss_fn is not None:
            loss = self.loss_fn(outputs['logits'], batch['label'])
        else:
            loss = outputs.get('loss', outputs['logits'].mean())

        # Backward pass
        loss.backward()
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return {"loss": loss.item()}

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, float]:
        """Perform a single validation step.

        Args:
            batch: Dictionary containing 'image' and 'label' tensors.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary with 'loss', 'correct', and 'total' keys.
        """
        outputs = self.model(**batch)

        # Calculate loss
        if self.loss_fn is not None:
            loss = self.loss_fn(outputs['logits'], batch['label'])
        else:
            loss = outputs.get('loss', torch.tensor(0.0))

        # Calculate accuracy
        _, predicted = outputs['logits'].max(1)
        total = batch['label'].size(0)
        correct = predicted.eq(batch['label']).sum().item()

        return {
            "loss": loss.item() if torch.is_tensor(loss) else loss,
            "correct": correct,
            "total": total
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Legacy method for backward compatibility.

        Deprecated: Use training_step() instead.

        Args:
            batch: Dictionary containing batch data.

        Returns:
            Dictionary with 'loss' key.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move batch to device
        batch = BatchProcessor.to_device(batch, self.device)

        # Forward pass
        outputs = self.model(batch['image'])
        loss = outputs["loss"]

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        batch_size=32,
        num_epochs=10,
        loss_fn=None,
        scheduler=None,
        checkpoint_frequency: int = 1,
        **kwargs
    ):
        """Train the model with checkpointing support.

        Overrides base train() to maintain backward compatibility
        with eval_dataset parameter name.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Optional evaluation dataset.
            batch_size: Batch size for data loaders.
            num_epochs: Number of training epochs.
            loss_fn: Loss function to use.
            scheduler: Learning rate scheduler.
            checkpoint_frequency: Save checkpoint every N epochs.
            **kwargs: Additional arguments passed to base train().

        Returns:
            Dictionary containing training history.
        """
        # Use CrossEntropyLoss as default if no loss_fn provided
        if loss_fn is None:
            loss_fn = torch.nn.CrossEntropyLoss()

        return super().train(
            train_dataset=train_dataset,
            val_dataset=eval_dataset,
            batch_size=batch_size,
            num_epochs=num_epochs,
            loss_fn=loss_fn,
            scheduler=scheduler,
            checkpoint_frequency=checkpoint_frequency,
            **kwargs
        )
