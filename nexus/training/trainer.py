import torch
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader
from ..utils.logging import Logger
from .checkpointing import CheckpointMixin

class Trainer(CheckpointMixin):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[Logger] = None,
        checkpoint_dir: Optional[str] = None
    ):
        self.model = model
        self.device = device
        self.logger = logger or Logger()
        self.checkpoint_dir = checkpoint_dir or "checkpoints"
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer(optimizer, learning_rate)
        self.model.to(device)
        
    def _setup_optimizer(self, optimizer_name: str, lr: float):
        if optimizer_name.lower() == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name.lower() == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
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
        """Train the model with checkpointing support."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        if eval_dataset:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        # Use CrossEntropyLoss as default if no loss_fn provided
        if loss_fn is None:
            loss_fn = torch.nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            num_batches = len(train_loader)
            
            # Add progress tracking
            for batch_idx, batch in enumerate(train_loader):
                # Handle both tuple and dict style batches
                if isinstance(batch, (list, tuple)):
                    images, labels = batch
                    batch = {'image': images, 'label': labels}
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                # Calculate loss using the loss function
                loss = loss_fn(outputs['logits'], batch['label'])
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if scheduler:
                    scheduler.step()
                
                total_loss += loss.item()
                
                # Log batch-level progress
                if (batch_idx + 1) % kwargs.get('log_interval', 10) == 0:
                    self.logger.info(
                        f"Epoch [{epoch+1}/{num_epochs}] "
                        f"Batch [{batch_idx+1}/{num_batches}] "
                        f"Loss: {loss.item():.4f}"
                    )
            
            # Log epoch-level metrics
            avg_loss = total_loss / len(train_loader)
            metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} "
                f"Average Loss: {avg_loss:.4f} "
                f"LR: {metrics['learning_rate']:.6f}"
            )
            
            # Evaluation
            if eval_dataset:
                self.model.eval()
                eval_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch in eval_loader:
                        # Handle both tuple and dict style batches
                        if isinstance(batch, (list, tuple)):
                            images, labels = batch
                            batch = {'image': images, 'label': labels}
                        
                        # Move batch to device
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        outputs = self.model(**batch)
                        
                        if loss_fn:
                            loss = loss_fn(outputs['logits'], batch['label'])
                        else:
                            loss = outputs.get('loss', 0)
                            
                        eval_loss += loss.item()
                        
                        _, predicted = outputs['logits'].max(1)
                        total += batch['label'].size(0)
                        correct += predicted.eq(batch['label']).sum().item()
                
                accuracy = 100. * correct / total
                avg_eval_loss = eval_loss / len(eval_loader)
                self.logger.info(f"Eval Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            # Save checkpoint if needed
            if checkpoint_frequency > 0 and (epoch + 1) % checkpoint_frequency == 0:
                metrics = {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                }
                if eval_dataset:
                    metrics.update({
                        "eval_loss": avg_eval_loss,
                        "accuracy": accuracy
                    })
                    
                checkpoint_path = self.save_checkpoint(
                    self.checkpoint_dir,
                    epoch + 1,
                    metrics
                )
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")
