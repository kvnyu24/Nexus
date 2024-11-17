import torch
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader
from ..utils.logging import Logger

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[Logger] = None
    ):
        self.model = model
        self.device = device
        self.logger = logger or Logger()
        
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
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs["loss"]
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()} 