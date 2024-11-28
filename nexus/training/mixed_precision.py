import torch
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any, Optional
from nexus.core.base import NexusModule

class MixedPrecisionTrainer:
    def __init__(
        self,
        model: NexusModule,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler or GradScaler()
        
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.optimizer.zero_grad()
        
        with autocast():
            outputs = self.model(**batch)
            loss = outputs["loss"]
        
        # Scale loss and compute gradients
        self.scaler.scale(loss).backward()
        
        # Unscale gradients and update parameters
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {"loss": loss.item()} 