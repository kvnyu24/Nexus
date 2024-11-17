import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class CosineWarmupScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 1e-7
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer)
        
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            lr_scale = step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            lr_scale = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr_scale = max(lr_scale, self.min_lr)
            
        return [base_lr * lr_scale for base_lr in self.base_lrs] 