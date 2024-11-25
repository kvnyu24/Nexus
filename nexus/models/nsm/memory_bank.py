import torch
import torch.nn as nn
from typing import Dict, Any
from ...core.base import NexusModule

class MemoryBank(NexusModule):
    def __init__(self, hidden_dim: int, bank_size: int):
        super().__init__({})
        
        # Initialize memory bank (following EnhancedReID pattern)
        self.register_buffer(
            "bank",
            torch.zeros(bank_size, hidden_dim)
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
    def update(self, features: torch.Tensor):
        batch_size = features.size(0)
        ptr = int(self.bank_ptr)
        
        # Update bank with new features
        if ptr + batch_size > self.bank.size(0):
            ptr = 0
            
        self.bank[ptr:ptr + batch_size] = features.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.bank.size(0)
        
    def get_bank(self) -> torch.Tensor:
        return self.bank 