import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from ...core.base import NexusModule

class MemoryBank(NexusModule):
    def __init__(self, hidden_dim: int, bank_size: int):
        super().__init__({})
        
        # Initialize memory bank (following EnhancedReID pattern)
        self.register_buffer(
            "feature_bank",
            torch.zeros(bank_size, hidden_dim)
        )
        self.register_buffer("importance_scores", torch.zeros(bank_size))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
        # Memory compression (following EnhancedSFT pattern)
        self.compressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Importance estimation (following DecisionMakingModule pattern)
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def update(self, features: torch.Tensor) -> None:
        """Update memory bank following EnhancedReID pattern"""
        batch_size = features.size(0)
        ptr = int(self.bank_ptr)
        
        # Compress features
        compressed_features = self.compressor(features)
        
        # Calculate importance scores
        importance = self.importance_head(features).squeeze(-1)
        
        # Update bank
        if ptr + batch_size > self.feature_bank.size(0):
            ptr = 0
            
        self.feature_bank[ptr:ptr + batch_size] = compressed_features.detach()
        self.importance_scores[ptr:ptr + batch_size] = importance.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.feature_bank.size(0)
        
    def get_bank(self, top_k: Optional[int] = None) -> torch.Tensor:
        """Get memory bank features, optionally returning top-k most important"""
        if top_k is not None:
            # Get top-k most important memories
            _, indices = torch.topk(self.importance_scores, k=min(top_k, len(self.importance_scores)))
            return self.feature_bank[indices]
        return self.feature_bank