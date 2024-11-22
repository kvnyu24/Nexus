import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ....core.base import NexusModule

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x shape: (sequence_length, batch_size, hidden_dim)
        attended, weights = self.attention(x, x, x, key_padding_mask=mask)
        return self.norm(attended + x), weights 