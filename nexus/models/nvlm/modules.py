from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from nexus.core.base import NexusModule

class CrossAttentionLayer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        hidden_size = config.get("hidden_size", 256)
        num_heads = config.get("num_heads", 8)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=config.get("attention_dropout", 0.1),
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
    def forward(self, x: torch.Tensor, context: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Cross attention
        residual = x
        x = self.norm1(x)
        x = self.cross_attn(x, context, context, 
                           attn_mask=attention_mask)[0]
        x = x + residual
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x 