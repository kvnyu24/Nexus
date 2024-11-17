import torch
import torch.nn as nn
from ..attention import MultiHeadSelfAttention, CrossAttention, MemoryEfficientAttention
from typing import Optional

class MultiModalTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        cross_attention_size: int,
        num_heads: int,
        dropout: float = 0.1,
        use_memory_efficient: bool = True
    ):
        super().__init__()
        
        # Self attention
        self.self_attn = (
            MemoryEfficientAttention(hidden_size, num_heads, dropout)
            if use_memory_efficient else
            MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        )
        
        # Cross attention
        self.cross_attn = CrossAttention(
            query_dim=hidden_size,
            key_dim=cross_attention_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Processing layers
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self attention
        x = x + self.self_attn(self.norm1(x), mask=self_mask)
        
        # Cross attention (if context is provided)
        if context is not None:
            x = x + self.cross_attn(self.norm2(x), context, mask=cross_mask)[0]
            
        # MLP
        x = x + self.mlp(self.norm3(x))
        
        return x 