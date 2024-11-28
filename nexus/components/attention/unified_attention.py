import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .flash_attention import FlashAttention
from .multi_head_attention import MultiHeadSelfAttention
from nexus.core.base import NexusModule

class UnifiedAttention(NexusModule):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_type: str = "default",
        use_flash_attention: bool = False,
        causal: bool = False
    ):
        super().__init__()
        
        self.attention_type = attention_type
        
        if use_flash_attention:
            self.attention = FlashAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                causal=causal
            )
        else:
            self.attention = MultiHeadSelfAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            )
            
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.attention(x, mask) 