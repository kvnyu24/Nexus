import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from .flash_attention import FlashAttention
from .multi_head_attention import MultiHeadSelfAttention
from .efficient_attention import MemoryEfficientAttention
from nexus.core.base import NexusModule

class UnifiedAttention(NexusModule):
    """A unified attention module that supports multiple attention implementations.
    
    This module provides a unified interface for different attention mechanisms including:
    - Standard multi-head self attention
    - Flash attention for improved memory efficiency
    - Memory efficient attention using chunked computation
    
    Args:
        hidden_size (int): Size of hidden dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
        attention_type (str): Type of attention - "default", "flash", or "efficient"
        use_flash_attention (bool): Whether to use flash attention when available
        causal (bool): Whether to apply causal masking
        chunk_size (int): Chunk size for efficient attention computation
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_type: str = "default",
        use_flash_attention: bool = False,
        causal: bool = False,
        chunk_size: int = 128
    ):
        super().__init__()
        
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
            
        self.attention_type = attention_type
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        if attention_type == "efficient":
            self.attention = MemoryEfficientAttention(
                hidden_size=hidden_size,
                num_heads=num_heads, 
                dropout=dropout,
                chunk_size=chunk_size
            )
        elif attention_type == "flash" or use_flash_attention:
            if not torch.cuda.is_available():
                print("Warning: Flash attention requested but CUDA not available. Falling back to standard attention.")
                self.attention = MultiHeadSelfAttention(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout
                )
            else:
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
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Forward pass of the attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size)
            mask (Optional[torch.Tensor]): Attention mask
            **kwargs: Additional arguments passed to the underlying attention implementation
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        if x.size(-1) != self.hidden_size:
            raise ValueError(f"Input hidden size {x.size(-1)} doesn't match configured hidden_size {self.hidden_size}")
            
        return self.attention(x, mask)