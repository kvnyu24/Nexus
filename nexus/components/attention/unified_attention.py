import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Tuple
from .flash_attention import FlashAttention
from .multi_head_attention import MultiHeadSelfAttention
from .efficient_attention import MemoryEfficientAttention
from nexus.core.base import NexusModule
from nexus.core.initialization import WeightInitializer
from nexus.utils.logging import Logger

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
        qkv_bias (bool): Whether to use bias in QKV projections
        return_attention_weights (bool): Whether to return attention weights
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_type: str = "default",
        use_flash_attention: bool = False,
        causal: bool = False,
        chunk_size: int = 128,
        qkv_bias: bool = True,
        return_attention_weights: bool = False
    ):
        super().__init__()
        
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
            
        self.attention_type = attention_type
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.return_attention_weights = return_attention_weights
        self.logger = Logger(self.__class__.__name__)
        
        if attention_type == "efficient":
            self.attention = MemoryEfficientAttention(
                hidden_size=hidden_size,
                num_heads=num_heads, 
                dropout=dropout,
                chunk_size=chunk_size,
                causal=causal,
                bias=qkv_bias
            )
        elif attention_type == "flash" or use_flash_attention:
            if not torch.cuda.is_available():
                self.logger.warning("Flash attention requested but CUDA not available. Falling back to standard attention.")
                self.attention = MultiHeadSelfAttention(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    causal=causal,
                    qkv_bias=qkv_bias
                )
            else:
                self.attention = FlashAttention(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    causal=causal,
                    bias=qkv_bias
                )
        else:
            self.attention = MultiHeadSelfAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                causal=causal,
                qkv_bias=qkv_bias
            )
            
        # Initialize weights using custom initializer
        WeightInitializer.initialize_weights(
            self,
            method='xavier_uniform',
            nonlinearity='linear'
        )
            
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, seq_len, seq_len)
            **kwargs: Additional arguments passed to the underlying attention implementation

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - Output tensor of shape (batch_size, seq_len, hidden_size)
                - Attention weights if return_attention_weights is True
        """
        if x.size(-1) != self.hidden_size:
            raise ValueError(f"Input hidden size {x.size(-1)} doesn't match configured hidden_size {self.hidden_size}")

        if self.return_attention_weights:
            output, attn_weights = self.attention(x, attention_mask, return_attention=True)
            return output, attn_weights

        return self.attention(x, attention_mask)