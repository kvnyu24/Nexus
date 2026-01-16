import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from nexus.core.base import NexusModule
from nexus.utils.logging import Logger
from nexus.utils.attention_utils import create_causal_mask
from .base_attention import BaseAttention
from .flash_attention import FlashAttention

class MemoryEfficientAttention(BaseAttention):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        chunk_size: int = 128,
        causal: bool = False,
        bias: bool = True
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attention=False
        )
        
        self.chunk_size = chunk_size
        self.causal = causal
        self.logger = Logger(self.__class__.__name__)
        
        # Override unified QKV projection with separate ones
        self.qkv_proj = None
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight) 
        nn.init.xavier_uniform_(self.v_proj.weight)
        
        # Flash attention for when available
        self.flash_attention = FlashAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            block_size=chunk_size,
            causal=causal,
            bias=bias
        )
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Try using flash attention first
        if torch.cuda.is_available():
            try:
                return self.flash_attention(x, attention_mask)
            except Exception as e:
                self.logger.warning(f"Flash attention failed, falling back to efficient attention: {e}")

        # Fall back to efficient attention implementation
        batch_size, seq_len, _ = x.shape

        # Project queries, keys and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = self._reshape_for_attention(q)
        k = self._reshape_for_attention(k)
        v = self._reshape_for_attention(v)

        # Initialize output tensor
        out = torch.zeros_like(q)

        # Chunked attention computation
        for i in range(0, seq_len, self.chunk_size):
            chunk_q = q[:, :, i:i+self.chunk_size]

            # Compute attention scores
            attn_weights = torch.matmul(chunk_q, k.transpose(-2, -1)) * self.scale

            # Apply causal mask if needed
            if self.causal:
                causal_mask = create_causal_mask(attn_weights.size(-1), dtype=torch.bool, device=attn_weights.device)
                causal_mask = causal_mask[-attn_weights.size(-2):, :]
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

            # Apply attention mask if provided
            if attention_mask is not None:
                if attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)  # Add head dimension
                attn_weights = attn_weights + attention_mask[:, :, i:i+self.chunk_size]

            # Compute attention probabilities
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Compute chunk output
            out[:, :, i:i+self.chunk_size] = torch.matmul(attn_weights, v)

        # Reshape and project output
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out
