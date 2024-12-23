import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from nexus.core.base import NexusModule
from .flash_attention import FlashAttention

class MemoryEfficientAttention(NexusModule):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        chunk_size: int = 128,
        causal: bool = False,
        bias: bool = True
    ):
        super().__init__()
        
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
            
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        
        # Separate QKV projections for better control
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight) 
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        # Flash attention for when available
        self.flash_attention = FlashAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            block_size=chunk_size,
            causal=causal,
            bias=bias
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Try using flash attention first
        if torch.cuda.is_available():
            try:
                return self.flash_attention(x, mask)
            except Exception as e:
                print(f"Flash attention failed, falling back to efficient attention: {e}")
                
        # Fall back to efficient attention implementation
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Initialize output tensor
        out = torch.zeros_like(q)
        
        # Chunked attention computation
        for i in range(0, seq_len, self.chunk_size):
            chunk_q = q[:, :, i:i+self.chunk_size]
            
            # Compute attention scores
            attn_weights = torch.matmul(chunk_q, k.transpose(-2, -1)) * self.scale
            
            # Apply causal mask if needed
            if self.causal:
                causal_mask = torch.triu(
                    torch.ones((attn_weights.size(-2), attn_weights.size(-1)), 
                             dtype=torch.bool, device=attn_weights.device),
                    diagonal=1
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            
            # Apply attention mask if provided
            if mask is not None:
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)  # Add head dimension
                attn_weights = attn_weights + mask[:, :, i:i+self.chunk_size]
                
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
