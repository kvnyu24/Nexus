import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from nexus.core.base import NexusModule
from ..embeddings import RotaryEmbedding, apply_rotary_pos_emb


class CrossAttention(NexusModule):
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        max_seq_len: int = 2048
    ):
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError(f"query_dim {query_dim} must be divisible by num_heads {num_heads}")
            
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=bias)
        self.to_k = nn.Linear(key_dim, query_dim, bias=bias)
        self.to_v = nn.Linear(key_dim, query_dim, bias=bias)
        self.to_out = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or context.dim() != 3:
            raise ValueError(f"Expected 3D tensors (batch, seq_len, dim), got x: {x.shape}, context: {context.shape}")
            
        batch_size, seq_len, _ = x.shape
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        sin, cos = self.rotary_emb(x)
        q, k = apply_rotary_pos_emb(q, k, sin, cos)
        
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            if mask.dim() != 2:
                raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
            attention = attention.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
        attention = self.dropout(F.softmax(attention, dim=-1))
        
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        out = self.to_out(out)
        
        return out, attention