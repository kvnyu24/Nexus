import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from nexus.core.base import NexusModule
class CrossAttention(NexusModule):
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
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
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attention = attention.masked_fill(mask.unsqueeze(1), float('-inf'))
            
        attention = F.softmax(attention, dim=-1)
        
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        out = self.to_out(out)
        
        return out, attention 