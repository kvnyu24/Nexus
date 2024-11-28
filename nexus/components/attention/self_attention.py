import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from nexus.core.base import NexusModule

class MultiHeadSelfAttention(NexusModule):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ):
        batch_size = hidden_states.size(0)
        
        # Project queries, keys, and values
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float)
        )
        
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf")
            )
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Compute output
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)
        
        output = self.out_proj(context)
        
        if return_attention:
            return output, attention_probs
        return output 