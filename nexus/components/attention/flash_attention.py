import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from einops import rearrange
from nexus.core.base import NexusModule

class FlashAttention(NexusModule):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        block_size: int = 1024,
        causal: bool = False
    ):
        super().__init__()
        
        # Validate input parameters
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
            
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.block_size = block_size
        self.causal = causal
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
            
        B, N, C = x.shape
        if C != self.hidden_size:
            raise ValueError(f"Input hidden size {C} doesn't match configured hidden_size {self.hidden_size}")
            
        # QKV transform
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d', 
                       three=3, h=self.num_heads)
        q, k, v = qkv.unbind(0)
        
        # Split sequence into blocks for efficient attention
        blocks = math.ceil(N / self.block_size)
        attention_outputs = []
        
        for i in range(blocks):
            start_idx = i * self.block_size
            end_idx = min((i + 1) * self.block_size, N)
            
            block_q = q[:, :, start_idx:end_idx]
            
            # For causal attention, only attend to previous blocks
            if self.causal:
                block_k = k[:, :, :end_idx]
                block_v = v[:, :, :end_idx]
            else:
                block_k = k
                block_v = v
            
            # Efficient attention computation
            attn_weights = torch.matmul(block_q, block_k.transpose(-2, -1)) * self.scale
            
            if self.causal and start_idx > 0:
                causal_mask = torch.ones(
                    (end_idx - start_idx, N), 
                    dtype=torch.bool, 
                    device=x.device
                ).triu_(start_idx)
                attn_weights.masked_fill_(causal_mask, float('-inf'))
            
            if mask is not None:
                if mask.dim() != 4:
                    raise ValueError(f"Attention mask should be 4D, got {mask.dim()}D")
                if mask.size(0) != B:
                    raise ValueError(f"Mask batch size {mask.size(0)} doesn't match input batch size {B}")
                attn_weights = attn_weights + mask[:, :, start_idx:end_idx]
            
            # Prevent NaN outputs from all -inf attention weights
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
            
            if self.dropout > 0 and self.training:
                attn_weights = nn.functional.dropout(
                    attn_weights,
                    p=self.dropout,
                    training=True
                )
            
            block_output = torch.matmul(attn_weights, block_v)
            attention_outputs.append(block_output)
        
        # Combine block outputs
        output = torch.cat(attention_outputs, dim=2)
        output = rearrange(output, 'b h n d -> b n (h d)')
        
        return self.out_proj(output)