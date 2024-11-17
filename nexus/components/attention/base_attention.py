import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from ...core.base import NexusModule


class BaseAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        use_flash_attention: bool = False,
        attention_scale: float = None
    ):
        super().__init__()
        
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            )
            
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_flash_attention = use_flash_attention
        self.scale = attention_scale or (self.head_dim ** -0.5)
        
        # Unified projections
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        # Try importing flash attention if requested
        if use_flash_attention:
            if torch.cuda.is_available():
                try:
                    from flash_attn import flash_attn_func
                    self.flash_attn_func = flash_attn_func
                except ImportError:
                    self.use_flash_attention = False
                    print("Flash Attention not available, falling back to standard attention")
        
    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.size()
        
        # Input validation
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                f"attention_mask should be 2D, got {attention_mask.dim()}D"
            )
        
        # Unified QKV projection
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for attention
        q = self._reshape_for_attention(q)
        k = self._reshape_for_attention(k)
        v = self._reshape_for_attention(v)
        
        if self.use_flash_attention and torch.cuda.is_available():
            # Use Flash Attention if available
            output = self.flash_attn_func(q, k, v, dropout_p=self.dropout.p)
            attention_weights = None
        else:
            # Standard scaled dot-product attention
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                # Expand mask for multiple heads
                expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_scores = attention_scores.masked_fill(
                    expanded_mask == 0,
                    float("-inf")
                )
            
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            output = torch.matmul(attention_weights, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.hidden_size)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attention_weights
        return output 