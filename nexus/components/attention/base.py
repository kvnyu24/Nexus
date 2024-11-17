import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union
from ...core.initialization import WeightInitializer

class UnifiedAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        attention_type: str = "default",
        use_flash_attention: bool = False,
        causal: bool = False,
        qkv_bias: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_type = attention_type
        self.use_flash_attention = use_flash_attention
        self.causal = causal
        
        # Unified QKV projection
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        WeightInitializer.initialize_weights(
            self,
            method='xavier_normal',
            nonlinearity='linear'
        )
    
    def _reshape_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, N, C = x.shape
        
        # Get query, key, value projections
        q, k, v = self._reshape_qkv(x)
        
        if self.use_flash_attention and torch.cuda.is_available():
            from flash_attn import flash_attn_func
            output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout.p,
                causal=self.causal
            )
            attention_weights = None
        else:
            # Standard scaled dot-product attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                attn = attn.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))
            
            if self.causal:
                causal_mask = torch.triu(
                    torch.ones(N, N, dtype=torch.bool, device=x.device), 
                    diagonal=1
                )
                attn = attn.masked_fill(causal_mask, float('-inf'))
            
            attention_weights = F.softmax(attn, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            output = attention_weights @ v
        
        # Reshape and project output
        output = output.transpose(1, 2).reshape(B, N, C)
        output = self.proj(output)
        output = self.dropout(output)
        
        if return_attention:
            return output, attention_weights
        return output 