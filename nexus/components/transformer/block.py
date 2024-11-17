import torch
import torch.nn as nn
from typing import Optional, Dict
from ..attention.base import UnifiedAttention

class UnifiedTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_flash_attention: bool = False,
        layer_scale: float = 1e-5,
        attention_type: str = "default",
        causal: bool = False
    ):
        super().__init__()
        
        # Attention
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = UnifiedAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=attention_dropout,
            attention_type=attention_type,
            use_flash_attention=use_flash_attention,
            causal=causal
        )
        
        # MLP
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Layer scaling
        self.layer_scale = layer_scale
        if layer_scale > 0:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_size))
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(hidden_size))
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Attention block
        attn_out = self.attention(self.norm1(x), attention_mask)
        if self.layer_scale > 0:
            x = x + self.gamma1 * attn_out
        else:
            x = x + attn_out
            
        # MLP block
        mlp_out = self.mlp(self.norm2(x))
        if self.layer_scale > 0:
            x = x + self.gamma2 * mlp_out
        else:
            x = x + mlp_out
            
        return {"hidden_states": x} 