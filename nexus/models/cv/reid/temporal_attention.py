import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from ....core.base import NexusModule

class TemporalAttention(NexusModule):
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
            
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention with improved initialization
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            add_bias_kv=True,
            add_zero_attn=True,
            batch_first=False
        )
        
        # Normalization and regularization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Optional feedforward network for enhanced feature transformation
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced temporal attention with residual connections and FFN
        
        Args:
            x: Input tensor of shape (sequence_length, batch_size, hidden_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of:
            - Output tensor of shape (sequence_length, batch_size, hidden_dim)
            - Attention weights of shape (batch_size, num_heads, sequence_length, sequence_length)
        """
        # Input validation
        if not torch.isfinite(x).all():
            raise ValueError("Input contains inf or nan values")
            
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got shape {x.shape}")
            
        if x.size(-1) != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim {self.hidden_dim}, got {x.size(-1)}")
            
        # Multi-head attention with skip connection
        attended, weights = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=mask,
            need_weights=True,
            average_attn_weights=False
        )
        
        x = self.norm1(x + self.dropout1(attended))
        
        # FFN with skip connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x, weights