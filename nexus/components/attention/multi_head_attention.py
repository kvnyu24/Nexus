import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .base_attention import BaseAttention
from ...core.initialization import WeightInitializer
from .rotary_embedding import RotaryEmbedding

class MultiHeadAttention(BaseAttention):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        add_zero_attn: bool = False,
        use_rotary: bool = False,
        attention_scale: Optional[float] = None
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            attention_scale=attention_scale
        )
        
        self.add_zero_attn = add_zero_attn
        self.use_rotary = use_rotary
        
        if use_rotary:
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=2048
            )
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project query, key, value
        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings if enabled
        if self.use_rotary:
            query, key = self.rotary_emb(query, key)
            
        # Add zero attention if requested
        if self.add_zero_attn:
            zero_attn_shape = (batch_size, 1, self.num_heads, self.head_dim)
            key = torch.cat([key, torch.zeros(zero_attn_shape, dtype=key.dtype, device=key.device)], dim=1)
            value = torch.cat([value, torch.zeros(zero_attn_shape, dtype=value.dtype, device=value.device)], dim=1)
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, 1), value=1)
                
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        if position_bias is not None:
            attention_scores = attention_scores + position_bias
            
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Convert scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Compute output
        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_length, self.hidden_size)
        
        output = self.out_proj(context)
        
        if use_cache:
            return output, (key, value)
        return output, None 