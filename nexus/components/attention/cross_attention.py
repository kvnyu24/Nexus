import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from nexus.core.base import NexusModule
from .base_attention import BaseAttention
from ..embeddings import RotaryEmbedding, apply_rotary_pos_emb


class CrossAttention(BaseAttention):
    def __init__(
        self,
        query_dim: int,
        key_dim: int, 
        num_heads: int,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
        max_seq_len: int = 2048
    ):
        super().__init__(
            hidden_size=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attention=use_flash_attention
        )
        
        # Override QKV projections for cross attention
        self.qkv_proj = None
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(key_dim, query_dim, bias=False)
        self.to_v = nn.Linear(key_dim, query_dim, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.dim() != 3 or context.dim() != 3:
            raise ValueError(f"Expected 3D tensors (batch, seq_len, dim), got x: {x.shape}, context: {context.shape}")

        batch_size, seq_len, _ = x.shape

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = self._reshape_for_attention(q)
        k = self._reshape_for_attention(k)
        v = self._reshape_for_attention(v)

        # Apply rotary embeddings
        sin, cos = self.rotary_emb(x)
        q, k = apply_rotary_pos_emb(q, k, sin, cos)

        if self.use_flash_attention and torch.cuda.is_available():
            output = self.flash_attn_func(q, k, v, dropout_p=self.dropout.p)
            attention_weights = None
        else:
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if attention_mask is not None:
                if attention_mask.dim() != 2:
                    raise ValueError(f"Expected 2D attention_mask, got shape {attention_mask.shape}")
                attention_scores = attention_scores.masked_fill(
                    attention_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf')
                )

            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            output = torch.matmul(attention_weights, v)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.hidden_size)
        output = self.out_proj(output)

        if return_attention:
            return output, attention_weights
        return output


class CrossAttentionLayer(NexusModule):
    """Cross attention layer with config-based initialization.

    This is a wrapper around CrossAttention for compatibility with
    config-dict-based model definitions.
    """

    def __init__(self, config: dict):
        super().__init__()
        hidden_size = config.get("hidden_size", 256)
        num_heads = config.get("num_heads", 8)
        dropout = config.get("dropout", 0.1)

        self.attention = CrossAttention(
            query_dim=hidden_size,
            key_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm style
        return x + self.attention(self.norm(x), context, attention_mask)