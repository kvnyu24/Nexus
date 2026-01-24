"""
Differential Attention implementation.

Computes the difference between two attention patterns to reduce noise and
improve signal quality. Introduced by Microsoft Research in 2024.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class DifferentialAttention(NexusModule):
    """Differential Attention.

    Computes attention as the difference between two separate attention patterns,
    helping to cancel out noise and focus on more relevant patterns.

    Formula: attn = softmax(Q1 @ K1.T) - λ * softmax(Q2 @ K2.T)

    Reference: https://arxiv.org/abs/2410.05258

    Args:
        dim: Model dimension
        num_heads: Number of attention heads (each head has 2 sub-heads)
        head_dim: Dimension per head
        dropout: Attention dropout probability
        bias: Whether to use bias in projections
        lambda_init: Initial value for lambda parameter
        lambda_learnable: Whether lambda is learnable
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        lambda_init: float = 0.8,
        lambda_learnable: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.dropout = dropout

        # Each "head" actually has 2 sub-heads for differential computation
        self.num_sub_heads = num_heads * 2
        self.scale = self.head_dim ** -0.5

        # Projections for both sub-attention patterns
        self.q_proj = nn.Linear(dim, self.num_sub_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.num_sub_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=bias)

        # Lambda parameter for weighting the subtraction
        if lambda_learnable:
            self.lambda_param = nn.Parameter(torch.ones(num_heads) * lambda_init)
        else:
            self.register_buffer('lambda_param', torch.ones(num_heads) * lambda_init)

        # Optional: Per-head lambda with learned scaling
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.ones(self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.ones(self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim))

        self.attn_dropout = nn.Dropout(dropout)

        # Layer norm for sub-attention (helps stabilize the differential)
        self.sub_norm = nn.LayerNorm(2 * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass computing differential attention.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape Q and K for sub-heads: (batch, seq, 2*num_heads, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_sub_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_sub_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch, heads, seq, dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply RoPE if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Split into two sub-attention patterns
        # Shape: (batch, num_heads, seq, head_dim) for each
        q1, q2 = query_states.chunk(2, dim=1)
        k1, k2 = key_states.chunk(2, dim=1)

        # Compute both attention patterns
        attn_weights_1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        attn_weights_2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale

        # Apply mask to both
        if attention_mask is not None:
            attn_weights_1 = attn_weights_1 + attention_mask
            attn_weights_2 = attn_weights_2 + attention_mask

        # Softmax
        attn_weights_1 = F.softmax(attn_weights_1, dim=-1, dtype=torch.float32)
        attn_weights_2 = F.softmax(attn_weights_2, dim=-1, dtype=torch.float32)

        # Compute lambda for each head
        # lambda = exp(lambda_q1 @ lambda_k1 - lambda_q2 @ lambda_k2)
        lambda_full = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1) -
            torch.sum(self.lambda_q2 * self.lambda_k2)
        )

        # Differential attention: attn1 - lambda * attn2
        attn_weights = attn_weights_1 - lambda_full * attn_weights_2
        attn_weights = attn_weights.to(query_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


class DiffAttn(DifferentialAttention):
    """Alias for DifferentialAttention."""
    pass
