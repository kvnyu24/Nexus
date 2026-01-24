"""
Grouped Query Attention (GQA) implementation.

GQA shares key-value heads across multiple query heads, reducing KV cache size
while maintaining model quality. This is the standard attention mechanism in
modern LLMs like Llama 2/3, Mistral, and Qwen.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class GroupedQueryAttention(NexusModule):
    """Grouped Query Attention (GQA).

    Shares KV heads across query head groups for memory efficiency.
    When num_kv_heads == num_heads, this is standard MHA.
    When num_kv_heads == 1, this is Multi-Query Attention (MQA).

    Used by: Llama 2/3, Mistral, Qwen, Gemma

    Reference: https://arxiv.org/abs/2305.13245

    Args:
        dim: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads (must divide num_heads evenly)
        head_dim: Dimension per head (if None, uses dim // num_heads)
        dropout: Attention dropout probability
        bias: Whether to use bias in projections
        max_position_embeddings: Maximum sequence length for RoPE cache
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        max_position_embeddings: int = 8192
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings

        # Validate that num_heads is divisible by num_kv_heads
        assert num_heads % num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"

        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5

        # Projection layers
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match number of query heads.

        Args:
            x: Tensor of shape (batch, num_kv_heads, seq_len, head_dim)
            n_rep: Number of times to repeat each KV head

        Returns:
            Tensor of shape (batch, num_heads, seq_len, head_dim)
        """
        if n_rep == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

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
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, dim)
            attention_mask: Attention mask of shape (batch, 1, seq_len, seq_len)
            position_embeddings: Tuple of (cos, sin) for RoPE
            past_key_value: Cached (key, value) tensors for incremental decoding
            use_cache: Whether to return updated cache
            output_attentions: Whether to return attention weights

        Returns:
            output: Attention output of shape (batch, seq_len, dim)
            attn_weights: Attention weights if output_attentions=True
            past_key_value: Updated cache if use_cache=True
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV heads to match query heads
        key_states = self._repeat_kv(key_states, self.num_kv_groups)
        value_states = self._repeat_kv(value_states, self.num_kv_groups)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to Q and K."""

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


class GQA(GroupedQueryAttention):
    """Alias for GroupedQueryAttention."""
    pass
