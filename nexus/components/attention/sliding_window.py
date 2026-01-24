"""
Sliding Window Attention implementation.

Restricts attention to a local window for linear complexity in sequence length.
Commonly combined with GQA in models like Mistral and Gemma.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class SlidingWindowAttention(NexusModule):
    """Sliding Window Attention.

    Each token attends only to the previous `window_size` tokens, reducing
    complexity from O(n²) to O(n * window_size).

    Used by: Mistral, Gemma 2, Longformer (local attention component)

    Reference: https://arxiv.org/abs/2310.06825 (Mistral)

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        window_size: Size of the sliding window (number of tokens to attend to)
        num_kv_heads: Number of KV heads (for GQA, default same as num_heads)
        head_dim: Dimension per head
        dropout: Attention dropout probability
        bias: Whether to use bias in projections
        causal: Whether to use causal masking within the window
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 4096,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.window_size = window_size
        self.dropout = dropout
        self.causal = causal

        assert num_heads % self.num_kv_heads == 0
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

    def _create_sliding_window_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Create a causal sliding window attention mask.

        Args:
            seq_len: Sequence length
            device: Device to create mask on
            dtype: Data type for mask

        Returns:
            Mask of shape (1, 1, seq_len, seq_len)
        """
        # Create position indices
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)

        # Sliding window: can attend if within window_size
        # Causal: can only attend to previous positions
        if self.causal:
            mask = (col_idx <= row_idx) & (row_idx - col_idx < self.window_size)
        else:
            mask = torch.abs(row_idx - col_idx) < self.window_size

        # Convert to attention mask (0 for attend, -inf for ignore)
        mask = mask.float().masked_fill(~mask, float('-inf')).masked_fill(mask, 0.0)
        return mask.unsqueeze(0).unsqueeze(0).to(dtype)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match query heads."""
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
            hidden_states: Input of shape (batch, seq_len, dim)
            attention_mask: Optional additional mask
            position_embeddings: Tuple of (cos, sin) for RoPE
            past_key_value: KV cache for incremental decoding
            use_cache: Whether to return updated cache
            output_attentions: Whether to return attention weights

        Returns:
            output: Shape (batch, seq_len, dim)
            attn_weights: If output_attentions
            past_key_value: If use_cache
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        kv_seq_len = seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV for GQA
        key_states = self._repeat_kv(key_states, self.num_kv_groups)
        value_states = self._repeat_kv(value_states, self.num_kv_groups)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale

        # Apply sliding window mask
        sliding_mask = self._create_sliding_window_mask(
            kv_seq_len, hidden_states.device, hidden_states.dtype
        )

        # For incremental decoding, only keep last row of mask
        if seq_len == 1 and kv_seq_len > 1:
            sliding_mask = sliding_mask[:, :, -1:, :]

        attn_weights = attn_weights + sliding_mask

        # Apply additional mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
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


class SWA(SlidingWindowAttention):
    """Alias for SlidingWindowAttention."""
    pass
