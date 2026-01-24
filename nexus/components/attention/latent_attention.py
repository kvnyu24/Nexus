"""
Multi-Head Latent Attention (MLA) implementation.

MLA compresses KV into a low-rank latent space for massive memory reduction
while maintaining model quality. Key innovation from DeepSeek V2/V3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class MultiHeadLatentAttention(NexusModule):
    """Multi-Head Latent Attention (MLA).

    Instead of storing full KV cache, MLA compresses KV into a low-rank latent
    representation, achieving 4-8x memory reduction. Uses decoupled RoPE where
    position information is only added to a subset of dimensions.

    Used by: DeepSeek V2, DeepSeek V3

    Reference: https://arxiv.org/abs/2405.04434 (DeepSeek-V2)

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        kv_lora_rank: Rank for KV compression (typically 512)
        q_lora_rank: Rank for Q compression (optional, if None uses full dim)
        qk_rope_head_dim: Dimension for RoPE in QK (subset of head_dim)
        qk_nope_head_dim: Dimension for non-positional QK
        v_head_dim: Value head dimension
        dropout: Attention dropout probability
        bias: Whether to use bias
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kv_lora_rank: int = 512,
        q_lora_rank: Optional[int] = None,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 128,
        v_head_dim: int = 128,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_rope_head_dim + qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.dropout = dropout

        self.scale = self.qk_head_dim ** -0.5

        # Q projections
        if q_lora_rank is not None:
            # Compressed Q path
            self.q_down_proj = nn.Linear(dim, q_lora_rank, bias=bias)
            self.q_up_proj = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=bias)
        else:
            self.q_proj = nn.Linear(dim, num_heads * self.qk_head_dim, bias=bias)

        # KV compression (down projection to latent space)
        self.kv_down_proj = nn.Linear(dim, kv_lora_rank, bias=bias)

        # KV decompression (up projection from latent space)
        # Produces both K (nope + rope parts) and V
        self.k_up_proj = nn.Linear(kv_lora_rank, num_heads * self.qk_head_dim, bias=bias)
        self.v_up_proj = nn.Linear(kv_lora_rank, num_heads * v_head_dim, bias=bias)

        # Additional K projection for RoPE portion (decoupled)
        self.k_rope_proj = nn.Linear(dim, num_heads * qk_rope_head_dim, bias=bias)

        # Output projection
        self.o_proj = nn.Linear(num_heads * v_head_dim, dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

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
        Forward pass with latent KV compression.

        The KV cache stores the compressed latent representation instead of full KV,
        significantly reducing memory usage.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q
        if self.q_lora_rank is not None:
            q_latent = self.q_down_proj(hidden_states)
            query_states = self.q_up_proj(q_latent)
        else:
            query_states = self.q_proj(hidden_states)

        # Reshape Q: (batch, seq, num_heads * qk_head_dim) -> (batch, num_heads, seq, qk_head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        query_states = query_states.transpose(1, 2)

        # Split Q into nope and rope parts
        q_nope, q_rope = query_states.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Compress KV to latent space
        kv_latent = self.kv_down_proj(hidden_states)

        # Get K rope part (directly from input, for decoupled RoPE)
        k_rope = self.k_rope_proj(hidden_states)
        k_rope = k_rope.view(batch_size, seq_len, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)

        # Apply RoPE to rope parts only
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Adjust cos/sin for rope head dim
            cos = cos[..., :self.qk_rope_head_dim]
            sin = sin[..., :self.qk_rope_head_dim]
            q_rope, k_rope = self._apply_rotary_pos_emb(q_rope, k_rope, cos, sin)

        # Handle KV cache (store latent + k_rope)
        if past_key_value is not None:
            past_latent, past_k_rope = past_key_value
            kv_latent = torch.cat([past_latent, kv_latent], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=2)

        past_key_value = (kv_latent, k_rope) if use_cache else None

        # Decompress KV from latent
        # Note: This is done after caching to save memory
        key_nope = self.k_up_proj(kv_latent)
        value_states = self.v_up_proj(kv_latent)

        # Reshape K_nope and V
        kv_seq_len = kv_latent.shape[1]
        key_nope = key_nope.view(batch_size, kv_seq_len, self.num_heads, self.qk_head_dim)
        key_nope = key_nope.transpose(1, 2)
        key_nope = key_nope[..., :self.qk_nope_head_dim]  # Take only nope portion

        value_states = value_states.view(batch_size, kv_seq_len, self.num_heads, self.v_head_dim)
        value_states = value_states.transpose(1, 2)

        # Concatenate K parts: nope + rope
        key_states = torch.cat([key_nope, k_rope], dim=-1)

        # Concatenate Q parts
        query_states = torch.cat([q_nope, q_rope], dim=-1)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)
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


class MLA(MultiHeadLatentAttention):
    """Alias for MultiHeadLatentAttention."""
    pass
