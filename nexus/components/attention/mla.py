"""
Multi-Head Latent Attention (MLA) with simplified low-rank KV compression.

MLA achieves ~93% KV cache reduction by compressing key-value pairs into a
compact latent vector before caching, then decompressing per-head K and V
at attention time. This design separates the caching cost from the number
of attention heads, enabling massive memory savings during inference.

Architecture:
    1. Compress: hidden_states -> down_proj -> latent (d_latent << d_model)
    2. Cache: store only the latent vector per token
    3. Decompress: latent -> up_proj_k, up_proj_v -> per-head K, V
    4. Attention: standard scaled dot-product over decompressed K, V

The latent bottleneck acts as an information bottleneck that retains only the
most relevant information for attention computation, while dramatically
reducing the per-token memory footprint of the KV cache.

Reference: https://arxiv.org/abs/2405.04434 (DeepSeek-V2: A Strong, Economical,
           and Efficient Mixture-of-Experts Language Model)

See Also:
    - latent_attention.py: Full DeepSeek-style MLA with decoupled RoPE
    - grouped_query.py: GQA, another KV cache reduction approach
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class MultiHeadLatentAttentionV2(NexusModule):
    """Multi-Head Latent Attention with low-rank KV compression.

    This is a simplified variant of MLA that focuses on the core idea:
    compress the full KV representation into a low-dimensional latent
    vector before caching. At attention time, the latent is decompressed
    back into per-head keys and values.

    KV cache memory comparison:
        Standard MHA: 2 * num_heads * head_dim * seq_len  (K + V)
        MLA:          d_latent * seq_len                   (latent only)

        With d_latent = num_heads * head_dim / 16, this is a ~93% reduction.

    Args:
        d_model: Model dimension (input/output size)
        num_heads: Number of attention heads
        d_latent: Dimension of the compressed latent vector. Controls the
            trade-off between cache size and model quality. Typical values
            are 1/8 to 1/16 of num_heads * head_dim.
        head_dim: Dimension per attention head. Defaults to d_model // num_heads.
        dropout: Attention dropout probability
        bias: Whether to use bias in linear projections
        use_layer_norm: Whether to apply layer norm to the latent representation
            for training stability

    Example:
        >>> mla = MultiHeadLatentAttentionV2(d_model=2048, num_heads=16, d_latent=128)
        >>> x = torch.randn(2, 512, 2048)
        >>> out, attn_w, cache = mla(x)
        >>> out.shape
        torch.Size([2, 512, 2048])
        >>> # Cache stores compressed latent instead of full KV
        >>> cache[0].shape  # latent: (2, 512, 128) instead of (2, 16, 512, 128)
        torch.Size([2, 512, 128])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_latent: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        use_layer_norm: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent
        self.head_dim = head_dim or (d_model // num_heads)
        self.dropout_p = dropout
        self.use_layer_norm = use_layer_norm

        # Validate parameters
        if d_model % num_heads != 0 and head_dim is None:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads}) "
                f"when head_dim is not specified"
            )
        if d_latent >= num_heads * self.head_dim:
            raise ValueError(
                f"d_latent ({d_latent}) should be smaller than the full KV dimension "
                f"({num_heads * self.head_dim}) for compression benefit"
            )

        self.scale = self.head_dim ** -0.5
        kv_dim = num_heads * self.head_dim

        # Query projection: d_model -> num_heads * head_dim
        self.q_proj = nn.Linear(d_model, kv_dim, bias=bias)

        # KV down-projection (compression): d_model -> d_latent
        self.kv_down_proj = nn.Linear(d_model, d_latent, bias=bias)

        # Optional layer norm on latent for stability
        if use_layer_norm:
            self.latent_norm = nn.LayerNorm(d_latent)

        # KV up-projections (decompression): d_latent -> num_heads * head_dim each
        self.k_up_proj = nn.Linear(d_latent, kv_dim, bias=bias)
        self.v_up_proj = nn.Linear(d_latent, kv_dim, bias=bias)

        # Output projection
        self.o_proj = nn.Linear(kv_dim, d_model, bias=bias)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)

        # Initialize projections
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with scaled initialization for stable training."""
        for proj in [self.q_proj, self.kv_down_proj, self.k_up_proj, self.v_up_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        # Output projection with smaller init for residual stability
        nn.init.xavier_uniform_(self.o_proj.weight, gain=1.0 / math.sqrt(2.0))
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)

    def compress_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compress hidden states into latent KV representation.

        Args:
            hidden_states: Input tensor (batch, seq_len, d_model)

        Returns:
            Latent KV tensor (batch, seq_len, d_latent)
        """
        latent = self.kv_down_proj(hidden_states)
        if self.use_layer_norm:
            latent = self.latent_norm(latent)
        return latent

    def decompress_kv(
        self, latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress latent into per-head key and value tensors.

        Args:
            latent: Latent KV tensor (batch, seq_len, d_latent)

        Returns:
            key_states: (batch, num_heads, seq_len, head_dim)
            value_states: (batch, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = latent.shape

        key_states = self.k_up_proj(latent)
        value_states = self.v_up_proj(latent)

        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        return key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with latent KV compression.

        The key insight is that only the compressed latent is cached, not the
        full per-head K and V. Decompression happens at attention time, which
        trades a small amount of compute for massive memory savings.

        Args:
            hidden_states: Input tensor (batch, seq_len, d_model)
            attention_mask: Mask tensor (batch, 1, seq_len, kv_seq_len)
            position_embeddings: Tuple of (cos, sin) for RoPE
            past_key_value: Cached latent tensor from previous steps
            use_cache: Whether to return the latent cache
            output_attentions: Whether to return attention weights

        Returns:
            output: Attention output (batch, seq_len, d_model)
            attn_weights: Attention weights if output_attentions, else None
            past_key_value: Cached latent tuple if use_cache, else None
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project queries
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Compress KV into latent
        kv_latent = self.compress_kv(hidden_states)

        # Handle KV cache: store and concatenate latents
        if past_key_value is not None:
            kv_latent = torch.cat([past_key_value[0], kv_latent], dim=1)

        new_cache = (kv_latent,) if use_cache else None

        # Decompress latent into per-head K, V
        key_states, value_states = self.decompress_kv(kv_latent)

        # Apply rotary position embeddings if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = self._apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(-2, -1)
        ) * self.scale

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        # Compute output
        attn_output = torch.matmul(attn_weights_dropped, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, new_cache

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

    def get_cache_size_ratio(self) -> float:
        """Return the cache size ratio compared to standard MHA.

        Returns:
            Ratio of MLA cache size to standard MHA cache size.
            For example, 0.07 means 93% reduction.
        """
        standard_cache = 2 * self.num_heads * self.head_dim  # K + V per token
        mla_cache = self.d_latent  # latent per token
        return mla_cache / standard_cache


class MLAV2(MultiHeadLatentAttentionV2):
    """Alias for MultiHeadLatentAttentionV2."""
    pass
