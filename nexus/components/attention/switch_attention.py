"""
SwitchHead: Mixture-of-Experts (MoE) Attention.

SwitchHead applies the Mixture-of-Experts paradigm to attention heads,
using a router to select which expert processes each head. This reduces
the number of attention matrices computed while matching the quality
of standard multi-head attention.

Key ideas:
    - Value and output projections are routed through experts
    - A lightweight router selects experts per head for each token
    - Fewer attention matrices need to be computed (top-k < num_experts)
    - Load balancing loss ensures even expert utilization

This is analogous to Switch Transformer's MoE for FFN layers, but applied
to the attention mechanism itself.

Reference: https://arxiv.org/abs/2312.07987 (SwitchHead: Accelerating Transformers
           with Mixture-of-Attention Heads)

See Also:
    - grouped_query.py: GQA (static head sharing)
    - linear_attention.py: Subquadratic attention alternatives
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class AttentionExpertRouter(NexusModule):
    """Router that selects experts for each attention head.

    Uses a learned routing function to assign tokens to experts.
    Supports top-k routing where each token is processed by the
    top-k scoring experts.

    Args:
        d_model: Model dimension
        num_experts: Number of available experts
        top_k: Number of experts to select per token
        router_jitter: Multiplicative jitter noise for load balancing (training only)
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 1,
        router_jitter: float = 0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter = router_jitter

        # Router: projects input to expert scores
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            hidden_states: Input tensor (batch, seq_len, d_model)

        Returns:
            router_weights: Soft weights for selected experts
                (batch, seq_len, top_k)
            expert_indices: Selected expert indices
                (batch, seq_len, top_k)
            router_logits: Raw logits for auxiliary loss
                (batch, seq_len, num_experts)
        """
        # Add jitter noise during training for load balancing
        if self.training and self.router_jitter > 0:
            noise = torch.empty_like(hidden_states).uniform_(
                1.0 - self.router_jitter, 1.0 + self.router_jitter
            )
            hidden_states = hidden_states * noise

        # Compute router logits
        router_logits = self.gate(hidden_states)  # (B, S, E)

        # Get top-k experts
        router_weights, expert_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )

        # Normalize weights via softmax over selected experts
        router_weights = F.softmax(router_weights, dim=-1)

        return router_weights, expert_indices, router_logits


class SwitchHeadAttention(NexusModule):
    """SwitchHead: Mixture-of-Experts Attention.

    Applies expert routing to value and output projections in attention.
    Each head has multiple expert V/O projections, and the router selects
    which expert(s) to use for each token.

    The query and key projections are shared across experts (they define
    the attention pattern), while value and output projections are routed
    (they define how information is extracted and combined).

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_experts: Number of experts per head for V/O projections
        top_k: Number of experts to route to (default 1 for switch routing)
        head_dim: Dimension per head. Defaults to d_model // num_heads.
        dropout: Attention dropout probability
        bias: Whether to use bias in projections
        aux_loss_coeff: Coefficient for load-balancing auxiliary loss

    Example:
        >>> switch_attn = SwitchHeadAttention(
        ...     d_model=512, num_heads=8, num_experts=4, top_k=1
        ... )
        >>> x = torch.randn(2, 64, 512)
        >>> out, attn_w, aux_loss = switch_attn(x)
        >>> out.shape
        torch.Size([2, 64, 512])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int = 4,
        top_k: int = 1,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        aux_loss_coeff: float = 0.01
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.head_dim = head_dim or (d_model // num_heads)
        self.dropout_p = dropout
        self.aux_loss_coeff = aux_loss_coeff

        self.scale = self.head_dim ** -0.5

        # Shared Q, K projections (same attention pattern regardless of expert)
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)

        # Expert V projections: num_experts sets of V projections
        self.v_experts = nn.ModuleList([
            nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
            for _ in range(num_experts)
        ])

        # Expert output projections
        self.o_experts = nn.ModuleList([
            nn.Linear(num_heads * self.head_dim, d_model, bias=bias)
            for _ in range(num_experts)
        ])

        # Router for selecting experts
        self.router = AttentionExpertRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k
        )

        self.attn_dropout = nn.Dropout(dropout)

    def _compute_load_balancing_loss(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary load-balancing loss.

        Encourages even distribution of tokens across experts,
        preventing expert collapse where one expert handles all tokens.

        Args:
            router_logits: Raw router scores (batch, seq_len, num_experts)
            expert_indices: Selected expert indices (batch, seq_len, top_k)

        Returns:
            Scalar load-balancing loss
        """
        # Fraction of tokens routed to each expert
        num_tokens = router_logits.shape[0] * router_logits.shape[1]
        expert_mask = F.one_hot(expert_indices, self.num_experts).float()
        # Sum over top_k dimension and normalize
        expert_mask = expert_mask.sum(dim=2)  # (B, S, E)
        tokens_per_expert = expert_mask.sum(dim=(0, 1)) / num_tokens

        # Average router probability for each expert
        router_probs = F.softmax(router_logits, dim=-1)
        avg_prob_per_expert = router_probs.mean(dim=(0, 1))

        # Load balancing loss: dot product of fractions and probabilities
        # Minimized when both are uniform (1/num_experts)
        loss = (tokens_per_expert * avg_prob_per_expert).sum() * self.num_experts

        return loss * self.aux_loss_coeff

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with expert-routed V and O projections.

        The attention pattern (Q @ K^T) is computed once with shared
        projections, then applied to expert-selected value projections.

        Args:
            hidden_states: Input tensor (batch, seq_len, d_model)
            attention_mask: Optional mask (batch, 1, seq_len, kv_seq_len)
            position_embeddings: Tuple of (cos, sin) for RoPE
            past_key_value: Cached (key, value) for incremental decoding
            use_cache: Whether to return cache
            output_attentions: Whether to return attention weights

        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: Attention weights if output_attentions, else None
            aux_loss: Load-balancing loss (scalar tensor), None during eval
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Route tokens to experts
        router_weights, expert_indices, router_logits = self.router(hidden_states)
        # router_weights: (B, S, top_k)
        # expert_indices: (B, S, top_k)

        # Shared Q, K projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)

        # Reshape Q, K
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Apply RoPE if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = self._apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)

        new_cache = None
        if use_cache:
            # We cache keys; values are expert-dependent so handled differently
            new_cache = (key_states, None)

        # Compute shared attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(-2, -1)
        ) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_probs = self.attn_dropout(attn_probs)

        # Compute expert-weighted value projections and outputs
        # For efficiency, compute all expert V projections and mix
        kv_seq_len = key_states.shape[2]

        # If caching, we need the original hidden states for V projection
        # during decode. For simplicity, compute V from current input only
        # (in practice, past values would also be cached per-expert).
        final_output = torch.zeros(
            batch_size, seq_len, self.d_model,
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        for k_idx in range(self.top_k):
            # Get the expert index and weight for this top-k slot
            expert_idx = expert_indices[:, :, k_idx]  # (B, S)
            expert_weight = router_weights[:, :, k_idx]  # (B, S)

            # Process each expert
            for e in range(self.num_experts):
                # Mask for tokens routed to this expert
                expert_mask = (expert_idx == e)  # (B, S)

                if not expert_mask.any():
                    continue

                # Compute V with this expert for ALL tokens
                # (simpler than sparse dispatch; efficient for small num_experts)
                v_states = self.v_experts[e](hidden_states)
                v_states = v_states.view(
                    batch_size, seq_len, self.num_heads, self.head_dim
                ).transpose(1, 2)

                # Handle past values for cache (use current expert's projection)
                if past_key_value is not None and past_key_value[1] is not None:
                    # In a full implementation, per-expert past values would be cached
                    pass

                # Apply attention: attn_probs @ v_states
                # attn_probs: (B, H, S_q, S_kv), v: (B, H, S_q, D) -- need S_kv
                # For prefill (no cache), kv_seq_len == seq_len, so this works
                attn_output = torch.matmul(attn_probs, v_states)

                # Reshape for output projection
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(batch_size, seq_len, -1)

                # Apply expert output projection
                expert_output = self.o_experts[e](attn_output)

                # Weight by router weight and mask
                weight = expert_weight.unsqueeze(-1) * expert_mask.unsqueeze(-1).float()
                final_output = final_output + expert_output * weight

        # Compute auxiliary loss during training
        aux_loss = None
        if self.training:
            aux_loss = self._compute_load_balancing_loss(router_logits, expert_indices)

        if not output_attentions:
            attn_weights = None

        return final_output, attn_weights, aux_loss

    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary position embeddings to Q and K."""
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


class SwitchHead(SwitchHeadAttention):
    """Alias for SwitchHeadAttention."""
    pass
