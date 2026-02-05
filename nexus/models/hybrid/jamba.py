"""
Jamba - Hybrid Transformer-Mamba-MoE Architecture.

Jamba is a production-scale hybrid architecture developed by AI21 Labs that
interleaves three types of layers:
1. Transformer attention layers (for precise retrieval and global context)
2. Mamba SSM layers (for efficient long-range sequence modeling)
3. Mixture-of-Experts (MoE) layers (for increased model capacity)

The interleaving pattern is configurable; a typical pattern uses attention
layers every few Mamba layers, with MoE applied periodically. For example:
    AMMAMMAMMAMM (A=attention, M=mamba)
with MoE applied to every other layer.

Key design choices:
- Mamba layers provide efficient O(N) sequence processing for most layers
- Attention layers are interspersed for tasks requiring precise retrieval
- MoE provides capacity scaling without proportional compute increase
- The ratio of Mamba:Attention layers is typically 7:1 or 3:1

Reference: Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model",
    AI21 Labs, 2024. https://arxiv.org/abs/2403.19887
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
from nexus.core.base import NexusModule


class JambaRMSNorm(NexusModule):
    """Root Mean Square Layer Normalization.

    RMSNorm is a simplification of LayerNorm that only normalizes by the
    RMS of the activations (no centering). Used throughout Jamba for
    efficiency.

    Args:
        d_model: Model dimension.
        eps: Epsilon for numerical stability (default: 1e-6).
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class JambaMambaLayer(NexusModule):
    """Mamba SSM layer for Jamba.

    Implements the Mamba selective SSM with gated MLP structure:
        x -> in_proj -> [branch, gate]
        branch -> conv1d -> SiLU -> SSM -> * SiLU(gate) -> out_proj

    This is the workhorse layer in Jamba, handling most of the sequence
    processing with O(N) complexity.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension (default: 16).
        d_conv: Convolution kernel size (default: 4).
        expand: Expansion factor (default: 2).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = d_model // 16

        # Input projection (branch + gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )

        # SSM parameters
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32)
                .repeat(self.d_inner, 1)
            )
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Input-dependent B, C, dt projections
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Optional SSM state.

        Returns:
            output: Shape (batch, seq_len, d_model).
            state: Updated SSM state.
        """
        batch_size, seq_len, _ = x.shape

        # Project and split
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Causal convolution
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        # SSM parameters from input
        x_dbl = self.x_proj(x_branch)
        delta, B, C = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))

        # Get A
        A = -torch.exp(self.A_log)

        # Run SSM
        if state is None:
            state = torch.zeros(
                batch_size, self.d_inner, self.d_state,
                device=x.device, dtype=x.dtype
            )

        outputs = []
        for t in range(seq_len):
            u_t = x_branch[:, t]
            delta_t = delta[:, t]
            B_t = B[:, t]
            C_t = C[:, t]

            dA = torch.exp(delta_t.unsqueeze(-1) * A)
            dB = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)
            state = dA * state + dB * u_t.unsqueeze(-1)
            y_t = torch.einsum('bdn,bn->bd', state, C_t) + self.D * u_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)

        # Gate and project
        y = y * F.silu(z)
        output = self.out_proj(y)

        return output, state


class JambaAttentionLayer(NexusModule):
    """Grouped-Query Attention layer for Jamba.

    Standard transformer attention with grouped-query attention (GQA)
    for KV cache efficiency. Interspersed between Mamba layers to
    provide precise retrieval capability.

    Uses RoPE (Rotary Position Embedding) for position encoding.

    Args:
        d_model: Model dimension.
        num_heads: Number of query heads (default: 32).
        num_kv_heads: Number of KV heads for GQA (default: 8).
        dropout: Attention dropout (default: 0.0).
        max_seq_len: Maximum sequence length for RoPE (default: 4096).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 4096
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # RoPE
        self._init_rope(max_seq_len)

    def _init_rope(self, max_seq_len: int):
        """Initialize rotary position embeddings."""
        dim = self.head_dim
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer('inv_freq', inv_freq)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def _apply_rope(
        self,
        x: torch.Tensor,
        offset: int = 0
    ) -> torch.Tensor:
        """Apply rotary position embeddings.

        Args:
            x: Shape (batch, heads, seq_len, head_dim).
            offset: Position offset for KV cache.

        Returns:
            x with RoPE applied.
        """
        seq_len = x.shape[2]
        cos = self.cos_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)

        # Split into pairs for rotation
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.stack(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
            dim=-1
        ).flatten(-2)

        return x_rotated

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            kv_cache: Optional (cached_keys, cached_values) tuple.
            attention_mask: Optional attention mask.

        Returns:
            output: Shape (batch, seq_len, d_model).
            kv_cache: Updated KV cache.
        """
        batch_size, seq_len, _ = x.shape

        # Project
        q = self.q_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Handle KV cache
        offset = 0
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            offset = k_cache.shape[2]
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # Apply RoPE
        q = self._apply_rope(q, offset=offset)
        k = self._apply_rope(k, offset=0)

        # Expand KV heads for GQA
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        kv_len = k.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool),
            diagonal=kv_len - seq_len + 1
        )
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.out_proj(output)

        new_kv_cache = (
            k[:, ::self.num_groups],  # Store only the KV head copies
            v[:, ::self.num_groups]
        )

        return output, new_kv_cache


class JambaMoELayer(NexusModule):
    """Mixture-of-Experts feed-forward layer for Jamba.

    Implements a top-k gated MoE where each token is routed to the top-k
    experts. The routing is learned and includes a load balancing auxiliary
    loss to prevent expert collapse.

    Args:
        d_model: Model dimension.
        num_experts: Total number of experts (default: 8).
        top_k: Number of experts per token (default: 2).
        ffn_expand: FFN expansion factor per expert (default: 4).
        dropout: Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        ffn_expand: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Expert FFNs (SwiGLU style)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * ffn_expand, bias=False),
                nn.SiLU(),
                nn.Linear(d_model * ffn_expand, d_model, bias=False),
                nn.Dropout(dropout)
            )
            for _ in range(num_experts)
        ])

        # Auxiliary loss coefficient
        self.aux_loss_coeff = 0.01

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with MoE routing.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            output: Shape (batch, seq_len, d_model).
            aux_info: Dict with 'router_logits' and 'aux_loss'.
        """
        batch_size, seq_len, d_model = x.shape

        # Compute routing probabilities
        router_logits = self.router(x)  # (batch, seq, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )

        # Normalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Process through experts
        # Flatten batch and seq dims for routing
        x_flat = x.view(-1, d_model)  # (batch * seq, d_model)
        top_k_weights_flat = top_k_weights.view(-1, self.top_k)
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)

        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_indices = top_k_indices_flat[:, k]  # (batch * seq,)
            weights_k = top_k_weights_flat[:, k]  # (batch * seq,)

            for expert_idx in range(self.num_experts):
                mask = expert_indices == expert_idx
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    output[mask] += weights_k[mask].unsqueeze(-1) * expert_output

        output = output.view(batch_size, seq_len, d_model)

        # Compute load balancing auxiliary loss
        aux_loss = self._compute_aux_loss(routing_weights)

        aux_info = {
            'router_logits': router_logits,
            'aux_loss': aux_loss
        }

        return output, aux_info

    def _compute_aux_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute load balancing auxiliary loss.

        Encourages equal utilization of all experts using the switch
        transformer auxiliary loss.

        Args:
            routing_weights: Shape (batch, seq, num_experts).

        Returns:
            Scalar auxiliary loss.
        """
        # Fraction of tokens routed to each expert
        tokens_per_expert = routing_weights.mean(dim=[0, 1])  # (num_experts,)

        # Target: uniform distribution
        uniform = torch.ones_like(tokens_per_expert) / self.num_experts

        # L2 distance from uniform
        aux_loss = self.aux_loss_coeff * (
            self.num_experts * (tokens_per_expert * tokens_per_expert).sum()
        )

        return aux_loss


class JambaDenseFFN(NexusModule):
    """Dense (non-MoE) feed-forward layer for Jamba.

    Standard SwiGLU-style FFN used in layers without MoE.

    Args:
        d_model: Model dimension.
        ffn_expand: Expansion factor (default: 4).
        dropout: Dropout (default: 0.0).
    """

    def __init__(
        self,
        d_model: int,
        ffn_expand: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        hidden_dim = d_model * ffn_expand

        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU activation.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            output: Shape (batch, seq_len, d_model).
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        output = self.down_proj(self.dropout(gate * up))
        return output


class JambaBlock(NexusModule):
    """Single Jamba block with configurable layer type.

    A Jamba block consists of:
    1. Pre-norm + Sequence mixer (either Attention or Mamba)
    2. Pre-norm + FFN (either dense or MoE)

    The layer type is determined by the block configuration.

    Args:
        d_model: Model dimension.
        layer_type: 'attention' or 'mamba'.
        use_moe: Whether to use MoE for the FFN (default: False).
        num_heads: Attention heads (for attention layers, default: 32).
        num_kv_heads: KV heads for GQA (default: 8).
        mamba_d_state: Mamba state dimension (default: 16).
        mamba_d_conv: Mamba conv kernel size (default: 4).
        mamba_expand: Mamba expansion factor (default: 2).
        num_experts: MoE expert count (default: 8).
        top_k: MoE top-k (default: 2).
        ffn_expand: FFN expansion (default: 4).
        dropout: Dropout (default: 0.0).
    """

    def __init__(
        self,
        d_model: int,
        layer_type: str = 'mamba',
        use_moe: bool = False,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        num_experts: int = 8,
        top_k: int = 2,
        ffn_expand: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_type = layer_type
        self.use_moe = use_moe

        # Sequence mixer
        self.norm1 = JambaRMSNorm(d_model)
        if layer_type == 'attention':
            self.mixer = JambaAttentionLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dropout=dropout
            )
        elif layer_type == 'mamba':
            self.mixer = JambaMambaLayer(
                d_model=d_model,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand
            )
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        self.dropout1 = nn.Dropout(dropout)

        # FFN (dense or MoE)
        self.norm2 = JambaRMSNorm(d_model)
        if use_moe:
            self.ffn = JambaMoELayer(
                d_model=d_model,
                num_experts=num_experts,
                top_k=top_k,
                ffn_expand=ffn_expand,
                dropout=dropout
            )
        else:
            self.ffn = JambaDenseFFN(
                d_model=d_model,
                ffn_expand=ffn_expand,
                dropout=dropout
            )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: SSM state for Mamba layers.
            kv_cache: KV cache for attention layers.
            attention_mask: Optional attention mask.

        Returns:
            output: Shape (batch, seq_len, d_model).
            layer_info: Dict containing updated state/cache and aux losses.
        """
        layer_info = {}

        # Sequence mixer with residual
        residual = x
        x = self.norm1(x)

        if self.layer_type == 'attention':
            x, new_kv_cache = self.mixer(x, kv_cache, attention_mask)
            layer_info['kv_cache'] = new_kv_cache
            layer_info['state'] = state  # Pass through unchanged
        else:
            x, new_state = self.mixer(x, state)
            layer_info['state'] = new_state
            layer_info['kv_cache'] = kv_cache  # Pass through unchanged

        x = self.dropout1(x)
        x = x + residual

        # FFN with residual
        residual = x
        x = self.norm2(x)

        if self.use_moe:
            x, aux_info = self.ffn(x)
            layer_info['aux_loss'] = aux_info.get('aux_loss', torch.tensor(0.0))
            layer_info['router_logits'] = aux_info.get('router_logits')
        else:
            x = self.ffn(x)
            layer_info['aux_loss'] = torch.tensor(0.0, device=x.device)

        x = self.dropout2(x)
        x = x + residual

        return x, layer_info


class JambaModel(NexusModule):
    """Full Jamba language model.

    Interleaves Transformer attention, Mamba SSM, and MoE layers in a
    configurable pattern. The default pattern alternates between attention
    and Mamba layers, with MoE applied periodically.

    Reference: Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language
        Model", AI21 Labs, 2024. https://arxiv.org/abs/2403.19887

    Args:
        d_model: Model dimension.
        num_layers: Total number of layers.
        num_heads: Attention heads (default: 32).
        num_kv_heads: KV heads for GQA (default: 8).
        vocab_size: Vocabulary size.
        num_experts: MoE expert count (default: 8).
        top_k: MoE top-k routing (default: 2).
        mamba_d_state: Mamba SSM state dimension (default: 16).
        mamba_d_conv: Mamba conv size (default: 4).
        mamba_expand: Mamba expansion (default: 2).
        ffn_expand: FFN expansion (default: 4).
        dropout: Dropout (default: 0.0).
        layer_pattern: String pattern like 'AMMAAMM' where A=attention,
            M=mamba. If None, uses every 4th layer as attention (default: None).
        moe_every_n: Apply MoE every N layers (default: 2).
            Set to 0 to disable MoE entirely.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        vocab_size: int = 32000,
        num_experts: int = 8,
        top_k: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        ffn_expand: int = 4,
        dropout: float = 0.0,
        layer_pattern: Optional[str] = None,
        moe_every_n: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # Determine layer pattern
        if layer_pattern is not None:
            # Repeat pattern to fill num_layers
            full_pattern = (layer_pattern * (num_layers // len(layer_pattern) + 1))
            pattern = full_pattern[:num_layers]
        else:
            # Default: every 4th layer is attention, rest are Mamba
            pattern = ''
            for i in range(num_layers):
                pattern += 'A' if i % 4 == 0 else 'M'

        self.layer_pattern = pattern

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Build layers according to pattern
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_type = 'attention' if pattern[i] == 'A' else 'mamba'
            use_moe = moe_every_n > 0 and (i + 1) % moe_every_n == 0

            self.layers.append(JambaBlock(
                d_model=d_model,
                layer_type=layer_type,
                use_moe=use_moe,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                num_experts=num_experts,
                top_k=top_k,
                ffn_expand=ffn_expand,
                dropout=dropout
            ))

        # Output
        self.final_norm = JambaRMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_states: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            attention_mask: Optional attention mask.
            layer_states: Optional list of per-layer state dicts.

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size).
            model_info: Dict with 'layer_states' and 'total_aux_loss'.
        """
        if layer_states is None:
            layer_states = [{'state': None, 'kv_cache': None}] * self.num_layers

        x = self.embedding(input_ids)

        total_aux_loss = torch.tensor(0.0, device=x.device)
        new_layer_states = []

        for i, layer in enumerate(self.layers):
            state_i = layer_states[i] if i < len(layer_states) else {}

            x, layer_info = layer(
                x,
                state=state_i.get('state'),
                kv_cache=state_i.get('kv_cache'),
                attention_mask=attention_mask
            )

            new_layer_states.append({
                'state': layer_info.get('state'),
                'kv_cache': layer_info.get('kv_cache')
            })

            aux_loss = layer_info.get('aux_loss', torch.tensor(0.0))
            if isinstance(aux_loss, torch.Tensor):
                total_aux_loss = total_aux_loss + aux_loss

        x = self.final_norm(x)
        logits = self.lm_head(x)

        model_info = {
            'layer_states': new_layer_states,
            'total_aux_loss': total_aux_loss,
            'layer_pattern': self.layer_pattern
        }

        return logits, model_info

    def get_layer_info(self) -> str:
        """Get a human-readable summary of the layer configuration.

        Returns:
            String describing the layer pattern.
        """
        info = f"Jamba Model ({self.num_layers} layers)\n"
        info += f"Pattern: {self.layer_pattern}\n"
        for i, (char, layer) in enumerate(zip(self.layer_pattern, self.layers)):
            layer_type = 'Attention' if char == 'A' else 'Mamba'
            moe_str = ' + MoE' if layer.use_moe else ' + Dense FFN'
            info += f"  Layer {i}: {layer_type}{moe_str}\n"
        return info
