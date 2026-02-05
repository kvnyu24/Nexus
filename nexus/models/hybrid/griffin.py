"""
Griffin - Hybrid Recurrence-Attention Architecture.

Griffin is a hybrid architecture that alternates between gated linear recurrence
blocks (the "Hawk" component) and local multi-query attention (MQA) blocks. The
design philosophy is that:

1. Gated linear recurrences efficiently handle long-range dependencies with
   O(1) per-step complexity, but may struggle with precise token-level retrieval.

2. Local attention (windowed) excels at precise short-range retrieval but is
   expensive for long contexts.

By interleaving these two mechanisms, Griffin achieves strong performance on both
long-range tasks and short-range retrieval tasks, while maintaining efficient
inference.

Key components:
- GatedLinearRecurrence: The "Hawk" recurrent block with input/forget gating
  and a Real-Gated Linear Recurrent Unit (RGLRU).
- LocalMultiQueryAttention: Windowed multi-query attention for local context.
- GriffinBlock: Combines recurrence and local attention in a single block.
- GriffinModel: Full language model with embedding, blocks, and LM head.

Reference: De et al., "Griffin: Mixing Gated Linear Recurrences with Local
    Attention for Efficient Language Models", Google DeepMind, 2024.
    https://arxiv.org/abs/2402.19427
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from nexus.core.base import NexusModule


class RealGatedLinearRecurrentUnit(NexusModule):
    """Real-Gated Linear Recurrent Unit (RGLRU).

    The core recurrent component of the Hawk/Griffin architecture. Unlike
    traditional gated RNNs (GRU, LSTM), RGLRU uses a diagonal recurrence
    matrix with learned gates, enabling efficient parallel scan computation.

    The recurrence is:
        a[t] = sigma(W_a @ x[t])          (recurrence gate, element-wise)
        h[t] = a[t] * h[t-1] + sqrt(1 - a[t]^2) * (W_x @ x[t])
        y[t] = h[t]

    The sqrt(1 - a^2) scaling ensures the recurrence preserves signal
    magnitude (unitarily-like behavior).

    Args:
        d_model: Input dimension.
        d_recurrence: Recurrence state dimension (default: same as d_model).
    """

    def __init__(self, d_model: int, d_recurrence: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.d_recurrence = d_recurrence or d_model

        # Input projection
        self.x_proj = nn.Linear(d_model, self.d_recurrence, bias=False)

        # Recurrence gate
        self.a_proj = nn.Linear(d_model, self.d_recurrence, bias=True)

        # Initialize gate bias to encourage remembering (high a values)
        nn.init.constant_(self.a_proj.bias, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Previous state of shape (batch, d_recurrence).

        Returns:
            output: Shape (batch, seq_len, d_recurrence).
            state: Updated state of shape (batch, d_recurrence).
        """
        batch_size, seq_len, _ = x.shape

        # Compute gate and input
        a = torch.sigmoid(self.a_proj(x))  # (batch, seq, d_rec)
        x_in = self.x_proj(x)  # (batch, seq, d_rec)

        # Scale input to preserve magnitude
        x_in = x_in * torch.sqrt(1 - a ** 2 + 1e-6)

        if state is None:
            state = torch.zeros(
                batch_size, self.d_recurrence,
                device=x.device, dtype=x.dtype
            )

        # Run recurrence
        if self.training:
            output, state = self._parallel_scan(a, x_in, state)
        else:
            output, state = self._sequential_scan(a, x_in, state)

        return output, state

    def _parallel_scan(
        self,
        a: torch.Tensor,
        x: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel scan for training using log-space cumsum trick.

        Args:
            a: Gates (batch, seq, d_rec).
            x: Inputs (batch, seq, d_rec).
            state: Initial state (batch, d_rec).

        Returns:
            output: (batch, seq, d_rec).
            final_state: (batch, d_rec).
        """
        batch_size, seq_len, d_rec = a.shape

        # Log-space computation for numerical stability
        log_a = torch.log(a + 1e-6)
        log_a_cumsum = torch.cumsum(log_a, dim=1)

        # Weighted inputs in log space
        weighted_x = x * torch.exp(-log_a_cumsum)
        cumsum_weighted_x = torch.cumsum(weighted_x, dim=1)

        # Scale back
        output = cumsum_weighted_x * torch.exp(log_a_cumsum)

        # Add initial state contribution
        state_contrib = state.unsqueeze(1) * torch.exp(log_a_cumsum)
        output = output + state_contrib

        final_state = output[:, -1]
        return output, final_state

    def _sequential_scan(
        self,
        a: torch.Tensor,
        x: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential scan for inference.

        Args:
            a: Gates (batch, seq, d_rec).
            x: Inputs (batch, seq, d_rec).
            state: Initial state (batch, d_rec).

        Returns:
            output: (batch, seq, d_rec).
            state: Updated state (batch, d_rec).
        """
        outputs = []
        for t in range(a.shape[1]):
            state = a[:, t] * state + x[:, t]
            outputs.append(state)
        output = torch.stack(outputs, dim=1)
        return output, state


class GatedLinearRecurrence(NexusModule):
    """Gated Linear Recurrence block (the "Hawk" component).

    Full gated recurrence block combining:
    1. Input projection with gating split
    2. Short convolution for local context
    3. RGLRU for sequence modeling
    4. Gated output

    Architecture:
        x -> in_proj -> [branch, gate]
        branch -> conv1d -> RGLRU -> * sigmoid(gate) -> out_proj -> y

    Args:
        d_model: Model dimension.
        d_recurrence: Recurrence state dimension (default: d_model).
        d_conv: Convolution kernel size (default: 4).
        expand: Expansion factor (default: 2).
    """

    def __init__(
        self,
        d_model: int,
        d_recurrence: Optional[int] = None,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_recurrence = d_recurrence or self.d_inner

        # Input projection (x and gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Short convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )

        # RGLRU
        self.rglru = RealGatedLinearRecurrentUnit(
            d_model=self.d_inner,
            d_recurrence=self.d_recurrence
        )

        # Project back if recurrence dim differs
        if self.d_recurrence != self.d_inner:
            self.rec_proj = nn.Linear(self.d_recurrence, self.d_inner, bias=False)
        else:
            self.rec_proj = nn.Identity()

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Optional recurrence state.

        Returns:
            output: Shape (batch, seq_len, d_model).
            state: Updated state.
        """
        batch_size, seq_len, _ = x.shape

        # Project and split
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Causal convolution
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]
        x_branch = x_branch.transpose(1, 2)

        # RGLRU
        y, state = self.rglru(x_branch, state)
        y = self.rec_proj(y)

        # Gated output
        y = y * torch.sigmoid(z)

        # Output projection
        output = self.out_proj(y)

        return output, state


class LocalMultiQueryAttention(NexusModule):
    """Local Multi-Query Attention (MQA) with sliding window.

    Implements multi-query attention restricted to a local window around
    each position. Multi-query attention uses multiple query heads but
    shared key/value heads, reducing KV cache size.

    In Griffin, this provides precise short-range retrieval that complements
    the long-range but imprecise recurrence.

    Args:
        d_model: Model dimension.
        num_heads: Number of query heads (default: 8).
        num_kv_heads: Number of key/value heads (default: 1 for MQA).
        window_size: Local attention window size (default: 128).
        dropout: Attention dropout (default: 0.0).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_kv_heads: int = 1,
        window_size: int = 128,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.head_dim = d_model // num_heads
        self.kv_head_dim = d_model // num_kv_heads if num_kv_heads > 0 else d_model

        assert d_model % num_heads == 0

        # Q projection (multiple heads)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # K, V projections (fewer heads for MQA)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with local windowed attention.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            kv_cache: Optional tuple of (cached_keys, cached_values).

        Returns:
            output: Shape (batch, seq_len, d_model).
            kv_cache: Updated KV cache.
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Expand KV heads for multi-query (repeat to match num_heads)
        heads_per_kv = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(heads_per_kv, dim=2)
        v = v.repeat_interleave(heads_per_kv, dim=2)

        # Handle KV cache for inference
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        # Keep only window_size most recent for local attention
        kv_len = k.shape[1]
        if kv_len > self.window_size:
            k = k[:, -self.window_size:]
            v = v[:, -self.window_size:]
            kv_len = self.window_size

        # Transpose for attention: (batch, heads, seq, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask within window
        q_len = q.shape[2]
        k_len = k.shape[2]
        causal_mask = torch.triu(
            torch.ones(q_len, k_len, device=x.device, dtype=torch.bool),
            diagonal=k_len - q_len + 1
        )
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        output = torch.matmul(attn, v)  # (batch, heads, seq, dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.out_proj(output)

        # Update cache (keep up to window_size)
        new_kv_cache = (
            k.transpose(1, 2)[:, -self.window_size:],
            v.transpose(1, 2)[:, -self.window_size:]
        )

        return output, new_kv_cache


class GriffinBlock(NexusModule):
    """Single Griffin block combining recurrence and local attention.

    Each Griffin block contains:
    1. Pre-norm + Gated Linear Recurrence (long-range)
    2. Pre-norm + Local Multi-Query Attention (short-range)
    3. Pre-norm + FFN

    The recurrence and attention are applied in sequence (not in parallel),
    allowing the attention to refine the recurrence output.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        window_size: Local attention window size (default: 128).
        d_conv: Convolution kernel size (default: 4).
        expand: Expansion factor for recurrence (default: 2).
        ffn_expand: FFN expansion factor (default: 4).
        dropout: Dropout probability (default: 0.0).
        use_attention: Whether to include local attention (default: True).
            Set to False for "Hawk" (recurrence-only) variant.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        window_size: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        ffn_expand: int = 4,
        dropout: float = 0.0,
        use_attention: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.use_attention = use_attention

        # Gated Linear Recurrence
        self.norm1 = nn.LayerNorm(d_model)
        self.recurrence = GatedLinearRecurrence(
            d_model=d_model,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout1 = nn.Dropout(dropout)

        # Local Multi-Query Attention (optional)
        if use_attention:
            self.norm2 = nn.LayerNorm(d_model)
            self.attention = LocalMultiQueryAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=1,
                window_size=window_size,
                dropout=dropout
            )
            self.dropout2 = nn.Dropout(dropout)

        # FFN
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expand, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Optional state dict with keys:
                - 'recurrence_state': State for RGLRU
                - 'kv_cache': KV cache for local attention

        Returns:
            output: Shape (batch, seq_len, d_model).
            state: Updated state dict.
        """
        if state is None:
            state = {'recurrence_state': None, 'kv_cache': None}

        # Recurrence with residual
        residual = x
        x = self.norm1(x)
        x, rec_state = self.recurrence(x, state.get('recurrence_state'))
        x = self.dropout1(x)
        x = x + residual

        # Local attention with residual (if enabled)
        kv_cache = None
        if self.use_attention:
            residual = x
            x = self.norm2(x)
            x, kv_cache = self.attention(x, state.get('kv_cache'))
            x = self.dropout2(x)
            x = x + residual

        # FFN with residual
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + residual

        new_state = {
            'recurrence_state': rec_state,
            'kv_cache': kv_cache
        }

        return x, new_state


class GriffinModel(NexusModule):
    """Full Griffin language model.

    Stacks multiple GriffinBlocks with embedding and LM head. Supports
    both the full Griffin (recurrence + attention) and the Hawk variant
    (recurrence only).

    Reference: De et al., "Griffin: Mixing Gated Linear Recurrences with
        Local Attention for Efficient Language Models", 2024.
        https://arxiv.org/abs/2402.19427

    Args:
        d_model: Model dimension.
        num_layers: Number of Griffin blocks.
        num_heads: Number of attention heads.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length (for position embeddings, if used).
        window_size: Local attention window size (default: 128).
        d_conv: Convolution kernel size (default: 4).
        expand: Expansion factor for recurrence (default: 2).
        ffn_expand: FFN expansion factor (default: 4).
        dropout: Dropout probability (default: 0.0).
        hawk_only: If True, disable attention (Hawk variant, default: False).
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        vocab_size: int,
        max_seq_len: int = 2048,
        window_size: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        ffn_expand: int = 4,
        dropout: float = 0.0,
        hawk_only: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_scale = d_model ** 0.5
        self.embed_dropout = nn.Dropout(dropout)

        # Griffin blocks
        self.blocks = nn.ModuleList([
            GriffinBlock(
                d_model=d_model,
                num_heads=num_heads,
                window_size=window_size,
                d_conv=d_conv,
                expand=expand,
                ffn_expand=ffn_expand,
                dropout=dropout,
                use_attention=not hawk_only
            )
            for _ in range(num_layers)
        ])

        # Output
        self.final_norm = nn.LayerNorm(d_model)
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
        states: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            states: Optional list of state dicts, one per layer.

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size).
            states: Updated states.
        """
        if states is None:
            states = [None] * self.num_layers

        x = self.embedding(input_ids) * self.embed_scale
        x = self.embed_dropout(x)

        new_states = []
        for i, block in enumerate(self.blocks):
            x, state_i = block(x, states[i])
            new_states.append(state_i)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_states
