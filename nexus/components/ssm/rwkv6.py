"""
RWKV-6 (Finch) - Receptance Weighted Key Value Architecture, Version 6.

RWKV-6 (codenamed "Finch") is the sixth iteration of the RWKV architecture,
combining the efficient inference of RNNs with the parallelizable training of
transformers. Key innovations in v6:

1. Matrix-valued recurrent states: Instead of vector-valued states, RWKV-6
   uses matrix-valued states that store key-value associations, similar to
   linear attention's outer product state.

2. Dynamic recurrence (data-dependent decay): The decay factor is no longer
   a static learned parameter but depends on the input through a learned
   transformation, enabling adaptive forgetting.

3. Token shift mechanism: Each timestep uses a weighted mix of the current
   and previous token embeddings, providing a simple form of local context.

4. Time mixing and channel mixing: The architecture uses two types of mixing
   blocks -- time mixing (recurrent, replaces attention) and channel mixing
   (feed-forward, replaces MLP).

Reference: Peng et al., "Eagle and Finch: RWKV with Matrix-Valued States
    and Dynamic Recurrence", 2024. https://arxiv.org/abs/2404.05892

See also: Peng et al., "RWKV: Reinventing RNNs for the Transformer Era",
    EMNLP 2023. https://arxiv.org/abs/2305.13048
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from nexus.core.base import NexusModule


class TokenShift(NexusModule):
    """Token shift mechanism for RWKV.

    Computes a weighted mix of the current token and the previous token:
        x_shifted = lerp(x[t], x[t-1], mix_ratio)

    where mix_ratio is a learnable per-channel parameter. This provides
    a simple, parameter-efficient way to incorporate local context without
    convolutions.

    In RWKV-6, different components (R, K, V, G, W) each have their own
    learnable mix ratios.

    Args:
        d_model: Model dimension.
        num_shifts: Number of independent shift parameters (default: 5
            for R, K, V, gate, and decay in time mixing).
    """

    def __init__(self, d_model: int, num_shifts: int = 5):
        super().__init__()
        self.d_model = d_model
        self.num_shifts = num_shifts

        # Learnable mix ratios for each shifted output
        self.mix = nn.Parameter(torch.zeros(num_shifts, d_model))

    def forward(
        self,
        x: torch.Tensor,
        last_x: Optional[torch.Tensor] = None
    ) -> Tuple[list, torch.Tensor]:
        """Apply token shift.

        Args:
            x: Current input of shape (batch, seq_len, d_model).
            last_x: Previous token of shape (batch, d_model) for recurrent mode.
                If None, uses zero for the first position.

        Returns:
            shifted: List of num_shifts tensors, each (batch, seq_len, d_model).
            last_token: Last token for next call, shape (batch, d_model).
        """
        batch_size, seq_len, d_model = x.shape

        # Construct shifted input (previous token for each position)
        if last_x is None:
            last_x = torch.zeros(batch_size, d_model, device=x.device, dtype=x.dtype)

        # Shift: prepend last_x, remove last token
        x_prev = torch.cat([
            last_x.unsqueeze(1),
            x[:, :-1, :]
        ], dim=1)  # (batch, seq, d_model)

        # Apply learnable mixing for each shift
        shifted = []
        for i in range(self.num_shifts):
            mix_i = torch.sigmoid(self.mix[i]).unsqueeze(0).unsqueeze(0)
            shifted_i = x * (1 - mix_i) + x_prev * mix_i
            shifted.append(shifted_i)

        # Return last token for next recurrent call
        last_token = x[:, -1, :]

        return shifted, last_token


class RWKV6TimeMixing(NexusModule):
    """RWKV-6 Time Mixing block (replaces attention).

    Implements the core RWKV-6 recurrence with matrix-valued states and
    data-dependent decay:

        wkv state: S[t] = diag(w[t]) * S[t-1] + k[t] outer v[t]
        output: o[t] = r[t] @ (S[t] @ 1) (simplified; actual uses attention-like readout)

    where:
    - r (receptance) acts as a query/gate
    - k (key) determines what to write
    - v (value) determines what information to store
    - w (decay) is data-dependent, controlling per-dimension forgetting
    - g (gate) provides an additional output gating mechanism

    Args:
        d_model: Model dimension.
        num_heads: Number of heads (default: 8).
        head_dim: Dimension per head. If None, computed as d_model // num_heads.
        layer_id: Layer index for initialization scaling (default: 0).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        layer_id: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or d_model // num_heads
        self.hidden_dim = self.num_heads * self.head_dim
        self.layer_id = layer_id

        assert self.hidden_dim == d_model or self.hidden_dim > 0

        # Token shift for R, K, V, Gate, Decay
        self.token_shift = TokenShift(d_model, num_shifts=5)

        # Linear projections
        self.r_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.g_proj = nn.Linear(d_model, self.hidden_dim, bias=False)

        # Data-dependent decay (RWKV-6 innovation)
        # The decay is computed as: w = base_w + dynamic_w(x)
        # This allows input-dependent forgetting
        self.w_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.w_base = nn.Parameter(torch.randn(self.hidden_dim) * 0.1)

        # Bonus term (per-head, for attention-like scoring)
        self.bonus = nn.Parameter(torch.zeros(num_heads, self.head_dim))

        # Output normalization (group norm across heads)
        self.group_norm = nn.GroupNorm(num_heads, self.hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Layer-dependent initialization."""
        # Scale initialization based on layer depth
        scale = 1.0 / (self.layer_id + 1)
        nn.init.uniform_(self.r_proj.weight, -scale, scale)
        nn.init.uniform_(self.k_proj.weight, -scale, scale)
        nn.init.uniform_(self.v_proj.weight, -scale, scale)
        nn.init.orthogonal_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of RWKV-6 time mixing.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Optional dict with keys:
                - 'wkv': Matrix state of shape (batch, num_heads, head_dim, head_dim)
                - 'last_x': Previous token of shape (batch, d_model)

        Returns:
            output: Output of shape (batch, seq_len, d_model).
            state: Updated state dict.
        """
        batch_size, seq_len, _ = x.shape

        # Initialize state
        if state is None:
            state = {
                'wkv': torch.zeros(
                    batch_size, self.num_heads, self.head_dim, self.head_dim,
                    device=x.device, dtype=x.dtype
                ),
                'last_x': None
            }

        # Token shift
        shifted, last_token = self.token_shift(x, state.get('last_x'))
        x_r, x_k, x_v, x_g, x_w = shifted

        # Compute R, K, V, G
        r = self.r_proj(x_r)  # (batch, seq, hidden)
        k = self.k_proj(x_k)
        v = self.v_proj(x_v)
        g = F.silu(self.g_proj(x_g))

        # Compute data-dependent decay w
        w = self.w_base.unsqueeze(0).unsqueeze(0) + self.w_proj(x_w)
        w = -F.softplus(w)  # Negative for decay (ensures decay <= 1 after exp)

        # Reshape to multi-head
        r = r.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        w = w.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Run WKV recurrence
        if self.training:
            output, wkv_state = self._forward_parallel(r, k, v, w, state['wkv'])
        else:
            output, wkv_state = self._forward_recurrent(r, k, v, w, state['wkv'])

        # Reshape output
        output = output.view(batch_size, seq_len, self.hidden_dim)

        # Apply group norm
        output = self.group_norm(output.transpose(1, 2)).transpose(1, 2)

        # Apply gate
        output = output * g

        # Project output
        output = self.out_proj(output)

        # Update state
        new_state = {
            'wkv': wkv_state,
            'last_x': last_token
        }

        return output, new_state

    def _forward_parallel(
        self,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        wkv_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel forward for training.

        Processes the sequence using chunk-wise recurrence for efficiency.

        Args:
            r: Receptance (batch, seq, heads, head_dim).
            k: Key (batch, seq, heads, head_dim).
            v: Value (batch, seq, heads, head_dim).
            w: Decay (batch, seq, heads, head_dim).
            wkv_state: State (batch, heads, head_dim, head_dim).

        Returns:
            output: (batch, seq, heads, head_dim).
            state: Updated state.
        """
        batch_size, seq_len, num_heads, head_dim = r.shape

        outputs = []
        state = wkv_state

        for t in range(seq_len):
            r_t = r[:, t]  # (batch, heads, head_dim)
            k_t = k[:, t]
            v_t = v[:, t]
            w_t = w[:, t]  # (batch, heads, head_dim)

            # Apply decay: state = diag(exp(w)) @ state
            decay = torch.exp(w_t)  # (batch, heads, head_dim)
            state = state * decay.unsqueeze(-1)  # broadcast over last dim

            # Add new key-value pair: state += k outer v
            kv = torch.einsum('bhd,bhe->bhde', k_t, v_t)
            state = state + kv

            # Read with receptance: output = r @ state + bonus * (k . r) * v
            output_t = torch.einsum('bhd,bhde->bhe', r_t, state)

            # Bonus term for current-token attention
            bonus_score = torch.einsum(
                'bhd,bhd->bh', k_t * self.bonus.unsqueeze(0), r_t
            )  # (batch, heads)
            output_t = output_t + bonus_score.unsqueeze(-1) * v_t

            outputs.append(output_t)

        output = torch.stack(outputs, dim=1)
        return output, state

    def _forward_recurrent(
        self,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        wkv_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recurrent forward for inference.

        Args:
            r: Receptance (batch, seq, heads, head_dim).
            k: Key (batch, seq, heads, head_dim).
            v: Value (batch, seq, heads, head_dim).
            w: Decay (batch, seq, heads, head_dim).
            wkv_state: State (batch, heads, head_dim, head_dim).

        Returns:
            output: (batch, seq, heads, head_dim).
            state: Updated state.
        """
        # Recurrent mode is identical to parallel in this case
        # (both process step-by-step), but kept separate for clarity
        return self._forward_parallel(r, k, v, w, wkv_state)


class RWKV6ChannelMixing(NexusModule):
    """RWKV-6 Channel Mixing block (replaces feed-forward network).

    Implements the RWKV channel mixing:
        x_shifted = token_shift(x)
        k = x_shifted @ W_k
        r = sigmoid(x_shifted @ W_r)
        output = r * (ReLU(k)^2 @ W_v)

    The squared ReLU activation provides strong nonlinearity while
    maintaining efficiency. The receptance r gates the output.

    Args:
        d_model: Model dimension.
        expand: Expansion factor for hidden dimension (default: 4).
        layer_id: Layer index for initialization (default: 0).
    """

    def __init__(
        self,
        d_model: int,
        expand: int = 4,
        layer_id: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = d_model * expand
        self.layer_id = layer_id

        # Token shift for K and R
        self.token_shift = TokenShift(d_model, num_shifts=2)

        # Projections
        self.k_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.r_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        scale = 1.0 / (self.layer_id + 1)
        nn.init.uniform_(self.k_proj.weight, -scale, scale)
        nn.init.uniform_(self.r_proj.weight, -scale, scale)
        nn.init.orthogonal_(self.v_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        last_x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of channel mixing.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            last_x: Previous token for token shift.

        Returns:
            output: Shape (batch, seq_len, d_model).
            last_token: Last token for next call.
        """
        # Token shift
        shifted, last_token = self.token_shift(x, last_x)
        x_k, x_r = shifted

        # Compute K and R
        k = self.k_proj(x_k)
        r = torch.sigmoid(self.r_proj(x_r))

        # Squared ReLU activation
        k = F.relu(k) ** 2

        # Value projection and gating
        output = r * self.v_proj(k)

        return output, last_token


class RWKV6Block(NexusModule):
    """Single RWKV-6 block combining time mixing and channel mixing.

    Each block consists of:
    1. LayerNorm -> Time Mixing (replaces attention)
    2. LayerNorm -> Channel Mixing (replaces FFN)

    Both use residual connections.

    Reference: Peng et al., "Eagle and Finch: RWKV with Matrix-Valued States
        and Dynamic Recurrence", 2024. https://arxiv.org/abs/2404.05892

    Args:
        d_model: Model dimension.
        num_heads: Number of heads for time mixing (default: 8).
        layer_id: Layer index for initialization (default: 0).
        ffn_expand: Channel mixing expansion factor (default: 4).
        dropout: Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        layer_id: int = 0,
        ffn_expand: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model

        # Time mixing (recurrent attention replacement)
        self.norm1 = nn.LayerNorm(d_model)
        self.time_mixing = RWKV6TimeMixing(
            d_model=d_model,
            num_heads=num_heads,
            layer_id=layer_id
        )
        self.dropout1 = nn.Dropout(dropout)

        # Channel mixing (FFN replacement)
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mixing = RWKV6ChannelMixing(
            d_model=d_model,
            expand=ffn_expand,
            layer_id=layer_id
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of RWKV-6 block.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Optional state dict with keys:
                - 'time_state': Dict for time mixing state
                - 'channel_last_x': Last token for channel mixing

        Returns:
            output: Output of shape (batch, seq_len, d_model).
            state: Updated state dict.
        """
        if state is None:
            state = {
                'time_state': None,
                'channel_last_x': None
            }

        # Time mixing with residual
        residual = x
        x_norm = self.norm1(x)
        tm_out, time_state = self.time_mixing(x_norm, state.get('time_state'))
        x = residual + self.dropout1(tm_out)

        # Channel mixing with residual
        residual = x
        x_norm = self.norm2(x)
        cm_out, channel_last_x = self.channel_mixing(
            x_norm, state.get('channel_last_x')
        )
        x = residual + self.dropout2(cm_out)

        new_state = {
            'time_state': time_state,
            'channel_last_x': channel_last_x
        }

        return x, new_state


class RWKV6Model(NexusModule):
    """Full RWKV-6 language model.

    Stacks multiple RWKV-6 blocks with token embedding and language model head.

    Reference: Peng et al., "Eagle and Finch: RWKV with Matrix-Valued States
        and Dynamic Recurrence", 2024. https://arxiv.org/abs/2404.05892

    Args:
        d_model: Model dimension.
        num_layers: Number of RWKV-6 blocks.
        vocab_size: Vocabulary size.
        num_heads: Number of heads per block (default: 8).
        ffn_expand: Channel mixing expansion (default: 4).
        dropout: Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        vocab_size: int,
        num_heads: int = 8,
        ffn_expand: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_norm = nn.LayerNorm(d_model)

        # RWKV-6 blocks
        self.blocks = nn.ModuleList([
            RWKV6Block(
                d_model=d_model,
                num_heads=num_heads,
                layer_id=i,
                ffn_expand=ffn_expand,
                dropout=dropout
            )
            for i in range(num_layers)
        ])

        # Output head
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """Forward pass of RWKV-6 model.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            states: Optional list of state dicts, one per layer.

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size).
            states: Updated list of state dicts.
        """
        if states is None:
            states = [None] * self.num_layers

        x = self.embedding(input_ids)
        x = self.embed_norm(x)

        new_states = []
        for i, block in enumerate(self.blocks):
            x, state_i = block(x, states[i])
            new_states.append(state_i)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_states
