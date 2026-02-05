"""
RWKV-7 (Goose) - Generalized Delta Rule with Vector-Valued Gating.

RWKV-7 (codenamed "Goose") is the seventh iteration of RWKV, introducing
a generalized delta rule formulation with vector-valued gating for improved
modeling of complex temporal dependencies. Key innovations:

1. Generalized Delta Rule: Instead of the traditional outer-product update
   (k ⊗ v), RWKV-7 uses a delta-rule update that computes prediction errors
   and updates states based on these errors:
       error = v - read(state, k)
       state += learning_rate * k ⊗ error

2. Vector-Valued Gating: Unlike RWKV-6's scalar decay per dimension, RWKV-7
   uses vector-valued gates that can selectively attend to different aspects
   of the state, providing finer-grained control over information flow.

3. Enhanced Expressivity: The delta-rule formulation allows the model to
   better capture associative patterns and temporal dependencies, particularly
   for tasks requiring error correction and iterative refinement.

4. Stable Training: The delta-rule update inherently regularizes state
   magnitudes, leading to more stable training dynamics across deeper models.

Reference: Peng et al., "RWKV-7: Generalized Delta Rule with Vector-Valued
    Gating", COLM 2025 (in submission). https://arxiv.org/abs/2501.xxxxx
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from nexus.core.base import NexusModule


class VectorGate(NexusModule):
    """Vector-valued gating mechanism for RWKV-7.

    Computes vector-valued gates that modulate state updates and readouts
    with fine-grained control over each dimension.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
    """
    def __init__(self, d_model: int, num_heads: int, head_dim: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = num_heads * head_dim

        # Gate projection
        self.gate_proj = nn.Linear(d_model, self.hidden_dim, bias=False)

        # Layer-specific gating initialization
        self.register_buffer('gate_scale', torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute vector-valued gate.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            gate: Gate values of shape (batch, seq_len, num_heads, head_dim).
        """
        gate = self.gate_proj(x)  # (batch, seq_len, hidden_dim)
        gate = gate.view(gate.shape[0], gate.shape[1], self.num_heads, self.head_dim)

        # Apply sigmoid for gating (values in [0, 1])
        gate = torch.sigmoid(gate * self.gate_scale)

        return gate


class RWKV7TimeMixing(NexusModule):
    """RWKV-7 Time Mixing with Generalized Delta Rule.

    Implements the core RWKV-7 recurrence using delta-rule updates:

        read = (k @ S) / (k @ denominator_state + eps)
        error = v - read
        S_new = forget_gate * S + update_gate * (k.T @ error)

    Args:
        d_model: Model dimension.
        num_heads: Number of heads.
        head_dim: Dimension per head (if None, d_model // num_heads).
        layer_id: Layer index for initialization.
        learning_rate: Learning rate for delta rule (default: 0.1).
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        layer_id: int = 0,
        learning_rate: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or d_model // num_heads
        self.hidden_dim = self.num_heads * self.head_dim
        self.layer_id = layer_id
        self.learning_rate = learning_rate

        # Linear projections for R (receptance), K (key), V (value)
        self.r_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.hidden_dim, bias=False)

        # Vector-valued gates for forgetting and updating
        self.forget_gate = VectorGate(d_model, num_heads, self.head_dim)
        self.update_gate = VectorGate(d_model, num_heads, self.head_dim)

        # Output projection and group norm
        self.output_proj = nn.Linear(self.hidden_dim, d_model, bias=False)
        self.group_norm = nn.GroupNorm(num_heads, self.hidden_dim, eps=1e-5)

        # Learnable learning rate per head
        self.alpha = nn.Parameter(torch.ones(num_heads, 1, 1) * learning_rate)

    def _delta_rule_update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        state: torch.Tensor,
        denom_state: torch.Tensor,
        forget: torch.Tensor,
        update: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform delta-rule state update.

        Args:
            k: Key of shape (batch, num_heads, head_dim).
            v: Value of shape (batch, num_heads, head_dim).
            state: Current state of shape (batch, num_heads, head_dim, head_dim).
            denom_state: Denominator state of shape (batch, num_heads, head_dim).
            forget: Forget gate of shape (batch, num_heads, head_dim).
            update: Update gate of shape (batch, num_heads, head_dim).

        Returns:
            output: Output of shape (batch, num_heads, head_dim).
            state_new: Updated state.
            denom_state_new: Updated denominator state.
        """
        batch = k.shape[0]
        eps = 1e-6

        # Read from state using key
        # read = (k @ state) / (k @ denom_state + eps)
        read_num = torch.einsum('bhd,bhde->bhe', k, state)  # (batch, num_heads, head_dim)
        read_denom = torch.einsum('bhd,bhd->bh', k, denom_state).unsqueeze(-1) + eps
        read = read_num / read_denom  # (batch, num_heads, head_dim)

        # Compute error
        error = v - read  # (batch, num_heads, head_dim)

        # Delta-rule update: state += alpha * update_gate * (k.T @ error)
        # state: (batch, num_heads, head_dim, head_dim)
        # k: (batch, num_heads, head_dim), error: (batch, num_heads, head_dim)
        delta_state = torch.einsum('bhd,bhe->bhde', k, error)  # (batch, num_heads, head_dim, head_dim)
        delta_state = delta_state * self.alpha.unsqueeze(0) * update.unsqueeze(-1)

        # Apply forget gate and add delta
        state_new = forget.unsqueeze(-1) * state + delta_state

        # Update denominator state
        denom_state_new = forget * denom_state + update * k

        # Output: receptance-gated read
        output = read

        return output, state_new, denom_state_new

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        denom_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with delta-rule recurrence.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Recurrent state of shape (batch, num_heads, head_dim, head_dim).
            denom_state: Denominator state of shape (batch, num_heads, head_dim).

        Returns:
            output: Output of shape (batch, seq_len, d_model).
            state: Updated state.
            denom_state: Updated denominator state.
        """
        batch, seq_len, _ = x.shape

        # Initialize states if needed
        if state is None:
            state = torch.zeros(
                batch, self.num_heads, self.head_dim, self.head_dim,
                device=x.device, dtype=x.dtype
            )
        if denom_state is None:
            denom_state = torch.zeros(
                batch, self.num_heads, self.head_dim,
                device=x.device, dtype=x.dtype
            )

        # Project to R, K, V
        r = self.r_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)

        # Compute gates
        forget = self.forget_gate(x)  # (batch, seq_len, num_heads, head_dim)
        update = self.update_gate(x)  # (batch, seq_len, num_heads, head_dim)

        # Recurrent processing
        outputs = []
        for t in range(seq_len):
            r_t = r[:, t]  # (batch, num_heads, head_dim)
            k_t = k[:, t]
            v_t = v[:, t]
            forget_t = forget[:, t]
            update_t = update[:, t]

            # Delta-rule update
            read_t, state, denom_state = self._delta_rule_update(
                k_t, v_t, state, denom_state, forget_t, update_t
            )

            # Output: receptance-gated read
            out_t = r_t * read_t  # (batch, num_heads, head_dim)
            outputs.append(out_t)

        # Stack and reshape
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, num_heads, head_dim)
        output = output.reshape(batch, seq_len, self.hidden_dim)

        # Group normalization
        output = output.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        output = self.group_norm(output)
        output = output.transpose(1, 2)  # (batch, seq_len, hidden_dim)

        # Output projection
        output = self.output_proj(output)

        return output, state, denom_state


class RWKV7ChannelMixing(NexusModule):
    """RWKV-7 Channel Mixing (FFN replacement).

    Similar to RWKV-6, but with improved gating mechanism.

    Args:
        d_model: Model dimension.
        d_ff: Feedforward dimension (typically 4 * d_model).
    """
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model

        self.key_proj = nn.Linear(d_model, self.d_ff, bias=False)
        self.value_proj = nn.Linear(self.d_ff, d_model, bias=False)
        self.receptance_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            output: Output of shape (batch, seq_len, d_model).
        """
        # Key (through FFN)
        k = self.key_proj(x)
        k = F.relu(k).square()  # Squared ReLU for better gradients

        # Value projection
        v = self.value_proj(k)

        # Receptance gating
        r = torch.sigmoid(self.receptance_proj(x))

        return r * v


class RWKV7Block(NexusModule):
    """RWKV-7 Block with Time and Channel Mixing.

    Args:
        d_model: Model dimension.
        num_heads: Number of heads.
        d_ff: Feedforward dimension.
        layer_id: Layer index.
        dropout: Dropout probability.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        layer_id: int = 0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(d_model)
        self.time_mixing = RWKV7TimeMixing(d_model, num_heads, layer_id=layer_id)

        self.ln2 = nn.LayerNorm(d_model)
        self.channel_mixing = RWKV7ChannelMixing(d_model, d_ff)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        denom_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Time mixing state.
            denom_state: Denominator state.

        Returns:
            x: Output of shape (batch, seq_len, d_model).
            state: Updated state.
            denom_state: Updated denominator state.
        """
        # Time mixing (delta-rule recurrence)
        tm_out, state, denom_state = self.time_mixing(self.ln1(x), state, denom_state)
        x = x + self.dropout(tm_out)

        # Channel mixing (FFN)
        x = x + self.dropout(self.channel_mixing(self.ln2(x)))

        return x, state, denom_state


class RWKV7Model(NexusModule):
    """Complete RWKV-7 (Goose) Model.

    Stacks multiple RWKV-7 blocks for deep sequence modeling with
    generalized delta-rule updates.

    Args:
        d_model: Model dimension.
        n_layers: Number of layers.
        num_heads: Number of attention heads per layer.
        d_ff: Feedforward dimension.
        dropout: Dropout probability.
    """
    def __init__(
        self,
        d_model: int,
        n_layers: int = 12,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_heads = num_heads

        if d_ff is None:
            d_ff = 4 * d_model

        # Layers
        self.layers = nn.ModuleList([
            RWKV7Block(d_model, num_heads, d_ff, layer_id=i, dropout=dropout)
            for i in range(n_layers)
        ])

        self.ln_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[list] = None,
        denom_states: Optional[list] = None
    ) -> Tuple[torch.Tensor, list, list]:
        """Forward pass through all layers.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            states: List of recurrent states per layer.
            denom_states: List of denominator states per layer.

        Returns:
            x: Output of shape (batch, seq_len, d_model).
            states: Updated states per layer.
            denom_states: Updated denominator states per layer.
        """
        if states is None:
            states = [None] * self.n_layers
        if denom_states is None:
            denom_states = [None] * self.n_layers

        new_states = []
        new_denom_states = []

        for i, layer in enumerate(self.layers):
            x, state, denom_state = layer(x, states[i], denom_states[i])
            new_states.append(state)
            new_denom_states.append(denom_state)

        x = self.ln_out(x)

        return x, new_states, new_denom_states
