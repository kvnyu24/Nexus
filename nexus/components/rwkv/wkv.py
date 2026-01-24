"""
RWKV: Receptance Weighted Key Value architecture components.

RWKV combines the benefits of RNNs (efficient inference) with
transformers (parallelizable training) using the WKV operator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class TokenShift(NexusModule):
    """Token Shift operation for RWKV.

    Linearly interpolates between current and previous token embeddings
    to provide local context without explicit convolution.

    RWKV-4/5: Simple linear interpolation (lerp)
    RWKV-6: Data-dependent interpolation (ddlerp) using LoRA-style projection

    Args:
        dim: Model dimension
        shift_amount: Amount to shift (negative for previous tokens)
        use_ddlerp: Whether to use data-dependent lerp (RWKV-6 style)
        lora_rank: Rank for ddlerp projection (if use_ddlerp)
    """

    def __init__(
        self,
        dim: int,
        shift_amount: int = -1,
        use_ddlerp: bool = False,
        lora_rank: int = 32
    ):
        super().__init__()
        self.dim = dim
        self.shift_amount = shift_amount
        self.use_ddlerp = use_ddlerp

        if use_ddlerp:
            # LoRA-style projection for data-dependent mixing
            self.lora_a = nn.Linear(dim, lora_rank, bias=False)
            self.lora_b = nn.Linear(lora_rank, dim, bias=False)
            # Initialize B to zero for residual-like behavior
            nn.init.zeros_(self.lora_b.weight)
        else:
            # Fixed mixing weights (learnable)
            self.mix = nn.Parameter(torch.ones(dim) * 0.5)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply token shift.

        Args:
            x: Input of shape (batch, seq_len, dim)
            state: Previous token state for incremental inference

        Returns:
            shifted: Token-shifted output
            new_state: State for next step
        """
        batch_size, seq_len, dim = x.shape

        if state is None:
            state = torch.zeros(batch_size, 1, dim, device=x.device, dtype=x.dtype)

        # Concatenate state with input for shifting
        x_with_state = torch.cat([state, x], dim=1)

        # Get shifted version (previous tokens)
        x_shifted = x_with_state[:, :-1, :]  # All but last

        # Update state
        new_state = x[:, -1:, :]

        if self.use_ddlerp:
            # Data-dependent mixing weights
            mix = torch.sigmoid(self.lora_b(self.lora_a(x)))
        else:
            mix = torch.sigmoid(self.mix)

        # Linear interpolation between current and shifted
        output = x * mix + x_shifted * (1 - mix)

        return output, new_state


class WKVOperator(NexusModule):
    """Weighted Key-Value (WKV) operator - core of RWKV.

    Computes attention-like weighted sums with linear complexity using
    recurrent formulation. The key innovation is the time-decay factor
    that allows efficient sequential computation.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads (RWKV-5/6)
        head_dim: Dimension per head
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        head_dim: Optional[int] = None
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)

        # Time decay (w in paper, learned per head/dim)
        self.time_decay = nn.Parameter(torch.ones(num_heads, self.head_dim) * -5.0)

        # Bonus for current token (u in paper)
        self.time_first = nn.Parameter(torch.ones(num_heads, self.head_dim) * 0.5)

    def forward(
        self,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute WKV attention.

        Args:
            r: Receptance of shape (batch, seq, num_heads, head_dim)
            k: Key of shape (batch, seq, num_heads, head_dim)
            v: Value of shape (batch, seq, num_heads, head_dim)
            state: Tuple of (wkv_state, max_state) for incremental inference

        Returns:
            output: WKV output
            new_state: Updated state
        """
        batch_size, seq_len, num_heads, head_dim = r.shape

        # Get time decay and first bonus
        w = self.time_decay  # (heads, head_dim)
        u = self.time_first  # (heads, head_dim)

        # Initialize state
        if state is None:
            # (batch, heads, head_dim, head_dim) - stores sum of exp(k)*v
            wkv_state = torch.zeros(
                batch_size, num_heads, head_dim, head_dim,
                device=r.device, dtype=r.dtype
            )
            # (batch, heads, head_dim) - stores max k for numerical stability
            max_state = torch.full(
                (batch_size, num_heads, head_dim),
                float('-inf'),
                device=r.device, dtype=r.dtype
            )
        else:
            wkv_state, max_state = state

        outputs = []

        for t in range(seq_len):
            r_t = r[:, t]  # (batch, heads, head_dim)
            k_t = k[:, t]
            v_t = v[:, t]

            # Update max for numerical stability
            max_new = torch.maximum(max_state + w, k_t)

            # Compute weighted state update
            exp_w = torch.exp(max_state + w - max_new)
            exp_k = torch.exp(k_t - max_new)

            # WKV computation with current token bonus
            wkv_new = exp_w.unsqueeze(-1) * wkv_state + exp_k.unsqueeze(-1) * v_t.unsqueeze(-2)

            # Include current token with bonus
            exp_u = torch.exp(u + k_t - max_new)
            wkv_with_bonus = wkv_new + exp_u.unsqueeze(-1) * v_t.unsqueeze(-2)

            # Compute output with receptance gating
            # output = r * (wkv / normalizer)
            normalizer = exp_w * (wkv_state.sum(dim=-1) + 1e-6) + exp_k + exp_u
            output_t = r_t * (torch.einsum('bhde,bhe->bhd', wkv_with_bonus, r_t) /
                            (normalizer + 1e-6))

            outputs.append(output_t)

            # Update state for next iteration
            wkv_state = wkv_new
            max_state = max_new

        output = torch.stack(outputs, dim=1)
        return output, (wkv_state, max_state)


class TimeMixing(NexusModule):
    """RWKV Time-Mixing block.

    Handles temporal dependencies using the WKV operator with token shifting.
    This is analogous to attention in transformers.

    Args:
        dim: Model dimension
        num_heads: Number of heads
        head_dim: Dimension per head
        layer_id: Layer index (affects initialization)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        layer_id: int = 0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.layer_id = layer_id

        # Token shift for RWKV mixing
        self.token_shift = TokenShift(dim, use_ddlerp=True)

        # Projections
        self.receptance = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.key = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.value = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.gate = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.output = nn.Linear(num_heads * self.head_dim, dim, bias=False)

        # WKV operator
        self.wkv = WKVOperator(dim, num_heads, self.head_dim)

        # Layer normalization
        self.ln = nn.LayerNorm(num_heads * self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass of time mixing.

        Args:
            x: Input of shape (batch, seq_len, dim)
            state: Dictionary containing shift_state and wkv_state

        Returns:
            output: Time-mixed output
            new_state: Updated state dictionary
        """
        batch_size, seq_len, _ = x.shape

        # Extract states
        if state is None:
            shift_state = None
            wkv_state = None
        else:
            shift_state = state.get('shift_state')
            wkv_state = state.get('wkv_state')

        # Token shift
        x_shifted, shift_state = self.token_shift(x, shift_state)

        # Compute RWKV projections
        r = self.receptance(x_shifted)
        k = self.key(x_shifted)
        v = self.value(x_shifted)
        g = self.gate(x)  # Gate uses original x

        # Reshape for multi-head
        r = r.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply receptance sigmoid
        r = torch.sigmoid(r)

        # WKV computation
        wkv_out, wkv_state = self.wkv(r, k, v, wkv_state)

        # Reshape and apply gate
        wkv_out = wkv_out.view(batch_size, seq_len, -1)
        wkv_out = self.ln(wkv_out)
        output = self.output(wkv_out * F.silu(g.view(batch_size, seq_len, -1)))

        new_state = {
            'shift_state': shift_state,
            'wkv_state': wkv_state
        }

        return output, new_state


class ChannelMixing(NexusModule):
    """RWKV Channel-Mixing block.

    FFN-like component that mixes information across channels/features.
    Uses squared ReLU activation.

    Args:
        dim: Model dimension
        hidden_dim: Hidden dimension (default: 4 * dim)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or (dim * 4)

        # Token shift
        self.token_shift = TokenShift(dim, use_ddlerp=True)

        # Projections
        self.key = nn.Linear(dim, self.hidden_dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(self.hidden_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of channel mixing.

        Args:
            x: Input of shape (batch, seq_len, dim)
            state: Previous token state

        Returns:
            output: Channel-mixed output
            new_state: Updated state
        """
        # Token shift
        x_shifted, new_state = self.token_shift(x, state)

        # Channel mixing: r * value(relu²(key(x)))
        k = self.key(x_shifted)
        k = torch.relu(k) ** 2  # Squared ReLU
        kv = self.value(k)

        r = torch.sigmoid(self.receptance(x_shifted))
        output = r * kv

        return output, new_state


class RWKVBlock(NexusModule):
    """Complete RWKV block combining time and channel mixing.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension for channel mixing
        layer_id: Layer index
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        hidden_dim: Optional[int] = None,
        layer_id: int = 0
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(dim)
        self.time_mixing = TimeMixing(dim, num_heads, layer_id=layer_id)

        self.ln2 = nn.LayerNorm(dim)
        self.channel_mixing = ChannelMixing(dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass of RWKV block.

        Args:
            x: Input tensor
            state: Block state dictionary

        Returns:
            output: Block output
            new_state: Updated state
        """
        if state is None:
            state = {}

        # Time mixing with residual
        time_out, time_state = self.time_mixing(self.ln1(x), state.get('time'))
        x = x + time_out

        # Channel mixing with residual
        channel_out, channel_state = self.channel_mixing(self.ln2(x), state.get('channel'))
        x = x + channel_out

        new_state = {
            'time': time_state,
            'channel': channel_state
        }

        return x, new_state
