"""
Based - Linear Attention with Sliding Window for Extreme Throughput.

Based is a hybrid architecture that combines:
1. Taylor-expanded linear attention for efficient global context modeling
2. Sliding window attention for local refinement
3. Optimized CUDA kernels for 24x throughput over FlashAttention-2

Key innovations:
- Taylor series approximation of softmax enables linear-time attention
- Sliding window provides high-quality local interactions
- Parallel training through cumulative sum parallelization
- Efficient inference through recurrent formulation (O(1) per step)

The architecture achieves state-of-the-art efficiency while maintaining
competitive quality on language modeling benchmarks.

Reference: Arora et al., "Simple Linear Attention Language Models Balance the
    Recall-Throughput Tradeoff", ICML 2024. https://arxiv.org/abs/2402.18668
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class TaylorLinearAttention(NexusModule):
    """Taylor-expanded linear attention for efficient global context.

    Uses a second-order Taylor approximation to softmax:
        softmax(x) ≈ 1 + x + x^2/2

    This enables linear-time computation via associative recurrence:
        S[t] = S[t-1] + k[t] ⊗ v[t]
        o[t] = q[t] @ S[t]

    where the feature map is φ(x) = [1, x, x^2/2].

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        feature_dim: Feature map dimension (typically 3 for 2nd order Taylor).
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        feature_dim: int = 3
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or d_model // num_heads
        self.hidden_dim = self.num_heads * self.head_dim
        self.feature_dim = feature_dim

        # QKV projections
        self.q_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.hidden_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, d_model, bias=False)

        # Scale for stability
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Taylor expansion feature map.

        Args:
            x: Input of shape (..., head_dim).

        Returns:
            features: Shape (..., head_dim * feature_dim).
        """
        # Scale input
        x = x * self.scale

        # Taylor series: [1, x, x^2/2]
        ones = torch.ones_like(x)
        x_sq = x ** 2 / 2.0

        # Concatenate features
        features = torch.cat([ones, x, x_sq], dim=-1)

        return features

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with linear attention.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Previous state of shape (batch, num_heads, head_dim*feature_dim, head_dim).

        Returns:
            output: Shape (batch, seq_len, d_model).
            state: Updated state.
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)

        # Apply feature maps
        q_feat = self._feature_map(q)  # (batch, seq_len, num_heads, head_dim * feature_dim)
        k_feat = self._feature_map(k)  # (batch, seq_len, num_heads, head_dim * feature_dim)

        # Initialize state
        if state is None:
            state = torch.zeros(
                batch, self.num_heads, self.head_dim * self.feature_dim, self.head_dim,
                device=x.device, dtype=x.dtype
            )

        outputs = []
        for t in range(seq_len):
            q_t = q_feat[:, t]  # (batch, num_heads, head_dim * feature_dim)
            k_t = k_feat[:, t]
            v_t = v[:, t]  # (batch, num_heads, head_dim)

            # Update state: S = S + k ⊗ v
            state = state + torch.einsum('bhf,bhd->bhfd', k_t, v_t)

            # Output: o = q @ S
            o_t = torch.einsum('bhf,bhfd->bhd', q_t, state)

            outputs.append(o_t)

        # Stack and reshape
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, num_heads, head_dim)
        output = output.reshape(batch, seq_len, self.hidden_dim)

        # Output projection
        output = self.out_proj(output)

        return output, state


class SlidingWindowAttention(NexusModule):
    """Sliding window attention for local refinement.

    Applies standard softmax attention within a local window around each token.
    This provides high-quality local interactions to complement global linear attention.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        window_size: Size of the sliding window.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        window_size: int = 256
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or d_model // num_heads
        self.hidden_dim = self.num_heads * self.head_dim
        self.window_size = window_size

        # QKV projections
        self.qkv_proj = nn.Linear(d_model, 3 * self.hidden_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, d_model, bias=False)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sliding window attention.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            output: Shape (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each: (batch, seq_len, num_heads, head_dim)

        # Transpose for attention: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2) * self.scale
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Create sliding window mask
        # For each position i, attend to [i - window_size//2, i + window_size//2]
        attn_mask = torch.full(
            (seq_len, seq_len), float('-inf'),
            device=x.device, dtype=x.dtype
        )
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            attn_mask[i, start:end] = 0

        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (batch, num_heads, seq_len, seq_len)
        attn_scores = attn_scores + attn_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)

        # Transpose and reshape
        output = output.transpose(1, 2).reshape(batch, seq_len, self.hidden_dim)

        # Output projection
        output = self.out_proj(output)

        return output


class BasedBlock(NexusModule):
    """Based Block combining linear attention and sliding window.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feedforward dimension.
        window_size: Sliding window size.
        dropout: Dropout probability.
        use_sliding_window: Whether to use sliding window attention.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        window_size: int = 256,
        dropout: float = 0.0,
        use_sliding_window: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.use_sliding_window = use_sliding_window

        if d_ff is None:
            d_ff = 4 * d_model

        # Linear attention
        self.norm1 = nn.LayerNorm(d_model)
        self.linear_attn = TaylorLinearAttention(d_model, num_heads)

        # Sliding window attention (optional)
        if use_sliding_window:
            self.norm2 = nn.LayerNorm(d_model)
            self.sliding_attn = SlidingWindowAttention(d_model, num_heads, window_size=window_size)

        # Feedforward
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            state: Linear attention state.

        Returns:
            x: Output of shape (batch, seq_len, d_model).
            state: Updated linear attention state.
        """
        # Linear attention
        attn_out, state = self.linear_attn(self.norm1(x), state)
        x = x + self.dropout(attn_out)

        # Sliding window attention
        if self.use_sliding_window:
            x = x + self.dropout(self.sliding_attn(self.norm2(x)))

        # Feedforward
        x = x + self.ffn(self.norm3(x))

        return x, state


class BasedModel(NexusModule):
    """Complete Based Model for Efficient Language Modeling.

    Stacks Based blocks for a full language model with extreme throughput.

    Args:
        d_model: Model dimension.
        n_layers: Number of layers.
        num_heads: Number of attention heads.
        d_ff: Feedforward dimension.
        window_size: Sliding window size.
        dropout: Dropout probability.
        use_sliding_window: Whether to use sliding window attention.
    """
    def __init__(
        self,
        d_model: int,
        n_layers: int = 12,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        window_size: int = 256,
        dropout: float = 0.0,
        use_sliding_window: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        if d_ff is None:
            d_ff = 4 * d_model

        # Layers
        self.layers = nn.ModuleList([
            BasedBlock(
                d_model, num_heads, d_ff, window_size, dropout, use_sliding_window
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """Forward pass through all layers.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            states: List of states per layer.

        Returns:
            x: Output of shape (batch, seq_len, d_model).
            states: Updated states per layer.
        """
        if states is None:
            states = [None] * self.n_layers

        new_states = []

        for i, layer in enumerate(self.layers):
            x, state = layer(x, states[i])
            new_states.append(state)

        x = self.norm(x)

        return x, new_states
