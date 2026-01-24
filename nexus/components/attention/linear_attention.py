"""
Linear Attention implementations.

Linear attention achieves O(n) complexity by using kernel feature maps
instead of the standard softmax attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Callable
from nexus.core.base import NexusModule


class LinearAttention(NexusModule):
    """Linear Attention with O(n) complexity.

    Uses kernel trick: instead of softmax(QK^T)V, computes φ(Q)(φ(K)^T V)
    where φ is a feature map. This allows computing in O(n) instead of O(n²).

    Reference: https://arxiv.org/abs/2006.16236 (Transformers are RNNs)

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        feature_map: Type of feature map ('elu', 'relu', 'softmax_kernel', 'favor')
        eps: Small constant for numerical stability
        dropout: Dropout probability
        bias: Whether to use bias
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        feature_map: str = 'elu',
        eps: float = 1e-6,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.feature_map_type = feature_map
        self.eps = eps

        # Select feature map
        self.feature_map = self._get_feature_map(feature_map)

        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def _get_feature_map(self, name: str) -> Callable:
        """Get feature map function by name."""
        if name == 'elu':
            return lambda x: F.elu(x) + 1
        elif name == 'relu':
            return F.relu
        elif name == 'softmax_kernel':
            # Approximates softmax via exp kernel
            return lambda x: torch.exp(x - x.max(dim=-1, keepdim=True).values)
        elif name == 'identity':
            return lambda x: x
        else:
            raise ValueError(f"Unknown feature map: {name}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        causal: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with linear attention.

        Args:
            hidden_states: Input of shape (batch, seq_len, dim)
            attention_mask: Not used in standard linear attention
            past_key_value: Cached (kv_state, k_sum) for incremental decoding
            use_cache: Whether to return cache
            causal: Whether to use causal masking

        Returns:
            output: Shape (batch, seq_len, dim)
            past_key_value: Updated cache if use_cache=True
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape: (batch, seq, num_heads, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply feature map
        query = self.feature_map(query)
        key = self.feature_map(key)

        if causal:
            output, new_cache = self._causal_linear_attention(
                query, key, value, past_key_value
            )
        else:
            output, new_cache = self._bidirectional_linear_attention(
                query, key, value
            )

        # Reshape and project output
        output = output.view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        output = self.dropout(output)

        past_key_value = new_cache if use_cache else None
        return output, past_key_value

    def _causal_linear_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Causal linear attention using cumulative sums.

        Computes: output[t] = (Σ_{i<=t} φ(k_i) ⊗ v_i) @ φ(q_t) / (Σ_{i<=t} φ(k_i)) @ φ(q_t)
        """
        batch_size, seq_len, num_heads, head_dim = query.shape

        # Initialize or retrieve cache
        if past_key_value is not None:
            kv_state, k_sum = past_key_value
        else:
            kv_state = torch.zeros(
                batch_size, num_heads, head_dim, head_dim,
                device=query.device, dtype=query.dtype
            )
            k_sum = torch.zeros(
                batch_size, num_heads, head_dim,
                device=query.device, dtype=query.dtype
            )

        outputs = []

        for t in range(seq_len):
            q_t = query[:, t]  # (batch, heads, dim)
            k_t = key[:, t]
            v_t = value[:, t]

            # Update running sums
            # kv_state += outer(k_t, v_t)
            kv_state = kv_state + torch.einsum('bhd,bhe->bhde', k_t, v_t)
            k_sum = k_sum + k_t

            # Compute output
            # numerator = kv_state @ q_t
            numerator = torch.einsum('bhde,bhd->bhe', kv_state, q_t)
            # denominator = k_sum @ q_t
            denominator = torch.einsum('bhd,bhd->bh', k_sum, q_t).unsqueeze(-1)

            output_t = numerator / (denominator + self.eps)
            outputs.append(output_t)

        output = torch.stack(outputs, dim=1)
        return output, (kv_state, k_sum)

    def _bidirectional_linear_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """
        Bidirectional linear attention.

        Computes: output = (φ(Q) @ (φ(K)^T @ V)) / (φ(Q) @ Σφ(K))
        """
        # Compute K^T @ V: (batch, heads, dim, dim)
        kv = torch.einsum('bshd,bshe->bhde', key, value)

        # Compute Q @ (K^T @ V): (batch, seq, heads, dim)
        numerator = torch.einsum('bshd,bhde->bshe', query, kv)

        # Compute normalization
        k_sum = key.sum(dim=1)  # (batch, heads, dim)
        denominator = torch.einsum('bshd,bhd->bsh', query, k_sum).unsqueeze(-1)

        output = numerator / (denominator + self.eps)
        return output, None


class CausalLinearAttention(LinearAttention):
    """Linear attention with causal masking by default."""

    def forward(self, hidden_states, **kwargs):
        kwargs['causal'] = True
        return super().forward(hidden_states, **kwargs)


class FAVORPlusAttention(NexusModule):
    """FAVOR+ (Fast Attention Via Orthogonal Random features).

    Uses random feature maps to approximate softmax attention with
    unbiased estimation. From the Performer paper.

    Reference: https://arxiv.org/abs/2009.14794

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        num_features: Number of random features (default: head_dim)
        ortho_features: Whether to use orthogonal random features
        redraw_features: Whether to redraw features on each forward
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        num_features: Optional[int] = None,
        ortho_features: bool = True,
        redraw_features: bool = False,
        bias: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.num_features = num_features or self.head_dim
        self.ortho_features = ortho_features
        self.redraw_features = redraw_features

        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=bias)

        # Random projection matrix
        self.register_buffer(
            'projection_matrix',
            self._create_projection_matrix()
        )

    def _create_projection_matrix(self) -> torch.Tensor:
        """Create (orthogonal) random projection matrix."""
        if self.ortho_features:
            # Create orthogonal random features
            num_blocks = math.ceil(self.num_features / self.head_dim)
            blocks = []
            for _ in range(num_blocks):
                random_matrix = torch.randn(self.head_dim, self.head_dim)
                q, _ = torch.linalg.qr(random_matrix)
                blocks.append(q)
            projection = torch.cat(blocks, dim=0)[:self.num_features]
        else:
            projection = torch.randn(self.num_features, self.head_dim)
            projection = projection / math.sqrt(self.head_dim)

        return projection

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FAVOR+ feature map.

        φ(x) = exp(x @ W - ||x||²/2) / sqrt(m)
        where W is the random projection matrix
        """
        # x: (batch, seq, heads, head_dim)
        # projection: (num_features, head_dim)

        # Project: (batch, seq, heads, num_features)
        x_proj = torch.einsum('bshd,fd->bshf', x, self.projection_matrix)

        # Normalize
        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True) / 2
        features = torch.exp(x_proj - x_norm_sq) / math.sqrt(self.num_features)

        return features

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal: bool = True
    ) -> torch.Tensor:
        """Forward pass with FAVOR+ attention."""
        if self.redraw_features:
            self.projection_matrix = self._create_projection_matrix().to(hidden_states.device)

        batch_size, seq_len, _ = hidden_states.shape

        # Project
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply feature map
        query = self._feature_map(query)
        key = self._feature_map(key)

        if causal:
            # Use cumulative sum for causal
            # This is O(n) instead of O(n²)
            kv = torch.einsum('bshf,bshd->bhfd', key, value).cumsum(dim=2)
            k_sum = key.cumsum(dim=1)

            # Gather for each position
            numerator = torch.einsum('bshf,bhfd->bshd', query, kv)
            denominator = torch.einsum('bshf,bshf->bsh', query, k_sum).unsqueeze(-1)
        else:
            kv = torch.einsum('bshf,bshd->bhfd', key, value)
            k_sum = key.sum(dim=1)

            numerator = torch.einsum('bshf,bhfd->bshd', query, kv)
            denominator = torch.einsum('bshf,bhf->bsh', query, k_sum).unsqueeze(-1)

        output = numerator / (denominator + 1e-6)
        output = output.view(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output
