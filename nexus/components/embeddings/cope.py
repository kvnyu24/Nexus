"""
Contextual Position Encoding (CoPE).

Position encoding that depends on context, not just position.
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class CoPE(NexusModule):
    """
    Contextual Position Encoding.

    Position encoding that depends on context, not just absolute position.
    CoPE computes positions based on the cumulative sum of "gates" that
    are derived from the content, allowing the model to learn position
    representations that are sensitive to the input content.

    Key idea: Instead of using fixed position indices, CoPE computes
    "soft" positions based on learned gates applied to the input.
    This allows positions to be content-dependent and enables better
    generalization across different sequence structures.

    Reference: https://arxiv.org/abs/2405.18719 (Contextual Position Encoding:
               Learning to Count What's Important)

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        max_pos: Maximum position (for position embedding table)
        gate_type: Type of gating ('sigmoid', 'softmax', 'linear')
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_pos: int = 2048,
        gate_type: str = 'sigmoid'
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.max_pos = max_pos
        self.gate_type = gate_type
        self.head_dim = dim // num_heads

        # Gate projection: computes content-based gates
        # Each token produces a gate value indicating "how much" it counts
        self.gate_proj = nn.Linear(dim, num_heads, bias=True)

        # Position embeddings table
        # These are looked up based on computed (soft) positions
        self.pos_emb = nn.Embedding(max_pos, self.head_dim)

        # Query projection for position computation
        # Maps queries to position query space
        self.pos_query = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.pos_query.weight, std=0.02)

    def compute_gates(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute content-based gate values.

        Args:
            x: Input tensor (batch, seq_len, dim)

        Returns:
            Gates of shape (batch, seq_len, num_heads)
        """
        gates = self.gate_proj(x)  # (batch, seq_len, num_heads)

        if self.gate_type == 'sigmoid':
            gates = torch.sigmoid(gates)
        elif self.gate_type == 'softmax':
            gates = torch.softmax(gates, dim=1)
        # 'linear' keeps gates as-is

        return gates

    def compute_positions(
        self,
        gates: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute soft positions from gates.

        The position of token i is the cumulative sum of gates up to i.
        This makes positions content-dependent.

        Args:
            gates: Gate values (batch, seq_len, num_heads)

        Returns:
            Positions of shape (batch, seq_len, num_heads)
        """
        # Cumulative sum gives "soft" positions
        positions = torch.cumsum(gates, dim=1)
        return positions

    def interpolate_pos_emb(
        self,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate position embeddings for soft (non-integer) positions.

        Uses linear interpolation between adjacent position embeddings.

        Args:
            positions: Soft positions (batch, seq_len, num_heads)

        Returns:
            Position embeddings (batch, seq_len, num_heads, head_dim)
        """
        batch_size, seq_len, num_heads = positions.shape

        # Clamp positions to valid range
        positions = torch.clamp(positions, 0, self.max_pos - 1.001)

        # Get floor and ceil positions
        pos_floor = positions.floor().long()
        pos_ceil = (pos_floor + 1).clamp(max=self.max_pos - 1)

        # Get embeddings for floor and ceil
        # Shape: (batch, seq_len, num_heads, head_dim)
        emb_floor = self.pos_emb(pos_floor)
        emb_ceil = self.pos_emb(pos_ceil)

        # Linear interpolation weight
        weight = (positions - pos_floor.float()).unsqueeze(-1)

        # Interpolate
        pos_emb = emb_floor * (1 - weight) + emb_ceil * weight

        return pos_emb

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        gates: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute CoPE position-aware attention bias.

        Args:
            q: Query tensor (batch, num_heads, seq_len, head_dim)
            k: Key tensor (batch, num_heads, seq_len, head_dim)
            x: Input tensor for gate computation (batch, seq_len, dim)
               Required if gates not provided
            gates: Pre-computed gates (batch, seq_len, num_heads)

        Returns:
            Position-aware attention bias (batch, num_heads, seq_len, seq_len)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Compute gates if not provided
        if gates is None:
            if x is None:
                raise ValueError("Either x or gates must be provided")
            gates = self.compute_gates(x)

        # Compute soft positions
        positions = self.compute_positions(gates)  # (batch, seq_len, num_heads)

        # Get position embeddings via interpolation
        pos_emb = self.interpolate_pos_emb(positions)  # (batch, seq_len, num_heads, head_dim)

        # Transpose for head-first format
        pos_emb = pos_emb.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)

        # Compute position queries
        q_pos = self.pos_query(q)  # (batch, num_heads, seq_len, head_dim)

        # Compute position bias: q_pos @ pos_emb^T
        # For each query position, compute similarity with all key positions
        pos_bias = torch.matmul(q_pos, pos_emb.transpose(-2, -1))

        # Scale by head dimension
        pos_bias = pos_bias / math.sqrt(head_dim)

        return pos_bias

    def get_relative_positions(
        self,
        gates: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relative positions between all pairs.

        Args:
            gates: Gate values (batch, seq_len, num_heads)

        Returns:
            Relative positions (batch, num_heads, seq_len, seq_len)
        """
        positions = self.compute_positions(gates)  # (batch, seq_len, num_heads)
        positions = positions.permute(0, 2, 1)  # (batch, num_heads, seq_len)

        # Compute relative positions: pos[i] - pos[j]
        relative_pos = positions.unsqueeze(-1) - positions.unsqueeze(-2)

        return relative_pos


class CoPEWithRoPE(NexusModule):
    """
    CoPE combined with RoPE for hybrid position encoding.

    Uses CoPE for content-dependent relative positions and RoPE for
    absolute position encoding, combining the benefits of both approaches.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        max_pos: Maximum position for CoPE
        rope_base: Base frequency for RoPE
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_pos: int = 2048,
        rope_base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # CoPE component
        self.cope = CoPE(dim, num_heads, max_pos)

        # RoPE frequencies
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq)

    def _compute_rope(
        self,
        seq_len: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin embeddings."""
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply combined CoPE and RoPE.

        Args:
            q: Query tensor (batch, num_heads, seq_len, head_dim)
            k: Key tensor (batch, num_heads, seq_len, head_dim)
            x: Input tensor (batch, seq_len, dim)

        Returns:
            q_rotated: Rotated queries
            k_rotated: Rotated keys
            cope_bias: CoPE attention bias
        """
        seq_len = q.shape[2]

        # Apply RoPE
        cos, sin = self._compute_rope(seq_len, q.device)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rotated = (q * cos) + (rotate_half(q) * sin)
        k_rotated = (k * cos) + (rotate_half(k) * sin)

        # Compute CoPE bias
        cope_bias = self.cope(q, k, x)

        return q_rotated, k_rotated, cope_bias
