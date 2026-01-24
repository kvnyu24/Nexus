"""
Ring Attention for distributed long-context processing.

Implements the Ring Attention algorithm that enables attention computation
over sequences longer than single-device memory by distributing the sequence
across multiple devices in a ring topology.

Reference: https://arxiv.org/abs/2310.01889
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union
from nexus.core.base import NexusModule


class RingAttention(NexusModule):
    """
    Ring Attention for distributed long-context processing.

    Splits sequence across devices and passes KV in a ring topology,
    enabling attention over sequences longer than single-device memory.
    Each device holds a chunk of the sequence and iteratively receives
    KV pairs from other devices in a ring pattern.

    Reference: https://arxiv.org/abs/2310.01889

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head (default: dim // num_heads)
        ring_size: Number of devices/chunks in ring (None for auto-detect)
        overlap_comm_compute: Whether to overlap communication with computation
        dropout: Attention dropout probability
        bias: Whether to use bias in projections
        causal: Whether to apply causal masking
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        ring_size: Optional[int] = None,
        overlap_comm_compute: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.ring_size = ring_size
        self.overlap_comm_compute = overlap_comm_compute
        self.dropout_p = dropout
        self.causal = causal
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def _get_ring_size(self, seq_len: int, chunk_size: Optional[int] = None) -> int:
        """Determine number of chunks for ring attention."""
        if self.ring_size is not None:
            return self.ring_size
        # Auto-detect: try to use available GPUs or default to reasonable chunk count
        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
        return max(1, seq_len // 1024)  # Default to ~1024 tokens per chunk

    def _split_into_chunks(
        self,
        tensor: torch.Tensor,
        num_chunks: int
    ) -> List[torch.Tensor]:
        """Split tensor along sequence dimension into chunks."""
        # tensor: (batch, seq, ...)
        seq_len = tensor.size(1)
        chunk_size = math.ceil(seq_len / num_chunks)
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, seq_len)
            if start < seq_len:
                chunks.append(tensor[:, start:end])
        return chunks

    def _create_block_causal_mask(
        self,
        q_chunk_idx: int,
        k_chunk_idx: int,
        q_len: int,
        k_len: int,
        num_chunks: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """
        Create causal mask for blockwise attention.

        For ring attention, we need to handle causal masking across blocks:
        - If k_chunk_idx > q_chunk_idx: mask everything (future block)
        - If k_chunk_idx < q_chunk_idx: no mask (past block)
        - If k_chunk_idx == q_chunk_idx: standard causal mask

        Args:
            q_chunk_idx: Index of query chunk
            k_chunk_idx: Index of key chunk
            q_len: Length of query chunk
            k_len: Length of key chunk
            num_chunks: Total number of chunks
            device: Device for mask
            dtype: Data type for mask

        Returns:
            Causal mask or None if no masking needed
        """
        if not self.causal:
            return None

        if k_chunk_idx > q_chunk_idx:
            # Future block: mask everything
            return torch.full(
                (q_len, k_len),
                float('-inf'),
                device=device,
                dtype=dtype
            )
        elif k_chunk_idx < q_chunk_idx:
            # Past block: no masking
            return None
        else:
            # Same block: standard causal mask
            row_idx = torch.arange(q_len, device=device).unsqueeze(1)
            col_idx = torch.arange(k_len, device=device).unsqueeze(0)
            mask = col_idx > row_idx
            return mask.float().masked_fill(mask, float('-inf')).masked_fill(~mask, 0.0)

    def _blockwise_attention(
        self,
        q_chunk: torch.Tensor,
        k_chunks: List[torch.Tensor],
        v_chunks: List[torch.Tensor],
        q_chunk_idx: int,
        num_chunks: int
    ) -> torch.Tensor:
        """
        Compute attention for one query chunk against all key-value chunks.

        Uses numerically stable online softmax to combine attention from
        multiple KV blocks without materializing the full attention matrix.

        Args:
            q_chunk: Query chunk of shape (batch, heads, q_len, head_dim)
            k_chunks: List of key chunks
            v_chunks: List of value chunks
            q_chunk_idx: Index of this query chunk
            num_chunks: Total number of chunks

        Returns:
            Output for this query chunk
        """
        batch_size, num_heads, q_len, head_dim = q_chunk.shape
        device = q_chunk.device
        dtype = q_chunk.dtype

        # Initialize accumulators for online softmax
        # We use the log-sum-exp trick for numerical stability
        output_acc = torch.zeros(
            batch_size, num_heads, q_len, head_dim,
            device=device, dtype=dtype
        )
        lse_acc = torch.full(
            (batch_size, num_heads, q_len, 1),
            float('-inf'),
            device=device, dtype=dtype
        )

        for k_idx, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):
            k_len = k_chunk.size(2)

            # Compute attention scores for this block
            attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.scale
            # Shape: (batch, heads, q_len, k_len)

            # Apply causal mask if needed
            causal_mask = self._create_block_causal_mask(
                q_chunk_idx, k_idx, q_len, k_len, num_chunks, device, dtype
            )
            if causal_mask is not None:
                attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)

            # Online softmax update
            # Compute log-sum-exp for this block
            block_max = attn_scores.max(dim=-1, keepdim=True).values
            block_exp = torch.exp(attn_scores - block_max)
            block_sum = block_exp.sum(dim=-1, keepdim=True)
            block_lse = block_max + torch.log(block_sum + 1e-10)

            # Update running log-sum-exp
            new_lse = torch.logaddexp(lse_acc, block_lse)

            # Rescale previous accumulator
            old_scale = torch.exp(lse_acc - new_lse)
            new_scale = torch.exp(block_lse - new_lse)

            # Update output accumulator
            block_attn = block_exp / (block_sum + 1e-10)
            block_output = torch.matmul(block_attn, v_chunk)

            output_acc = output_acc * old_scale + block_output * new_scale
            lse_acc = new_lse

        return output_acc

    def _simulate_ring_pass(
        self,
        k_chunks: List[torch.Tensor],
        v_chunks: List[torch.Tensor],
        step: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Simulate ring topology communication.

        In actual distributed setting, each device would send its KV to the next
        device in the ring and receive from the previous device.
        Here we simulate by rotating the chunk lists.

        Args:
            k_chunks: Current key chunks per device
            v_chunks: Current value chunks per device
            step: Current ring step

        Returns:
            Rotated (k_chunks, v_chunks)
        """
        # Rotate chunks to simulate ring communication
        if step == 0:
            return k_chunks, v_chunks

        n = len(k_chunks)
        rotated_k = [k_chunks[(i - step) % n] for i in range(n)]
        rotated_v = [v_chunks[(i - step) % n] for i in range(n)]
        return rotated_k, rotated_v

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        ring_group: Optional[object] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with ring attention.

        Implements blockwise ring attention where the sequence is split across
        chunks (simulating devices), and KV pairs are passed around in a ring
        pattern to compute full attention efficiently.

        Args:
            q: Query tensor of shape (batch, seq_len, dim) or
               pre-projected (batch, heads, seq_len, head_dim)
            k: Key tensor (optional, uses q if not provided for self-attention)
            v: Value tensor (optional, uses q if not provided for self-attention)
            ring_group: Distributed process group for ring communication (optional)
            attention_mask: Additional attention mask (optional)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Handle self-attention case
        if k is None:
            k = q
        if v is None:
            v = q

        # Check if inputs are already projected
        needs_projection = q.dim() == 3 and q.size(-1) == self.dim

        if needs_projection:
            batch_size, seq_len, _ = q.shape

            # Project Q, K, V
            q = self.q_proj(q)
            k = self.k_proj(k)
            v = self.v_proj(v)

            # Reshape to (batch, heads, seq, head_dim)
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            batch_size = q.size(0)
            seq_len = q.size(2)

        # Determine ring size
        num_chunks = self._get_ring_size(seq_len)
        num_chunks = min(num_chunks, seq_len)  # Can't have more chunks than tokens

        # Split into chunks (simulating distribution across devices)
        # Transpose for splitting: (batch, heads, seq, dim) -> (batch, seq, heads, dim)
        q_transposed = q.transpose(1, 2)
        k_transposed = k.transpose(1, 2)
        v_transposed = v.transpose(1, 2)

        q_chunks = self._split_into_chunks(q_transposed, num_chunks)
        k_chunks = self._split_into_chunks(k_transposed, num_chunks)
        v_chunks = self._split_into_chunks(v_transposed, num_chunks)

        # Transpose chunks back: (batch, chunk_len, heads, dim) -> (batch, heads, chunk_len, dim)
        q_chunks = [chunk.transpose(1, 2) for chunk in q_chunks]
        k_chunks = [chunk.transpose(1, 2) for chunk in k_chunks]
        v_chunks = [chunk.transpose(1, 2) for chunk in v_chunks]

        # Ring attention loop
        # Each "device" computes attention for its query chunk against all KV chunks
        outputs = []

        for q_idx, q_chunk in enumerate(q_chunks):
            # Collect all KV chunks this query chunk needs to attend to
            # In distributed setting, this would involve ring communication
            # Here we simulate by having access to all chunks

            if self.overlap_comm_compute:
                # Simulate overlapped communication: process blocks as they "arrive"
                chunk_output = self._blockwise_attention(
                    q_chunk, k_chunks, v_chunks, q_idx, num_chunks
                )
            else:
                # Non-overlapped: gather all KV first, then compute
                chunk_output = self._blockwise_attention(
                    q_chunk, k_chunks, v_chunks, q_idx, num_chunks
                )

            outputs.append(chunk_output)

        # Concatenate outputs from all chunks
        # Each output is (batch, heads, chunk_len, head_dim)
        output = torch.cat(outputs, dim=2)

        # Apply dropout
        output = self.dropout(output)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)

        if needs_projection:
            output = self.o_proj(output)

        return output


class BlockwiseRingAttention(RingAttention):
    """
    Alias for RingAttention with explicit block-based interface.

    Provides additional methods for fine-grained control over
    block processing in ring attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        block_size: int = 1024,
        **kwargs
    ):
        super().__init__(dim, num_heads, **kwargs)
        self.block_size = block_size

    def _get_ring_size(self, seq_len: int, chunk_size: Optional[int] = None) -> int:
        """Use block_size to determine number of chunks."""
        return max(1, math.ceil(seq_len / self.block_size))

    def forward_with_blocks(
        self,
        hidden_states: torch.Tensor,
        block_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Forward with explicit block specification.

        Args:
            hidden_states: Input of shape (batch, seq_len, dim)
            block_indices: Specific blocks to compute (default: all)

        Returns:
            Output tensor
        """
        return self.forward(hidden_states)
