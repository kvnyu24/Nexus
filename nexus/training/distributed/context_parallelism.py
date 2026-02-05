"""Context Parallelism: Sequence-Length Parallelism for Long Context Training.

Context Parallelism (CP) enables training models with extremely long contexts
by partitioning the sequence dimension across multiple GPUs. This is different
from tensor parallelism (which partitions model weights) and data parallelism
(which partitions batch dimension).

Reference:
    "Ring Attention with Blockwise Transformers for Near-Infinite Context"
    Liu et al., 2023
    https://arxiv.org/abs/2310.01889

    "DeepSpeed Ulysses: System Optimizations for Enabling Training of
    Extreme Long Sequence Transformer Models"
    Jacobs et al., Microsoft, 2024

Key features:
    - Partition sequence dimension across GPUs
    - Enables training with contexts > 1M tokens
    - Compatible with Flash Attention
    - Can be combined with tensor and data parallelism
    - Ring-based communication for memory efficiency

Example:
    >>> from nexus.training.distributed.context_parallelism import (
    ...     ContextParallelAttention,
    ...     init_context_parallel_group,
    ... )
    >>>
    >>> # Initialize CP group (e.g., 4-way sequence parallelism)
    >>> cp_group = init_context_parallel_group(cp_size=4)
    >>>
    >>> # Use in attention layer
    >>> attention = ContextParallelAttention(
    ...     hidden_size=4096,
    ...     num_heads=32,
    ...     cp_group=cp_group,
    ... )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple
from nexus.core.base import NexusModule
from nexus.utils.logging import Logger
import math


class ContextParallelGroup:
    """Context parallel process group manager.

    Manages the process group for context parallelism and provides
    utilities for sequence partitioning and communication.

    Args:
        cp_size: Size of context parallel group.
        rank: Rank within the group.
        group: Process group.
    """

    def __init__(self, cp_size: int, rank: int, group):
        self.cp_size = cp_size
        self.rank = rank
        self.group = group
        self.logger = Logger(f"ContextParallel[rank={rank}]")

    def partition_sequence(
        self,
        tensor: torch.Tensor,
        seq_dim: int = 1,
    ) -> torch.Tensor:
        """Partition sequence across CP group.

        Args:
            tensor: Input tensor with sequence dimension.
            seq_dim: Dimension index for sequence.

        Returns:
            Local sequence shard.
        """
        seq_len = tensor.shape[seq_dim]
        if seq_len % self.cp_size != 0:
            raise ValueError(
                f"Sequence length {seq_len} must be divisible by CP size {self.cp_size}"
            )

        # Compute shard size and indices
        shard_size = seq_len // self.cp_size
        start_idx = self.rank * shard_size
        end_idx = start_idx + shard_size

        # Slice along sequence dimension
        slices = [slice(None)] * tensor.ndim
        slices[seq_dim] = slice(start_idx, end_idx)

        return tensor[tuple(slices)]

    def gather_sequence(
        self,
        tensor: torch.Tensor,
        seq_dim: int = 1,
    ) -> torch.Tensor:
        """Gather sequence from all CP ranks.

        Args:
            tensor: Local sequence shard.
            seq_dim: Sequence dimension.

        Returns:
            Full sequence.
        """
        # All-gather along sequence dimension
        output_size = list(tensor.shape)
        output_size[seq_dim] *= self.cp_size

        # Gather from all ranks
        gathered_list = [
            torch.zeros_like(tensor) for _ in range(self.cp_size)
        ]
        dist.all_gather(gathered_list, tensor, group=self.group)

        # Concatenate along sequence dimension
        return torch.cat(gathered_list, dim=seq_dim)


def init_context_parallel_group(
    cp_size: int,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
) -> ContextParallelGroup:
    """Initialize context parallel process group.

    Args:
        cp_size: Size of context parallel group (must divide world_size).
        world_size: Total world size (default: from dist).
        rank: Global rank (default: from dist).

    Returns:
        ContextParallelGroup instance.

    Example:
        >>> # 8 GPUs total, 4-way CP, 2-way DP
        >>> cp_group = init_context_parallel_group(cp_size=4)
    """
    if not dist.is_initialized():
        raise RuntimeError("Context parallelism requires distributed training.")

    if world_size is None:
        world_size = dist.get_world_size()
    if rank is None:
        rank = dist.get_rank()

    if world_size % cp_size != 0:
        raise ValueError(f"World size {world_size} must be divisible by CP size {cp_size}")

    # Create CP groups
    num_cp_groups = world_size // cp_size
    cp_group = None

    for i in range(num_cp_groups):
        ranks = list(range(i * cp_size, (i + 1) * cp_size))
        group = dist.new_group(ranks)

        if rank in ranks:
            cp_rank = rank % cp_size
            cp_group = ContextParallelGroup(cp_size, cp_rank, group)

    if cp_group is None:
        raise RuntimeError("Failed to create CP group for this rank")

    return cp_group


class RingAttentionCommunicator:
    """Ring-based communication for context parallel attention.

    Implements ring-style communication pattern where each GPU:
    1. Computes attention with local KV
    2. Sends KV to next GPU in ring
    3. Receives KV from previous GPU in ring
    4. Repeats until all KV blocks are processed
    """

    def __init__(self, cp_group: ContextParallelGroup):
        self.cp_group = cp_group
        self.cp_size = cp_group.cp_size
        self.rank = cp_group.rank
        self.group = cp_group.group

    def ring_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        """Compute attention using ring communication.

        Args:
            query: Local query (B, L_local, H, D).
            key: Local key (B, L_local, H, D).
            value: Local value (B, L_local, H, D).
            causal: Whether to use causal masking.

        Returns:
            Attention output (B, L_local, H, D).
        """
        batch_size, seq_len_local, num_heads, head_dim = query.shape
        device = query.device

        # Initialize output and normalizer
        output = torch.zeros_like(query)
        normalizer = torch.zeros(
            batch_size, seq_len_local, num_heads, 1,
            device=device, dtype=query.dtype
        )

        # Current KV blocks
        current_key = key
        current_value = value

        # Ring communication loop
        for step in range(self.cp_size):
            # Compute which sequence block we're processing
            # For causal attention, we need to mask future blocks
            block_offset = (self.rank - step) % self.cp_size

            # Compute attention scores with current KV block
            # QK^T / sqrt(d)
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(query, current_key.transpose(-2, -1)) * scale

            # Apply causal mask if needed
            if causal:
                # Create block-level causal mask
                q_offset = self.rank * seq_len_local
                k_offset = block_offset * seq_len_local

                # Mask future positions
                if q_offset < k_offset:
                    # Future block - mask everything
                    scores = scores.masked_fill(
                        torch.ones_like(scores, dtype=torch.bool),
                        float('-inf')
                    )
                elif q_offset == k_offset:
                    # Same block - apply intra-block causal mask
                    mask = torch.triu(
                        torch.ones(seq_len_local, seq_len_local, device=device),
                        diagonal=1
                    ).bool()
                    scores = scores.masked_fill(mask, float('-inf'))
                # else: past block - no masking needed

            # Compute attention weights
            attn_weights = torch.softmax(scores, dim=-1)

            # Weighted sum of values
            block_output = torch.matmul(attn_weights, current_value)

            # Accumulate (for numerical stability with softmax across blocks)
            max_score = scores.max(dim=-1, keepdim=True)[0]
            exp_scores = torch.exp(scores - max_score)
            block_sum = exp_scores.sum(dim=-1, keepdim=True)

            output += block_output * block_sum
            normalizer += block_sum

            # Ring communication: send KV to next, receive from previous
            if step < self.cp_size - 1:
                next_rank = (self.rank + 1) % self.cp_size
                prev_rank = (self.rank - 1) % self.cp_size

                # Prepare buffers
                send_key = current_key
                send_value = current_value
                recv_key = torch.zeros_like(current_key)
                recv_value = torch.zeros_like(current_value)

                # Send to next, receive from previous
                send_reqs = [
                    dist.isend(send_key, dst=next_rank, group=self.group),
                    dist.isend(send_value, dst=next_rank, group=self.group),
                ]
                recv_reqs = [
                    dist.irecv(recv_key, src=prev_rank, group=self.group),
                    dist.irecv(recv_value, src=prev_rank, group=self.group),
                ]

                # Wait for communication
                for req in send_reqs + recv_reqs:
                    req.wait()

                current_key = recv_key
                current_value = recv_value

        # Normalize output
        output = output / (normalizer + 1e-8)

        return output


class ContextParallelAttention(NexusModule):
    """Multi-head attention with context parallelism.

    Implements sequence-parallel attention using ring communication.
    Each GPU holds a portion of the sequence and computes attention
    across all sequence blocks through ring-based KV communication.

    Args:
        hidden_size: Hidden dimension.
        num_heads: Number of attention heads.
        cp_group: Context parallel group.
        dropout: Dropout probability.
        causal: Whether to use causal masking.

    Example:
        >>> cp_group = init_context_parallel_group(cp_size=4)
        >>> attention = ContextParallelAttention(
        ...     hidden_size=2048,
        ...     num_heads=16,
        ...     cp_group=cp_group,
        ... )
        >>> # Input: local sequence shard (B, L_local, D)
        >>> output = attention(x)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        cp_group: ContextParallelGroup,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.cp_group = cp_group
        self.dropout = dropout
        self.causal = causal

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            )

        # QKV projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Ring communicator
        self.ring_comm = RingAttentionCommunicator(cp_group)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with context parallel attention.

        Args:
            x: Input tensor (B, L_local, D) - local sequence shard.
            attention_mask: Optional attention mask (currently not supported with CP).

        Returns:
            Output tensor (B, L_local, D).
        """
        batch_size, seq_len_local, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, L, H, D_head)
        q = q.view(batch_size, seq_len_local, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len_local, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len_local, self.num_heads, self.head_dim)

        # Ring attention
        attn_output = self.ring_comm.ring_forward(q, k, v, causal=self.causal)

        # Reshape and project
        attn_output = attn_output.reshape(batch_size, seq_len_local, self.hidden_size)
        output = self.out_proj(attn_output)

        if self.dropout > 0:
            output = F.dropout(output, p=self.dropout, training=self.training)

        return output


def estimate_context_parallel_memory_savings(
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    cp_size: int,
) -> dict:
    """Estimate memory savings from context parallelism.

    Args:
        seq_len: Total sequence length.
        hidden_size: Model hidden size.
        num_layers: Number of transformer layers.
        batch_size: Batch size.
        cp_size: Context parallel size.

    Returns:
        Dictionary with memory estimates (in GB).
    """
    # Activation memory per layer (QKV + attention scores)
    # Without CP
    qkv_memory = batch_size * seq_len * hidden_size * 3 * 4  # bytes (FP32)
    attn_scores_memory = batch_size * seq_len * seq_len * 4  # bytes

    total_without_cp = (qkv_memory + attn_scores_memory) * num_layers / 1e9  # GB

    # With CP
    seq_len_local = seq_len // cp_size
    qkv_memory_cp = batch_size * seq_len_local * hidden_size * 3 * 4
    attn_scores_memory_cp = batch_size * seq_len_local * seq_len_local * 4

    total_with_cp = (qkv_memory_cp + attn_scores_memory_cp) * num_layers / 1e9  # GB

    return {
        "without_cp_gb": total_without_cp,
        "with_cp_gb": total_with_cp,
        "savings_gb": total_without_cp - total_with_cp,
        "reduction_factor": total_without_cp / total_with_cp,
        "max_seq_len_without_cp": seq_len,
        "max_seq_len_with_cp": seq_len * cp_size,  # Theoretical
    }
