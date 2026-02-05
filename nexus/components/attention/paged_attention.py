"""
PagedAttention: OS-inspired virtual memory management for KV cache.

PagedAttention partitions the KV cache into fixed-size blocks that can be
stored non-contiguously in GPU memory, similar to how operating systems
manage virtual memory with paging. This eliminates memory fragmentation
and enables near-optimal GPU memory utilization during inference.

Key components:
    - BlockTable: Maps logical block indices to physical block locations
    - PagedKVCache: Manages allocation, freeing, and access of physical blocks
    - PagedAttention: Computes attention over non-contiguous paged KV blocks

This approach enables:
    - Near-zero memory waste from fragmentation
    - Efficient memory sharing across sequences (e.g., beam search, prefix caching)
    - Dynamic memory allocation matching actual sequence lengths

Reference: https://arxiv.org/abs/2309.06180 (Efficient Memory Management for Large
           Language Model Serving with PagedAttention, vLLM)

See Also:
    - flash_attention.py: Memory-efficient attention computation
    - chunked_prefill.py: Chunked prefill scheduling for batched inference
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
from nexus.core.base import NexusModule


class BlockTable(NexusModule):
    """Maps logical KV cache blocks to physical GPU memory blocks.

    Each sequence maintains a block table that translates logical block
    indices (sequential) to physical block indices (potentially scattered
    in GPU memory). This enables non-contiguous storage of the KV cache.

    The block table supports copy-on-write semantics for memory sharing:
    multiple sequences can point to the same physical block until one
    needs to modify it.

    Args:
        num_blocks: Total number of physical blocks available
        block_size: Number of tokens per block
        num_heads: Number of attention heads
        head_dim: Dimension per attention head

    Example:
        >>> table = BlockTable(num_blocks=1024, block_size=16, num_heads=32, head_dim=128)
        >>> # Allocate 3 blocks for a sequence
        >>> block_ids = table.allocate(num_blocks=3)
        >>> block_ids
        [0, 1, 2]
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Track which physical blocks are free
        self.free_blocks: List[int] = list(range(num_blocks))
        # Map: sequence_id -> list of physical block indices
        self.block_tables: Dict[int, List[int]] = {}
        # Reference counts for copy-on-write
        self.ref_counts = torch.zeros(num_blocks, dtype=torch.int32)

    def allocate(self, num_blocks: int, seq_id: int = 0) -> List[int]:
        """Allocate physical blocks for a sequence.

        Args:
            num_blocks: Number of blocks to allocate
            seq_id: Sequence identifier

        Returns:
            List of allocated physical block indices

        Raises:
            RuntimeError: If not enough free blocks available
        """
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError(
                f"Cannot allocate {num_blocks} blocks, only "
                f"{len(self.free_blocks)} free blocks available"
            )

        allocated = []
        for _ in range(num_blocks):
            block_id = self.free_blocks.pop(0)
            allocated.append(block_id)
            self.ref_counts[block_id] = 1

        if seq_id not in self.block_tables:
            self.block_tables[seq_id] = []
        self.block_tables[seq_id].extend(allocated)

        return allocated

    def free(self, seq_id: int) -> None:
        """Free all blocks associated with a sequence.

        Decrements reference counts and returns blocks with zero
        references to the free pool.

        Args:
            seq_id: Sequence identifier to free
        """
        if seq_id not in self.block_tables:
            return

        for block_id in self.block_tables[seq_id]:
            self.ref_counts[block_id] -= 1
            if self.ref_counts[block_id] == 0:
                self.free_blocks.append(block_id)

        del self.block_tables[seq_id]

    def get_block_ids(self, seq_id: int) -> List[int]:
        """Get physical block indices for a sequence.

        Args:
            seq_id: Sequence identifier

        Returns:
            List of physical block indices
        """
        return self.block_tables.get(seq_id, [])

    def fork(self, src_seq_id: int, dst_seq_id: int) -> None:
        """Fork block table for copy-on-write sharing (e.g., beam search).

        The destination sequence shares the same physical blocks as the
        source, with reference counts incremented.

        Args:
            src_seq_id: Source sequence to fork from
            dst_seq_id: Destination sequence
        """
        if src_seq_id not in self.block_tables:
            raise ValueError(f"Source sequence {src_seq_id} not found")

        src_blocks = self.block_tables[src_seq_id]
        self.block_tables[dst_seq_id] = list(src_blocks)

        for block_id in src_blocks:
            self.ref_counts[block_id] += 1

    @property
    def num_free_blocks(self) -> int:
        """Number of available free blocks."""
        return len(self.free_blocks)

    def forward(self, seq_id: int) -> torch.Tensor:
        """Return block table as a tensor for a sequence.

        Args:
            seq_id: Sequence identifier

        Returns:
            Tensor of physical block indices
        """
        block_ids = self.get_block_ids(seq_id)
        return torch.tensor(block_ids, dtype=torch.long)


class PagedKVCache(NexusModule):
    """Manages non-contiguous physical blocks for KV cache storage.

    Provides a block-level interface for storing and retrieving key-value
    pairs. Physical blocks can be scattered in GPU memory, and the cache
    handles the mapping transparently.

    Args:
        num_blocks: Total number of physical blocks
        block_size: Number of tokens per block
        num_heads: Number of KV heads
        head_dim: Dimension per head
        dtype: Data type for cache tensors

    Example:
        >>> cache = PagedKVCache(num_blocks=1024, block_size=16, num_heads=32, head_dim=128)
        >>> # Write KV to block 5, slot 3
        >>> cache.write(block_idx=5, slot_idx=3, key=k_tensor, value=v_tensor)
        >>> # Read all filled slots from blocks [5, 12, 7]
        >>> keys, values = cache.read(block_ids=[5, 12, 7], seq_len=48)
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        num_heads: int = 32,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Pre-allocate physical block storage
        # Shape: (num_blocks, block_size, num_heads, head_dim) for K and V
        self.register_buffer(
            'k_cache',
            torch.zeros(num_blocks, block_size, num_heads, head_dim, dtype=dtype)
        )
        self.register_buffer(
            'v_cache',
            torch.zeros(num_blocks, block_size, num_heads, head_dim, dtype=dtype)
        )

        # Track how many slots are filled in each block
        self.register_buffer(
            'block_fill_count',
            torch.zeros(num_blocks, dtype=torch.int32)
        )

    def write(
        self,
        block_idx: int,
        slot_idx: int,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> None:
        """Write a single token's KV to a specific slot in a block.

        Args:
            block_idx: Physical block index
            slot_idx: Slot index within the block (0 to block_size-1)
            key: Key tensor (num_heads, head_dim) or (1, num_heads, head_dim)
            value: Value tensor (num_heads, head_dim) or (1, num_heads, head_dim)
        """
        if key.dim() == 3:
            key = key.squeeze(0)
        if value.dim() == 3:
            value = value.squeeze(0)

        self.k_cache[block_idx, slot_idx] = key
        self.v_cache[block_idx, slot_idx] = value
        self.block_fill_count[block_idx] = max(
            self.block_fill_count[block_idx].item(), slot_idx + 1
        )

    def write_batch(
        self,
        block_ids: List[int],
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> None:
        """Write a batch of tokens' KV across blocks.

        Tokens are written sequentially across the provided blocks.

        Args:
            block_ids: List of physical block indices
            keys: Key tensor (seq_len, num_heads, head_dim)
            values: Value tensor (seq_len, num_heads, head_dim)
        """
        seq_len = keys.shape[0]
        for i in range(seq_len):
            block_idx = block_ids[i // self.block_size]
            slot_idx = i % self.block_size
            self.write(block_idx, slot_idx, keys[i], values[i])

    def read(
        self,
        block_ids: List[int],
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read KV from a sequence of blocks.

        Gathers keys and values from the specified blocks in order,
        returning a contiguous tensor.

        Args:
            block_ids: Ordered list of physical block indices for a sequence
            seq_len: Total number of tokens to read

        Returns:
            keys: (seq_len, num_heads, head_dim)
            values: (seq_len, num_heads, head_dim)
        """
        all_keys = []
        all_values = []
        tokens_read = 0

        for block_id in block_ids:
            tokens_in_block = min(self.block_size, seq_len - tokens_read)
            if tokens_in_block <= 0:
                break

            all_keys.append(self.k_cache[block_id, :tokens_in_block])
            all_values.append(self.v_cache[block_id, :tokens_in_block])
            tokens_read += tokens_in_block

        keys = torch.cat(all_keys, dim=0)
        values = torch.cat(all_values, dim=0)

        return keys, values

    def clear_block(self, block_idx: int) -> None:
        """Clear a physical block's contents.

        Args:
            block_idx: Physical block index to clear
        """
        self.k_cache[block_idx].zero_()
        self.v_cache[block_idx].zero_()
        self.block_fill_count[block_idx] = 0

    def forward(
        self,
        block_ids: List[int],
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read KV cache for given blocks.

        Args:
            block_ids: Physical block indices
            seq_len: Number of tokens to read

        Returns:
            Tuple of (keys, values)
        """
        return self.read(block_ids, seq_len)


class PagedAttention(NexusModule):
    """Attention computation over paged (non-contiguous) KV cache blocks.

    Integrates the BlockTable and PagedKVCache to provide a complete
    attention mechanism with paged memory management. During inference,
    new tokens are appended to the paged cache, and attention is computed
    by gathering KV from the relevant physical blocks.

    This approach is inspired by OS virtual memory:
        - BlockTable ~ page table (logical -> physical mapping)
        - PagedKVCache ~ physical memory (block-level storage)
        - PagedAttention ~ TLB + memory access (transparent paging)

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head. Defaults to d_model // num_heads.
        block_size: Number of tokens per physical block (default 16)
        num_blocks: Total physical blocks available
        num_kv_heads: Number of KV heads (for GQA compatibility). Defaults
            to num_heads.
        dropout: Attention dropout probability
        bias: Whether to use bias in projections

    Example:
        >>> attn = PagedAttention(
        ...     d_model=2048, num_heads=16, block_size=16, num_blocks=1024
        ... )
        >>> x = torch.randn(1, 32, 2048)
        >>> out, weights, cache_state = attn(x, seq_id=0)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        block_size: int = 16,
        num_blocks: int = 1024,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or (d_model // num_heads)
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads or num_heads
        self.dropout_p = dropout

        assert num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"

        self.num_kv_groups = num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5

        # Projection layers
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        # Paged memory components
        self.block_table = BlockTable(
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=self.num_kv_heads,
            head_dim=self.head_dim
        )
        self.kv_cache = PagedKVCache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=self.num_kv_heads,
            head_dim=self.head_dim
        )

        # Track sequence lengths for cache management
        self.seq_lengths: Dict[int, int] = {}

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match query heads for GQA."""
        if n_rep == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def _allocate_blocks_for_tokens(self, seq_id: int, num_new_tokens: int) -> None:
        """Ensure enough blocks are allocated for the new tokens.

        Args:
            seq_id: Sequence identifier
            num_new_tokens: Number of new tokens being added
        """
        current_len = self.seq_lengths.get(seq_id, 0)
        new_total_len = current_len + num_new_tokens

        current_blocks = len(self.block_table.get_block_ids(seq_id))
        needed_blocks = math.ceil(new_total_len / self.block_size)
        additional_blocks = needed_blocks - current_blocks

        if additional_blocks > 0:
            self.block_table.allocate(additional_blocks, seq_id)

    def _write_to_cache(
        self,
        seq_id: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor
    ) -> None:
        """Write new KV states to the paged cache.

        Args:
            seq_id: Sequence identifier
            key_states: (1, num_kv_heads, new_seq_len, head_dim)
            value_states: (1, num_kv_heads, new_seq_len, head_dim)
        """
        new_seq_len = key_states.shape[2]
        current_len = self.seq_lengths.get(seq_id, 0)
        block_ids = self.block_table.get_block_ids(seq_id)

        # Write each new token to the appropriate block and slot
        for i in range(new_seq_len):
            pos = current_len + i
            block_idx = block_ids[pos // self.block_size]
            slot_idx = pos % self.block_size
            self.kv_cache.write(
                block_idx, slot_idx,
                key_states[0, :, i, :],    # (num_kv_heads, head_dim)
                value_states[0, :, i, :]   # (num_kv_heads, head_dim)
            )

        self.seq_lengths[seq_id] = current_len + new_seq_len

    def _read_from_cache(
        self, seq_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read full KV cache for a sequence from paged blocks.

        Args:
            seq_id: Sequence identifier

        Returns:
            keys: (1, num_kv_heads, total_seq_len, head_dim)
            values: (1, num_kv_heads, total_seq_len, head_dim)
        """
        total_len = self.seq_lengths.get(seq_id, 0)
        block_ids = self.block_table.get_block_ids(seq_id)

        keys, values = self.kv_cache.read(block_ids, total_len)
        # keys, values: (total_len, num_kv_heads, head_dim)

        keys = keys.unsqueeze(0).permute(0, 2, 1, 3)    # (1, num_kv_heads, total_len, head_dim)
        values = values.unsqueeze(0).permute(0, 2, 1, 3)

        return keys, values

    def free_sequence(self, seq_id: int) -> None:
        """Free all memory associated with a sequence.

        Args:
            seq_id: Sequence identifier to free
        """
        self.block_table.free(seq_id)
        if seq_id in self.seq_lengths:
            del self.seq_lengths[seq_id]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        seq_id: int = 0,
        use_cache: bool = True,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass with paged KV cache management.

        During prefill (first call), allocates blocks and fills the cache.
        During decode (subsequent calls), appends to existing blocks.

        Args:
            hidden_states: Input tensor (batch, seq_len, d_model).
                Currently supports batch_size=1 for paged caching.
            attention_mask: Optional mask (batch, 1, seq_len, kv_seq_len)
            position_embeddings: Tuple of (cos, sin) for RoPE
            seq_id: Sequence identifier for cache management
            use_cache: Whether to use the paged KV cache
            output_attentions: Whether to return attention weights

        Returns:
            output: Attention output (batch, seq_len, d_model)
            attn_weights: If output_attentions, else None
            cache_state: Dict with cache metadata if use_cache, else None
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Apply rotary position embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = self._apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if use_cache:
            # Allocate blocks and write to paged cache
            self._allocate_blocks_for_tokens(seq_id, seq_len)
            self._write_to_cache(seq_id, key_states, value_states)

            # Read full KV from cache (includes all previous tokens)
            key_states, value_states = self._read_from_cache(seq_id)
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)

        # Repeat KV for GQA
        key_states = self._repeat_kv(key_states, self.num_kv_groups)
        value_states = self._repeat_kv(value_states, self.num_kv_groups)

        # Compute attention
        attn_weights = torch.matmul(
            query_states, key_states.transpose(-2, -1)
        ) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights_dropped, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        cache_state = None
        if use_cache:
            cache_state = {
                'seq_id': seq_id,
                'seq_len': self.seq_lengths[seq_id],
                'num_blocks': len(self.block_table.get_block_ids(seq_id)),
                'block_ids': self.block_table.get_block_ids(seq_id),
            }

        return attn_output, attn_weights, cache_state

    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary position embeddings to Q and K."""
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
