"""
KV Cache management utilities for efficient LLM inference.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from nexus.core.base import NexusModule


class KVCache:
    """Key-Value cache for transformer inference.

    Stores and manages cached key/value tensors to avoid
    recomputation during autoregressive generation.

    Args:
        num_layers: Number of transformer layers
        max_batch_size: Maximum batch size to support
        max_seq_len: Maximum sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dtype: Data type for cache
        device: Device for cache
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device('cuda')
    ):
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Preallocate cache tensors
        cache_shape = (max_batch_size, num_heads, max_seq_len, head_dim)
        self.k_cache = [
            torch.zeros(cache_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(cache_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

        # Track current sequence lengths per batch item
        self.seq_lens = torch.zeros(max_batch_size, dtype=torch.long, device=device)

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value and return full cached tensors.

        Args:
            layer_idx: Which layer's cache to update
            key: New keys (batch, num_heads, seq_len, head_dim)
            value: New values (batch, num_heads, seq_len, head_dim)
            start_pos: Starting position for update (if None, uses current seq_len)

        Returns:
            Full cached keys and values including new entries
        """
        batch_size, _, seq_len, _ = key.shape

        if start_pos is None:
            start_pos = self.seq_lens[0].item()

        # Update cache
        self.k_cache[layer_idx][:batch_size, :, start_pos:start_pos + seq_len, :] = key
        self.v_cache[layer_idx][:batch_size, :, start_pos:start_pos + seq_len, :] = value

        # Update sequence lengths
        new_seq_len = start_pos + seq_len
        self.seq_lens[:batch_size] = new_seq_len

        # Return full cached tensors up to current position
        return (
            self.k_cache[layer_idx][:batch_size, :, :new_seq_len, :],
            self.v_cache[layer_idx][:batch_size, :, :new_seq_len, :]
        )

    def get(
        self,
        layer_idx: int,
        batch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached key/value for a layer."""
        batch_size = batch_size or self.max_batch_size
        seq_len = self.seq_lens[0].item()
        return (
            self.k_cache[layer_idx][:batch_size, :, :seq_len, :],
            self.v_cache[layer_idx][:batch_size, :, :seq_len, :]
        )

    def clear(self):
        """Clear all cached values."""
        for layer_idx in range(self.num_layers):
            self.k_cache[layer_idx].zero_()
            self.v_cache[layer_idx].zero_()
        self.seq_lens.zero_()

    def trim(self, new_seq_len: int):
        """Trim cache to shorter sequence length."""
        self.seq_lens.clamp_(max=new_seq_len)


class PagedKVCache:
    """Paged KV cache for memory-efficient inference.

    Uses OS-style paging to manage KV cache memory, enabling:
    - Dynamic memory allocation
    - Memory sharing between sequences
    - Reduced memory fragmentation

    Reference: vLLM PagedAttention

    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per head
        block_size: Number of tokens per block
        num_blocks: Total number of blocks to allocate
        dtype: Data type
        device: Device
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 1024,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device('cuda')
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.device = device

        # Physical block storage: (num_blocks, num_heads, block_size, head_dim)
        block_shape = (num_blocks, num_heads, block_size, head_dim)
        self.k_blocks = [
            torch.zeros(block_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_blocks = [
            torch.zeros(block_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

        # Free block list
        self.free_blocks = list(range(num_blocks))

        # Block tables: maps sequence_id -> list of block indices
        self.block_tables: Dict[int, List[int]] = {}

    def allocate(self, seq_id: int, num_tokens: int) -> List[int]:
        """
        Allocate blocks for a sequence.

        Args:
            seq_id: Sequence identifier
            num_tokens: Number of tokens to allocate for

        Returns:
            List of allocated block indices
        """
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError("Not enough free blocks")

        allocated = []
        for _ in range(num_blocks_needed):
            block_idx = self.free_blocks.pop()
            allocated.append(block_idx)

        self.block_tables[seq_id] = allocated
        return allocated

    def free(self, seq_id: int):
        """Free blocks allocated to a sequence."""
        if seq_id in self.block_tables:
            blocks = self.block_tables.pop(seq_id)
            self.free_blocks.extend(blocks)

    def update(
        self,
        layer_idx: int,
        seq_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
        position: int
    ):
        """
        Update cache for a specific position.

        Args:
            layer_idx: Layer index
            seq_id: Sequence identifier
            key: Key tensor (1, num_heads, 1, head_dim)
            value: Value tensor (1, num_heads, 1, head_dim)
            position: Token position
        """
        blocks = self.block_tables.get(seq_id, [])

        # Compute which block and offset
        block_idx = position // self.block_size
        offset = position % self.block_size

        # Ensure we have enough blocks
        while block_idx >= len(blocks):
            if not self.free_blocks:
                raise RuntimeError("Not enough free blocks")
            new_block = self.free_blocks.pop()
            blocks.append(new_block)
            self.block_tables[seq_id] = blocks

        # Update the block
        physical_block = blocks[block_idx]
        self.k_blocks[layer_idx][physical_block, :, offset, :] = key.squeeze(0).squeeze(1)
        self.v_blocks[layer_idx][physical_block, :, offset, :] = value.squeeze(0).squeeze(1)

    def get_kv(
        self,
        layer_idx: int,
        seq_id: int,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve cached KV for a sequence.

        Args:
            layer_idx: Layer index
            seq_id: Sequence identifier
            seq_len: Current sequence length

        Returns:
            Key and value tensors (1, num_heads, seq_len, head_dim)
        """
        blocks = self.block_tables.get(seq_id, [])

        k_list = []
        v_list = []

        for pos in range(seq_len):
            block_idx = pos // self.block_size
            offset = pos % self.block_size

            if block_idx < len(blocks):
                physical_block = blocks[block_idx]
                k_list.append(self.k_blocks[layer_idx][physical_block, :, offset, :])
                v_list.append(self.v_blocks[layer_idx][physical_block, :, offset, :])

        if k_list:
            k = torch.stack(k_list, dim=1).unsqueeze(0)  # (1, seq, heads, dim) -> (1, heads, seq, dim)
            v = torch.stack(v_list, dim=1).unsqueeze(0)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            return k, v
        else:
            return None, None


class StaticKVCache(NexusModule):
    """Static KV cache with fixed size allocation.

    Simple cache implementation for single-sequence inference.

    Args:
        max_seq_len: Maximum sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        num_layers: Number of layers
    """

    def __init__(
        self,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        num_layers: int
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers

        # Initialize as None, will be allocated on first use
        self.cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self.current_len = 0

    def allocate(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Allocate cache memory."""
        self.cache = []
        for _ in range(self.num_layers):
            k = torch.zeros(
                batch_size, self.num_heads, self.max_seq_len, self.head_dim,
                device=device, dtype=dtype
            )
            v = torch.zeros(
                batch_size, self.num_heads, self.max_seq_len, self.head_dim,
                device=device, dtype=dtype
            )
            self.cache.append((k, v))
        self.current_len = 0

    def update_and_get(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache and return full KV."""
        if self.cache is None:
            raise RuntimeError("Cache not allocated")

        batch_size, num_heads, seq_len, head_dim = key.shape
        k_cache, v_cache = self.cache[layer_idx]

        # Update
        k_cache[:batch_size, :, self.current_len:self.current_len + seq_len, :] = key
        v_cache[:batch_size, :, self.current_len:self.current_len + seq_len, :] = value

        # Return full cached KV
        return (
            k_cache[:batch_size, :, :self.current_len + seq_len, :],
            v_cache[:batch_size, :, :self.current_len + seq_len, :]
        )

    def step(self, num_tokens: int = 1):
        """Advance current position."""
        self.current_len += num_tokens

    def reset(self):
        """Reset cache position (keep allocation)."""
        self.current_len = 0
        if self.cache is not None:
            for k, v in self.cache:
                k.zero_()
                v.zero_()
