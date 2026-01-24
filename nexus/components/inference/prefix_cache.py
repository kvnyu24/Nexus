"""
Prefix Caching for efficient reuse of computed KV cache.

Enables reusing KV cache for common prefixes like system prompts,
few-shot examples, and shared context across multiple requests.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from collections import OrderedDict
import hashlib

from nexus.core.base import NexusModule


class PrefixCache(NexusModule):
    """
    Prefix Caching for reusing KV cache of common prefixes.

    Stores computed KV for shared prefixes (system prompts, few-shot examples)
    and reuses them across requests. This significantly reduces computation
    for requests that share common prefixes.

    Used by: vLLM, SGLang, TensorRT-LLM

    Features:
    - Hash-based prefix matching for O(1) lookup
    - LRU eviction policy for cache management
    - Support for variable-length prefix matching
    - Thread-safe operations for concurrent requests

    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per head
        max_prefixes: Maximum number of prefixes to cache
        hash_tokens: Number of tokens to hash for prefix matching
        dtype: Data type for cached tensors
        device: Device for cache storage

    Example:
        >>> cache = PrefixCache(
        ...     num_layers=32,
        ...     num_heads=32,
        ...     head_dim=128,
        ...     max_prefixes=100
        ... )
        >>> # Store system prompt KV cache
        >>> prefix_ids = tokenizer.encode("You are a helpful assistant.")
        >>> kv_cache = compute_kv(prefix_ids)
        >>> cache.add_prefix(prefix_ids, kv_cache)
        >>>
        >>> # Reuse for new request
        >>> input_ids = tokenizer.encode("You are a helpful assistant. Hello!")
        >>> match = cache.get_prefix(input_ids)
        >>> if match:
        ...     kv, matched_len = match
        ...     # Skip computation for first matched_len tokens
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_prefixes: int = 100,
        hash_tokens: int = 32,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_prefixes = max_prefixes
        self.hash_tokens = hash_tokens
        self.dtype = dtype
        self.device = device

        # LRU cache: hash -> (prefix_ids, kv_cache, access_count)
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _compute_hash(self, token_ids: List[int]) -> str:
        """
        Compute hash key for prefix matching.

        Uses first hash_tokens tokens for efficient lookup while
        still allowing variable-length prefixes.

        Args:
            token_ids: List of token IDs

        Returns:
            Hash string for the prefix
        """
        # Use first N tokens for hash
        hash_ids = token_ids[:self.hash_tokens]
        hash_str = ','.join(map(str, hash_ids))
        return hashlib.md5(hash_str.encode()).hexdigest()

    def _compute_full_hash(self, token_ids: List[int]) -> str:
        """Compute hash of full prefix for exact matching."""
        hash_str = ','.join(map(str, token_ids))
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def add_prefix(
        self,
        prefix_ids: List[int],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store prefix KV cache with hash key.

        Args:
            prefix_ids: Token IDs of the prefix
            kv_cache: List of (key, value) tuples per layer
                     Each tensor shape: (1, num_heads, seq_len, head_dim)
            metadata: Optional metadata to store with prefix

        Returns:
            Hash key for the stored prefix
        """
        if len(prefix_ids) < 1:
            raise ValueError("Prefix must contain at least 1 token")

        # Compute hashes
        lookup_hash = self._compute_hash(prefix_ids)
        full_hash = self._compute_full_hash(prefix_ids)

        # Check if already cached (exact match)
        if lookup_hash in self._cache:
            existing = self._cache[lookup_hash]
            if existing['full_hash'] == full_hash:
                # Move to end (most recently used)
                self._cache.move_to_end(lookup_hash)
                existing['access_count'] += 1
                return lookup_hash

        # Evict if at capacity
        while len(self._cache) >= self.max_prefixes:
            self._evict_lru()

        # Clone and store KV cache
        stored_kv = []
        for k, v in kv_cache:
            stored_kv.append((
                k.clone().to(self.device, self.dtype),
                v.clone().to(self.device, self.dtype)
            ))

        self._cache[lookup_hash] = {
            'prefix_ids': list(prefix_ids),
            'full_hash': full_hash,
            'kv_cache': stored_kv,
            'seq_len': len(prefix_ids),
            'access_count': 1,
            'metadata': metadata or {}
        }

        return lookup_hash

    def get_prefix(
        self,
        input_ids: List[int],
        min_match_ratio: float = 0.5
    ) -> Optional[Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int]]:
        """
        Check if input starts with a cached prefix.

        Args:
            input_ids: Token IDs to check for prefix match
            min_match_ratio: Minimum ratio of prefix tokens to match

        Returns:
            Tuple of (kv_cache, num_matched_tokens) if match found, None otherwise
        """
        if len(input_ids) < 1:
            self._misses += 1
            return None

        # Compute lookup hash
        lookup_hash = self._compute_hash(input_ids)

        if lookup_hash not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[lookup_hash]
        prefix_ids = entry['prefix_ids']

        # Verify exact prefix match
        matched_tokens = 0
        for i, (inp_id, prefix_id) in enumerate(zip(input_ids, prefix_ids)):
            if inp_id != prefix_id:
                break
            matched_tokens = i + 1

        # Check if match meets minimum threshold
        if matched_tokens < len(prefix_ids) * min_match_ratio:
            self._misses += 1
            return None

        # Update LRU order and access count
        self._cache.move_to_end(lookup_hash)
        entry['access_count'] += 1
        self._hits += 1

        # Return KV cache up to matched length
        kv_cache = []
        for k, v in entry['kv_cache']:
            kv_cache.append((
                k[:, :, :matched_tokens, :].clone(),
                v[:, :, :matched_tokens, :].clone()
            ))

        return kv_cache, matched_tokens

    def get_prefix_by_hash(
        self,
        prefix_hash: str
    ) -> Optional[Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int]]:
        """
        Get prefix by its hash key directly.

        Args:
            prefix_hash: Hash key returned by add_prefix

        Returns:
            Tuple of (kv_cache, seq_len) if found, None otherwise
        """
        if prefix_hash not in self._cache:
            return None

        entry = self._cache[prefix_hash]
        self._cache.move_to_end(prefix_hash)
        entry['access_count'] += 1

        kv_cache = [(k.clone(), v.clone()) for k, v in entry['kv_cache']]
        return kv_cache, entry['seq_len']

    def _evict_lru(self):
        """Evict least recently used entry."""
        if self._cache:
            # OrderedDict maintains insertion order; first item is LRU
            evicted_key, evicted_entry = self._cache.popitem(last=False)
            # Free memory
            del evicted_entry['kv_cache']
            self._evictions += 1

    def remove_prefix(self, prefix_hash: str) -> bool:
        """
        Remove a prefix from cache.

        Args:
            prefix_hash: Hash key of prefix to remove

        Returns:
            True if prefix was removed, False if not found
        """
        if prefix_hash in self._cache:
            del self._cache[prefix_hash]
            return True
        return False

    def clear(self):
        """Clear all cached prefixes."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        # Calculate memory usage
        total_memory = 0
        for entry in self._cache.values():
            for k, v in entry['kv_cache']:
                total_memory += k.numel() * k.element_size()
                total_memory += v.numel() * v.element_size()

        return {
            'num_prefixes': len(self._cache),
            'max_prefixes': self.max_prefixes,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'evictions': self._evictions,
            'memory_mb': total_memory / (1024 ** 2)
        }

    def list_prefixes(self) -> List[Dict[str, Any]]:
        """
        List all cached prefixes with metadata.

        Returns:
            List of prefix information dictionaries
        """
        prefixes = []
        for hash_key, entry in self._cache.items():
            prefixes.append({
                'hash': hash_key,
                'seq_len': entry['seq_len'],
                'access_count': entry['access_count'],
                'metadata': entry['metadata'],
                'prefix_preview': entry['prefix_ids'][:10]  # First 10 tokens
            })
        return prefixes

    def forward(self, *args, **kwargs):
        """Not used - cache is managed through add_prefix/get_prefix methods."""
        raise NotImplementedError("Use add_prefix() and get_prefix() methods instead")


class RadixPrefixCache(NexusModule):
    """
    Radix tree-based prefix cache for efficient prefix sharing.

    Uses a radix tree (trie) structure to enable sharing of partial
    prefixes between different cached sequences. More memory efficient
    than simple prefix caching when many prefixes share common roots.

    Used by: SGLang RadixAttention

    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per head
        block_size: Number of tokens per cache block
        max_blocks: Maximum number of blocks to allocate
        dtype: Data type for cached tensors
        device: Device for cache storage
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_blocks: int = 1024,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.dtype = dtype
        self.device = device

        # Block storage
        block_shape = (max_blocks, num_heads, block_size, head_dim)
        self.k_blocks = [
            torch.zeros(block_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_blocks = [
            torch.zeros(block_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

        # Free block list
        self.free_blocks = list(range(max_blocks))

        # Radix tree: each node contains
        # - children: dict mapping token_id -> child_node
        # - block_idx: index of KV block (if this node ends a block)
        # - ref_count: number of sequences using this node
        self._root = self._create_node()

    def _create_node(self) -> Dict[str, Any]:
        """Create a new radix tree node."""
        return {
            'children': {},
            'block_idx': None,
            'ref_count': 0,
            'tokens': []  # Tokens stored at this node
        }

    def insert(
        self,
        token_ids: List[int],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> List[int]:
        """
        Insert a prefix into the radix tree.

        Args:
            token_ids: Token IDs of the prefix
            kv_cache: KV cache for the prefix

        Returns:
            List of block indices for the prefix
        """
        block_indices = []
        current_node = self._root

        for block_start in range(0, len(token_ids), self.block_size):
            block_end = min(block_start + self.block_size, len(token_ids))
            block_tokens = tuple(token_ids[block_start:block_end])

            # Find or create path for this block
            if block_tokens not in current_node['children']:
                # Allocate new block
                if not self.free_blocks:
                    raise RuntimeError("No free blocks available")

                block_idx = self.free_blocks.pop()

                # Store KV data
                seq_len = block_end - block_start
                for layer_idx in range(self.num_layers):
                    k, v = kv_cache[layer_idx]
                    self.k_blocks[layer_idx][block_idx, :, :seq_len, :] = \
                        k[:, :, block_start:block_end, :].squeeze(0)
                    self.v_blocks[layer_idx][block_idx, :, :seq_len, :] = \
                        v[:, :, block_start:block_end, :].squeeze(0)

                # Create new node
                new_node = self._create_node()
                new_node['block_idx'] = block_idx
                new_node['tokens'] = list(block_tokens)
                current_node['children'][block_tokens] = new_node

            child_node = current_node['children'][block_tokens]
            child_node['ref_count'] += 1
            block_indices.append(child_node['block_idx'])
            current_node = child_node

        return block_indices

    def match(
        self,
        token_ids: List[int]
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int]:
        """
        Find longest matching prefix in the radix tree.

        Args:
            token_ids: Token IDs to match

        Returns:
            Tuple of (kv_cache, num_matched_tokens)
        """
        matched_blocks = []
        matched_tokens = 0
        current_node = self._root

        for block_start in range(0, len(token_ids), self.block_size):
            block_end = min(block_start + self.block_size, len(token_ids))
            block_tokens = tuple(token_ids[block_start:block_end])

            if block_tokens in current_node['children']:
                child_node = current_node['children'][block_tokens]
                matched_blocks.append(child_node['block_idx'])
                matched_tokens = block_end
                current_node = child_node
            else:
                # Check for partial match
                for key in current_node['children']:
                    if block_tokens[:len(key)] == key[:len(block_tokens)]:
                        # Partial match
                        child_node = current_node['children'][key]
                        matched_blocks.append(child_node['block_idx'])
                        # Count actual matching tokens
                        match_len = 0
                        for t1, t2 in zip(block_tokens, key):
                            if t1 == t2:
                                match_len += 1
                            else:
                                break
                        matched_tokens = block_start + match_len
                        break
                break

        # Gather KV cache from matched blocks
        if not matched_blocks:
            return [], 0

        kv_cache = []
        for layer_idx in range(self.num_layers):
            k_list = []
            v_list = []
            for block_idx in matched_blocks:
                k_list.append(self.k_blocks[layer_idx][block_idx])
                v_list.append(self.v_blocks[layer_idx][block_idx])

            k = torch.cat(k_list, dim=1)[:, :matched_tokens, :]
            v = torch.cat(v_list, dim=1)[:, :matched_tokens, :]
            kv_cache.append((k.unsqueeze(0), v.unsqueeze(0)))

        return kv_cache, matched_tokens

    def release(self, token_ids: List[int]):
        """
        Release reference to a prefix.

        Args:
            token_ids: Token IDs of the prefix to release
        """
        current_node = self._root

        for block_start in range(0, len(token_ids), self.block_size):
            block_end = min(block_start + self.block_size, len(token_ids))
            block_tokens = tuple(token_ids[block_start:block_end])

            if block_tokens not in current_node['children']:
                break

            child_node = current_node['children'][block_tokens]
            child_node['ref_count'] -= 1

            # Free block if no more references
            if child_node['ref_count'] <= 0:
                if child_node['block_idx'] is not None:
                    self.free_blocks.append(child_node['block_idx'])
                del current_node['children'][block_tokens]
                break

            current_node = child_node

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        used_blocks = self.max_blocks - len(self.free_blocks)
        return {
            'total_blocks': self.max_blocks,
            'used_blocks': used_blocks,
            'free_blocks': len(self.free_blocks),
            'memory_mb': (used_blocks * self.num_layers * 2 *
                         self.num_heads * self.block_size * self.head_dim *
                         2) / (1024 ** 2)  # Assuming FP16
        }

    def forward(self, *args, **kwargs):
        """Not used - cache is managed through insert/match methods."""
        raise NotImplementedError("Use insert() and match() methods instead")
