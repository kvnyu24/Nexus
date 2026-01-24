"""
Chunked Prefill for processing long prompts efficiently.

Implements chunked prefill to process long prompts by splitting them
into manageable chunks and processing iteratively, building the KV cache
incrementally to avoid memory spikes.

Reference: vLLM and other inference engines use this technique.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any, Union
from nexus.core.base import NexusModule


class ChunkedPrefill(NexusModule):
    """
    Chunked Prefill for processing long prompts efficiently.

    Splits long prompts into chunks and processes iteratively,
    managing KV cache incrementally to avoid memory spikes.
    This is essential for serving long-context models where
    prefilling the entire prompt at once would exceed memory.

    Args:
        chunk_size: Number of tokens per chunk (default: 512)
        attention_module: The underlying attention module to use
        max_seq_len: Maximum sequence length to support
        overlap_chunks: Number of tokens to overlap between chunks for context
        use_gradient_checkpointing: Whether to checkpoint attention computation
    """

    def __init__(
        self,
        chunk_size: int = 512,
        attention_module: Optional[nn.Module] = None,
        max_seq_len: int = 131072,
        overlap_chunks: int = 0,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len
        self.overlap_chunks = overlap_chunks
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Store reference to attention module
        self.attention = attention_module

        # KV cache management
        self._kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._cache_seq_len: int = 0

    def set_attention_module(self, attention_module: nn.Module) -> None:
        """Set or update the underlying attention module."""
        self.attention = attention_module

    def clear_cache(self) -> None:
        """Clear the KV cache."""
        self._kv_cache = None
        self._cache_seq_len = 0

    def get_cache_size(self) -> int:
        """Get current cache sequence length."""
        return self._cache_seq_len

    def _compute_chunk_boundaries(
        self,
        seq_len: int,
        chunk_size: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Compute start and end positions for each chunk.

        Args:
            seq_len: Total sequence length
            chunk_size: Tokens per chunk (uses self.chunk_size if None)

        Returns:
            List of (start, end) tuples for each chunk
        """
        chunk_size = chunk_size or self.chunk_size
        boundaries = []

        start = 0
        while start < seq_len:
            end = min(start + chunk_size, seq_len)
            boundaries.append((start, end))
            # Move to next chunk, accounting for overlap
            start = end - self.overlap_chunks if self.overlap_chunks > 0 else end

        return boundaries

    def _update_kv_cache(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache with new key-value pairs.

        Args:
            new_k: New keys of shape (batch, heads, new_len, head_dim)
            new_v: New values of shape (batch, heads, new_len, head_dim)

        Returns:
            Updated (keys, values) tuple
        """
        if self._kv_cache is None:
            self._kv_cache = (new_k, new_v)
            self._cache_seq_len = new_k.size(2)
        else:
            cached_k, cached_v = self._kv_cache
            self._kv_cache = (
                torch.cat([cached_k, new_k], dim=2),
                torch.cat([cached_v, new_v], dim=2)
            )
            self._cache_seq_len += new_k.size(2)

        return self._kv_cache

    def _attention_with_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention using provided Q against cached K, V.

        Args:
            q: Query tensor of shape (batch, heads, q_len, head_dim)
            k: Full key tensor (cached + new)
            v: Full value tensor (cached + new)
            causal: Whether to apply causal masking
            attention_mask: Additional attention mask

        Returns:
            Attention output of shape (batch, heads, q_len, head_dim)
        """
        batch_size, num_heads, q_len, head_dim = q.shape
        k_len = k.size(2)
        scale = head_dim ** -0.5

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        # Shape: (batch, heads, q_len, k_len)

        # Apply causal mask
        if causal:
            # For prefill: q positions are at the end of the cached sequence
            # Each query at position i can attend to positions 0..i
            q_positions = torch.arange(
                k_len - q_len, k_len, device=q.device
            ).unsqueeze(1)
            k_positions = torch.arange(k_len, device=q.device).unsqueeze(0)
            causal_mask = k_positions > q_positions
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply additional mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax and output
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(q.dtype)
        output = torch.matmul(attn_weights, v)

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        chunk_size: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        causal: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process input in chunks, building KV cache incrementally.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, dim)
            chunk_size: Override chunk size for this call
            attention_mask: Additional attention mask
            position_ids: Position IDs for rotary embeddings
            use_cache: Whether to return the KV cache
            causal: Whether to use causal attention

        Returns:
            output: Tensor of shape (batch, seq_len, dim)
            kv_cache: If use_cache, returns (keys, values) tuple
        """
        if self.attention is None:
            raise ValueError("Attention module not set. Call set_attention_module first.")

        batch_size, seq_len, dim = hidden_states.shape
        chunk_size = chunk_size or self.chunk_size

        # If sequence is short enough, process normally
        if seq_len <= chunk_size:
            output = self._process_single_chunk(
                hidden_states, attention_mask, causal
            )
            if use_cache:
                return output, self._kv_cache
            return output

        # Compute chunk boundaries
        boundaries = self._compute_chunk_boundaries(seq_len, chunk_size)

        # Clear cache at start of new sequence
        self.clear_cache()

        # Process each chunk
        outputs = []

        for chunk_idx, (start, end) in enumerate(boundaries):
            chunk = hidden_states[:, start:end]

            if self.use_gradient_checkpointing and self.training:
                output = torch.utils.checkpoint.checkpoint(
                    self._process_chunk_with_cache,
                    chunk, chunk_idx, len(boundaries), causal,
                    use_reentrant=False
                )
            else:
                output = self._process_chunk_with_cache(
                    chunk, chunk_idx, len(boundaries), causal
                )

            # Handle overlap: only keep non-overlapping part except for last chunk
            if self.overlap_chunks > 0 and chunk_idx < len(boundaries) - 1:
                output = output[:, :-self.overlap_chunks]

            outputs.append(output)

        # Concatenate all chunk outputs
        output = torch.cat(outputs, dim=1)

        if use_cache:
            return output, self._kv_cache
        return output

    def _process_single_chunk(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        causal: bool
    ) -> torch.Tensor:
        """Process a single chunk through the attention module."""
        # Delegate to the underlying attention module
        if hasattr(self.attention, 'forward'):
            # Try to pass through to attention module
            result = self.attention(hidden_states)
            if isinstance(result, tuple):
                return result[0]
            return result
        else:
            raise ValueError("Attention module must have a forward method")

    def _process_chunk_with_cache(
        self,
        chunk: torch.Tensor,
        chunk_idx: int,
        total_chunks: int,
        causal: bool
    ) -> torch.Tensor:
        """
        Process a chunk while managing KV cache.

        This is the core of chunked prefill: we project Q, K, V for the chunk,
        append K, V to cache, and compute attention of Q against full cache.
        """
        batch_size, chunk_len, dim = chunk.shape

        # Get projections from attention module
        if hasattr(self.attention, 'q_proj'):
            q = self.attention.q_proj(chunk)
            k = self.attention.k_proj(chunk)
            v = self.attention.v_proj(chunk)

            # Get dimensions
            if hasattr(self.attention, 'num_heads'):
                num_heads = self.attention.num_heads
            else:
                num_heads = q.size(-1) // (dim // 8)  # Estimate

            if hasattr(self.attention, 'head_dim'):
                head_dim = self.attention.head_dim
            else:
                head_dim = q.size(-1) // num_heads

            # Reshape to (batch, heads, seq, head_dim)
            q = q.view(batch_size, chunk_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, chunk_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, chunk_len, num_heads, head_dim).transpose(1, 2)

            # Update cache with new K, V
            full_k, full_v = self._update_kv_cache(k, v)

            # Compute attention: Q against full K, V
            output = self._attention_with_cache(q, full_k, full_v, causal)

            # Reshape output
            output = output.transpose(1, 2).contiguous().view(batch_size, chunk_len, -1)

            # Apply output projection if available
            if hasattr(self.attention, 'o_proj'):
                output = self.attention.o_proj(output)
            elif hasattr(self.attention, 'out_proj'):
                output = self.attention.out_proj(output)

            return output
        else:
            # Fallback: use attention module directly
            return self._process_single_chunk(chunk, None, causal)

    def prefill(
        self,
        prompt_hidden_states: torch.Tensor,
        chunk_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prefill prompt and return output with KV cache.

        Convenience method that always returns cache for use in generation.

        Args:
            prompt_hidden_states: Prompt embeddings of shape (batch, prompt_len, dim)
            chunk_size: Optional override for chunk size

        Returns:
            Tuple of (output, (key_cache, value_cache))
        """
        self.clear_cache()
        return self.forward(
            prompt_hidden_states,
            chunk_size=chunk_size,
            use_cache=True
        )

    def decode_step(
        self,
        token_hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single decode step using cached KV.

        Args:
            token_hidden_states: Single token embedding of shape (batch, 1, dim)

        Returns:
            Tuple of (output, updated_cache)
        """
        return self.forward(
            token_hidden_states,
            chunk_size=1,
            use_cache=True
        )


class ChunkedPrefillScheduler:
    """
    Scheduler for managing chunked prefill across multiple sequences.

    Implements scheduling logic for batching prefill chunks from
    multiple sequences to maximize throughput.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        max_batch_tokens: int = 8192,
        max_batch_sequences: int = 32
    ):
        self.chunk_size = chunk_size
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_sequences = max_batch_sequences
        self._pending_sequences: List[Dict[str, Any]] = []

    def add_sequence(
        self,
        sequence_id: str,
        hidden_states: torch.Tensor,
        priority: int = 0
    ) -> None:
        """Add a sequence to the scheduling queue."""
        self._pending_sequences.append({
            'id': sequence_id,
            'hidden_states': hidden_states,
            'priority': priority,
            'position': 0,  # Current position in sequence
            'length': hidden_states.size(1)
        })
        # Sort by priority (higher first)
        self._pending_sequences.sort(key=lambda x: -x['priority'])

    def get_next_batch(self) -> Optional[Dict[str, Any]]:
        """
        Get next batch of chunks to process.

        Returns dict with:
            - 'chunks': List of (sequence_id, chunk_tensor) tuples
            - 'total_tokens': Total tokens in batch
        """
        if not self._pending_sequences:
            return None

        batch_chunks = []
        total_tokens = 0

        for seq in self._pending_sequences[:]:
            remaining = seq['length'] - seq['position']
            if remaining <= 0:
                self._pending_sequences.remove(seq)
                continue

            chunk_len = min(remaining, self.chunk_size)
            if total_tokens + chunk_len > self.max_batch_tokens:
                break
            if len(batch_chunks) >= self.max_batch_sequences:
                break

            start = seq['position']
            end = start + chunk_len
            chunk = seq['hidden_states'][:, start:end]

            batch_chunks.append((seq['id'], chunk))
            total_tokens += chunk_len
            seq['position'] = end

        if not batch_chunks:
            return None

        return {
            'chunks': batch_chunks,
            'total_tokens': total_tokens
        }

    def has_pending(self) -> bool:
        """Check if there are pending sequences."""
        return len(self._pending_sequences) > 0
