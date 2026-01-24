"""
Sparse Attention implementations.

Sparse attention reduces computational complexity from O(n^2) to O(n * k) where k
is the number of positions each token attends to. Supports various sparsity patterns
including local windows, strided patterns, and learned patterns.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Literal, List
from nexus.core.base import NexusModule


class SparseAttention(NexusModule):
    """Fine-Grained Sparse Attention.

    Attends to a sparse subset of positions based on configurable sparsity patterns.
    Reduces complexity from O(n^2) to O(n * k) where k is pattern-dependent.

    Used by: DeepSeek V3 (DSA), Longformer, BigBird

    References:
        - Longformer: https://arxiv.org/abs/2004.05150
        - BigBird: https://arxiv.org/abs/2007.14062
        - DeepSeek V3: https://arxiv.org/abs/2412.19437

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        sparsity_pattern: Type of sparsity pattern
            - 'local': Attend to local window only
            - 'strided': Attend to every k-th position
            - 'local_global': Local window + global tokens
            - 'bigbird': Local + global + random (BigBird pattern)
        local_window: Size of local attention window
        global_tokens: Number of global tokens (for local_global and bigbird)
        num_random: Number of random attention connections (for bigbird)
        stride: Stride for strided attention pattern
        head_dim: Dimension per head (default: dim // num_heads)
        dropout: Attention dropout probability
        bias: Whether to use bias in projections
        causal: Whether to use causal masking
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sparsity_pattern: Literal['local', 'strided', 'local_global', 'bigbird'] = 'local',
        local_window: int = 256,
        global_tokens: int = 1,
        num_random: int = 3,
        stride: int = 64,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.sparsity_pattern = sparsity_pattern
        self.local_window = local_window
        self.global_tokens = global_tokens
        self.num_random = num_random
        self.stride = stride
        self.dropout_p = dropout
        self.causal = causal

        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        # For learned sparsity (optional extension)
        self._random_indices_cache = None
        self._cache_seq_len = 0

    def _create_sparse_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Create sparse attention mask based on pattern.

        Args:
            seq_len: Sequence length
            device: Device for mask
            dtype: Data type for mask

        Returns:
            Mask of shape (1, 1, seq_len, seq_len), 0 for attend, -inf for ignore
        """
        # Initialize mask with all positions masked
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)

        # Create position indices
        row_idx = torch.arange(seq_len, device=device)
        col_idx = torch.arange(seq_len, device=device)

        if self.sparsity_pattern == 'local':
            # Local window attention
            for i in range(seq_len):
                start = max(0, i - self.local_window + 1) if self.causal else max(0, i - self.local_window // 2)
                end = i + 1 if self.causal else min(seq_len, i + self.local_window // 2 + 1)
                mask[i, start:end] = 0.0

        elif self.sparsity_pattern == 'strided':
            # Strided attention: attend to every stride-th position
            for i in range(seq_len):
                # Local window
                start = max(0, i - self.local_window + 1) if self.causal else max(0, i - self.local_window // 2)
                end = i + 1 if self.causal else min(seq_len, i + self.local_window // 2 + 1)
                mask[i, start:end] = 0.0

                # Strided positions
                if self.causal:
                    strided_positions = torch.arange(0, i + 1, self.stride, device=device)
                else:
                    strided_positions = torch.arange(0, seq_len, self.stride, device=device)
                mask[i, strided_positions] = 0.0

        elif self.sparsity_pattern == 'local_global':
            # Local window + global tokens
            for i in range(seq_len):
                # Local window
                start = max(0, i - self.local_window + 1) if self.causal else max(0, i - self.local_window // 2)
                end = i + 1 if self.causal else min(seq_len, i + self.local_window // 2 + 1)
                mask[i, start:end] = 0.0

            # Global tokens: first global_tokens positions attend to/from all
            # Global tokens attend to all (respecting causality)
            for i in range(min(self.global_tokens, seq_len)):
                if self.causal:
                    mask[i, :i+1] = 0.0
                else:
                    mask[i, :] = 0.0

            # All tokens attend to global tokens
            if self.causal:
                # Can only attend to global tokens that come before
                for i in range(seq_len):
                    mask[i, :min(self.global_tokens, i+1)] = 0.0
            else:
                mask[:, :self.global_tokens] = 0.0

        elif self.sparsity_pattern == 'bigbird':
            # BigBird: local + global + random
            for i in range(seq_len):
                # Local window
                start = max(0, i - self.local_window + 1) if self.causal else max(0, i - self.local_window // 2)
                end = i + 1 if self.causal else min(seq_len, i + self.local_window // 2 + 1)
                mask[i, start:end] = 0.0

            # Global tokens
            for i in range(min(self.global_tokens, seq_len)):
                if self.causal:
                    mask[i, :i+1] = 0.0
                else:
                    mask[i, :] = 0.0
            if self.causal:
                for i in range(seq_len):
                    mask[i, :min(self.global_tokens, i+1)] = 0.0
            else:
                mask[:, :self.global_tokens] = 0.0

            # Random attention connections
            if self.num_random > 0:
                for i in range(seq_len):
                    max_pos = i if self.causal else seq_len - 1
                    if max_pos > 0:
                        num_random = min(self.num_random, max_pos)
                        random_positions = torch.randperm(max_pos + 1, device=device)[:num_random]
                        if self.causal:
                            random_positions = random_positions[random_positions <= i]
                        mask[i, random_positions] = 0.0
        else:
            raise ValueError(f"Unknown sparsity pattern: {self.sparsity_pattern}")

        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with sparse attention.

        Args:
            hidden_states: Input of shape (batch, seq_len, dim)
            attention_mask: Optional additional mask
            position_embeddings: Tuple of (cos, sin) for RoPE
            past_key_value: KV cache for incremental decoding
            use_cache: Whether to return updated cache
            output_attentions: Whether to return attention weights

        Returns:
            output: Shape (batch, seq_len, dim)
            attn_weights: If output_attentions
            past_key_value: If use_cache
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        kv_seq_len = seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale

        # Apply sparse mask
        sparse_mask = self._create_sparse_mask(kv_seq_len, hidden_states.device, hidden_states.dtype)

        # For incremental decoding, only keep last row(s) of mask
        if seq_len < kv_seq_len:
            sparse_mask = sparse_mask[:, :, -seq_len:, :]

        attn_weights = attn_weights + sparse_mask

        # Apply additional mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        # Output
        attn_output = torch.matmul(attn_weights_dropped, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary position embeddings."""
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


class BlockSparseAttention(NexusModule):
    """Block-Sparse Attention.

    Performs attention on blocks of tokens rather than individual tokens.
    More hardware-efficient than fine-grained sparse attention due to
    better memory access patterns and GPU utilization.

    References:
        - Sparse Transformers: https://arxiv.org/abs/1904.10509
        - BigBird: https://arxiv.org/abs/2007.14062

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        block_size: Size of each attention block
        num_global_blocks: Number of global blocks that attend to all
        num_random_blocks: Number of random block connections
        num_sliding_blocks: Number of sliding window blocks
        head_dim: Dimension per head
        dropout: Attention dropout probability
        bias: Whether to use bias in projections
        causal: Whether to use causal masking
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        block_size: int = 64,
        num_global_blocks: int = 1,
        num_random_blocks: int = 1,
        num_sliding_blocks: int = 3,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.block_size = block_size
        self.num_global_blocks = num_global_blocks
        self.num_random_blocks = num_random_blocks
        self.num_sliding_blocks = num_sliding_blocks
        self.dropout_p = dropout
        self.causal = causal

        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

    def _get_block_indices(
        self,
        num_blocks: int,
        device: torch.device
    ) -> List[Tuple[int, int]]:
        """Get list of (query_block, key_block) pairs to compute attention for.

        Returns list of block index pairs based on the sparsity pattern.
        """
        block_pairs = []

        for q_block in range(num_blocks):
            # Global blocks: first num_global_blocks attend to all (respecting causality)
            if q_block < self.num_global_blocks:
                max_k = q_block + 1 if self.causal else num_blocks
                for k_block in range(max_k):
                    block_pairs.append((q_block, k_block))
                continue

            # All blocks attend to global blocks
            for k_block in range(min(self.num_global_blocks, q_block + 1 if self.causal else num_blocks)):
                block_pairs.append((q_block, k_block))

            # Sliding window blocks
            start = max(self.num_global_blocks, q_block - self.num_sliding_blocks + 1)
            end = q_block + 1 if self.causal else min(num_blocks, q_block + self.num_sliding_blocks)
            for k_block in range(start, end):
                if (q_block, k_block) not in block_pairs:
                    block_pairs.append((q_block, k_block))

            # Random blocks
            if self.num_random_blocks > 0:
                max_k = q_block if self.causal else num_blocks
                available = [k for k in range(max_k)
                           if (q_block, k) not in block_pairs and k >= self.num_global_blocks]
                if available:
                    num_random = min(self.num_random_blocks, len(available))
                    random_blocks = torch.randperm(len(available), device=device)[:num_random]
                    for idx in random_blocks:
                        block_pairs.append((q_block, available[idx]))

        return block_pairs

    def _pad_to_blocks(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pad tensor to be divisible by block_size.

        Args:
            x: Tensor of shape (batch, seq_len, dim)

        Returns:
            Padded tensor and original sequence length
        """
        batch_size, seq_len, dim = x.shape
        pad_len = (self.block_size - seq_len % self.block_size) % self.block_size

        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len), value=0.0)

        return x, seq_len

    def _reshape_to_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor from (batch, heads, seq, dim) to (batch, heads, num_blocks, block_size, dim)."""
        batch_size, num_heads, seq_len, head_dim = x.shape
        num_blocks = seq_len // self.block_size
        return x.view(batch_size, num_heads, num_blocks, self.block_size, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with block-sparse attention.

        Args:
            hidden_states: Input of shape (batch, seq_len, dim)
            attention_mask: Optional additional mask
            position_embeddings: Tuple of (cos, sin) for RoPE
            output_attentions: Whether to return attention weights

        Returns:
            output: Shape (batch, seq_len, dim)
            attn_weights: If output_attentions (sparse representation)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Pad to block size
        hidden_states_padded, orig_seq_len = self._pad_to_blocks(hidden_states)
        padded_seq_len = hidden_states_padded.shape[1]
        num_blocks = padded_seq_len // self.block_size

        # Project Q, K, V
        query_states = self.q_proj(hidden_states_padded)
        key_states = self.k_proj(hidden_states_padded)
        value_states = self.v_proj(hidden_states_padded)

        # Reshape to (batch, heads, seq, head_dim)
        query_states = query_states.view(batch_size, padded_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, padded_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, padded_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Adjust for padded length
            if cos.shape[2] < padded_seq_len:
                cos = F.pad(cos, (0, 0, 0, padded_seq_len - cos.shape[2]))
                sin = F.pad(sin, (0, 0, 0, padded_seq_len - sin.shape[2]))
            query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Reshape to blocks: (batch, heads, num_blocks, block_size, head_dim)
        query_blocks = self._reshape_to_blocks(query_states)
        key_blocks = self._reshape_to_blocks(key_states)
        value_blocks = self._reshape_to_blocks(value_states)

        # Get block pairs to compute
        block_pairs = self._get_block_indices(num_blocks, hidden_states.device)

        # Initialize output
        output = torch.zeros_like(query_states)
        normalizer = torch.zeros(batch_size, self.num_heads, padded_seq_len, 1,
                                device=hidden_states.device, dtype=hidden_states.dtype)

        attn_weights_all = None
        if output_attentions:
            attn_weights_all = torch.zeros(batch_size, self.num_heads, padded_seq_len, padded_seq_len,
                                          device=hidden_states.device, dtype=hidden_states.dtype)

        # Compute attention for each block pair
        for q_idx, k_idx in block_pairs:
            q_start = q_idx * self.block_size
            q_end = (q_idx + 1) * self.block_size
            k_start = k_idx * self.block_size
            k_end = (k_idx + 1) * self.block_size

            # Get blocks
            q_block = query_blocks[:, :, q_idx]  # (batch, heads, block_size, head_dim)
            k_block = key_blocks[:, :, k_idx]
            v_block = value_blocks[:, :, k_idx]

            # Compute attention for this block pair
            attn_scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale

            # Apply causal mask within blocks if needed
            if self.causal and q_idx == k_idx:
                causal_mask = torch.triu(
                    torch.ones(self.block_size, self.block_size, device=hidden_states.device),
                    diagonal=1
                ).bool()
                attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
            elif self.causal and q_idx < k_idx:
                # Skip: query block comes before key block in causal setting
                continue

            # Apply padding mask
            if orig_seq_len < padded_seq_len:
                if k_end > orig_seq_len:
                    pad_start = max(0, orig_seq_len - k_start)
                    attn_scores[:, :, :, pad_start:] = float('-inf')

            # Compute attention weights (before softmax for proper normalization)
            attn_scores_exp = torch.exp(attn_scores - attn_scores.max(dim=-1, keepdim=True).values)

            # Mask out -inf positions
            attn_scores_exp = attn_scores_exp.masked_fill(attn_scores == float('-inf'), 0.0)

            # Accumulate
            output[:, :, q_start:q_end] += torch.matmul(attn_scores_exp, v_block)
            normalizer[:, :, q_start:q_end] += attn_scores_exp.sum(dim=-1, keepdim=True)

            if output_attentions:
                attn_weights_all[:, :, q_start:q_end, k_start:k_end] = attn_scores_exp

        # Normalize
        output = output / (normalizer + 1e-6)

        # Apply dropout
        output = self.attn_dropout(output)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, padded_seq_len, -1)
        output = self.o_proj(output)

        # Remove padding
        output = output[:, :orig_seq_len, :]

        if output_attentions:
            # Normalize attention weights
            attn_weights_all = attn_weights_all / (normalizer.transpose(2, 3) + 1e-6)
            attn_weights_all = attn_weights_all[:, :, :orig_seq_len, :orig_seq_len]

        return output, attn_weights_all

    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary position embeddings."""
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


class DSA(SparseAttention):
    """Alias for SparseAttention (DeepSeek-style Sparse Attention)."""
    pass
