"""
Context Compression for long sequences.

Implements context compression techniques that compress older context
into summary tokens, enabling effective context length extension
beyond the model's native context window.

References:
- AutoCompressor: https://arxiv.org/abs/2305.14788
- Gisting: https://arxiv.org/abs/2304.08467
- LongLLM Lingua: https://arxiv.org/abs/2310.06839
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union
from nexus.core.base import NexusModule


class ContextCompression(NexusModule):
    """
    Context compression for long sequences.

    Compresses older context into summary tokens, allowing
    effective context length extension. Uses learned compression
    to distill important information from token sequences.

    The compression process:
    1. Segment the context into chunks
    2. Each chunk is compressed into a fixed number of summary tokens
    3. Summary tokens are used as compressed context for future attention

    Args:
        dim: Model dimension
        compression_ratio: How much to compress (e.g., 4 = 4:1 ratio)
        num_summary_tokens: Number of summary tokens per segment
        num_heads: Number of attention heads for compression
        segment_size: Size of each segment to compress (default: compression_ratio * num_summary_tokens)
        accumulate_summaries: Whether to accumulate summaries across segments
        dropout: Dropout probability
    """

    def __init__(
        self,
        dim: int,
        compression_ratio: int = 4,
        num_summary_tokens: int = 64,
        num_heads: int = 8,
        segment_size: Optional[int] = None,
        accumulate_summaries: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.compression_ratio = compression_ratio
        self.num_summary_tokens = num_summary_tokens
        self.num_heads = num_heads
        self.segment_size = segment_size or (compression_ratio * num_summary_tokens)
        self.accumulate_summaries = accumulate_summaries
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        # Learnable summary tokens (soft prompts for compression)
        self.summary_tokens = nn.Parameter(
            torch.randn(1, num_summary_tokens, dim) * 0.02
        )

        # Cross-attention for compression: summary tokens attend to context
        self.compress_q_proj = nn.Linear(dim, dim)
        self.compress_k_proj = nn.Linear(dim, dim)
        self.compress_v_proj = nn.Linear(dim, dim)
        self.compress_o_proj = nn.Linear(dim, dim)

        # Layer norm for compression
        self.compress_norm = nn.LayerNorm(dim)

        # Optional: MLP for additional processing of summaries
        self.summary_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.mlp_norm = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # For accumulated summaries
        self._accumulated_summaries: Optional[torch.Tensor] = None

    def reset_accumulated_summaries(self) -> None:
        """Reset accumulated summaries for new sequence."""
        self._accumulated_summaries = None

    def get_accumulated_summaries(self) -> Optional[torch.Tensor]:
        """Get current accumulated summaries."""
        return self._accumulated_summaries

    def _compress_segment(
        self,
        segment: torch.Tensor,
        summary_init: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compress a segment into summary tokens using cross-attention.

        Args:
            segment: Tensor of shape (batch, segment_len, dim)
            summary_init: Optional initial summary tokens

        Returns:
            Summary tokens of shape (batch, num_summary_tokens, dim)
        """
        batch_size = segment.size(0)

        # Initialize summary tokens
        if summary_init is None:
            summaries = self.summary_tokens.expand(batch_size, -1, -1)
        else:
            summaries = summary_init

        # Cross-attention: summaries query, segment provides keys/values
        q = self.compress_q_proj(summaries)
        k = self.compress_k_proj(segment)
        v = self.compress_v_proj(segment)

        # Reshape for multi-head attention
        q = q.view(batch_size, self.num_summary_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_weights = self.dropout(attn_weights.to(q.dtype))

        # Compute output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, self.num_summary_tokens, -1)
        attn_output = self.compress_o_proj(attn_output)

        # Residual connection and norm
        summaries = self.compress_norm(summaries + attn_output)

        # MLP
        summaries = self.mlp_norm(summaries + self.summary_mlp(summaries))

        return summaries

    def compress(
        self,
        context: torch.Tensor,
        return_all_summaries: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Compress context into summary tokens.

        Args:
            context: Input context of shape (batch, seq_len, dim)
            return_all_summaries: Whether to return summaries for each segment

        Returns:
            If return_all_summaries:
                Tuple of (final_summaries, list_of_segment_summaries)
            Else:
                Final summary tokens of shape (batch, num_summary_tokens, dim)
        """
        batch_size, seq_len, _ = context.shape

        # Compute number of segments
        num_segments = max(1, seq_len // self.segment_size)

        all_summaries = []
        current_summary = None

        for seg_idx in range(num_segments):
            start = seg_idx * self.segment_size
            end = min(start + self.segment_size, seq_len)
            segment = context[:, start:end]

            # Compress segment, optionally using previous summary as init
            if self.accumulate_summaries and current_summary is not None:
                # Incorporate previous summaries into context for compression
                extended_segment = torch.cat([current_summary, segment], dim=1)
                segment_summary = self._compress_segment(extended_segment)
            else:
                segment_summary = self._compress_segment(segment, current_summary)

            all_summaries.append(segment_summary)
            current_summary = segment_summary

        # Handle remaining tokens
        remaining_start = num_segments * self.segment_size
        if remaining_start < seq_len:
            remaining = context[:, remaining_start:]
            if current_summary is not None:
                extended_remaining = torch.cat([current_summary, remaining], dim=1)
                final_summary = self._compress_segment(extended_remaining)
            else:
                final_summary = self._compress_segment(remaining)
            all_summaries.append(final_summary)
            current_summary = final_summary

        # Update accumulated summaries if enabled
        if self.accumulate_summaries:
            self._accumulated_summaries = current_summary

        if return_all_summaries:
            return current_summary, all_summaries
        return current_summary

    def forward(
        self,
        hidden_states: torch.Tensor,
        compress_prefix: bool = True,
        prefix_length: Optional[int] = None,
        return_compressed: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional context compression.

        Can operate in two modes:
        1. Compress entire input into summary tokens
        2. Compress prefix, keep suffix as-is, return concatenated

        Args:
            hidden_states: Input of shape (batch, seq_len, dim)
            compress_prefix: If True, compress the prefix portion
            prefix_length: Length of prefix to compress (default: all but segment_size tokens)
            return_compressed: If True, return compressed representation separately

        Returns:
            If return_compressed:
                Tuple of (output, compressed_context)
            Else:
                Output tensor combining compressed prefix and suffix
        """
        batch_size, seq_len, _ = hidden_states.shape

        if not compress_prefix:
            # Just compress everything
            compressed = self.compress(hidden_states)
            if return_compressed:
                return compressed, compressed
            return compressed

        # Determine prefix length
        if prefix_length is None:
            # Keep last segment_size tokens uncompressed
            prefix_length = max(0, seq_len - self.segment_size)

        if prefix_length <= 0:
            # Nothing to compress
            if return_compressed:
                return hidden_states, None
            return hidden_states

        # Split into prefix (to compress) and suffix (to keep)
        prefix = hidden_states[:, :prefix_length]
        suffix = hidden_states[:, prefix_length:]

        # Compress prefix
        compressed_prefix = self.compress(prefix)

        # Concatenate compressed prefix with suffix
        output = torch.cat([compressed_prefix, suffix], dim=1)

        if return_compressed:
            return output, compressed_prefix
        return output


class HierarchicalContextCompression(NexusModule):
    """
    Hierarchical context compression with multiple levels.

    Implements a tree-like compression structure where context
    is progressively compressed at multiple scales.

    Args:
        dim: Model dimension
        num_levels: Number of compression levels
        compression_ratio_per_level: Compression ratio at each level
        num_summary_tokens_per_level: Summary tokens at each level
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        dim: int,
        num_levels: int = 3,
        compression_ratio_per_level: int = 4,
        num_summary_tokens_per_level: int = 32,
        num_heads: int = 8
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels

        # Create compression layers for each level
        self.compressors = nn.ModuleList([
            ContextCompression(
                dim=dim,
                compression_ratio=compression_ratio_per_level,
                num_summary_tokens=num_summary_tokens_per_level,
                num_heads=num_heads,
                accumulate_summaries=True
            )
            for _ in range(num_levels)
        ])

        # Level embeddings to distinguish compression levels
        self.level_embeddings = nn.Parameter(
            torch.randn(num_levels, dim) * 0.02
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_all_levels: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Hierarchically compress context.

        Args:
            hidden_states: Input of shape (batch, seq_len, dim)
            return_all_levels: Whether to return summaries at all levels

        Returns:
            Final compressed representation or tuple with all level summaries
        """
        all_level_summaries = []
        current_context = hidden_states

        for level, compressor in enumerate(self.compressors):
            # Compress current context
            compressed = compressor.compress(current_context)

            # Add level embedding
            compressed = compressed + self.level_embeddings[level].unsqueeze(0).unsqueeze(0)

            all_level_summaries.append(compressed)
            current_context = compressed

        if return_all_levels:
            return compressed, all_level_summaries
        return compressed


class AdaptiveContextCompression(NexusModule):
    """
    Adaptive context compression based on importance scoring.

    Learns to identify important tokens and compress less important
    regions more aggressively.

    Args:
        dim: Model dimension
        min_compression_ratio: Minimum compression (for important regions)
        max_compression_ratio: Maximum compression (for less important regions)
        num_summary_tokens: Base number of summary tokens
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        dim: int,
        min_compression_ratio: int = 2,
        max_compression_ratio: int = 8,
        num_summary_tokens: int = 64,
        num_heads: int = 8
    ):
        super().__init__()
        self.dim = dim
        self.min_compression_ratio = min_compression_ratio
        self.max_compression_ratio = max_compression_ratio
        self.num_summary_tokens = num_summary_tokens

        # Importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

        # Base compressor
        self.compressor = ContextCompression(
            dim=dim,
            compression_ratio=min_compression_ratio,
            num_summary_tokens=num_summary_tokens,
            num_heads=num_heads
        )

    def compute_importance(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute importance scores for each token.

        Args:
            hidden_states: Input of shape (batch, seq_len, dim)

        Returns:
            Importance scores of shape (batch, seq_len)
        """
        scores = self.importance_scorer(hidden_states).squeeze(-1)
        return scores

    def forward(
        self,
        hidden_states: torch.Tensor,
        importance_threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptively compress based on importance.

        Args:
            hidden_states: Input of shape (batch, seq_len, dim)
            importance_threshold: Threshold for keeping tokens uncompressed

        Returns:
            Tuple of (compressed_output, importance_scores)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute importance scores
        importance = self.compute_importance(hidden_states)

        # Identify important vs. less important regions
        important_mask = importance > importance_threshold

        # For simplicity, we compress everything but weight by importance
        # More sophisticated: segment and compress differently
        compressed = self.compressor.compress(hidden_states)

        # Weight summaries by average importance
        avg_importance = importance.mean(dim=1, keepdim=True).unsqueeze(-1)
        compressed = compressed * (0.5 + avg_importance)

        return compressed, importance
