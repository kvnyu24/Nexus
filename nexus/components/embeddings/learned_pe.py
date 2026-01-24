"""
Learned Positional Encoding.

Standard learned absolute positional embeddings as used by GPT-2, BERT, etc.
"""
import torch
import torch.nn as nn
from typing import Optional
from nexus.core.base import NexusModule


class LearnedPositionalEncoding(NexusModule):
    """
    Learned absolute positional embeddings.

    Standard learned PE used by GPT-2, BERT, etc. Each position has a
    learnable embedding vector that is added to the token embeddings.

    Used by: GPT-2, BERT, RoBERTa, DistilBERT

    Reference:
        - GPT-2: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
        - BERT: https://arxiv.org/abs/1810.04805

    Args:
        max_seq_len: Maximum sequence length
        dim: Embedding dimension
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        max_seq_len: int,
        dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dim = dim

        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(p=dropout)

        # Initialize positions buffer for efficient indexing
        positions = torch.arange(max_seq_len)
        self.register_buffer('positions', positions)

        # Initialize embeddings (small random values)
        self._init_weights()

    def _init_weights(self):
        """Initialize position embeddings with small random values."""
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        offset: int = 0
    ) -> torch.Tensor:
        """
        Add positional embeddings to input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, dim) or (batch, seq_len)
            position_ids: Optional custom position indices (batch, seq_len)
            offset: Starting position offset (useful for incremental decoding)

        Returns:
            Tensor with positional embeddings added (same shape as x if 3D,
            or (batch, seq_len, dim) if x is 2D)
        """
        if x.dim() == 2:
            batch_size, seq_len = x.shape
        else:
            batch_size, seq_len = x.shape[:2]

        # Validate sequence length
        if seq_len + offset > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} + offset {offset} exceeds "
                f"maximum sequence length {self.max_seq_len}"
            )

        # Get position indices
        if position_ids is None:
            position_ids = self.positions[offset:offset + seq_len]
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Get position embeddings
        pos_embeddings = self.position_embeddings(position_ids)

        # Add to input (if 3D) or just return embeddings (if 2D)
        if x.dim() == 3:
            output = x + pos_embeddings
        else:
            output = pos_embeddings

        return self.dropout(output)

    def get_position_embeddings(
        self,
        seq_len: int,
        offset: int = 0,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Get position embeddings without adding to input.

        Args:
            seq_len: Sequence length
            offset: Starting position offset
            device: Target device

        Returns:
            Position embeddings of shape (1, seq_len, dim)
        """
        position_ids = self.positions[offset:offset + seq_len].unsqueeze(0)
        if device is not None:
            position_ids = position_ids.to(device)
        return self.position_embeddings(position_ids)
