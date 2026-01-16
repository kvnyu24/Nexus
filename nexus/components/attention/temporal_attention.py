import torch
import torch.nn as nn
from typing import Optional, Tuple
from nexus.core.base import NexusModule
from nexus.utils.attention_utils import create_causal_mask

class TemporalAttention(NexusModule):
    """Temporal attention module for sequence modeling in quantitative models.

    This module provides multi-head attention specialized for time series data,
    with optional positional encoding and causal masking.

    Args:
        hidden_dim: Dimension of hidden states
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.1)
        causal: Whether to apply causal masking (default: True)
        batch_first: Whether batch dimension is first (default: True)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        causal: bool = True,
        batch_first: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.causal = causal

        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            attention_mask: Optional attention mask
            key_padding_mask: Optional padding mask

        Returns:
            Tuple of (output, attention_weights)
        """
        # Generate causal mask if needed
        if self.causal and attention_mask is None:
            seq_len = x.size(1)
            attention_mask = create_causal_mask(seq_len, dtype=torch.bool, device=x.device)

        # Apply attention with residual connection
        attended, weights = self.attention(
            x, x, x,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )

        # Residual + layer norm
        output = self.layer_norm(x + self.dropout(attended))

        return output, weights
