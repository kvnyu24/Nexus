"""
Attention utility functions for mask generation and manipulation.

This module provides centralized utilities for:
- Causal mask generation
- Attention mask application
- Mask expansion for multi-head attention
"""

import torch
from typing import Optional, Literal


def create_causal_mask(
    seq_length: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate an upper triangular causal mask for autoregressive attention.

    The mask prevents attending to future tokens by setting future positions
    to -inf, which becomes 0 after softmax.

    Args:
        seq_length: Length of the sequence
        dtype: Data type for the mask tensor
        device: Device to place the mask on

    Returns:
        Causal mask tensor of shape (seq_length, seq_length) with -inf above
        the diagonal and 0 on and below the diagonal.

    Example:
        >>> mask = create_causal_mask(4, torch.float32)
        >>> # Returns:
        >>> # [[0, -inf, -inf, -inf],
        >>> #  [0,    0, -inf, -inf],
        >>> #  [0,    0,    0, -inf],
        >>> #  [0,    0,    0,    0]]
    """
    mask = torch.triu(
        torch.ones((seq_length, seq_length), dtype=dtype, device=device) * float("-inf"),
        diagonal=1
    )
    return mask


def apply_attention_mask(
    attention_scores: torch.Tensor,
    mask: torch.Tensor,
    method: Literal["add", "fill"] = "add"
) -> torch.Tensor:
    """
    Apply an attention mask to attention scores with consistent dimensionality handling.

    Supports both additive masking (for pre-computed -inf masks) and fill masking
    (for boolean masks).

    Args:
        attention_scores: Attention scores tensor of shape
            (batch_size, num_heads, seq_length, seq_length) or
            (batch_size, seq_length, seq_length)
        mask: Attention mask tensor. For 'add' method, should contain 0 for
            positions to keep and -inf for positions to mask. For 'fill' method,
            should be boolean with True for positions to mask.
        method: Masking method - 'add' for additive masking, 'fill' for fill masking

    Returns:
        Masked attention scores tensor with same shape as input

    Raises:
        ValueError: If method is not 'add' or 'fill'

    Example:
        >>> scores = torch.randn(2, 8, 16, 16)  # (batch, heads, seq, seq)
        >>> mask = create_causal_mask(16, scores.dtype, scores.device)
        >>> masked_scores = apply_attention_mask(scores, mask, method='add')
    """
    if method not in ("add", "fill"):
        raise ValueError(f"method must be 'add' or 'fill', got '{method}'")

    # Expand mask dimensions to match attention_scores if needed
    scores_dim = attention_scores.dim()
    mask_dim = mask.dim()

    # Handle dimension expansion
    if mask_dim < scores_dim:
        # Add leading dimensions as needed
        for _ in range(scores_dim - mask_dim):
            mask = mask.unsqueeze(0)

    # Broadcast mask to match attention_scores shape
    if method == "add":
        return attention_scores + mask
    else:  # method == "fill"
        return attention_scores.masked_fill(mask, float("-inf"))


def expand_attention_mask(
    mask: torch.Tensor,
    num_heads: int,
    target_dim: int = 4
) -> torch.Tensor:
    """
    Expand an attention mask for multi-head attention.

    Takes a 2D or 3D mask and expands it to have a head dimension, making it
    compatible with multi-head attention computations.

    Args:
        mask: Attention mask tensor of shape:
            - (seq_length, seq_length) - 2D mask
            - (batch_size, seq_length) - 2D padding mask
            - (batch_size, seq_length, seq_length) - 3D mask
        num_heads: Number of attention heads
        target_dim: Target number of dimensions (default 4 for multi-head attention)

    Returns:
        Expanded mask tensor of shape (batch_size, num_heads, seq_length, seq_length)
        or (1, num_heads, seq_length, seq_length) if no batch dimension provided

    Example:
        >>> # 2D causal mask
        >>> mask_2d = create_causal_mask(16, torch.float32)
        >>> expanded = expand_attention_mask(mask_2d, num_heads=8)
        >>> # Shape: (1, 8, 16, 16)

        >>> # 3D batch mask
        >>> mask_3d = torch.randn(2, 16, 16)
        >>> expanded = expand_attention_mask(mask_3d, num_heads=8)
        >>> # Shape: (2, 8, 16, 16)
    """
    original_dim = mask.dim()

    if original_dim == 2:
        # Could be (seq, seq) or (batch, seq)
        # Check if it's a square matrix (likely seq x seq causal mask)
        if mask.shape[0] == mask.shape[1]:
            # (seq_length, seq_length) -> (1, num_heads, seq_length, seq_length)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.expand(-1, num_heads, -1, -1)
        else:
            # (batch_size, seq_length) padding mask
            # -> (batch_size, 1, 1, seq_length) for broadcasting
            mask = mask.unsqueeze(1).unsqueeze(2)
            # Expand heads dimension
            mask = mask.expand(-1, num_heads, -1, -1)
    elif original_dim == 3:
        # (batch_size, seq_length, seq_length) -> (batch_size, num_heads, seq_length, seq_length)
        mask = mask.unsqueeze(1)
        mask = mask.expand(-1, num_heads, -1, -1)
    elif original_dim == 4:
        # Already has head dimension, just ensure correct number of heads
        if mask.shape[1] == 1:
            mask = mask.expand(-1, num_heads, -1, -1)
    else:
        raise ValueError(
            f"mask must be 2D, 3D, or 4D tensor, got {original_dim}D"
        )

    return mask


def combine_masks(
    causal_mask: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    custom_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None
) -> Optional[torch.Tensor]:
    """
    Combine multiple attention masks into a single mask.

    All masks are combined using addition, assuming they use -inf for masked
    positions and 0 for unmasked positions.

    Args:
        causal_mask: Causal (autoregressive) mask
        padding_mask: Padding mask for variable-length sequences
        custom_mask: Any additional custom mask
        dtype: Data type for the combined mask
        device: Device for the combined mask

    Returns:
        Combined mask tensor, or None if all input masks are None

    Example:
        >>> causal = create_causal_mask(16, torch.float32, device)
        >>> padding = torch.zeros(2, 16, device=device)
        >>> padding[:, 10:] = float("-inf")  # Mask positions 10-15
        >>> combined = combine_masks(causal_mask=causal, padding_mask=padding)
    """
    masks = [m for m in [causal_mask, padding_mask, custom_mask] if m is not None]

    if not masks:
        return None

    if len(masks) == 1:
        return masks[0].to(dtype=dtype, device=device) if device else masks[0].to(dtype=dtype)

    # Start with the first mask
    combined = masks[0].clone()

    # Add remaining masks, broadcasting as needed
    for mask in masks[1:]:
        combined = combined + mask

    if device is not None:
        combined = combined.to(dtype=dtype, device=device)
    else:
        combined = combined.to(dtype=dtype)

    return combined
