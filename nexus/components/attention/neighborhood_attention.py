"""
Neighborhood Attention: Sliding-window self-attention for local neighborhoods.

Neighborhood Attention (NA) restricts each token's attention to a local
neighborhood defined by a kernel around that token. This reduces the
quadratic complexity of standard attention to linear, making it particularly
suitable for vision tasks and long sequences where global attention is
unnecessary.

Key features:
    - 1D Neighborhood Attention: For sequence (NLP) data
    - 2D Neighborhood Attention: For spatial (vision) data
    - Dilated variant: Expands the receptive field without increasing kernel size
    - Boundary handling: Tokens near edges attend to available neighbors

The attention pattern resembles a convolution-like local window that slides
across the sequence/image, but with data-dependent (attention) weights
instead of fixed convolutional filters.

Reference: https://arxiv.org/abs/2204.07143 (Neighborhood Attention Transformer)
           https://arxiv.org/abs/2209.15001 (Dilated Neighborhood Attention Transformer)

See Also:
    - sliding_window.py: Causal sliding window for autoregressive models
    - flash_attention.py: Memory-efficient global attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from nexus.core.base import NexusModule


class NeighborhoodAttention1D(NexusModule):
    """1D Neighborhood Attention for sequence data.

    Each token attends only to its local neighborhood of size `kernel_size`,
    centered on itself. With dilation, the neighborhood is expanded by
    spacing tokens `dilation` apart.

    Effective receptive field = kernel_size + (kernel_size - 1) * (dilation - 1)

    Args:
        d_model: Model dimension (input/output size)
        num_heads: Number of attention heads
        kernel_size: Size of the local attention window (must be odd)
        dilation: Dilation factor for expanded receptive field (default 1)
        dropout: Attention dropout probability
        bias: Whether to use bias in projections
        head_dim: Dimension per head. Defaults to d_model // num_heads.

    Example:
        >>> na = NeighborhoodAttention1D(d_model=256, num_heads=8, kernel_size=7)
        >>> x = torch.randn(2, 100, 256)
        >>> out, attn = na(x)
        >>> out.shape
        torch.Size([2, 100, 256])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        head_dim: Optional[int] = None
    ):
        super().__init__()

        assert kernel_size % 2 == 1, \
            f"kernel_size must be odd, got {kernel_size}"
        assert dilation >= 1, \
            f"dilation must be >= 1, got {dilation}"

        self.d_model = d_model
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.head_dim = head_dim or (d_model // num_heads)
        self.dropout_p = dropout

        self.scale = self.head_dim ** -0.5
        self.half_kernel = kernel_size // 2

        # Projections
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        # Learnable relative position bias
        self.rpb = nn.Parameter(
            torch.zeros(num_heads, kernel_size)
        )
        nn.init.trunc_normal_(self.rpb, std=0.02)

    def _gather_neighbors(
        self,
        x: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """Gather local neighborhood tokens for each position.

        For each token at position i, gathers tokens at positions
        [i - half_kernel*dilation, ..., i, ..., i + half_kernel*dilation]
        with step size `dilation`.

        Args:
            x: Input tensor (batch, num_heads, seq_len, head_dim)
            seq_len: Sequence length

        Returns:
            Neighborhoods (batch, num_heads, seq_len, kernel_size, head_dim)
        """
        batch_size, num_heads, _, head_dim = x.shape

        # Pad the sequence for boundary handling
        pad_size = self.half_kernel * self.dilation
        # Pad on both sides along the sequence dimension
        x_padded = F.pad(x, (0, 0, pad_size, pad_size), mode='constant', value=0)

        # Gather neighborhoods using unfold
        # After padding, sequence dimension is at index 2
        # unfold(dimension, size, step) - we need to extract windows
        neighborhoods = []
        for offset in range(-self.half_kernel, self.half_kernel + 1):
            idx = pad_size + offset * self.dilation
            # Gather the position for all sequence elements
            indices = torch.arange(seq_len, device=x.device) + idx
            indices = indices.clamp(0, x_padded.shape[2] - 1)
            gathered = x_padded[:, :, indices, :]  # (B, H, S, D)
            neighborhoods.append(gathered)

        # Stack: (B, H, S, kernel_size, D)
        neighborhoods = torch.stack(neighborhoods, dim=3)
        return neighborhoods

    def _compute_neighbor_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Compute validity mask for neighborhood positions.

        Marks which neighbors are valid (within sequence bounds)
        for each position.

        Args:
            seq_len: Sequence length
            device: Device for tensor

        Returns:
            Mask (seq_len, kernel_size) where True means valid
        """
        positions = torch.arange(seq_len, device=device)
        offsets = torch.arange(
            -self.half_kernel, self.half_kernel + 1, device=device
        ) * self.dilation

        # Neighbor positions for each sequence position
        neighbor_pos = positions.unsqueeze(1) + offsets.unsqueeze(0)

        # Valid if within [0, seq_len)
        valid_mask = (neighbor_pos >= 0) & (neighbor_pos < seq_len)
        return valid_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: Input tensor (batch, seq_len, d_model)
            attention_mask: Optional mask (not typically used with NA)
            output_attentions: Whether to return attention weights

        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, num_heads, seq_len, kernel_size) if output_attentions
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape: (B, S, H, D)  ->  (B, H, S, D)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Gather K and V neighborhoods
        k_neighbors = self._gather_neighbors(k, seq_len)  # (B, H, S, K, D)
        v_neighbors = self._gather_neighbors(v, seq_len)  # (B, H, S, K, D)

        # Compute attention scores: q @ k_neighbors^T
        # q: (B, H, S, D) -> (B, H, S, 1, D)
        # k_neighbors: (B, H, S, K, D) -> transpose last two -> (B, H, S, D, K)
        attn_scores = torch.einsum(
            'bhsd,bhskd->bhsk', q, k_neighbors
        ) * self.scale

        # Add relative position bias
        attn_scores = attn_scores + self.rpb.unsqueeze(0).unsqueeze(2)

        # Mask invalid neighbors (out of bounds)
        valid_mask = self._compute_neighbor_mask(seq_len, hidden_states.device)
        attn_scores = attn_scores.masked_fill(
            ~valid_mask.unsqueeze(0).unsqueeze(0), float('-inf')
        )

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute output: attn_weights @ v_neighbors
        # attn_weights: (B, H, S, K)
        # v_neighbors: (B, H, S, K, D)
        output = torch.einsum('bhsk,bhskd->bhsd', attn_weights, v_neighbors)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)
        output = self.o_proj(output)

        if not output_attentions:
            attn_weights = None

        return output, attn_weights


class NeighborhoodAttention2D(NexusModule):
    """2D Neighborhood Attention for spatial (image/feature map) data.

    Each spatial location attends to a 2D local neighborhood of size
    kernel_size x kernel_size. Supports dilation for larger receptive
    fields without increasing the kernel size.

    Args:
        d_model: Model dimension (channel dimension)
        num_heads: Number of attention heads
        kernel_size: Size of the 2D attention window (must be odd)
        dilation: Dilation factor (default 1)
        dropout: Attention dropout probability
        bias: Whether to use bias in projections
        head_dim: Dimension per head. Defaults to d_model // num_heads.

    Example:
        >>> na2d = NeighborhoodAttention2D(d_model=256, num_heads=8, kernel_size=7)
        >>> x = torch.randn(2, 14, 14, 256)  # (B, H, W, C)
        >>> out, attn = na2d(x)
        >>> out.shape
        torch.Size([2, 14, 14, 256])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        head_dim: Optional[int] = None
    ):
        super().__init__()

        assert kernel_size % 2 == 1, \
            f"kernel_size must be odd, got {kernel_size}"

        self.d_model = d_model
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.head_dim = head_dim or (d_model // num_heads)
        self.dropout_p = dropout

        self.scale = self.head_dim ** -0.5
        self.half_kernel = kernel_size // 2
        self.window_area = kernel_size * kernel_size

        # Projections
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        # Learnable 2D relative position bias
        self.rpb = nn.Parameter(
            torch.zeros(num_heads, kernel_size * kernel_size)
        )
        nn.init.trunc_normal_(self.rpb, std=0.02)

    def _gather_2d_neighbors(
        self,
        x: torch.Tensor,
        height: int,
        width: int
    ) -> torch.Tensor:
        """Gather 2D local neighborhoods for each spatial position.

        Args:
            x: Input (batch, num_heads, height*width, head_dim)
            height: Spatial height
            width: Spatial width

        Returns:
            Neighborhoods (batch, num_heads, height*width, kernel_size^2, head_dim)
        """
        batch_size, num_heads, _, head_dim = x.shape

        # Reshape to spatial layout
        x_spatial = x.view(batch_size, num_heads, height, width, head_dim)

        # Pad spatially
        pad_h = self.half_kernel * self.dilation
        pad_w = self.half_kernel * self.dilation
        x_padded = F.pad(
            x_spatial, (0, 0, pad_w, pad_w, pad_h, pad_h), mode='constant', value=0
        )

        neighborhoods = []
        for dy in range(-self.half_kernel, self.half_kernel + 1):
            for dx in range(-self.half_kernel, self.half_kernel + 1):
                y_idx = pad_h + dy * self.dilation
                x_idx = pad_w + dx * self.dilation

                y_indices = torch.arange(height, device=x.device) + y_idx
                x_indices = torch.arange(width, device=x.device) + x_idx

                y_indices = y_indices.clamp(0, x_padded.shape[2] - 1)
                x_indices = x_indices.clamp(0, x_padded.shape[3] - 1)

                # Gather using advanced indexing
                gathered = x_padded[:, :, y_indices[:, None], x_indices[None, :], :]
                # (B, H, height, width, D) -> (B, H, height*width, D)
                gathered = gathered.reshape(batch_size, num_heads, height * width, head_dim)
                neighborhoods.append(gathered)

        # Stack: (B, H, H*W, K^2, D)
        neighborhoods = torch.stack(neighborhoods, dim=3)
        return neighborhoods

    def _compute_2d_neighbor_mask(
        self,
        height: int,
        width: int,
        device: torch.device
    ) -> torch.Tensor:
        """Compute validity mask for 2D neighborhood positions.

        Args:
            height: Spatial height
            width: Spatial width
            device: Device for tensor

        Returns:
            Mask (height*width, kernel_size^2) where True means valid
        """
        y_pos = torch.arange(height, device=device)
        x_pos = torch.arange(width, device=device)

        # Create 2D position grid
        gy, gx = torch.meshgrid(y_pos, x_pos, indexing='ij')
        positions_y = gy.reshape(-1)  # (H*W,)
        positions_x = gx.reshape(-1)  # (H*W,)

        offsets = []
        for dy in range(-self.half_kernel, self.half_kernel + 1):
            for dx in range(-self.half_kernel, self.half_kernel + 1):
                offsets.append((dy * self.dilation, dx * self.dilation))

        valid_masks = []
        for dy, dx in offsets:
            ny = positions_y + dy
            nx = positions_x + dx
            valid = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
            valid_masks.append(valid)

        # (H*W, K^2)
        return torch.stack(valid_masks, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: Input tensor (batch, height, width, d_model)
            attention_mask: Optional mask
            output_attentions: Whether to return attention weights

        Returns:
            output: (batch, height, width, d_model)
            attn_weights: If output_attentions
        """
        batch_size, height, width, _ = hidden_states.shape
        num_tokens = height * width

        # Flatten spatial dimensions
        x_flat = hidden_states.reshape(batch_size, num_tokens, self.d_model)

        # Project Q, K, V
        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        # Reshape: (B, N, H, D) -> (B, H, N, D)
        q = q.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Gather 2D neighborhoods
        k_neighbors = self._gather_2d_neighbors(k, height, width)
        v_neighbors = self._gather_2d_neighbors(v, height, width)

        # Compute attention scores
        attn_scores = torch.einsum(
            'bhnd,bhnkd->bhnk', q, k_neighbors
        ) * self.scale

        # Add 2D relative position bias
        attn_scores = attn_scores + self.rpb.unsqueeze(0).unsqueeze(2)

        # Mask invalid neighbors
        valid_mask = self._compute_2d_neighbor_mask(height, width, hidden_states.device)
        attn_scores = attn_scores.masked_fill(
            ~valid_mask.unsqueeze(0).unsqueeze(0), float('-inf')
        )

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute output
        output = torch.einsum('bhnk,bhnkd->bhnd', attn_weights, v_neighbors)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, num_tokens, -1)
        output = self.o_proj(output)

        # Restore spatial shape
        output = output.view(batch_size, height, width, self.d_model)

        if not output_attentions:
            attn_weights = None

        return output, attn_weights


class NeighborhoodAttention(NeighborhoodAttention1D):
    """Alias for NeighborhoodAttention1D (default 1D variant)."""
    pass


class NA1D(NeighborhoodAttention1D):
    """Short alias for NeighborhoodAttention1D."""
    pass


class NA2D(NeighborhoodAttention2D):
    """Short alias for NeighborhoodAttention2D."""
    pass
