"""
Zamba - Mamba Backbone with Shared Attention Block.

Zamba is a hybrid architecture designed by Zyphra that uses:
1. Mamba blocks as the primary sequence modeling mechanism (efficient backbone)
2. Shared attention blocks interspersed for precise token interactions
3. Weight sharing across attention layers to reduce parameters

Key innovations:
- Primarily Mamba-based for efficiency (O(1) inference complexity)
- Strategic attention insertion for quality where precision matters
- Shared attention parameters amortize cost across multiple layers
- Achieves strong quality-efficiency tradeoff

The architecture is particularly effective for language modeling tasks
requiring both long-range dependencies and precise short-range reasoning.

Reference: Zyphra AI, "Zamba: A Compact 7B SSM Hybrid Model Outperforming
    Larger Models", 2024. https://www.zyphra.com/zamba
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from nexus.core.base import NexusModule


class MambaBlock(NexusModule):
    """Simplified Mamba block for Zamba.

    Uses selective state-space model with input-dependent dynamics.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        d_conv: Convolution width.
        expand_factor: Expansion factor for hidden dimension.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand_factor

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Initialize A (state matrix) - complex diagonal
        A = torch.arange(1, d_state + 1).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A.float()))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            output: Output of shape (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        # Input projection and split
        xz = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)

        # Convolution
        x_conv = x_in.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)

        # Activation
        x_conv = F.silu(x_conv)

        # SSM parameters (input-dependent)
        x_ssm = self.x_proj(x_conv)  # (batch, seq_len, d_state * 2)
        B, C = x_ssm.chunk(2, dim=-1)  # Each: (batch, seq_len, d_state)

        # Delta (input-dependent step size)
        delta = F.softplus(self.dt_proj(x_conv))  # (batch, seq_len, d_inner)

        # SSM computation (simplified selective scan)
        # In practice, this would use efficient CUDA kernels
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Discretize
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        # Scan (simplified - actual implementation uses parallel scan)
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB[:, t] * x_conv[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1) + self.D * x_conv[:, t]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)

        # Gating
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        return output


class SharedAttentionBlock(NexusModule):
    """Shared multi-head attention block used across multiple layers.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model

        # QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            mask: Optional causal mask.

        Returns:
            output: Output of shape (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Transpose
        q = q.transpose(1, 2) * self.scale
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        if mask is not None:
            attn_scores = attn_scores + mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        # Reshape
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.out_proj(output)

        return output


class ZambaBlock(NexusModule):
    """Zamba block - either Mamba or shared attention.

    Args:
        d_model: Model dimension.
        block_type: 'mamba' or 'attention'.
        shared_attention: Shared attention module (if block_type='attention').
        d_ff: Feedforward dimension.
        dropout: Dropout probability.
    """
    def __init__(
        self,
        d_model: int,
        block_type: str = 'mamba',
        shared_attention: Optional[SharedAttentionBlock] = None,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        **mamba_kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.block_type = block_type

        if d_ff is None:
            d_ff = 4 * d_model

        # Normalization
        self.norm1 = nn.LayerNorm(d_model)

        # Main block
        if block_type == 'mamba':
            self.main_block = MambaBlock(d_model, **mamba_kwargs)
        elif block_type == 'attention':
            assert shared_attention is not None, "Shared attention must be provided"
            self.main_block = shared_attention
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        # Feedforward
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            mask: Optional attention mask (only used for attention blocks).

        Returns:
            x: Output of shape (batch, seq_len, d_model).
        """
        # Main block
        if self.block_type == 'attention':
            x = x + self.dropout(self.main_block(self.norm1(x), mask))
        else:
            x = x + self.dropout(self.main_block(self.norm1(x)))

        # Feedforward
        x = x + self.ffn(self.norm2(x))

        return x


class ZambaModel(NexusModule):
    """Complete Zamba Model with Mamba Backbone and Shared Attention.

    Args:
        d_model: Model dimension.
        n_layers: Total number of layers.
        num_heads: Number of attention heads.
        d_ff: Feedforward dimension.
        dropout: Dropout probability.
        attention_every_n: Insert attention every N layers (e.g., 6).
        d_state: Mamba state dimension.
        d_conv: Mamba convolution width.
    """
    def __init__(
        self,
        d_model: int,
        n_layers: int = 24,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        attention_every_n: int = 6,
        d_state: int = 16,
        d_conv: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        if d_ff is None:
            d_ff = 4 * d_model

        # Shared attention block (reused across layers)
        self.shared_attention = SharedAttentionBlock(d_model, num_heads, dropout)

        # Create layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # Insert attention every N-th layer
            block_type = 'attention' if (i + 1) % attention_every_n == 0 else 'mamba'

            self.layers.append(
                ZambaBlock(
                    d_model=d_model,
                    block_type=block_type,
                    shared_attention=self.shared_attention if block_type == 'attention' else None,
                    d_ff=d_ff,
                    dropout=dropout,
                    d_state=d_state,
                    d_conv=d_conv
                )
            )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through all layers.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            mask: Optional attention mask.

        Returns:
            x: Output of shape (batch, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        return x
