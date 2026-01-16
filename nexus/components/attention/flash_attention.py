import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange
from nexus.core.base import NexusModule
from nexus.utils.attention_utils import create_causal_mask

class FlashAttention(NexusModule):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        block_size: int = 1024,
        causal: bool = False
    ):
        super().__init__()
        
        # Validate input parameters
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
            
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.block_size = block_size
        self.causal = causal
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (B, N, C)
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (B, 1, N, K)

        Returns:
            torch.Tensor: Output tensor of shape (B, N, C)
        """
        if hidden_states.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got {hidden_states.dim()}D")

        B, N, C = hidden_states.shape
        if C != self.hidden_size:
            raise ValueError(f"Input hidden size {C} doesn't match configured hidden_size {self.hidden_size}")

        # Project queries, keys, and values
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        # Scale queries
        q = q * self.scale

        # Initialize output tensor
        output = torch.zeros_like(v, device=hidden_states.device, dtype=hidden_states.dtype)

        # Process in blocks for memory efficiency
        for start in range(0, N, self.block_size):
            end = start + self.block_size
            q_block = q[:, :, start:end]  # (B, H, block_size, D)

            if self.causal:
                k_block = k[:, :, :end]  # Causal: attend to all previous tokens
                v_block = v[:, :, :end]
            else:
                k_block = k
                v_block = v

            # Compute attention scores
            attn_scores = torch.matmul(q_block, k_block.transpose(-2, -1))  # (B, H, block_size, K)

            if self.causal:
                causal_mask = create_causal_mask(attn_scores.size(-1), dtype=torch.bool, device=attn_scores.device)
                causal_mask = causal_mask[-attn_scores.size(-2):, :]
                attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

            if attention_mask is not None:
                # Ensure mask shape compatibility
                if attention_mask.dim() != 4:
                    raise ValueError(f"Attention mask should be 4D, got {attention_mask.dim()}D")
                if attention_mask.size(0) != B:
                    raise ValueError(f"Attention mask batch size {attention_mask.size(0)} doesn't match input batch size {B}")
                attn_scores += attention_mask[:, None, start:end, :k_block.size(-2)]

            # Apply softmax to get attention probabilities
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            # Compute attention output
            attn_output = torch.matmul(attn_probs, v_block)  # (B, H, block_size, D)
            output[:, :, start:end, :] = attn_output

        # Reshape and project output
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.out_proj(output)  # (B, N, C)

        return output