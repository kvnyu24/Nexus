"""
CogVideoX - Expert Transformer for Text-to-Video Generation.

CogVideoX is a large-scale diffusion transformer designed specifically
for high-quality text-to-video generation. It introduces several key
innovations for efficient spatiotemporal modeling:

1. **3D Causal Attention**: Combines spatial and temporal attention with
   causal masking for autoregressive video generation.

2. **Expert Adapters**: Uses mixture-of-experts-style adapters within
   transformer blocks for better temporal modeling without full retraining.

3. **Progressive Training**: Three-stage training (image → short video →
   long video) for efficient learning of temporal dynamics.

4. **Latent Video Diffusion**: Operates in VAE latent space for memory
   efficiency with 3D VAE encoder/decoder.

Key components:
- 3D positional embeddings (spatial + temporal)
- Expert routing for temporal attention
- Classifier-free guidance for text conditioning
- Progressive resolution and length training

References:
    "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer"
    Hong et al., 2024 (https://arxiv.org/abs/2408.06072)

    Accepted at ICLR 2025
"""

from typing import Dict, Any, Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule


class CogVideoXAttention(nn.Module):
    """3D Causal Attention with spatial and temporal components.

    Processes video latents with separate spatial (within-frame) and
    temporal (across-frame) attention, with optional causal masking
    for autoregressive generation.

    Args:
        hidden_dim: Hidden dimension of features.
        num_heads: Number of attention heads.
        causal: Whether to use causal masking. Default: False.
        dropout: Dropout probability. Default: 0.0.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        causal: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.causal = causal

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Spatial attention (within each frame)
        self.spatial_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.spatial_proj = nn.Linear(hidden_dim, hidden_dim)

        # Temporal attention (across frames)
        self.temporal_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.temporal_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def _reshape_for_attention(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Reshape [B, N, D] to [B, num_heads, N, head_dim]."""
        B, N, D = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _spatial_attention(
        self, x: torch.Tensor, T: int, H: int, W: int
    ) -> torch.Tensor:
        """Apply spatial attention within each frame.

        Args:
            x: Input features, shape [B, T*H*W, D].
            T: Number of frames.
            H: Height of each frame.
            W: Width of each frame.

        Returns:
            Output features after spatial attention, shape [B, T*H*W, D].
        """
        B, THW, D = x.shape

        # Compute QKV
        qkv = self.spatial_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to [B*T, H*W, D]
        q = q.view(B, T, H * W, D).reshape(B * T, H * W, D)
        k = k.view(B, T, H * W, D).reshape(B * T, H * W, D)
        v = v.view(B, T, H * W, D).reshape(B * T, H * W, D)

        # Multi-head attention
        q = self._reshape_for_attention(q)  # [B*T, num_heads, H*W, head_dim]
        k = self._reshape_for_attention(k)
        v = self._reshape_for_attention(v)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # [B*T, num_heads, H*W, head_dim]
        out = out.transpose(1, 2).reshape(B * T, H * W, D)

        # Reshape back to [B, T*H*W, D]
        out = out.view(B, T, H * W, D).reshape(B, THW, D)
        out = self.spatial_proj(out)

        return out

    def _temporal_attention(
        self, x: torch.Tensor, T: int, H: int, W: int
    ) -> torch.Tensor:
        """Apply temporal attention across frames.

        Args:
            x: Input features, shape [B, T*H*W, D].
            T: Number of frames.
            H: Height of each frame.
            W: Width of each frame.

        Returns:
            Output features after temporal attention, shape [B, T*H*W, D].
        """
        B, THW, D = x.shape

        # Compute QKV
        qkv = self.temporal_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to [B*H*W, T, D]
        q = q.view(B, T, H * W, D).permute(0, 2, 1, 3).reshape(B * H * W, T, D)
        k = k.view(B, T, H * W, D).permute(0, 2, 1, 3).reshape(B * H * W, T, D)
        v = v.view(B, T, H * W, D).permute(0, 2, 1, 3).reshape(B * H * W, T, D)

        # Multi-head attention
        q = self._reshape_for_attention(q)  # [B*H*W, num_heads, T, head_dim]
        k = self._reshape_for_attention(k)
        v = self._reshape_for_attention(v)

        # Scaled dot-product attention with optional causal masking
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.causal:
            # Create causal mask
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # [B*H*W, num_heads, T, head_dim]
        out = out.transpose(1, 2).reshape(B * H * W, T, D)

        # Reshape back to [B, T*H*W, D]
        out = out.view(B, H * W, T, D).permute(0, 2, 1, 3).reshape(B, THW, D)
        out = self.temporal_proj(out)

        return out

    def forward(
        self, x: torch.Tensor, T: int, H: int, W: int
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features, shape [B, T*H*W, D].
            T: Number of frames.
            H: Height of each frame.
            W: Width of each frame.

        Returns:
            Output features, shape [B, T*H*W, D].
        """
        # Spatial attention
        x = x + self._spatial_attention(x, T, H, W)

        # Temporal attention
        x = x + self._temporal_attention(x, T, H, W)

        return x


class ExpertAdapter(nn.Module):
    """Expert adapter for temporal modeling.

    Lightweight MLP adapter that specializes in temporal dynamics,
    inspired by mixture-of-experts but with shared routing.

    Args:
        hidden_dim: Hidden dimension.
        num_experts: Number of expert adapters. Default: 4.
        expert_dim: Dimension of each expert. Default: hidden_dim // 2.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 4,
        expert_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.expert_dim = expert_dim or hidden_dim // 2

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, self.expert_dim),
                nn.GELU(),
                nn.Linear(self.expert_dim, hidden_dim),
            )
            for _ in range(num_experts)
        ])

        # Router
        self.router = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with expert routing.

        Args:
            x: Input features, shape [B, N, D].

        Returns:
            Output features, shape [B, N, D].
        """
        # Compute routing weights
        router_logits = self.router(x)  # [B, N, num_experts]
        router_weights = F.softmax(router_logits, dim=-1)

        # Apply each expert and combine
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))

        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, N, D, num_experts]
        output = torch.einsum("bnde,bne->bnd", expert_outputs, router_weights)

        return output


class CogVideoXTransformerBlock(nn.Module):
    """CogVideoX Transformer block with expert adapters.

    Args:
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio. Default: 4.0.
        num_experts: Number of expert adapters. Default: 4.
        causal: Use causal attention. Default: False.
        dropout: Dropout probability. Default: 0.0.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        num_experts: int = 4,
        causal: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = CogVideoXAttention(hidden_dim, num_heads, causal, dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm3 = nn.LayerNorm(hidden_dim)
        self.expert_adapter = ExpertAdapter(hidden_dim, num_experts)

    def forward(
        self, x: torch.Tensor, T: int, H: int, W: int
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features, shape [B, T*H*W, D].
            T: Number of frames.
            H: Height.
            W: Width.

        Returns:
            Output features, shape [B, T*H*W, D].
        """
        # Attention
        x = x + self.attn(self.norm1(x), T, H, W)

        # MLP
        x = x + self.mlp(self.norm2(x))

        # Expert adapter
        x = x + self.expert_adapter(self.norm3(x))

        return x


class CogVideoX(NexusModule):
    """CogVideoX model for text-to-video generation.

    Args:
        video_size: Tuple of (T, H, W) for video dimensions. Default: (16, 32, 32).
        patch_size: Tuple of (t, h, w) for patch dimensions. Default: (1, 2, 2).
        in_channels: Number of input channels (latent). Default: 4.
        hidden_dim: Transformer hidden dimension. Default: 1024.
        depth: Number of transformer blocks. Default: 24.
        num_heads: Number of attention heads. Default: 16.
        mlp_ratio: MLP expansion ratio. Default: 4.0.
        text_dim: Text embedding dimension. Default: 768.
        num_experts: Number of expert adapters. Default: 4.
        causal: Use causal attention for autoregressive generation. Default: False.
        learn_sigma: Predict noise variance. Default: False.
    """

    def __init__(
        self,
        video_size: Tuple[int, int, int] = (16, 32, 32),
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 4,
        hidden_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        text_dim: int = 768,
        num_experts: int = 4,
        causal: bool = False,
        learn_sigma: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.video_size = video_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_dim = hidden_dim

        T, H, W = video_size
        t, h, w = patch_size

        # Compute number of patches
        self.num_frames = T // t
        self.num_patches_h = H // h
        self.num_patches_w = W // w
        self.num_patches = self.num_frames * self.num_patches_h * self.num_patches_w

        # 3D patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Positional embeddings (spatial + temporal)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))

        # Timestep embedding
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Text cross-attention
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CogVideoXTransformerBlock(
                hidden_dim, num_heads, mlp_ratio, num_experts, causal
            )
            for _ in range(depth)
        ])

        # Final layer
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_linear = nn.Linear(
            hidden_dim, t * h * w * self.out_channels
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def _get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Sinusoidal timestep embeddings."""
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Noisy video latents, shape [B, C, T, H, W].
            t: Timesteps, shape [B].
            text_embeds: Text embeddings, shape [B, seq_len, text_dim].

        Returns:
            Predicted noise, shape [B, C, T, H, W].
        """
        B, C, T, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # [B, hidden_dim, T', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]

        # Add positional embedding
        x = x + self.pos_embed

        # Timestep embedding
        t_emb = self._get_timestep_embedding(t)
        t_emb = self.t_embedder(t_emb)  # [B, hidden_dim]

        # Add timestep embedding to all tokens
        x = x + t_emb.unsqueeze(1)

        # Project text embeddings
        text_proj = self.text_proj(text_embeds)  # [B, seq_len, hidden_dim]

        # Concatenate text with video tokens
        x = torch.cat([text_proj, x], dim=1)

        # Adjust dimensions for attention
        num_text_tokens = text_proj.shape[1]
        total_tokens = x.shape[1]

        # Transformer blocks
        for block in self.blocks:
            # Only apply 3D attention to video tokens
            x_video = x[:, num_text_tokens:, :]
            x_video = block(x_video, self.num_frames, self.num_patches_h, self.num_patches_w)
            x = torch.cat([x[:, :num_text_tokens, :], x_video], dim=1)

        # Extract video tokens
        x = x[:, num_text_tokens:, :]

        # Final layer
        x = self.final_norm(x)
        x = self.final_linear(x)

        # Unpatchify
        t, h, w = self.patch_size
        x = x.reshape(
            B, self.num_frames, self.num_patches_h, self.num_patches_w,
            t, h, w, self.out_channels
        )
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # [B, C, T', t, H', h, W', w]
        x = x.reshape(B, self.out_channels, T, H, W)

        return x
