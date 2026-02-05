"""
VideoPoet - Autoregressive Multimodal Video Generation.

VideoPoet is a large language model (LLM) approach to video generation
that treats video, audio, and text as discrete tokens. It supports multiple
video-centric tasks in a unified autoregressive framework:

1. Text-to-video generation
2. Image-to-video (animation)
3. Video stylization and editing
4. Video inpainting and outpainting
5. Video-to-audio generation

Key innovations:
- **Unified tokenization**: Videos tokenized with MAGVIT-v2 (C-ViViT)
- **Multimodal LLM**: Transformer decoder processes mixed token sequences
- **Multitask training**: Single model for all video tasks via task prompts
- **Temporal coherence**: Causal attention maintains frame consistency

The model uses a two-stage approach:
1. Tokenization: Compress video/audio/image into discrete tokens
2. Generation: Autoregressive transformer generates token sequences

References:
    "VideoPoet: A Large Language Model for Zero-Shot Video Generation"
    Kondratyuk et al., Google Research, 2023
    (https://arxiv.org/abs/2312.14125)

    Accepted at ICML 2024
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule


class VideoTokenizer(nn.Module):
    """Video tokenizer based on MAGVIT-v2 / C-ViViT.

    Compresses video frames into discrete tokens using a VQ-VAE-style
    approach with causal 3D convolutions.

    Args:
        in_channels: Number of input channels (3 for RGB). Default: 3.
        latent_dim: Dimension of latent features. Default: 256.
        num_embeddings: Size of discrete codebook. Default: 8192.
        num_res_blocks: Number of residual blocks. Default: 2.
        channel_mult: Channel multipliers per resolution. Default: (1, 2, 4, 4).
        temporal_downsample: Whether to downsample temporally. Default: True.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        num_embeddings: int = 8192,
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        temporal_downsample: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.temporal_downsample = temporal_downsample

        # Encoder
        channels = [latent_dim * m for m in channel_mult]
        self.encoder = self._build_encoder(in_channels, channels, num_res_blocks)

        # Codebook
        self.codebook = nn.Embedding(num_embeddings, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

        # Decoder
        self.decoder = self._build_decoder(channels, in_channels, num_res_blocks)

    def _build_encoder(
        self, in_channels: int, channels: List[int], num_res_blocks: int
    ) -> nn.ModuleList:
        """Build encoder network."""
        layers = nn.ModuleList()

        # Initial conv
        layers.append(nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1))

        # Downsampling blocks
        in_ch = channels[0]
        for out_ch in channels[1:]:
            # Residual blocks
            for _ in range(num_res_blocks):
                layers.append(ResBlock3D(in_ch, in_ch))

            # Downsample
            stride = (2, 2, 2) if self.temporal_downsample else (1, 2, 2)
            layers.append(nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1))
            in_ch = out_ch

        # Final residual blocks
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(in_ch, in_ch))

        # Project to latent dim
        layers.append(nn.Conv3d(in_ch, self.latent_dim, kernel_size=1))

        return layers

    def _build_decoder(
        self, channels: List[int], out_channels: int, num_res_blocks: int
    ) -> nn.ModuleList:
        """Build decoder network."""
        layers = nn.ModuleList()

        # Project from latent
        channels_reversed = channels[::-1]
        layers.append(nn.Conv3d(self.latent_dim, channels_reversed[0], kernel_size=1))

        # Initial residual blocks
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(channels_reversed[0], channels_reversed[0]))

        # Upsampling blocks
        in_ch = channels_reversed[0]
        for out_ch in channels_reversed[1:]:
            # Upsample
            stride = (2, 2, 2) if self.temporal_downsample else (1, 2, 2)
            layers.append(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, output_padding=(stride[0]-1, stride[1]-1, stride[2]-1)))

            # Residual blocks
            for _ in range(num_res_blocks):
                layers.append(ResBlock3D(out_ch, out_ch))
            in_ch = out_ch

        # Final conv to output
        layers.append(nn.Conv3d(in_ch, out_channels, kernel_size=3, padding=1))

        return layers

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video to latent features.

        Args:
            x: Video tensor, shape [B, C, T, H, W].

        Returns:
            Latent features, shape [B, latent_dim, T', H', W'].
        """
        for layer in self.encoder:
            x = layer(x)
        return x

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize latents to discrete tokens.

        Args:
            z: Continuous latents, shape [B, latent_dim, T, H, W].

        Returns:
            Tuple of (quantized latents, token indices).
        """
        B, C, T, H, W = z.shape

        # Flatten spatial/temporal dimensions
        z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, C)  # [B*T*H*W, C]

        # Compute distances to codebook
        distances = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.codebook.weight ** 2, dim=1) - \
                    2 * torch.matmul(z_flat, self.codebook.weight.t())

        # Get nearest codebook entry
        indices = torch.argmin(distances, dim=1)  # [B*T*H*W]
        z_q = self.codebook(indices)  # [B*T*H*W, C]

        # Reshape back
        z_q = z_q.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        indices = indices.view(B, T, H, W)

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents to video.

        Args:
            z_q: Quantized latents, shape [B, latent_dim, T, H, W].

        Returns:
            Reconstructed video, shape [B, C, T, H, W].
        """
        x = z_q
        for layer in self.decoder:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encoding-quantization-decoding pass.

        Args:
            x: Input video, shape [B, C, T, H, W].

        Returns:
            Tuple of (reconstructed video, quantization loss, token indices).
        """
        z = self.encode(x)
        z_q, indices = self.quantize(z)
        x_recon = self.decode(z_q)

        # VQ loss
        commitment_loss = F.mse_loss(z, z_q.detach())
        codebook_loss = F.mse_loss(z.detach(), z_q)
        vq_loss = codebook_loss + 0.25 * commitment_loss

        return x_recon, vq_loss, indices


class ResBlock3D(nn.Module):
    """3D Residual block for video processing."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.activation = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x + residual)
        return x


class VideoPoetTransformer(nn.Module):
    """Causal transformer decoder for autoregressive token generation.

    Args:
        vocab_size: Size of token vocabulary (video + audio + text). Default: 32000.
        hidden_dim: Hidden dimension. Default: 2048.
        num_layers: Number of transformer layers. Default: 24.
        num_heads: Number of attention heads. Default: 16.
        mlp_ratio: MLP expansion ratio. Default: 4.0.
        dropout: Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 2048,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)

        # Positional embedding
        self.pos_embed = nn.Embedding(8192, hidden_dim)  # Max sequence length

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embed.weight

    def forward(
        self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: Token indices, shape [B, seq_len].
            attention_mask: Optional mask, shape [B, seq_len].

        Returns:
            Logits over vocabulary, shape [B, seq_len, vocab_size].
        """
        B, seq_len = token_ids.shape
        device = token_ids.device

        # Embeddings
        x = self.token_embed(token_ids)
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

        # Transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask, attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with causal attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False
        )
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class VideoPoet(NexusModule):
    """VideoPoet: Autoregressive multimodal video generation.

    Unified model for text-to-video, image-to-video, video editing,
    and video-to-audio generation via discrete token sequences.

    Args:
        tokenizer: Video tokenizer (MAGVIT-v2 style). If None, creates default.
        vocab_size: Total vocabulary size (video + audio + text). Default: 32000.
        video_vocab_size: Size of video token vocabulary. Default: 8192.
        audio_vocab_size: Size of audio token vocabulary. Default: 8192.
        hidden_dim: Transformer hidden dimension. Default: 2048.
        num_layers: Number of transformer layers. Default: 24.
        num_heads: Number of attention heads. Default: 16.
        mlp_ratio: MLP expansion ratio. Default: 4.0.
        dropout: Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        tokenizer: Optional[VideoTokenizer] = None,
        vocab_size: int = 32000,
        video_vocab_size: int = 8192,
        audio_vocab_size: int = 8192,
        hidden_dim: int = 2048,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.video_vocab_size = video_vocab_size
        self.audio_vocab_size = audio_vocab_size

        # Tokenizer
        self.tokenizer = tokenizer or VideoTokenizer(num_embeddings=video_vocab_size)

        # Transformer
        self.transformer = VideoPoetTransformer(
            vocab_size, hidden_dim, num_layers, num_heads, mlp_ratio, dropout
        )

        # Task embeddings for different generation tasks
        self.task_tokens = nn.Parameter(torch.randn(5, hidden_dim))  # 5 tasks

    def forward(
        self,
        video: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        task_id: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            video: Input video, shape [B, C, T, H, W].
            text_tokens: Optional text tokens, shape [B, text_len].
            task_id: Task identifier (0=text-to-video, 1=image-to-video, etc.).

        Returns:
            Dictionary with losses and predictions.
        """
        # Tokenize video
        video_recon, vq_loss, video_tokens = self.tokenizer(video)

        B, T, H, W = video_tokens.shape

        # Flatten video tokens
        video_tokens_flat = video_tokens.reshape(B, T * H * W)

        # Concatenate task token, text tokens, and video tokens
        if text_tokens is not None:
            token_seq = torch.cat([text_tokens, video_tokens_flat], dim=1)
        else:
            token_seq = video_tokens_flat

        # Forward through transformer (teacher forcing)
        logits = self.transformer(token_seq[:, :-1])

        # Compute language modeling loss
        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            token_seq[:, 1:].reshape(-1),
            ignore_index=-100
        )

        total_loss = lm_loss + 0.1 * vq_loss

        return {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "vq_loss": vq_loss,
            "video_recon": video_recon,
        }

    @torch.no_grad()
    def generate(
        self,
        text_tokens: torch.Tensor,
        num_frames: int = 16,
        video_size: Tuple[int, int] = (16, 16),
        task_id: int = 0,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """Generate video from text.

        Args:
            text_tokens: Text token sequence, shape [B, text_len].
            num_frames: Number of frames to generate. Default: 16.
            video_size: Spatial size (H, W) in token space. Default: (16, 16).
            task_id: Task ID. Default: 0 (text-to-video).
            temperature: Sampling temperature. Default: 1.0.
            top_p: Nucleus sampling parameter. Default: 0.95.

        Returns:
            Generated video, shape [B, 3, T, H*downsample, W*downsample].
        """
        B = text_tokens.shape[0]
        device = text_tokens.device
        H, W = video_size

        # Start with text tokens
        token_seq = text_tokens

        # Generate video tokens autoregressively
        num_video_tokens = num_frames * H * W

        for _ in range(num_video_tokens):
            # Get logits
            logits = self.transformer(token_seq)[:, -1, :]  # [B, vocab_size]

            # Apply temperature
            logits = logits / temperature

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            token_seq = torch.cat([token_seq, next_token], dim=1)

        # Extract video tokens
        video_tokens = token_seq[:, -num_video_tokens:].reshape(B, num_frames, H, W)

        # Decode video tokens
        video_tokens_embeddings = self.tokenizer.codebook(video_tokens)
        video_tokens_embeddings = video_tokens_embeddings.permute(0, 4, 1, 2, 3)
        video = self.tokenizer.decode(video_tokens_embeddings)

        return video
