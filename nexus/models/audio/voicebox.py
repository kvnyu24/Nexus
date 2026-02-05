"""
Voicebox - Flow Matching for Text-Guided Speech Generation.

Voicebox is a non-autoregressive generative model for speech synthesis
that uses flow matching (continuous normalizing flows) to generate
high-quality, diverse speech from text. It supports multiple speech
generation tasks in a unified framework:

1. Text-to-Speech (TTS)
2. Voice Conversion
3. Speech Editing (infilling)
4. Noise Removal / Restoration
5. Style Transfer

Key innovations:
- **Flow Matching Training**: Trains continuous normalizing flows without
  simulation, much more efficient than diffusion models.
- **In-context Learning**: Conditions on audio prompts for style transfer
  and voice conversion (zero-shot TTS).
- **Masked Prediction**: Supports infilling for speech editing.
- **Audio Codec**: Uses EnCodec for discrete-continuous hybrid representation.

Voicebox achieves state-of-the-art quality while being 20x faster than
autoregressive models like VALL-E.

References:
    "Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale"
    Le et al., Meta AI, 2023 (https://arxiv.org/abs/2306.15687)

    Accepted at ICLR 2024
"""

from typing import Dict, Any, Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule


class ConvolutionalFlowMatching(nn.Module):
    """Convolutional U-Net backbone for flow matching in audio space.

    Uses a U-Net architecture with 1D convolutions for processing
    mel-spectrograms or codec features.

    Args:
        in_channels: Input channels (e.g., 80 for mel-spectrogram). Default: 80.
        hidden_channels: Base hidden channels. Default: 256.
        num_layers: Number of U-Net layers. Default: 4.
        kernel_size: Convolution kernel size. Default: 3.
        use_attention: Add attention layers. Default: True.
    """

    def __init__(
        self,
        in_channels: int = 80,
        hidden_channels: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Time embedding for flow timestep
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.SiLU(),
            nn.Linear(hidden_channels * 4, hidden_channels * 4),
        )

        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        in_ch = in_channels
        for i in range(num_layers):
            out_ch = hidden_channels * (2 ** i)
            self.encoder.append(
                ResidualBlock1D(in_ch, out_ch, hidden_channels * 4, kernel_size)
            )
            if use_attention and i >= num_layers // 2:
                self.encoder.append(AttentionBlock1D(out_ch))
            self.encoder.append(Downsample1D(out_ch))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock1D(in_ch, in_ch, hidden_channels * 4, kernel_size),
            AttentionBlock1D(in_ch) if use_attention else nn.Identity(),
            ResidualBlock1D(in_ch, in_ch, hidden_channels * 4, kernel_size),
        ])

        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            out_ch = hidden_channels * (2 ** i) if i > 0 else hidden_channels
            self.decoder.append(Upsample1D(in_ch))
            self.decoder.append(
                ResidualBlock1D(in_ch + out_ch, out_ch, hidden_channels * 4, kernel_size)
            )
            if use_attention and i >= num_layers // 2:
                self.decoder.append(AttentionBlock1D(out_ch))
            in_ch = out_ch

        # Output projection
        self.out_conv = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)

    def _get_time_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal time embedding."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input audio features, shape [B, C, T].
            t: Flow timestep, shape [B].
            condition: Optional conditioning (e.g., text, audio prompt), shape [B, C', T].

        Returns:
            Predicted velocity field, shape [B, C, T].
        """
        # Time embedding
        t_emb = self._get_time_embedding(t, self.hidden_channels)
        t_emb = self.time_embed(t_emb)

        # Concatenate condition if provided
        if condition is not None:
            x = torch.cat([x, condition], dim=1)

        # Encoder
        skip_connections = []
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, Downsample1D):
                x = layer(x)
            else:
                if isinstance(layer, ResidualBlock1D):
                    skip_connections.append(x)
                x = layer(x, t_emb) if isinstance(layer, ResidualBlock1D) else layer(x)

        # Bottleneck
        for layer in self.bottleneck:
            x = layer(x, t_emb) if isinstance(layer, ResidualBlock1D) else layer(x)

        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, Upsample1D):
                x = layer(x)
            elif isinstance(layer, ResidualBlock1D):
                skip = skip_connections.pop()
                # Match dimensions if needed
                if x.shape[-1] != skip.shape[-1]:
                    min_len = min(x.shape[-1], skip.shape[-1])
                    x = x[..., :min_len]
                    skip = skip[..., :min_len]
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t_emb)
            else:
                x = layer(x)

        # Output
        x = self.out_conv(x)
        return x


class ResidualBlock1D(nn.Module):
    """1D convolutional residual block with time conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.activation = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        # Add time embedding
        t_proj = self.time_mlp(t_emb)[:, :, None]
        x = x + t_proj

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x + residual)

        return x


class Downsample1D(nn.Module):
    """1D downsampling via strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """1D upsampling via transposed convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionBlock1D(nn.Module):
    """1D self-attention block."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        residual = x

        x = self.norm(x)
        x = x.permute(0, 2, 1)  # [B, T, C]
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x.permute(0, 2, 1)  # [B, C, T]

        return x + residual


class Voicebox(NexusModule):
    """Voicebox: Flow matching for text-guided speech generation.

    Supports multiple speech generation tasks via conditional flow matching.

    Args:
        mel_channels: Number of mel-spectrogram channels. Default: 80.
        hidden_channels: Hidden channels in U-Net. Default: 256.
        num_layers: Number of U-Net layers. Default: 4.
        text_dim: Text embedding dimension. Default: 512.
        use_audio_prompt: Enable in-context learning from audio prompts. Default: True.
        sigma_min: Minimum noise level for flow. Default: 1e-5.
        sigma_data: Data scaling parameter. Default: 1.0.
    """

    def __init__(
        self,
        mel_channels: int = 80,
        hidden_channels: int = 256,
        num_layers: int = 4,
        text_dim: int = 512,
        use_audio_prompt: bool = True,
        sigma_min: float = 1e-5,
        sigma_data: float = 1.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.mel_channels = mel_channels
        self.use_audio_prompt = use_audio_prompt
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data

        # Text encoder projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Audio prompt encoder (if using in-context learning)
        if use_audio_prompt:
            self.prompt_encoder = nn.Sequential(
                nn.Conv1d(mel_channels, hidden_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            )

        # Flow matching model
        condition_channels = hidden_channels if use_audio_prompt else 0
        self.flow_model = ConvolutionalFlowMatching(
            in_channels=mel_channels + condition_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )

    def _get_condition(
        self,
        text_embeds: torch.Tensor,
        audio_prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prepare conditioning signal.

        Args:
            text_embeds: Text embeddings, shape [B, seq_len, text_dim].
            audio_prompt: Optional audio prompt, shape [B, mel_channels, T_prompt].

        Returns:
            Conditioning signal, shape [B, C, T].
        """
        # Project text embeddings
        text_cond = self.text_proj(text_embeds)  # [B, seq_len, hidden_channels]
        text_cond = text_cond.permute(0, 2, 1)  # [B, hidden_channels, seq_len]

        # Encode audio prompt if provided
        if self.use_audio_prompt and audio_prompt is not None:
            audio_cond = self.prompt_encoder(audio_prompt)
            # Concatenate along channel dimension
            condition = torch.cat([text_cond, audio_cond], dim=1)
        else:
            condition = text_cond

        return condition

    def forward(
        self,
        mel_target: torch.Tensor,
        text_embeds: torch.Tensor,
        audio_prompt: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with flow matching loss.

        Args:
            mel_target: Target mel-spectrogram, shape [B, mel_channels, T].
            text_embeds: Text embeddings, shape [B, seq_len, text_dim].
            audio_prompt: Optional audio prompt for in-context learning.
            mask: Optional mask for inpainting (1=generate, 0=keep).

        Returns:
            Dictionary with loss and info.
        """
        B, C, T = mel_target.shape
        device = mel_target.device

        # Sample flow timestep uniformly
        t = torch.rand(B, device=device)

        # Sample noise
        noise = torch.randn_like(mel_target)

        # Linear interpolation between noise and data
        mel_t = (1 - t[:, None, None]) * noise + t[:, None, None] * mel_target

        # Apply mask if provided (for inpainting)
        if mask is not None:
            mel_t = mask * mel_t + (1 - mask) * mel_target

        # Get conditioning
        condition = self._get_condition(text_embeds, audio_prompt)

        # Upsample condition to match mel length if needed
        if condition.shape[-1] != T:
            condition = F.interpolate(condition, size=T, mode='linear', align_corners=False)

        # Predict velocity
        velocity_pred = self.flow_model(mel_t, t, condition)

        # Target velocity is mel_target - noise
        velocity_target = mel_target - noise

        # Flow matching loss
        loss = F.mse_loss(velocity_pred, velocity_target)

        return {
            "loss": loss,
            "loss_dict": {"loss": loss.item()},
        }

    @torch.no_grad()
    def generate(
        self,
        text_embeds: torch.Tensor,
        audio_prompt: Optional[torch.Tensor] = None,
        duration: int = 200,
        num_steps: int = 10,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate speech from text using flow matching.

        Args:
            text_embeds: Text embeddings, shape [B, seq_len, text_dim].
            audio_prompt: Optional audio prompt for voice cloning.
            duration: Duration in time steps. Default: 200.
            num_steps: Number of ODE solver steps. Default: 10.
            mask: Optional mask for inpainting.
            temperature: Sampling temperature. Default: 1.0.

        Returns:
            Generated mel-spectrogram, shape [B, mel_channels, duration].
        """
        device = next(self.parameters()).device
        B = text_embeds.shape[0]

        # Start from noise
        mel = torch.randn(
            B, self.mel_channels, duration, device=device
        ) * temperature

        # If mask provided, initialize with masked content
        if mask is not None:
            mel = mask * mel

        # Get conditioning
        condition = self._get_condition(text_embeds, audio_prompt)

        # Upsample condition to match duration
        if condition.shape[-1] != duration:
            condition = F.interpolate(condition, size=duration, mode='linear', align_corners=False)

        # Euler ODE solver
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)

            # Predict velocity
            velocity = self.flow_model(mel, t, condition)

            # Apply mask if provided
            if mask is not None:
                velocity = mask * velocity

            # Euler step
            mel = mel + velocity * dt

        return mel

    @torch.no_grad()
    def infill(
        self,
        mel_context: torch.Tensor,
        mask: torch.Tensor,
        text_embeds: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Speech editing via infilling.

        Args:
            mel_context: Partial mel-spectrogram with context, shape [B, C, T].
            mask: Binary mask (1=generate, 0=keep), shape [B, 1, T].
            text_embeds: Text embeddings for masked region.
            num_steps: Number of ODE steps. Default: 10.

        Returns:
            Completed mel-spectrogram, shape [B, C, T].
        """
        return self.generate(
            text_embeds=text_embeds,
            duration=mel_context.shape[-1],
            num_steps=num_steps,
            mask=mask,
        )
