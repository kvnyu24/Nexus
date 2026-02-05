"""
NaturalSpeech 3 - Factorized Diffusion Model for Zero-Shot TTS.

NaturalSpeech 3 achieves state-of-the-art zero-shot text-to-speech through
a novel factorized speech representation and a latent diffusion architecture.

Key innovations:
1. **Factorized Vector Quantization (FVQ)**: Decomposes speech into
   content, prosody, and timbre subspaces using factorized codebooks.

2. **Factorized Diffusion**: Separate diffusion models for each attribute,
   enabling fine-grained control and disentanglement.

3. **Two-Stage Architecture**:
   - Stage 1: Text → Content tokens (phoneme-to-token mapping)
   - Stage 2: Content + Prosody + Timbre → Speech (diffusion in latent space)

4. **Prosody Predictor**: Duration and pitch prediction from text.

5. **Zero-Shot Voice Cloning**: Extracts timbre from reference audio
   for speaker adaptation without fine-tuning.

Achieves human-level naturalness on LibriSpeech and supports:
- Zero-shot TTS
- Voice conversion
- Speech editing
- Emotion and style control

References:
    "NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized
     Codec and Diffusion Models"
    Ju et al., Microsoft Research, 2024 (https://arxiv.org/abs/2403.03100)

    Accepted at ICML 2024
"""

from typing import Dict, Any, Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule


class FactorizedVectorQuantizer(nn.Module):
    """Factorized Vector Quantizer for speech decomposition.

    Decomposes speech into three orthogonal subspaces:
    - Content (what is said): phonetic information
    - Prosody (how it's said): rhythm, stress, intonation
    - Timbre (who says it): speaker identity

    Args:
        in_channels: Input feature dimension. Default: 512.
        content_vocab: Content codebook size. Default: 1024.
        prosody_vocab: Prosody codebook size. Default: 1024.
        timbre_vocab: Timbre codebook size. Default: 512.
        latent_dim: Dimension of each latent subspace. Default: 256.
    """

    def __init__(
        self,
        in_channels: int = 512,
        content_vocab: int = 1024,
        prosody_vocab: int = 1024,
        timbre_vocab: int = 512,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Projection to factorized subspaces
        self.content_proj = nn.Linear(in_channels, latent_dim)
        self.prosody_proj = nn.Linear(in_channels, latent_dim)
        self.timbre_proj = nn.Linear(in_channels, latent_dim)

        # Separate codebooks for each factor
        self.content_codebook = nn.Embedding(content_vocab, latent_dim)
        self.prosody_codebook = nn.Embedding(prosody_vocab, latent_dim)
        self.timbre_codebook = nn.Embedding(timbre_vocab, latent_dim)

        # Initialize codebooks
        nn.init.uniform_(self.content_codebook.weight, -1/content_vocab, 1/content_vocab)
        nn.init.uniform_(self.prosody_codebook.weight, -1/prosody_vocab, 1/prosody_vocab)
        nn.init.uniform_(self.timbre_codebook.weight, -1/timbre_vocab, 1/timbre_vocab)

        # Reconstruction projection
        self.recon_proj = nn.Linear(3 * latent_dim, in_channels)

    def quantize_factor(
        self, z: torch.Tensor, codebook: nn.Embedding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a single factor.

        Args:
            z: Continuous latent, shape [..., latent_dim].
            codebook: Codebook embeddings.

        Returns:
            Tuple of (quantized latent, indices).
        """
        # Flatten spatial dimensions
        z_flat = z.reshape(-1, self.latent_dim)

        # Compute distances to codebook
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True) +
            torch.sum(codebook.weight ** 2, dim=1) -
            2 * torch.matmul(z_flat, codebook.weight.t())
        )

        # Get nearest codebook entry
        indices = torch.argmin(distances, dim=1)
        z_q = codebook(indices)

        # Reshape back
        z_q = z_q.view(z.shape)
        indices = indices.view(z.shape[:-1])

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, indices

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with factorized quantization.

        Args:
            x: Input features, shape [B, T, in_channels].

        Returns:
            Tuple of (reconstructed features, dict of quantization info).
        """
        # Project to subspaces
        z_content = self.content_proj(x)
        z_prosody = self.prosody_proj(x)
        z_timbre = self.timbre_proj(x)

        # Quantize each factor
        z_content_q, content_indices = self.quantize_factor(z_content, self.content_codebook)
        z_prosody_q, prosody_indices = self.quantize_factor(z_prosody, self.prosody_codebook)
        z_timbre_q, timbre_indices = self.quantize_factor(z_timbre, self.timbre_codebook)

        # Concatenate quantized factors
        z_q = torch.cat([z_content_q, z_prosody_q, z_timbre_q], dim=-1)

        # Reconstruct
        x_recon = self.recon_proj(z_q)

        # Compute VQ losses
        commitment_loss = (
            F.mse_loss(z_content, z_content_q.detach()) +
            F.mse_loss(z_prosody, z_prosody_q.detach()) +
            F.mse_loss(z_timbre, z_timbre_q.detach())
        ) / 3

        codebook_loss = (
            F.mse_loss(z_content.detach(), z_content_q) +
            F.mse_loss(z_prosody.detach(), z_prosody_q) +
            F.mse_loss(z_timbre.detach(), z_timbre_q)
        ) / 3

        vq_loss = codebook_loss + 0.25 * commitment_loss

        info = {
            "vq_loss": vq_loss,
            "content_indices": content_indices,
            "prosody_indices": prosody_indices,
            "timbre_indices": timbre_indices,
            "z_content": z_content_q,
            "z_prosody": z_prosody_q,
            "z_timbre": z_timbre_q,
        }

        return x_recon, info


class ProsodyPredictor(nn.Module):
    """Predicts prosody (duration, pitch) from text.

    Args:
        text_dim: Text embedding dimension. Default: 512.
        hidden_dim: Hidden dimension. Default: 256.
        num_layers: Number of LSTM layers. Default: 2.
    """

    def __init__(
        self,
        text_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()

        # Duration predictor
        self.duration_predictor = nn.Sequential(
            nn.Conv1d(text_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        # Pitch predictor
        self.pitch_predictor = nn.LSTM(
            text_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )
        self.pitch_proj = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self, text_embeds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict duration and pitch.

        Args:
            text_embeds: Text embeddings, shape [B, T, text_dim].

        Returns:
            Tuple of (duration [B, T], pitch [B, T]).
        """
        # Duration prediction
        x = text_embeds.transpose(1, 2)  # [B, text_dim, T]
        duration = self.duration_predictor(x).squeeze(1)  # [B, T]
        duration = F.softplus(duration)  # Ensure positive

        # Pitch prediction
        pitch_hidden, _ = self.pitch_predictor(text_embeds)
        pitch = self.pitch_proj(pitch_hidden).squeeze(-1)  # [B, T]

        return duration, pitch


class FactorizedDiffusion(nn.Module):
    """Diffusion model for factorized latent space.

    Separate diffusion models for content, prosody, and timbre allow
    independent control and better disentanglement.

    Args:
        latent_dim: Dimension of each latent factor. Default: 256.
        hidden_dim: Hidden dimension. Default: 512.
        num_layers: Number of transformer layers. Default: 8.
        num_heads: Number of attention heads. Default: 8.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Input projection (3 factors concatenated)
        self.input_proj = nn.Linear(3 * latent_dim, hidden_dim)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                hidden_dim, num_heads, dim_feedforward=hidden_dim * 4,
                dropout=0.1, activation='gelu', batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Output projection (back to 3 factors)
        self.output_proj = nn.Linear(hidden_dim, 3 * latent_dim)

    def _get_timestep_embedding(
        self, timesteps: torch.Tensor, dim: int
    ) -> torch.Tensor:
        """Sinusoidal timestep embeddings."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(
        self,
        z_content: torch.Tensor,
        z_prosody: torch.Tensor,
        z_timbre: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict noise for each factor.

        Args:
            z_content: Noisy content latent, shape [B, T, latent_dim].
            z_prosody: Noisy prosody latent, shape [B, T, latent_dim].
            z_timbre: Noisy timbre latent, shape [B, T, latent_dim].
            t: Timesteps, shape [B].

        Returns:
            Tuple of predicted noise for (content, prosody, timbre).
        """
        B, T, D = z_content.shape

        # Concatenate factors
        z = torch.cat([z_content, z_prosody, z_timbre], dim=-1)

        # Project to hidden dimension
        x = self.input_proj(z)

        # Add time embedding
        t_emb = self._get_timestep_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb).unsqueeze(1)  # [B, 1, hidden_dim]
        x = x + t_emb

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Project to output
        x = self.output_proj(x)

        # Split back to factors
        noise_content, noise_prosody, noise_timbre = x.chunk(3, dim=-1)

        return noise_content, noise_prosody, noise_timbre


class NaturalSpeech3(NexusModule):
    """NaturalSpeech 3: Zero-shot TTS with factorized codec and diffusion.

    Two-stage architecture:
    1. Text → Content tokens (phoneme-to-token mapping)
    2. Content + Prosody + Timbre → Speech (factorized diffusion)

    Args:
        text_dim: Text embedding dimension. Default: 512.
        feature_dim: Audio feature dimension. Default: 512.
        latent_dim: Factorized latent dimension. Default: 256.
        content_vocab: Content codebook size. Default: 1024.
        prosody_vocab: Prosody codebook size. Default: 1024.
        timbre_vocab: Timbre codebook size. Default: 512.
        hidden_dim: Diffusion model hidden dimension. Default: 512.
        num_diffusion_layers: Number of diffusion transformer layers. Default: 8.
        num_diffusion_steps: Number of diffusion steps for training. Default: 1000.
    """

    def __init__(
        self,
        text_dim: int = 512,
        feature_dim: int = 512,
        latent_dim: int = 256,
        content_vocab: int = 1024,
        prosody_vocab: int = 1024,
        timbre_vocab: int = 512,
        hidden_dim: int = 512,
        num_diffusion_layers: int = 8,
        num_diffusion_steps: int = 1000,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.num_diffusion_steps = num_diffusion_steps

        # Factorized vector quantizer
        self.fvq = FactorizedVectorQuantizer(
            feature_dim, content_vocab, prosody_vocab, timbre_vocab, latent_dim
        )

        # Prosody predictor
        self.prosody_predictor = ProsodyPredictor(text_dim, hidden_dim // 2)

        # Text to content mapper
        self.text_to_content = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Factorized diffusion model
        self.diffusion = FactorizedDiffusion(
            latent_dim, hidden_dim, num_diffusion_layers
        )

        # Timbre encoder (extract from reference audio)
        self.timbre_encoder = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def encode_speech(
        self, speech_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Encode speech into factorized latents.

        Args:
            speech_features: Speech features, shape [B, T, feature_dim].

        Returns:
            Dictionary with factorized latents and indices.
        """
        recon, info = self.fvq(speech_features)
        return info

    def extract_timbre(self, reference_audio: torch.Tensor) -> torch.Tensor:
        """Extract timbre from reference audio for zero-shot TTS.

        Args:
            reference_audio: Reference audio features, shape [B, T, feature_dim].

        Returns:
            Timbre embedding, shape [B, latent_dim].
        """
        x = reference_audio.transpose(1, 2)  # [B, feature_dim, T]
        timbre = self.timbre_encoder(x)
        return timbre

    def forward(
        self,
        text_embeds: torch.Tensor,
        speech_features: torch.Tensor,
        reference_audio: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            text_embeds: Text embeddings, shape [B, T_text, text_dim].
            speech_features: Target speech features, shape [B, T_audio, feature_dim].
            reference_audio: Optional reference for timbre extraction.

        Returns:
            Dictionary with losses.
        """
        B, T_audio, _ = speech_features.shape

        # Encode target speech
        recon, fvq_info = self.fvq(speech_features)
        vq_loss = fvq_info["vq_loss"]

        z_content_target = fvq_info["z_content"]
        z_prosody_target = fvq_info["z_prosody"]
        z_timbre_target = fvq_info["z_timbre"]

        # Predict prosody from text
        duration, pitch = self.prosody_predictor(text_embeds)

        # Map text to content
        z_content_pred = self.text_to_content(text_embeds)

        # Align text to audio length (use duration)
        # Simplified: just repeat or interpolate
        z_content_pred = F.interpolate(
            z_content_pred.transpose(1, 2),
            size=T_audio,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

        # Content prediction loss
        content_loss = F.mse_loss(z_content_pred, z_content_target.detach())

        # Sample diffusion timestep
        t = torch.randint(0, self.num_diffusion_steps, (B,), device=speech_features.device)
        t_normalized = t.float() / self.num_diffusion_steps

        # Add noise to latents
        noise_content = torch.randn_like(z_content_target)
        noise_prosody = torch.randn_like(z_prosody_target)
        noise_timbre = torch.randn_like(z_timbre_target)

        alpha = (1 - t_normalized).view(-1, 1, 1)
        z_content_noisy = alpha * z_content_target + (1 - alpha) * noise_content
        z_prosody_noisy = alpha * z_prosody_target + (1 - alpha) * noise_prosody
        z_timbre_noisy = alpha * z_timbre_target + (1 - alpha) * noise_timbre

        # Predict noise
        noise_pred_content, noise_pred_prosody, noise_pred_timbre = self.diffusion(
            z_content_noisy, z_prosody_noisy, z_timbre_noisy, t_normalized
        )

        # Diffusion loss
        diffusion_loss = (
            F.mse_loss(noise_pred_content, noise_content) +
            F.mse_loss(noise_pred_prosody, noise_prosody) +
            F.mse_loss(noise_pred_timbre, noise_timbre)
        ) / 3

        # Total loss
        total_loss = vq_loss + content_loss + diffusion_loss

        return {
            "loss": total_loss,
            "vq_loss": vq_loss,
            "content_loss": content_loss,
            "diffusion_loss": diffusion_loss,
        }

    @torch.no_grad()
    def synthesize(
        self,
        text_embeds: torch.Tensor,
        reference_audio: torch.Tensor,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Zero-shot TTS synthesis.

        Args:
            text_embeds: Text embeddings, shape [B, T_text, text_dim].
            reference_audio: Reference audio for timbre, shape [B, T_ref, feature_dim].
            num_steps: Number of diffusion steps. Default: 50.

        Returns:
            Generated speech latents, shape [B, T_audio, 3*latent_dim].
        """
        B, T_text, _ = text_embeds.shape
        device = text_embeds.device

        # Predict prosody
        duration, pitch = self.prosody_predictor(text_embeds)

        # Map text to content
        z_content = self.text_to_content(text_embeds)

        # Determine audio length from duration
        T_audio = int(duration.sum(dim=1).mean().item())

        # Align content to audio length
        z_content = F.interpolate(
            z_content.transpose(1, 2),
            size=T_audio,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

        # Extract timbre from reference
        timbre_emb = self.extract_timbre(reference_audio)
        z_timbre = timbre_emb.unsqueeze(1).expand(B, T_audio, -1)

        # Initialize prosody from noise
        z_prosody = torch.randn(B, T_audio, self.fvq.latent_dim, device=device)

        # Diffusion sampling (DDPM)
        for i in reversed(range(num_steps)):
            t = torch.full((B,), i / num_steps, device=device)

            # Predict noise
            noise_content, noise_prosody, noise_timbre = self.diffusion(
                z_content, z_prosody, z_timbre, t
            )

            # Denoise prosody (keep content and timbre fixed)
            alpha = 1 - t.view(-1, 1, 1)
            z_prosody = (z_prosody - (1 - alpha) * noise_prosody) / alpha.clamp(min=1e-8)

            # Add noise for next step (except last)
            if i > 0:
                z_prosody = z_prosody + torch.randn_like(z_prosody) * 0.1

        # Concatenate factors
        z_combined = torch.cat([z_content, z_prosody, z_timbre], dim=-1)

        return z_combined
