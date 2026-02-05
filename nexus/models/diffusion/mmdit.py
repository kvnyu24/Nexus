"""
Multimodal Diffusion Transformer (MMDiT).

A dual-stream transformer architecture for diffusion models that processes
image and text tokens through separate parameter streams with joint
cross-modal attention. This architecture forms the basis for Stable
Diffusion 3 and FLUX-style models.

Key innovations:
- Dual-stream design: separate weights for image and text modalities
- Joint attention: both streams attend to the concatenated sequence,
  enabling cross-modal information flow
- adaLN-Zero conditioning on timestep (same as DiT)
- Modality-specific MLP and normalization layers
- Text conditioning integrated directly into the architecture rather
  than through cross-attention alone

Reference: "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
           Esser et al., 2024 (https://arxiv.org/abs/2403.03206)
           "FLUX" - Black Forest Labs, 2024
"""

from typing import Dict, Any, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule


class PatchEmbed(nn.Module):
    """Patch embedding for latent images in MMDiT.

    Converts 2D latent feature maps into sequences of patch tokens
    by splitting into non-overlapping patches and projecting.

    Args:
        input_size: Spatial resolution of the input latent.
        patch_size: Size of each non-overlapping patch.
        in_channels: Number of input latent channels.
        hidden_dim: Output embedding dimension.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_dim: int = 1536,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent tensor of shape (B, C, H, W).

        Returns:
            Patch embeddings of shape (B, num_patches, hidden_dim).
        """
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedder for MMDiT conditioning.

    Args:
        hidden_dim: Output embedding dimension.
        frequency_dim: Dimension of the sinusoidal encoding.
    """

    def __init__(self, hidden_dim: int, frequency_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_dim = frequency_dim

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.sinusoidal_embedding(t, self.frequency_dim)
        return self.mlp(t_freq)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer normalization modulation."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class JointAttention(nn.Module):
    """Joint multi-head attention across image and text streams.

    Both modalities are projected with their own Q/K/V matrices, then
    the key-value pairs are concatenated so that each token can attend
    to tokens from both modalities. The output is split back into
    separate image and text streams.

    This enables bidirectional cross-modal information flow while
    keeping per-modality parameterization.

    Args:
        hidden_dim: Hidden dimension for both streams.
        num_heads: Number of attention heads.
        dropout: Attention dropout probability.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Separate Q/K/V projections for each modality
        self.q_img = nn.Linear(hidden_dim, hidden_dim)
        self.k_img = nn.Linear(hidden_dim, hidden_dim)
        self.v_img = nn.Linear(hidden_dim, hidden_dim)

        self.q_txt = nn.Linear(hidden_dim, hidden_dim)
        self.k_txt = nn.Linear(hidden_dim, hidden_dim)
        self.v_txt = nn.Linear(hidden_dim, hidden_dim)

        # Separate output projections
        self.out_img = nn.Linear(hidden_dim, hidden_dim)
        self.out_txt = nn.Linear(hidden_dim, hidden_dim)

        self.attn_drop = nn.Dropout(dropout)

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, N, D) -> (B, H, N, D_h)."""
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        img_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img_tokens: Image stream of shape (B, N_img, D).
            txt_tokens: Text stream of shape (B, N_txt, D).

        Returns:
            Tuple of (updated_img_tokens, updated_txt_tokens), both
            retaining their original sequence lengths.
        """
        B = img_tokens.shape[0]
        N_img = img_tokens.shape[1]
        N_txt = txt_tokens.shape[1]

        # Project queries, keys, values for each modality
        q_img = self._reshape_for_heads(self.q_img(img_tokens))
        k_img = self._reshape_for_heads(self.k_img(img_tokens))
        v_img = self._reshape_for_heads(self.v_img(img_tokens))

        q_txt = self._reshape_for_heads(self.q_txt(txt_tokens))
        k_txt = self._reshape_for_heads(self.k_txt(txt_tokens))
        v_txt = self._reshape_for_heads(self.v_txt(txt_tokens))

        # Concatenate keys and values from both modalities
        k_joint = torch.cat([k_img, k_txt], dim=2)  # (B, H, N_img+N_txt, D_h)
        v_joint = torch.cat([v_img, v_txt], dim=2)

        # Image stream attends to joint sequence
        attn_img = torch.matmul(q_img, k_joint.transpose(-2, -1)) * self.scale
        attn_img = F.softmax(attn_img, dim=-1)
        attn_img = self.attn_drop(attn_img)
        out_img = torch.matmul(attn_img, v_joint)

        # Text stream attends to joint sequence
        attn_txt = torch.matmul(q_txt, k_joint.transpose(-2, -1)) * self.scale
        attn_txt = F.softmax(attn_txt, dim=-1)
        attn_txt = self.attn_drop(attn_txt)
        out_txt = torch.matmul(attn_txt, v_joint)

        # Reshape back and project
        out_img = out_img.transpose(1, 2).contiguous().view(B, N_img, -1)
        out_txt = out_txt.transpose(1, 2).contiguous().view(B, N_txt, -1)

        out_img = self.out_img(out_img)
        out_txt = self.out_txt(out_txt)

        return out_img, out_txt


class MMDiTBlock(nn.Module):
    """Dual-stream transformer block with joint attention and adaLN-Zero.

    Each block maintains separate normalization, modulation, and MLP
    parameters for image and text streams, but shares information
    through joint attention over the concatenated sequence.

    The adaLN-Zero mechanism conditions each stream on the timestep
    embedding, with gates initialized to zero for identity initialization.

    Args:
        hidden_dim: Hidden dimension for both streams.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        mlp_hidden = int(hidden_dim * mlp_ratio)

        # Image stream layers
        self.norm1_img = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2_img = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp_img = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # Text stream layers
        self.norm1_txt = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2_txt = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp_txt = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # Joint attention
        self.joint_attn = JointAttention(hidden_dim, num_heads, dropout)

        # adaLN-Zero modulation for image stream (6 params: shift, scale, gate for attn+mlp)
        self.adaLN_img = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_img[-1].weight)
        nn.init.zeros_(self.adaLN_img[-1].bias)

        # adaLN-Zero modulation for text stream
        self.adaLN_txt = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_txt[-1].weight)
        nn.init.zeros_(self.adaLN_txt[-1].bias)

    def forward(
        self,
        img_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img_tokens: Image stream of shape (B, N_img, D).
            txt_tokens: Text stream of shape (B, N_txt, D).
            c: Conditioning vector of shape (B, D).

        Returns:
            Tuple of (updated_img_tokens, updated_txt_tokens).
        """
        # Compute modulation parameters for both streams
        shift_attn_i, scale_attn_i, gate_attn_i, shift_mlp_i, scale_mlp_i, gate_mlp_i = (
            self.adaLN_img(c).chunk(6, dim=-1)
        )
        shift_attn_t, scale_attn_t, gate_attn_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = (
            self.adaLN_txt(c).chunk(6, dim=-1)
        )

        # Pre-normalize and modulate for attention
        img_norm = modulate(self.norm1_img(img_tokens), shift_attn_i, scale_attn_i)
        txt_norm = modulate(self.norm1_txt(txt_tokens), shift_attn_t, scale_attn_t)

        # Joint attention
        attn_img, attn_txt = self.joint_attn(img_norm, txt_norm)

        # Residual + gating for attention
        img_tokens = img_tokens + gate_attn_i.unsqueeze(1) * attn_img
        txt_tokens = txt_tokens + gate_attn_t.unsqueeze(1) * attn_txt

        # Pre-normalize and modulate for MLP
        img_norm = modulate(self.norm2_img(img_tokens), shift_mlp_i, scale_mlp_i)
        txt_norm = modulate(self.norm2_txt(txt_tokens), shift_mlp_t, scale_mlp_t)

        # MLP + residual + gating
        img_tokens = img_tokens + gate_mlp_i.unsqueeze(1) * self.mlp_img(img_norm)
        txt_tokens = txt_tokens + gate_mlp_t.unsqueeze(1) * self.mlp_txt(txt_norm)

        return img_tokens, txt_tokens


class FinalLayer(nn.Module):
    """Final projection layer for the image stream in MMDiT.

    Applies adaLN-Zero modulated normalization followed by linear
    projection back to patch pixel space.

    Args:
        hidden_dim: Input hidden dimension.
        patch_size: Spatial size of each patch.
        out_channels: Number of output channels per patch pixel.
    """

    def __init__(self, hidden_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class MMDiT(NexusModule):
    """Multimodal Diffusion Transformer (MMDiT).

    A dual-stream transformer architecture for text-conditioned image
    generation via diffusion. The image and text streams maintain
    separate parameters (normalization, MLP, Q/K/V projections) but
    share information through joint attention at each layer.

    This is the backbone architecture underlying Stable Diffusion 3
    and FLUX models, designed for rectified flow / flow matching
    training objectives.

    Architecture overview:
    1. Patchify image latents and project text embeddings
    2. Add positional embeddings to image tokens
    3. Process through N dual-stream MMDiTBlocks with joint attention
    4. Final layer: project image tokens back to pixel space
    5. Unpatchify to reconstruct spatial layout

    Default configuration (MMDiT-Large):
        hidden_dim=1536, depth=24, num_heads=24, text_dim=4096, patch_size=2

    Reference: "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
               Esser et al., 2024 (https://arxiv.org/abs/2403.03206)

    Args:
        config: Dictionary containing model hyperparameters.
            - hidden_dim (int): Transformer hidden dimension. Default: 1536.
            - depth (int): Number of MMDiT blocks. Default: 24.
            - num_heads (int): Number of attention heads. Default: 24.
            - text_dim (int): Input text embedding dimension. Default: 4096.
            - input_size (int): Spatial resolution of input latent. Default: 32.
            - patch_size (int): Patch size. Default: 2.
            - in_channels (int): Number of input latent channels. Default: 4.
            - mlp_ratio (float): MLP expansion ratio. Default: 4.0.
            - dropout (float): Dropout probability. Default: 0.0.
            - max_text_len (int): Maximum text sequence length. Default: 77.
            - num_timesteps (int): Number of diffusion timesteps. Default: 1000.
            - beta_start (float): Start of beta schedule. Default: 0.0001.
            - beta_end (float): End of beta schedule. Default: 0.02.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Model hyperparameters
        self.hidden_dim = config.get("hidden_dim", 1536)
        self.depth = config.get("depth", 24)
        self.num_heads = config.get("num_heads", 24)
        self.text_dim = config.get("text_dim", 4096)
        self.input_size = config.get("input_size", 32)
        self.patch_size = config.get("patch_size", 2)
        self.in_channels = config.get("in_channels", 4)
        self.mlp_ratio = config.get("mlp_ratio", 4.0)
        self.dropout = config.get("dropout", 0.0)
        self.max_text_len = config.get("max_text_len", 77)

        # Diffusion schedule
        self.num_timesteps = config.get("num_timesteps", 1000)
        beta_start = config.get("beta_start", 0.0001)
        beta_end = config.get("beta_end", 0.02)
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Patch embedding for image stream
        self.patch_embed = PatchEmbed(
            input_size=self.input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            hidden_dim=self.hidden_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Positional embedding for image patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.hidden_dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Text projection: map external text embeddings to hidden_dim
        self.text_proj = nn.Linear(self.text_dim, self.hidden_dim)

        # Positional embedding for text tokens
        self.text_pos_embed = nn.Parameter(
            torch.zeros(1, self.max_text_len, self.hidden_dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.text_pos_embed, std=0.02)

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(self.hidden_dim)

        # Dual-stream transformer blocks
        self.blocks = nn.ModuleList([
            MMDiTBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
            )
            for _ in range(self.depth)
        ])

        # Final output layer (image stream only)
        self.final_layer = FinalLayer(
            self.hidden_dim, self.patch_size, self.in_channels
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        # Patch embedding
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(self.patch_embed.proj.bias)

        # Text projection
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)

        # Timestep embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct spatial image from patch sequence.

        Args:
            x: Patch predictions of shape (B, N, patch_size^2 * C).

        Returns:
            Image tensor of shape (B, C, H, W).
        """
        c = self.in_channels
        p = self.patch_size
        h = w = self.input_size // p
        x = x.reshape(-1, h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(-1, c, h * p, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embeds: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            x: Noisy latent images of shape (B, C, H, W).
            t: Diffusion timesteps of shape (B,).
            text_embeds: Pre-computed text embeddings of shape (B, L, text_dim).

        Returns:
            Dictionary with:
                - "prediction": Velocity/noise prediction of shape (B, C, H, W).
                - "img_hidden_states": Final image hidden states.
                - "txt_hidden_states": Final text hidden states.
        """
        # Prepare image tokens
        img_tokens = self.patch_embed(x) + self.pos_embed

        # Prepare text tokens
        L = text_embeds.shape[1]
        txt_tokens = self.text_proj(text_embeds) + self.text_pos_embed[:, :L, :]

        # Conditioning vector from timestep
        c = self.t_embedder(t)

        # Process through dual-stream blocks
        for block in self.blocks:
            img_tokens, txt_tokens = block(img_tokens, txt_tokens, c)

        # Final projection (image stream only)
        output = self.final_layer(img_tokens, c)
        output = self.unpatchify(output)

        return {
            "prediction": output,
            "img_hidden_states": img_tokens,
            "txt_hidden_states": txt_tokens,
        }

    def compute_loss(
        self,
        x_start: torch.Tensor,
        text_embeds: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the diffusion training loss (epsilon prediction).

        Args:
            x_start: Clean latent images of shape (B, C, H, W).
            text_embeds: Text embeddings of shape (B, L, text_dim).
            noise: Optional pre-sampled noise of shape (B, C, H, W).

        Returns:
            Dictionary with:
                - "loss": Scalar MSE loss.
                - "noise_pred": Predicted noise.
                - "noise_target": Target noise.
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

        output = self.forward(x_noisy, t, text_embeds)
        noise_pred = output["prediction"]

        loss = F.mse_loss(noise_pred, noise)

        return {
            "loss": loss,
            "noise_pred": noise_pred,
            "noise_target": noise,
        }

    @torch.no_grad()
    def sample(
        self,
        text_embeds: torch.Tensor,
        cfg_scale: float = 7.5,
        null_text_embeds: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate samples using DDPM with classifier-free guidance.

        Args:
            text_embeds: Text condition embeddings of shape (B, L, text_dim).
            cfg_scale: Guidance scale (1.0 = no guidance).
            null_text_embeds: Unconditional text embeddings for CFG.
                If None, uses zeros.
            num_steps: Number of denoising steps.

        Returns:
            Generated latent images of shape (B, C, H, W).
        """
        batch_size = text_embeds.shape[0]
        device = text_embeds.device
        num_steps = num_steps or self.num_timesteps

        if null_text_embeds is None:
            null_text_embeds = torch.zeros_like(text_embeds)

        x = torch.randn(
            batch_size, self.in_channels, self.input_size, self.input_size,
            device=device,
        )

        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps, device=device
        ).long()

        for t_val in timesteps:
            t_batch = t_val.expand(batch_size)

            # Conditional prediction
            out_cond = self.forward(x, t_batch, text_embeds)["prediction"]
            # Unconditional prediction
            out_uncond = self.forward(x, t_batch, null_text_embeds)["prediction"]

            # CFG
            pred = out_uncond + cfg_scale * (out_cond - out_uncond)

            # DDPM update
            alpha_bar = self.alphas_cumprod[t_val]
            alpha_bar_prev = self.alphas_cumprod[t_val - 1] if t_val > 0 else torch.tensor(1.0, device=device)
            beta_t = self.betas[t_val]

            x0_pred = (x - torch.sqrt(1.0 - alpha_bar) * pred) / torch.sqrt(alpha_bar)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            coeff1 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar)
            coeff2 = torch.sqrt(1.0 - beta_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            mean = coeff1 * x0_pred + coeff2 * x

            if t_val > 0:
                noise = torch.randn_like(x)
                variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean

        return x
