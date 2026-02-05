"""
Diffusion Transformer (DiT) - Scalable Diffusion Models with Transformers.

Replaces the U-Net backbone commonly used in diffusion models with a
transformer architecture, enabling better scaling properties and
leveraging the success of Vision Transformers (ViT) for image generation.

Key innovations:
- Patch-based tokenization of latent images
- Adaptive Layer Normalization (adaLN-Zero) for conditioning on timestep
  and class labels, initialized so each block acts as identity at init
- Standard transformer blocks with multi-head self-attention and MLP
- Final linear decoder to reconstruct noise/velocity predictions

Reference: "Scalable Diffusion Models with Transformers"
           Peebles & Xie, 2023 (https://arxiv.org/abs/2212.09748)
"""

from typing import Dict, Any, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule


class PatchEmbed(nn.Module):
    """Convert latent images to sequences of patch embeddings.

    Splits a 2D feature map into non-overlapping patches and projects
    each patch to a fixed-dimensional embedding vector. This mirrors
    the patch embedding used in Vision Transformer (ViT).

    Args:
        input_size: Spatial resolution of the input latent (assumes square).
        patch_size: Size of each non-overlapping patch.
        in_channels: Number of input channels in the latent.
        hidden_dim: Output embedding dimension per patch.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_dim: int = 1152,
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
        # (B, hidden_dim, H/P, W/P) -> (B, hidden_dim, num_patches) -> (B, num_patches, hidden_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer normalization modulation.

    Given pre-normalized activations x, applies affine modulation:
        x_out = x * (1 + scale) + shift

    This is the core operation of adaLN-Zero conditioning.
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embed scalar diffusion timesteps into vector representations.

    Uses sinusoidal positional encoding followed by a two-layer MLP
    to project timestep scalars into the model hidden dimension.

    Args:
        hidden_dim: Output embedding dimension.
        frequency_dim: Dimension of the sinusoidal frequency embedding.
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
        """Create sinusoidal timestep embeddings.

        Args:
            t: Timestep tensor of shape (B,).
            dim: Embedding dimension (must be even).

        Returns:
            Sinusoidal embeddings of shape (B, dim).
        """
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep indices of shape (B,).

        Returns:
            Timestep embeddings of shape (B, hidden_dim).
        """
        t_freq = self.sinusoidal_embedding(t, self.frequency_dim)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embed class labels into vector representations for conditioning.

    Supports classifier-free guidance by randomly dropping labels during
    training (replacing with a learned null embedding).

    Args:
        num_classes: Number of discrete class labels.
        hidden_dim: Output embedding dimension.
        dropout_prob: Probability of dropping the label (for CFG training).
    """

    def __init__(self, num_classes: int, hidden_dim: int, dropout_prob: float = 0.1):
        super().__init__()
        use_cfg = dropout_prob > 0.0
        self.embedding_table = nn.Embedding(num_classes + int(use_cfg), hidden_dim)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Replace labels with the null class index for classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(
        self,
        labels: torch.Tensor,
        train: bool = True,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            labels: Class label indices of shape (B,).
            train: Whether in training mode (enables label dropout).
            force_drop_ids: Optional mask to force label dropping.

        Returns:
            Label embeddings of shape (B, hidden_dim).
        """
        use_dropout = self.dropout_prob > 0.0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class DiTBlock(nn.Module):
    """Transformer block with adaptive Layer Normalization (adaLN-Zero).

    Each block consists of:
    1. LayerNorm + adaLN-Zero modulation -> Multi-head self-attention
    2. LayerNorm + adaLN-Zero modulation -> Pointwise feed-forward MLP

    The adaLN-Zero mechanism modulates activations based on the conditioning
    signal (timestep + class embedding). All modulation parameters (shift,
    scale, gate) are predicted from the conditioning vector. The gate
    parameters are initialized to zero so that each block starts as an
    identity function, enabling stable deep network training.

    Args:
        hidden_dim: Hidden dimension of the transformer.
        num_heads: Number of attention heads.
        mlp_ratio: Expansion ratio for the MLP hidden dimension.
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
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # adaLN-Zero modulation: predict 6 modulation parameters
        # (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        # Initialize gate parameters to zero (identity initialization)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token sequence of shape (B, N, D).
            c: Conditioning vector of shape (B, D).

        Returns:
            Updated token sequence of shape (B, N, D).
        """
        # Predict all modulation parameters from conditioning
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # Attention block with adaLN-Zero
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP block with adaLN-Zero
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class FinalLayer(nn.Module):
    """Final layer of DiT: adaLN-Zero modulated linear projection.

    Applies a final adaptive normalization followed by a linear projection
    to map from hidden dimension back to the patch pixel space.

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
        # Initialize to zero for identity-like behavior
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token sequence of shape (B, N, D).
            c: Conditioning vector of shape (B, D).

        Returns:
            Projected output of shape (B, N, patch_size^2 * out_channels).
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(NexusModule):
    """Diffusion Transformer (DiT) for image generation.

    A transformer-based architecture for diffusion models that replaces
    the conventional U-Net backbone. The model processes latent image
    patches as a sequence, conditioning on timestep and class label
    through adaptive layer normalization (adaLN-Zero).

    Architecture overview:
    1. Patchify: split latent image into non-overlapping patches
    2. Add positional embeddings
    3. Process through N DiTBlocks with adaLN-Zero conditioning
    4. Final layer: project back to pixel space
    5. Unpatchify: reconstruct spatial layout

    Default configuration matches DiT-XL/2:
        input_size=32, patch_size=2, in_channels=4, hidden_dim=1152,
        depth=28, num_heads=16, num_classes=1000

    Reference: "Scalable Diffusion Models with Transformers"
               Peebles & Xie, 2023 (https://arxiv.org/abs/2212.09748)

    Args:
        config: Dictionary containing model hyperparameters.
            - input_size (int): Spatial resolution of input latent. Default: 32.
            - patch_size (int): Patch size for tokenization. Default: 2.
            - in_channels (int): Number of input latent channels. Default: 4.
            - hidden_dim (int): Transformer hidden dimension. Default: 1152.
            - depth (int): Number of DiT transformer blocks. Default: 28.
            - num_heads (int): Number of attention heads. Default: 16.
            - num_classes (int): Number of class labels for conditioning. Default: 1000.
            - mlp_ratio (float): MLP expansion ratio. Default: 4.0.
            - dropout (float): Dropout probability. Default: 0.0.
            - class_dropout_prob (float): Label dropout for CFG. Default: 0.1.
            - learn_sigma (bool): Whether to predict variance. Default: True.
            - num_timesteps (int): Number of diffusion timesteps. Default: 1000.
            - beta_start (float): Start of linear beta schedule. Default: 0.0001.
            - beta_end (float): End of linear beta schedule. Default: 0.02.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Model hyperparameters
        self.input_size = config.get("input_size", 32)
        self.patch_size = config.get("patch_size", 2)
        self.in_channels = config.get("in_channels", 4)
        self.hidden_dim = config.get("hidden_dim", 1152)
        self.depth = config.get("depth", 28)
        self.num_heads = config.get("num_heads", 16)
        self.num_classes = config.get("num_classes", 1000)
        self.mlp_ratio = config.get("mlp_ratio", 4.0)
        self.dropout = config.get("dropout", 0.0)
        self.class_dropout_prob = config.get("class_dropout_prob", 0.1)
        self.learn_sigma = config.get("learn_sigma", True)
        self.out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels

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

        # Patch embedding
        self.patch_embed = PatchEmbed(
            input_size=self.input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            hidden_dim=self.hidden_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Positional embedding (learned)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.hidden_dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Conditioning embedders
        self.t_embedder = TimestepEmbedder(self.hidden_dim)
        self.y_embedder = LabelEmbedder(self.num_classes, self.hidden_dim, self.class_dropout_prob)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
            )
            for _ in range(self.depth)
        ])

        # Final output layer
        self.final_layer = FinalLayer(self.hidden_dim, self.patch_size, self.out_channels)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights following DiT paper conventions."""
        # Initialize patch embedding like a linear layer
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(self.patch_embed.proj.bias)

        # Initialize timestep and label embedders
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize transformer block weights
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.in_proj_weight)
            nn.init.xavier_uniform_(block.attn.out_proj.weight)
            nn.init.zeros_(block.attn.out_proj.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct spatial image from patch sequence.

        Args:
            x: Patch predictions of shape (B, N, patch_size^2 * C).

        Returns:
            Image tensor of shape (B, C, H, W).
        """
        c = self.out_channels
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
        y: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training: predict noise (and optionally variance).

        Args:
            x: Noisy latent images of shape (B, C, H, W).
            t: Diffusion timesteps of shape (B,).
            y: Class labels of shape (B,).

        Returns:
            Dictionary with:
                - "prediction": Model output of shape (B, C_out, H, W).
                - "hidden_states": Final hidden states before unpatchify.
        """
        # Patchify and add positional embedding
        x = self.patch_embed(x) + self.pos_embed

        # Compute conditioning vector: timestep + class label
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, train=self.training)
        c = t_emb + y_emb

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, c)

        hidden_states = x

        # Final layer and unpatchify
        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        return {
            "prediction": x,
            "hidden_states": hidden_states,
        }

    def compute_loss(
        self,
        x_start: torch.Tensor,
        y: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the diffusion training loss.

        Samples random timesteps, adds noise to data, and computes MSE
        between predicted and actual noise.

        Args:
            x_start: Clean latent images of shape (B, C, H, W).
            y: Class labels of shape (B,).
            noise: Optional pre-sampled noise (B, C, H, W).

        Returns:
            Dictionary with:
                - "loss": Scalar training loss.
                - "noise_pred": Predicted noise (B, C, H, W).
                - "noise_target": Target noise (B, C, H, W).
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_start)

        # Create noisy samples: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

        # Predict noise
        output = self.forward(x_noisy, t, y)
        prediction = output["prediction"]

        if self.learn_sigma:
            # Split into noise prediction and variance prediction
            noise_pred, var_pred = prediction.chunk(2, dim=1)
        else:
            noise_pred = prediction

        # Simple MSE loss on noise prediction
        loss = F.mse_loss(noise_pred, noise)

        return {
            "loss": loss,
            "noise_pred": noise_pred,
            "noise_target": noise,
        }

    @torch.no_grad()
    def sample(
        self,
        y: torch.Tensor,
        cfg_scale: float = 4.0,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate samples using DDPM sampling with classifier-free guidance.

        Args:
            y: Class labels of shape (B,).
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance).
            num_steps: Number of denoising steps (defaults to num_timesteps).

        Returns:
            Generated latent images of shape (B, C, H, W).
        """
        batch_size = y.shape[0]
        device = y.device
        num_steps = num_steps or self.num_timesteps

        # Start from pure noise
        x = torch.randn(
            batch_size, self.in_channels, self.input_size, self.input_size,
            device=device,
        )

        # Prepare unconditional labels for CFG
        y_null = torch.full_like(y, self.num_classes)

        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps, device=device
        ).long()

        for i, t in enumerate(timesteps):
            t_batch = t.expand(batch_size)

            # Conditional and unconditional predictions
            x_input = torch.cat([x, x], dim=0)
            t_input = torch.cat([t_batch, t_batch], dim=0)
            y_input = torch.cat([y, y_null], dim=0)

            output = self.forward(x_input, t_input, y_input)
            pred = output["prediction"]

            if self.learn_sigma:
                pred, _ = pred.chunk(2, dim=1)

            pred_cond, pred_uncond = pred.chunk(2, dim=0)

            # Classifier-free guidance
            pred_guided = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

            # DDPM update step
            alpha_bar = self.alphas_cumprod[t]
            alpha_bar_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
            beta_t = self.betas[t]

            # Predicted x_0
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar) * pred_guided) / torch.sqrt(alpha_bar)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # Posterior mean
            coeff1 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar)
            coeff2 = torch.sqrt(1.0 - beta_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            mean = coeff1 * x0_pred + coeff2 * x

            # Add noise (except at final step)
            if t > 0:
                noise = torch.randn_like(x)
                variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean

        return x
