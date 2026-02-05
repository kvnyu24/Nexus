"""Genie: Generative Interactive Environments.

Reference: "Genie: Generative Interactive Environments" (Bruce et al., Google DeepMind, 2024)
Reference: "Genie 2: A Large-Scale Foundation World Model" (Google DeepMind, 2024)

Genie is a foundation world model that can generate interactive, playable
virtual environments from a single image or text prompt. Unlike traditional
world models trained on action-labeled data, Genie learns from unlabeled
internet videos through self-supervised learning.

Key innovations:
    - Spatiotemporal Transformer (STT): Models video dynamics with attention
    - Latent Action Model (LAM): Discovers latent actions without labels
    - Video Tokenizer: Encodes frames into discrete tokens
    - Dynamics Model: Predicts next frame given current frame + latent action

Genie 2 scales to:
    - 11B parameters
    - Generates up to 60 seconds of interactive video
    - Supports 3D world generation
    - Enables embodied agent training

Architecture:
    - VideoTokenizer: Encodes frames to discrete tokens (VQ-VAE style)
    - LatentActionModel: Infers latent actions from consecutive frames
    - DynamicsModel: Spatiotemporal transformer for frame prediction
    - GenieModel: Complete generative world model

Key properties:
    - Learns from unlabeled video (no action annotations required)
    - Generates controllable, interactive environments
    - Supports video game-like interactions
    - Can be used for agent training and data augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from ...core.base import NexusModule
import math


class VideoTokenizer(NexusModule):
    """Video tokenizer using VQ-VAE for discrete latent representation.

    Encodes video frames into discrete tokens using vector quantization.
    This allows the dynamics model to operate in a discrete token space
    rather than continuous pixel space.

    Args:
        config: Configuration dictionary with:
            - input_channels: Number of input channels. Default: 3.
            - hidden_dim: Hidden dimension. Default: 128.
            - num_embeddings: Codebook size. Default: 1024.
            - embedding_dim: Embedding dimension. Default: 256.
            - commitment_cost: VQ commitment cost. Default: 0.25.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.input_channels = config.get("input_channels", 3)
        self.hidden_dim = config.get("hidden_dim", 128)
        self.num_embeddings = config.get("num_embeddings", 1024)
        self.embedding_dim = config.get("embedding_dim", 256)
        self.commitment_cost = config.get("commitment_cost", 0.25)

        # Encoder: downsample images to latent tokens
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, self.hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.embedding_dim, 3, stride=1, padding=1),
        )

        # Vector quantization codebook
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.num_embeddings, 1.0 / self.num_embeddings
        )

        # Decoder: upsample latent tokens back to images
        self.decoder = nn.Sequential(
            nn.Conv2d(self.embedding_dim, self.hidden_dim, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.hidden_dim, self.input_channels, 4, stride=2, padding=1),
        )

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vector quantization.

        Args:
            z: Continuous latent (B, D, H, W).

        Returns:
            Tuple of (quantized, token_ids, vq_loss).
        """
        # Flatten spatial dimensions
        z_flat = z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
        z_flat = z_flat.view(-1, self.embedding_dim)  # (B*H*W, D)

        # Compute distances to codebook vectors
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        # Get nearest codebook entry
        token_ids = torch.argmin(distances, dim=1)  # (B*H*W,)
        quantized_flat = self.embedding(token_ids)  # (B*H*W, D)

        # Reshape back
        quantized = quantized_flat.view(z.shape[0], z.shape[2], z.shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)

        # VQ loss: commitment + codebook loss
        commitment_loss = F.mse_loss(quantized.detach(), z)
        codebook_loss = F.mse_loss(quantized, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        # Reshape token IDs
        token_ids = token_ids.view(z.shape[0], z.shape[2], z.shape[3])

        return quantized, token_ids, vq_loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and decode with quantization.

        Args:
            x: Input frames (B, C, H, W).

        Returns:
            Tuple of (reconstructed, token_ids, vq_loss).
        """
        # Encode
        z = self.encoder(x)

        # Quantize
        z_q, token_ids, vq_loss = self.quantize(z)

        # Decode
        x_recon = self.decoder(z_q)

        return x_recon, token_ids, vq_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode frames to token IDs.

        Args:
            x: Input frames (B, C, H, W).

        Returns:
            Token IDs (B, H', W').
        """
        z = self.encoder(x)
        _, token_ids, _ = self.quantize(z)
        return token_ids

    def decode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Decode token IDs to frames.

        Args:
            token_ids: Token IDs (B, H', W').

        Returns:
            Reconstructed frames (B, C, H, W).
        """
        # Look up embeddings
        z_q = self.embedding(token_ids)  # (B, H', W', D)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # (B, D, H', W')

        # Decode
        x_recon = self.decoder(z_q)

        return x_recon


class LatentActionModel(NexusModule):
    """Latent Action Model (LAM): Infers latent actions from frame transitions.

    Learns a discrete latent action space from unlabeled video by predicting
    which latent action was taken between consecutive frames. This enables
    control without requiring action labels in the training data.

    Args:
        config: Configuration dictionary with:
            - token_dim: Dimension of input tokens. Default: 256.
            - hidden_dim: Hidden dimension. Default: 512.
            - num_latent_actions: Number of latent actions. Default: 8.
            - num_layers: Number of transformer layers. Default: 4.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.token_dim = config.get("token_dim", 256)
        self.hidden_dim = config.get("hidden_dim", 512)
        self.num_latent_actions = config.get("num_latent_actions", 8)
        self.num_layers = config.get("num_layers", 4)

        # Token embedding projection
        self.token_proj = nn.Linear(self.token_dim, self.hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 1000, self.hidden_dim)  # Max sequence length
        )

        # Transformer for processing frame pairs
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Action classifier
        self.action_head = nn.Linear(self.hidden_dim, self.num_latent_actions)

    def forward(
        self,
        tokens_t: torch.Tensor,
        tokens_t1: torch.Tensor
    ) -> torch.Tensor:
        """Infer latent action from consecutive frames.

        Args:
            tokens_t: Current frame tokens (B, H, W).
            tokens_t1: Next frame tokens (B, H, W).

        Returns:
            Latent action logits (B, num_latent_actions).
        """
        batch_size = tokens_t.shape[0]

        # Flatten spatial dimensions
        tokens_t_flat = tokens_t.view(batch_size, -1)  # (B, H*W)
        tokens_t1_flat = tokens_t1.view(batch_size, -1)  # (B, H*W)

        # Concatenate frame pair
        tokens = torch.stack([tokens_t_flat, tokens_t1_flat], dim=1)  # (B, 2, H*W)

        # Project to hidden dim (simplified: treat each position as token)
        # In full implementation, would look up embeddings
        # For now, treat token IDs as features
        tokens = tokens.float().unsqueeze(-1).expand(-1, -1, -1, self.token_dim)
        tokens = tokens.reshape(batch_size, -1, self.token_dim)  # (B, 2*H*W, token_dim)

        x = self.token_proj(tokens)  # (B, 2*H*W, hidden_dim)

        # Add positional encoding
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transform
        x = self.transformer(x)  # (B, 2*H*W, hidden_dim)

        # Pool and classify
        x = x.mean(dim=1)  # (B, hidden_dim)
        action_logits = self.action_head(x)  # (B, num_latent_actions)

        return action_logits


class SpatiotemporalTransformer(NexusModule):
    """Spatiotemporal Transformer (STT) for video dynamics prediction.

    Predicts the next frame tokens given current frame tokens and latent action.
    Uses masked attention to ensure autoregressive generation.

    Args:
        config: Configuration dictionary with:
            - num_embeddings: Vocabulary size (codebook size). Default: 1024.
            - embedding_dim: Token embedding dimension. Default: 256.
            - hidden_dim: Transformer hidden dimension. Default: 512.
            - num_heads: Number of attention heads. Default: 8.
            - num_layers: Number of transformer layers. Default: 12.
            - num_latent_actions: Number of latent actions. Default: 8.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.num_embeddings = config.get("num_embeddings", 1024)
        self.embedding_dim = config.get("embedding_dim", 256)
        self.hidden_dim = config.get("hidden_dim", 512)
        self.num_heads = config.get("num_heads", 8)
        self.num_layers = config.get("num_layers", 12)
        self.num_latent_actions = config.get("num_latent_actions", 8)

        # Token embedding
        self.token_embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # Action embedding
        self.action_embedding = nn.Embedding(self.num_latent_actions, self.embedding_dim)

        # Project to hidden dim
        self.input_proj = nn.Linear(self.embedding_dim, self.hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 10000, self.hidden_dim)  # Large enough for tokens
        )

        # Transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        # Output head
        self.output_head = nn.Linear(self.hidden_dim, self.num_embeddings)

    def forward(
        self,
        current_tokens: torch.Tensor,
        latent_action: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict next frame tokens.

        Args:
            current_tokens: Current frame tokens (B, H, W).
            latent_action: Latent action (B,) as indices.
            target_tokens: Optional target tokens for teacher forcing (B, H, W).

        Returns:
            Next frame token logits (B, H*W, num_embeddings).
        """
        batch_size, h, w = current_tokens.shape
        seq_len = h * w

        # Flatten spatial dimensions
        current_flat = current_tokens.view(batch_size, -1)  # (B, H*W)

        # Embed tokens
        token_embeds = self.token_embedding(current_flat)  # (B, H*W, embedding_dim)

        # Embed action and prepend
        action_embeds = self.action_embedding(latent_action).unsqueeze(1)  # (B, 1, embedding_dim)
        memory = torch.cat([action_embeds, token_embeds], dim=1)  # (B, 1+H*W, embedding_dim)

        # Project to hidden dim
        memory = self.input_proj(memory)  # (B, 1+H*W, hidden_dim)

        # Add positional encoding
        memory_len = memory.shape[1]
        memory = memory + self.pos_encoding[:, :memory_len, :]

        # For autoregressive generation, use target tokens if provided
        if target_tokens is not None:
            target_flat = target_tokens.view(batch_size, -1)  # (B, H*W)
            target_embeds = self.token_embedding(target_flat)
            tgt = self.input_proj(target_embeds)
            tgt = tgt + self.pos_encoding[:, :seq_len, :]

            # Create causal mask
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        else:
            # Generate autoregressively (simplified: use memory as target)
            tgt = memory[:, 1:, :]  # Skip action token
            causal_mask = None

        # Transformer decode
        output = self.transformer(tgt, memory, tgt_mask=causal_mask)  # (B, H*W, hidden_dim)

        # Predict tokens
        logits = self.output_head(output)  # (B, H*W, num_embeddings)

        return logits


class GenieModel(NexusModule):
    """Genie: Generative Interactive Environments.

    Complete foundation world model that learns from unlabeled video and
    generates interactive, controllable environments.

    Training procedure:
        1. Tokenize video frames into discrete tokens (VQ-VAE)
        2. Infer latent actions between consecutive frames (LAM)
        3. Train dynamics model to predict next frame given current + action (STT)
        4. Generate new frames autoregressively for interaction

    Args:
        config: Configuration dictionary with:
            - num_embeddings: Codebook size. Default: 1024.
            - embedding_dim: Token embedding dimension. Default: 256.
            - hidden_dim: Model hidden dimension. Default: 512.
            - num_latent_actions: Number of latent actions. Default: 8.
            - num_layers: Number of transformer layers. Default: 12.
            - Additional tokenizer and LAM config options.

    Example:
        >>> config = {
        ...     "num_embeddings": 1024,
        ...     "num_latent_actions": 8,
        ...     "hidden_dim": 512
        ... }
        >>> model = GenieModel(config)
        >>> frames = torch.randn(4, 3, 64, 64)  # (B, C, H, W)
        >>> next_frames = torch.randn(4, 3, 64, 64)
        >>> loss, metrics = model(frames, next_frames)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Video tokenizer
        self.tokenizer = VideoTokenizer(config)

        # Latent action model
        lam_config = {
            "token_dim": config.get("embedding_dim", 256),
            "hidden_dim": config.get("hidden_dim", 512),
            "num_latent_actions": config.get("num_latent_actions", 8),
            "num_layers": config.get("lam_layers", 4),
        }
        self.action_model = LatentActionModel(lam_config)

        # Dynamics model (STT)
        stt_config = {
            "num_embeddings": config.get("num_embeddings", 1024),
            "embedding_dim": config.get("embedding_dim", 256),
            "hidden_dim": config.get("hidden_dim", 512),
            "num_heads": config.get("num_heads", 8),
            "num_layers": config.get("num_layers", 12),
            "num_latent_actions": config.get("num_latent_actions", 8),
        }
        self.dynamics_model = SpatiotemporalTransformer(stt_config)

    def forward(
        self,
        current_frames: torch.Tensor,
        next_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Training forward pass.

        Args:
            current_frames: Current frames (B, C, H, W).
            next_frames: Next frames (B, C, H, W).

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        # Tokenize frames
        recon_current, tokens_current, vq_loss_current = self.tokenizer(current_frames)
        recon_next, tokens_next, vq_loss_next = self.tokenizer(next_frames)

        # VQ loss
        vq_loss = vq_loss_current + vq_loss_next

        # Reconstruction loss
        recon_loss = F.mse_loss(recon_current, current_frames) + F.mse_loss(recon_next, next_frames)

        # Infer latent action
        action_logits = self.action_model(tokens_current, tokens_next)

        # Sample action (during training, use Gumbel-Softmax)
        action_probs = F.gumbel_softmax(action_logits, tau=1.0, hard=True)
        latent_action = action_probs.argmax(dim=-1)

        # Predict next frame tokens
        next_logits = self.dynamics_model(
            tokens_current,
            latent_action,
            target_tokens=tokens_next
        )

        # Dynamics loss
        dynamics_loss = F.cross_entropy(
            next_logits.reshape(-1, next_logits.shape[-1]),
            tokens_next.view(-1)
        )

        # Action distribution loss (encourage diverse actions)
        action_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()

        # Total loss
        total_loss = recon_loss + vq_loss + dynamics_loss - 0.01 * action_entropy

        metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "vq_loss": vq_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "action_entropy": action_entropy.item(),
        }

        return total_loss, metrics

    @torch.no_grad()
    def generate(
        self,
        initial_frame: torch.Tensor,
        actions: torch.Tensor,
        num_steps: int
    ) -> List[torch.Tensor]:
        """Generate video by rolling out the world model.

        Args:
            initial_frame: Starting frame (1, C, H, W).
            actions: Sequence of latent actions (num_steps,).
            num_steps: Number of steps to generate.

        Returns:
            List of generated frames.
        """
        # Set to inference mode (PyTorch method for disabling dropout/batch norm updates)
        self.train(mode=False)

        # Tokenize initial frame
        tokens = self.tokenizer.encode(initial_frame)  # (1, H', W')

        generated_frames = [initial_frame]

        for step in range(num_steps):
            # Get action for this step
            action = actions[step:step+1]  # (1,)

            # Predict next tokens
            next_logits = self.dynamics_model(tokens, action)  # (1, H'*W', vocab_size)

            # Sample next tokens
            next_tokens = next_logits.argmax(dim=-1)  # (1, H'*W')
            next_tokens = next_tokens.view(1, tokens.shape[1], tokens.shape[2])  # (1, H', W')

            # Decode tokens to frame
            next_frame = self.tokenizer.decode_tokens(next_tokens)

            generated_frames.append(next_frame)

            # Update current tokens
            tokens = next_tokens

        return generated_frames
