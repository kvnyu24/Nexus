"""
MusicGen - Controllable Music Generation with Language Models.

MusicGen is a single-stage transformer language model that generates
high-quality music conditioned on text descriptions and melodic features.
It uses a novel multi-stream modeling approach that allows efficient
parallel prediction of audio codec tokens.

Key innovations:
1. **Delayed Pattern Strategy**: Parallel prediction of multiple codebook
   tokens from EnCodec with a delay pattern to handle dependencies.

2. **Melody Conditioning**: Optional conditioning on melodic features
   (chromagram) extracted from reference audio.

3. **Single-Stage Generation**: Unlike cascaded models, MusicGen generates
   all codec tokens in one pass, making it much faster.

4. **Multi-Resolution Modeling**: Handles EnCodec's 8 codebooks (50Hz each)
   representing different frequency bands.

Supports:
- Text-to-music generation
- Melody-guided generation (melody + text)
- Continuation and variation generation

References:
    "Simple and Controllable Music Generation"
    Copet et al., Meta AI, 2023 (https://arxiv.org/abs/2306.05284)

    Accepted at NeurIPS 2023
"""

from typing import Dict, Any, Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule


class DelayedPatternProvider:
    """Manages the delayed pattern for parallel codebook prediction.

    EnCodec produces 8 codebooks, each with 2048 tokens. Instead of
    predicting them sequentially (slow), we use a delay pattern that
    allows parallel prediction while respecting dependencies.

    Pattern example (K=4 codebooks, delay=1):
        Time:  0  1  2  3  4  5
        C0:    x  x  x  x  x  x
        C1:    _  x  x  x  x  x
        C2:    _  _  x  x  x  x
        C3:    _  _  _  x  x  x

    Args:
        num_codebooks: Number of codebooks (typically 8 for EnCodec). Default: 8.
        delay: Delay between codebook predictions. Default: 1.
    """

    def __init__(self, num_codebooks: int = 8, delay: int = 1):
        self.num_codebooks = num_codebooks
        self.delay = delay

    def build_pattern(self, seq_len: int) -> torch.Tensor:
        """Build delay pattern mask.

        Args:
            seq_len: Sequence length.

        Returns:
            Pattern mask, shape [num_codebooks, seq_len + total_delay].
        """
        total_delay = self.delay * (self.num_codebooks - 1)
        extended_len = seq_len + total_delay

        pattern = torch.zeros(self.num_codebooks, extended_len, dtype=torch.long)

        for k in range(self.num_codebooks):
            start = k * self.delay
            pattern[k, start:start + seq_len] = 1

        return pattern

    def revert_pattern(
        self, tokens: torch.Tensor, pattern: torch.Tensor
    ) -> torch.Tensor:
        """Revert delayed sequence to aligned codebook tokens.

        Args:
            tokens: Delayed token sequence, shape [B, num_codebooks, extended_len].
            pattern: Pattern mask from build_pattern.

        Returns:
            Aligned tokens, shape [B, num_codebooks, seq_len].
        """
        B, K, extended_len = tokens.shape
        seq_len = pattern[0].sum().item()

        aligned = torch.zeros(B, K, seq_len, dtype=tokens.dtype, device=tokens.device)

        for k in range(K):
            start = k * self.delay
            aligned[:, k, :] = tokens[:, k, start:start + seq_len]

        return aligned


class MelodyConditioner(nn.Module):
    """Conditions on melodic features (chromagram).

    Extracts chromagram from reference audio and projects it to
    conditioning signals that guide music generation.

    Args:
        chroma_dim: Chromagram feature dimension (typically 12). Default: 12.
        hidden_dim: Hidden dimension for projection. Default: 1024.
        sample_rate: Audio sample rate. Default: 32000.
    """

    def __init__(
        self,
        chroma_dim: int = 12,
        hidden_dim: int = 1024,
        sample_rate: int = 32000,
    ):
        super().__init__()
        self.chroma_dim = chroma_dim
        self.sample_rate = sample_rate

        # Project chromagram to hidden dimension
        self.proj = nn.Sequential(
            nn.Linear(chroma_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, chroma: torch.Tensor) -> torch.Tensor:
        """Project chromagram features.

        Args:
            chroma: Chromagram features, shape [B, T, chroma_dim].

        Returns:
            Melody conditioning, shape [B, T, hidden_dim].
        """
        return self.proj(chroma)


class MusicGenTransformer(nn.Module):
    """Transformer decoder for music generation with delayed pattern.

    Args:
        vocab_size: Codebook vocabulary size. Default: 2048.
        num_codebooks: Number of codebooks. Default: 8.
        hidden_dim: Hidden dimension. Default: 1024.
        num_layers: Number of transformer layers. Default: 24.
        num_heads: Number of attention heads. Default: 16.
        mlp_ratio: MLP expansion ratio. Default: 4.0.
        dropout: Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        vocab_size: int = 2048,
        num_codebooks: int = 8,
        hidden_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.hidden_dim = hidden_dim

        # Separate embeddings for each codebook
        self.codebook_embeds = nn.ModuleList([
            nn.Embedding(vocab_size + 1, hidden_dim)  # +1 for padding/special token
            for _ in range(num_codebooks)
        ])

        # Positional embedding
        self.pos_embed = nn.Embedding(4096, hidden_dim)  # Max sequence length

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Output heads for each codebook
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size)
            for _ in range(num_codebooks)
        ])

    def forward(
        self,
        tokens: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: Codebook tokens, shape [B, num_codebooks, seq_len].
            condition: Optional conditioning (text + melody), shape [B, cond_len, hidden_dim].

        Returns:
            Logits for each codebook, shape [B, num_codebooks, seq_len, vocab_size].
        """
        B, K, seq_len = tokens.shape
        device = tokens.device

        # Embed tokens from each codebook and sum
        x = torch.zeros(B, seq_len, self.hidden_dim, device=device)
        for k in range(K):
            x = x + self.codebook_embeds[k](tokens[:, k, :])

        # Add positional embedding
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos)

        # Prepend conditioning if provided
        if condition is not None:
            x = torch.cat([condition, x], dim=1)
            cond_len = condition.shape[1]
        else:
            cond_len = 0

        # Create causal mask
        total_len = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(total_len, total_len, device=device, dtype=torch.bool),
            diagonal=1
        )

        # Transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask)

        # Extract sequence tokens (remove conditioning)
        x = x[:, cond_len:, :]
        x = self.norm(x)

        # Predict logits for each codebook
        logits = torch.stack([head(x) for head in self.output_heads], dim=1)

        return logits


class TransformerLayer(nn.Module):
    """Standard transformer decoder layer."""

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

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class MusicGen(NexusModule):
    """MusicGen: Controllable music generation with language models.

    Generates music from text descriptions and optional melody conditioning
    using a transformer language model with delayed pattern prediction.

    Args:
        vocab_size: EnCodec codebook size. Default: 2048.
        num_codebooks: Number of EnCodec codebooks. Default: 8.
        hidden_dim: Transformer hidden dimension. Default: 1024.
        num_layers: Number of transformer layers. Default: 24.
        num_heads: Number of attention heads. Default: 16.
        text_dim: Text embedding dimension. Default: 768.
        use_melody: Enable melody conditioning. Default: True.
        delay: Delay for pattern provider. Default: 1.
    """

    def __init__(
        self,
        vocab_size: int = 2048,
        num_codebooks: int = 8,
        hidden_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        text_dim: int = 768,
        use_melody: bool = True,
        delay: int = 1,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.use_melody = use_melody

        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Melody conditioner
        if use_melody:
            self.melody_conditioner = MelodyConditioner(hidden_dim=hidden_dim)

        # Delayed pattern provider
        self.pattern_provider = DelayedPatternProvider(num_codebooks, delay)

        # Transformer
        self.transformer = MusicGenTransformer(
            vocab_size, num_codebooks, hidden_dim, num_layers, num_heads
        )

    def forward(
        self,
        tokens: torch.Tensor,
        text_embeds: torch.Tensor,
        chroma: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            tokens: Target codebook tokens, shape [B, num_codebooks, seq_len].
            text_embeds: Text embeddings, shape [B, text_len, text_dim].
            chroma: Optional chromagram, shape [B, seq_len, 12].

        Returns:
            Dictionary with loss and predictions.
        """
        B, K, seq_len = tokens.shape
        device = tokens.device

        # Build delay pattern
        pattern = self.pattern_provider.build_pattern(seq_len).to(device)

        # Apply pattern to create delayed input sequence
        extended_len = pattern.shape[1]
        tokens_delayed = torch.zeros(
            B, K, extended_len, dtype=tokens.dtype, device=device
        )
        for k in range(K):
            start = k * self.pattern_provider.delay
            tokens_delayed[:, k, start:start + seq_len] = tokens[:, k, :]

        # Prepare conditioning
        text_cond = self.text_proj(text_embeds)

        if self.use_melody and chroma is not None:
            melody_cond = self.melody_conditioner(chroma)
            # Concatenate text and melody conditioning
            condition = torch.cat([text_cond, melody_cond], dim=1)
        else:
            condition = text_cond

        # Forward through transformer (teacher forcing)
        logits = self.transformer(tokens_delayed[:, :, :-1], condition)

        # Compute loss (cross-entropy for each codebook)
        total_loss = 0
        for k in range(K):
            # Get target tokens for this codebook with proper alignment
            start = k * self.pattern_provider.delay
            target = tokens_delayed[:, k, start + 1:start + seq_len + 1]

            # Crop logits to match target length
            logits_k = logits[:, k, start:start + seq_len, :]

            # Cross-entropy loss
            loss_k = F.cross_entropy(
                logits_k.reshape(-1, self.vocab_size),
                target.reshape(-1),
                ignore_index=self.vocab_size  # Padding token
            )
            total_loss += loss_k

        total_loss = total_loss / K

        return {
            "loss": total_loss,
            "loss_dict": {"loss": total_loss.item()},
        }

    @torch.no_grad()
    def generate(
        self,
        text_embeds: torch.Tensor,
        chroma: Optional[torch.Tensor] = None,
        duration: int = 250,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
    ) -> torch.Tensor:
        """Generate music from text and optional melody.

        Args:
            text_embeds: Text embeddings, shape [B, text_len, text_dim].
            chroma: Optional chromagram for melody guidance, shape [B, duration, 12].
            duration: Duration in codec frames (50Hz). Default: 250 (5 seconds).
            temperature: Sampling temperature. Default: 1.0.
            top_k: Top-k sampling. Default: 250.
            top_p: Nucleus sampling threshold. Default: 0.0 (disabled).

        Returns:
            Generated codebook tokens, shape [B, num_codebooks, duration].
        """
        B = text_embeds.shape[0]
        device = next(self.parameters()).device

        # Prepare conditioning
        text_cond = self.text_proj(text_embeds)

        if self.use_melody and chroma is not None:
            melody_cond = self.melody_conditioner(chroma)
            condition = torch.cat([text_cond, melody_cond], dim=1)
        else:
            condition = text_cond

        # Build pattern
        pattern = self.pattern_provider.build_pattern(duration).to(device)
        extended_len = pattern.shape[1]

        # Initialize token sequence
        tokens = torch.full(
            (B, self.num_codebooks, extended_len),
            self.vocab_size,  # Padding token
            dtype=torch.long,
            device=device
        )

        # Generate autoregressively
        for t in range(extended_len):
            # Check which codebooks are active at this timestep
            active_codebooks = (pattern[:, t] == 1).nonzero(as_tuple=False).squeeze(-1)

            if len(active_codebooks) == 0:
                continue

            # Get logits from transformer
            logits = self.transformer(tokens[:, :, :t + 1], condition)

            # Sample for each active codebook
            for k in active_codebooks:
                logits_k = logits[:, k, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits_k < torch.topk(logits_k, top_k)[0][..., -1, None]
                    logits_k[indices_to_remove] = float('-inf')

                # Top-p filtering
                if top_p > 0:
                    sorted_logits, sorted_indices = torch.sort(logits_k, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits_k[indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(logits_k, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                tokens[:, k, t] = next_token

        # Revert pattern to get aligned tokens
        tokens_aligned = self.pattern_provider.revert_pattern(tokens, pattern)

        return tokens_aligned
