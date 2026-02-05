"""
VALL-E: Neural Codec Language Modeling for Text-to-Speech.

Treats text-to-speech as a conditional language modeling problem
over discrete neural audio codec tokens (e.g., from EnCodec).
The model generates audio by predicting sequences of codec tokens
conditioned on phoneme/text input and an acoustic prompt.

Architecture:
- Autoregressive (AR) decoder: Generates the first codebook level
  token-by-token, conditioned on text and acoustic prompt.
- Non-autoregressive (NAR) decoder: Generates remaining codebook
  levels in parallel, conditioned on text, acoustic prompt, and
  previously generated codebook levels.

This two-stage approach balances quality (AR for coarse structure)
with efficiency (NAR for fine details).

Reference: "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
           Wang et al., 2023 (https://arxiv.org/abs/2301.02111)
"""

from typing import Dict, Any, Optional, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule


class CodecTokenizer(NexusModule):
    """Interface for neural audio codec tokenization (EnCodec-style).

    Wraps a pre-trained neural audio codec (such as EnCodec or SoundStream)
    that converts raw audio waveforms into discrete token sequences across
    multiple codebook levels via residual vector quantization (RVQ).

    Each codebook level captures progressively finer acoustic details:
    - Level 0: Coarse structure (prosody, speaker identity)
    - Level 1-7: Fine details (spectral detail, harmonics)

    This module provides the encode/decode interface. In practice, the
    actual codec model would be loaded from a pre-trained checkpoint.

    Args:
        config: Dictionary containing tokenizer hyperparameters.
            - codec_dim (int): Codebook embedding dimension. Default: 8.
            - num_codebooks (int): Number of RVQ codebook levels. Default: 8.
            - codebook_size (int): Number of entries per codebook. Default: 1024.
            - sample_rate (int): Audio sample rate in Hz. Default: 24000.
            - hop_length (int): Codec frame hop length. Default: 320.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.codec_dim = config.get("codec_dim", 8)
        self.num_codebooks = config.get("num_codebooks", 8)
        self.codebook_size = config.get("codebook_size", 1024)
        self.sample_rate = config.get("sample_rate", 24000)
        self.hop_length = config.get("hop_length", 320)

        # Codebook embeddings (one embedding table per codebook level)
        self.codebooks = nn.ModuleList([
            nn.Embedding(self.codebook_size, self.codec_dim)
            for _ in range(self.num_codebooks)
        ])

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to codec token indices.

        In practice this would run the full codec encoder + RVQ.
        Here we provide the interface specification.

        Args:
            audio: Waveform of shape (B, T_samples).

        Returns:
            Token indices of shape (B, num_codebooks, T_frames).
        """
        raise NotImplementedError(
            "CodecTokenizer.encode requires a pre-trained codec model. "
            "Load a pre-trained EnCodec or SoundStream checkpoint."
        )

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode codec token indices back to audio waveform.

        Args:
            tokens: Token indices of shape (B, num_codebooks, T_frames).

        Returns:
            Reconstructed waveform of shape (B, T_samples).
        """
        raise NotImplementedError(
            "CodecTokenizer.decode requires a pre-trained codec model."
        )

    def embed_tokens(self, tokens: torch.Tensor, codebook_idx: int) -> torch.Tensor:
        """Look up codebook embeddings for a specific codebook level.

        Args:
            tokens: Token indices of shape (B, T).
            codebook_idx: Which codebook level to embed from.

        Returns:
            Embeddings of shape (B, T, codec_dim).
        """
        return self.codebooks[codebook_idx](tokens)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed tokens from all codebook levels and sum.

        Args:
            tokens: Token indices of shape (B, num_codebooks, T).

        Returns:
            Summed embeddings of shape (B, T, codec_dim).
        """
        embeddings = torch.zeros(
            tokens.shape[0], tokens.shape[2], self.codec_dim,
            device=tokens.device, dtype=torch.float32
        )
        for i in range(self.num_codebooks):
            embeddings = embeddings + self.codebooks[i](tokens[:, i, :])
        return embeddings


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input of shape (B, T, D).

        Returns:
            Output with positional encoding added.
        """
        return x + self.pe[:, :x.shape[1], :]


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm architecture.

    Args:
        d_model: Hidden dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        dropout: Dropout probability.
        causal: Whether to use causal (autoregressive) attention mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate a causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask

    def forward(
        self,
        x: torch.Tensor,
        cross_attn_kv: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, T, D).
            cross_attn_kv: Optional cross-attention key/value source.
            key_padding_mask: Optional padding mask.

        Returns:
            Output of shape (B, T, D).
        """
        # Self-attention with optional causal masking
        x_norm = self.norm1(x)
        attn_mask = None
        if self.causal:
            attn_mask = self._get_causal_mask(x.shape[1], x.device)

        if cross_attn_kv is not None:
            attn_out, _ = self.attn(
                x_norm, cross_attn_kv, cross_attn_kv,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        else:
            attn_out, _ = self.attn(
                x_norm, x_norm, x_norm,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        x = x + attn_out

        # Feed-forward
        x = x + self.mlp(self.norm2(x))
        return x


class AutoregressiveDecoder(NexusModule):
    """Autoregressive decoder for the first codebook level.

    Generates codec tokens at the first (coarsest) codebook level
    one token at a time, conditioned on:
    - Text/phoneme embeddings
    - Acoustic prompt (a few seconds of reference audio codec tokens)

    Uses causal (left-to-right) attention to ensure autoregressive
    generation. The text and prompt are prepended to the target sequence
    as prefix context.

    Reference: "VALL-E" (https://arxiv.org/abs/2301.02111)

    Args:
        config: Dictionary containing hyperparameters.
            - vocab_size (int): Text vocabulary size.
            - codebook_size (int): Codec codebook size. Default: 1024.
            - d_model (int): Hidden dimension. Default: 1024.
            - num_layers (int): Number of transformer layers. Default: 12.
            - num_heads (int): Number of attention heads. Default: 16.
            - dropout (float): Dropout probability. Default: 0.1.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.vocab_size = config["vocab_size"]
        self.codebook_size = config.get("codebook_size", 1024)
        self.d_model = config.get("d_model", 1024)
        self.num_layers = config.get("num_layers", 12)
        self.num_heads = config.get("num_heads", 16)
        self.dropout = config.get("dropout", 0.1)

        # Text embedding
        self.text_embed = nn.Embedding(self.vocab_size, self.d_model)

        # Audio codec token embedding (first codebook level)
        self.audio_embed = nn.Embedding(self.codebook_size + 1, self.d_model)  # +1 for BOS

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(self.d_model)

        # Transformer layers (causal)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dropout=self.dropout,
                causal=True,
            )
            for _ in range(self.num_layers)
        ])

        # Output projection to codebook tokens
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.codebook_size)

        # Special tokens
        self.bos_token_id = self.codebook_size  # Use last index as BOS

    def forward(
        self,
        text_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training (teacher-forced).

        Constructs the input sequence as:
            [text_embeddings | prompt_codec_embeddings | BOS | target_codec_embeddings]

        Args:
            text_tokens: Text/phoneme token indices of shape (B, T_text).
            prompt_tokens: Acoustic prompt codec tokens (level 0) of shape (B, T_prompt).
            target_tokens: Target codec tokens (level 0) of shape (B, T_target).
                If None, only processes text + prompt (for generation).
            text_padding_mask: Padding mask for text of shape (B, T_text).

        Returns:
            Dictionary with:
                - "logits": Token logits of shape (B, T_target, codebook_size).
                - "hidden_states": Final hidden states.
        """
        B = text_tokens.shape[0]
        device = text_tokens.device

        # Embed text
        text_emb = self.text_embed(text_tokens)  # (B, T_text, D)

        # Embed prompt codec tokens
        prompt_emb = self.audio_embed(prompt_tokens)  # (B, T_prompt, D)

        # BOS token
        bos = self.audio_embed(
            torch.full((B, 1), self.bos_token_id, device=device, dtype=torch.long)
        )  # (B, 1, D)

        if target_tokens is not None:
            # Teacher-forced training: embed target tokens (shifted right)
            target_emb = self.audio_embed(target_tokens[:, :-1])  # (B, T_target-1, D)
            sequence = torch.cat([text_emb, prompt_emb, bos, target_emb], dim=1)
        else:
            sequence = torch.cat([text_emb, prompt_emb, bos], dim=1)

        # Add positional encoding
        sequence = self.pos_enc(sequence)

        # Process through transformer
        for layer in self.layers:
            sequence = layer(sequence)

        # Extract the target region (after text + prompt + BOS)
        prefix_len = text_tokens.shape[1] + prompt_tokens.shape[1] + 1
        if target_tokens is not None:
            target_hidden = sequence[:, prefix_len - 1:prefix_len - 1 + target_tokens.shape[1], :]
        else:
            target_hidden = sequence[:, -1:, :]

        # Project to logits
        target_hidden = self.output_norm(target_hidden)
        logits = self.output_proj(target_hidden)

        return {
            "logits": logits,
            "hidden_states": sequence,
        }

    @torch.no_grad()
    def generate(
        self,
        text_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        max_length: int = 1000,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively generate first codebook level tokens.

        Args:
            text_tokens: Text/phoneme tokens of shape (B, T_text).
            prompt_tokens: Acoustic prompt tokens of shape (B, T_prompt).
            max_length: Maximum generation length.
            temperature: Sampling temperature.
            top_k: Top-k sampling (0 = disabled).
            top_p: Nucleus sampling threshold (1.0 = disabled).

        Returns:
            Generated codec tokens of shape (B, T_generated).
        """
        B = text_tokens.shape[0]
        device = text_tokens.device

        generated = torch.full(
            (B, 1), self.bos_token_id, device=device, dtype=torch.long
        )

        for _ in range(max_length):
            # Forward pass
            output = self.forward(text_tokens, prompt_tokens, target_tokens=None)
            # The implementation above needs target_tokens for proper indexing
            # in generation mode. We re-implement the core loop here.

            text_emb = self.text_embed(text_tokens)
            prompt_emb = self.audio_embed(prompt_tokens)
            gen_emb = self.audio_embed(generated)

            sequence = torch.cat([text_emb, prompt_emb, gen_emb], dim=1)
            sequence = self.pos_enc(sequence)

            for layer in self.layers:
                sequence = layer(sequence)

            # Get logits for the last position
            last_hidden = self.output_norm(sequence[:, -1:, :])
            logits = self.output_proj(last_hidden).squeeze(1)  # (B, codebook_size)

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            generated = torch.cat([generated, next_token], dim=1)

        # Remove BOS token
        return generated[:, 1:]


class NonAutoregressiveDecoder(NexusModule):
    """Non-autoregressive decoder for remaining codebook levels.

    Generates codec tokens at codebook levels 1 through (num_codebooks-1)
    in parallel, conditioned on:
    - Text/phoneme embeddings
    - Acoustic prompt tokens (all levels)
    - Previously generated codebook levels

    Each level is generated conditioned on all levels below it,
    enabling fast parallel generation of fine acoustic details.

    Reference: "VALL-E" (https://arxiv.org/abs/2301.02111)

    Args:
        config: Dictionary containing hyperparameters.
            - vocab_size (int): Text vocabulary size.
            - codebook_size (int): Codec codebook size. Default: 1024.
            - num_codebooks (int): Total number of codebook levels. Default: 8.
            - d_model (int): Hidden dimension. Default: 1024.
            - num_layers (int): Number of transformer layers. Default: 12.
            - num_heads (int): Number of attention heads. Default: 16.
            - dropout (float): Dropout probability. Default: 0.1.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.vocab_size = config["vocab_size"]
        self.codebook_size = config.get("codebook_size", 1024)
        self.num_codebooks = config.get("num_codebooks", 8)
        self.d_model = config.get("d_model", 1024)
        self.num_layers = config.get("num_layers", 12)
        self.num_heads = config.get("num_heads", 16)
        self.dropout = config.get("dropout", 0.1)

        # Text embedding
        self.text_embed = nn.Embedding(self.vocab_size, self.d_model)

        # Per-codebook audio embeddings
        self.audio_embeds = nn.ModuleList([
            nn.Embedding(self.codebook_size, self.d_model)
            for _ in range(self.num_codebooks)
        ])

        # Codebook level embedding (to indicate which level is being predicted)
        self.level_embed = nn.Embedding(self.num_codebooks, self.d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(self.d_model)

        # Transformer layers (bidirectional - no causal mask)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dropout=self.dropout,
                causal=False,
            )
            for _ in range(self.num_layers)
        ])

        # Per-level output projections
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_projs = nn.ModuleList([
            nn.Linear(self.d_model, self.codebook_size)
            for _ in range(self.num_codebooks - 1)  # Levels 1 to num_codebooks-1
        ])

    def forward(
        self,
        text_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        codec_tokens: torch.Tensor,
        target_level: int,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for one codebook level.

        Args:
            text_tokens: Text/phoneme tokens of shape (B, T_text).
            prompt_tokens: Prompt codec tokens (all levels) of shape (B, num_codebooks, T_prompt).
            codec_tokens: Previously generated codec tokens of shape (B, levels_so_far, T_audio).
            target_level: Which codebook level to predict (1 to num_codebooks-1).
            text_padding_mask: Padding mask for text.

        Returns:
            Dictionary with:
                - "logits": Token logits of shape (B, T_audio, codebook_size).
                - "hidden_states": Final hidden states.
        """
        B = text_tokens.shape[0]
        device = text_tokens.device
        T_audio = codec_tokens.shape[2]

        # Embed text
        text_emb = self.text_embed(text_tokens)

        # Embed and sum prompt tokens across all codebook levels
        prompt_emb = torch.zeros(
            B, prompt_tokens.shape[2], self.d_model, device=device
        )
        for i in range(self.num_codebooks):
            prompt_emb = prompt_emb + self.audio_embeds[i](prompt_tokens[:, i, :])

        # Embed and sum previously generated codebook levels
        audio_emb = torch.zeros(B, T_audio, self.d_model, device=device)
        num_levels = codec_tokens.shape[1]
        for i in range(num_levels):
            audio_emb = audio_emb + self.audio_embeds[i](codec_tokens[:, i, :])

        # Add level embedding to audio tokens
        level_emb = self.level_embed(
            torch.full((1,), target_level, device=device, dtype=torch.long)
        )
        audio_emb = audio_emb + level_emb

        # Concatenate: [text | prompt | audio]
        sequence = torch.cat([text_emb, prompt_emb, audio_emb], dim=1)
        sequence = self.pos_enc(sequence)

        # Process through bidirectional transformer
        for layer in self.layers:
            sequence = layer(sequence)

        # Extract audio region
        audio_start = text_tokens.shape[1] + prompt_tokens.shape[2]
        audio_hidden = sequence[:, audio_start:audio_start + T_audio, :]
        audio_hidden = self.output_norm(audio_hidden)

        # Project to logits for the target level
        logits = self.output_projs[target_level - 1](audio_hidden)

        return {
            "logits": logits,
            "hidden_states": audio_hidden,
        }

    @torch.no_grad()
    def generate_level(
        self,
        text_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        codec_tokens: torch.Tensor,
        target_level: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate tokens for one codebook level in parallel (non-autoregressive).

        Args:
            text_tokens: Text tokens of shape (B, T_text).
            prompt_tokens: Prompt tokens of shape (B, num_codebooks, T_prompt).
            codec_tokens: Previously generated tokens of shape (B, levels_so_far, T_audio).
            target_level: Codebook level to generate.
            temperature: Sampling temperature.

        Returns:
            Generated tokens of shape (B, T_audio).
        """
        output = self.forward(text_tokens, prompt_tokens, codec_tokens, target_level)
        logits = output["logits"] / temperature
        probs = F.softmax(logits, dim=-1)

        # Sample from the distribution for each position independently
        B, T, V = probs.shape
        tokens = torch.multinomial(probs.view(-1, V), num_samples=1).view(B, T)
        return tokens


class VALLE(NexusModule):
    """VALL-E: Neural Codec Language Model for Zero-Shot Text-to-Speech.

    A two-stage model that generates speech from text by predicting
    neural audio codec tokens:

    Stage 1 (AR): Generates first codebook level autoregressively,
    capturing coarse prosodic and speaker characteristics.

    Stage 2 (NAR): Generates remaining codebook levels in parallel,
    filling in fine acoustic details conditioned on the coarse tokens.

    Zero-shot capability: Given a 3-second acoustic prompt of an unseen
    speaker, VALL-E can synthesize speech in that speaker's voice for
    arbitrary text, without fine-tuning.

    Reference: "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
               Wang et al., 2023 (https://arxiv.org/abs/2301.02111)

    Args:
        config: Dictionary containing model hyperparameters.
            - vocab_size (int): Text/phoneme vocabulary size. Required.
            - codec_dim (int): Codec embedding dimension. Default: 8.
            - num_codebooks (int): Number of RVQ codebook levels. Default: 8.
            - codebook_size (int): Entries per codebook. Default: 1024.
            - d_model (int): Transformer hidden dimension. Default: 1024.
            - num_layers (int): Transformer layers per decoder. Default: 12.
            - num_heads (int): Attention heads. Default: 16.
            - dropout (float): Dropout probability. Default: 0.1.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.vocab_size = config["vocab_size"]
        self.num_codebooks = config.get("num_codebooks", 8)
        self.codebook_size = config.get("codebook_size", 1024)
        self.d_model = config.get("d_model", 1024)

        # Codec tokenizer interface
        self.codec = CodecTokenizer(config)

        # Autoregressive decoder (first codebook level)
        self.ar_decoder = AutoregressiveDecoder(config)

        # Non-autoregressive decoder (remaining codebook levels)
        self.nar_decoder = NonAutoregressiveDecoder(config)

    def forward(
        self,
        text_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training both stages.

        Args:
            text_tokens: Text/phoneme token indices of shape (B, T_text).
            prompt_tokens: Acoustic prompt codec tokens of shape (B, num_codebooks, T_prompt).
            target_tokens: Target audio codec tokens of shape (B, num_codebooks, T_target).
            text_padding_mask: Optional text padding mask.

        Returns:
            Dictionary with:
                - "ar_logits": AR decoder logits for level 0.
                - "nar_logits": List of NAR decoder logits for levels 1+.
                - "ar_loss": AR cross-entropy loss.
                - "nar_loss": NAR cross-entropy loss (averaged over levels).
                - "loss": Total combined loss.
        """
        B, K, T_target = target_tokens.shape
        device = text_tokens.device

        # Stage 1: AR decoder for first codebook level
        ar_output = self.ar_decoder(
            text_tokens=text_tokens,
            prompt_tokens=prompt_tokens[:, 0, :],
            target_tokens=target_tokens[:, 0, :],
            text_padding_mask=text_padding_mask,
        )
        ar_logits = ar_output["logits"]

        # AR loss: cross-entropy on first codebook level
        ar_loss = F.cross_entropy(
            ar_logits.reshape(-1, self.codebook_size),
            target_tokens[:, 0, :].reshape(-1),
        )

        # Stage 2: NAR decoder for remaining codebook levels
        nar_logits_list = []
        nar_loss = torch.tensor(0.0, device=device)

        for level in range(1, self.num_codebooks):
            # Condition on all previously generated levels
            codec_context = target_tokens[:, :level, :]

            nar_output = self.nar_decoder(
                text_tokens=text_tokens,
                prompt_tokens=prompt_tokens,
                codec_tokens=codec_context,
                target_level=level,
                text_padding_mask=text_padding_mask,
            )
            nar_logits = nar_output["logits"]
            nar_logits_list.append(nar_logits)

            # NAR loss for this level
            level_loss = F.cross_entropy(
                nar_logits.reshape(-1, self.codebook_size),
                target_tokens[:, level, :].reshape(-1),
            )
            nar_loss = nar_loss + level_loss

        nar_loss = nar_loss / max(self.num_codebooks - 1, 1)

        # Combined loss
        total_loss = ar_loss + nar_loss

        return {
            "ar_logits": ar_logits,
            "nar_logits": nar_logits_list,
            "ar_loss": ar_loss,
            "nar_loss": nar_loss,
            "loss": total_loss,
        }

    def compute_loss(
        self,
        text_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss (alias for forward).

        Args:
            text_tokens: Text tokens of shape (B, T_text).
            prompt_tokens: Prompt codec tokens of shape (B, num_codebooks, T_prompt).
            target_tokens: Target codec tokens of shape (B, num_codebooks, T_target).
            text_padding_mask: Optional text padding mask.

        Returns:
            Dictionary with loss values and logits.
        """
        return self.forward(text_tokens, prompt_tokens, target_tokens, text_padding_mask)

    @torch.no_grad()
    def generate(
        self,
        text_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        max_length: int = 1000,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        nar_temperature: float = 0.5,
    ) -> torch.Tensor:
        """Generate audio codec tokens from text and acoustic prompt.

        Two-stage generation:
        1. AR decoder generates first codebook level autoregressively
        2. NAR decoder generates remaining levels in parallel

        Args:
            text_tokens: Text/phoneme tokens of shape (B, T_text).
            prompt_tokens: Acoustic prompt tokens of shape (B, num_codebooks, T_prompt).
            max_length: Maximum length for AR generation.
            temperature: AR sampling temperature.
            top_k: AR top-k sampling.
            top_p: AR nucleus sampling threshold.
            nar_temperature: NAR sampling temperature.

        Returns:
            Generated codec tokens of shape (B, num_codebooks, T_generated).
        """
        B = text_tokens.shape[0]
        device = text_tokens.device

        # Stage 1: Generate first codebook level (autoregressive)
        level_0_tokens = self.ar_decoder.generate(
            text_tokens=text_tokens,
            prompt_tokens=prompt_tokens[:, 0, :],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )  # (B, T_generated)

        T_generated = level_0_tokens.shape[1]

        # Initialize all codebook levels
        all_tokens = torch.zeros(
            B, self.num_codebooks, T_generated,
            device=device, dtype=torch.long,
        )
        all_tokens[:, 0, :] = level_0_tokens

        # Stage 2: Generate remaining levels (non-autoregressive)
        for level in range(1, self.num_codebooks):
            level_tokens = self.nar_decoder.generate_level(
                text_tokens=text_tokens,
                prompt_tokens=prompt_tokens,
                codec_tokens=all_tokens[:, :level, :],
                target_level=level,
                temperature=nar_temperature,
            )
            all_tokens[:, level, :] = level_tokens

        return all_tokens
