"""
SoundStorm: Efficient Parallel Audio Generation.

Implements the SoundStorm model for fast, parallel audio generation
from semantic tokens. Uses a bidirectional transformer with
confidence-based parallel decoding (MaskGIT-style) to generate
neural audio codec tokens iteratively.

Unlike autoregressive models that generate tokens one at a time,
SoundStorm generates all tokens simultaneously and iteratively
unmasks the most confident predictions, achieving orders of
magnitude speedup while maintaining quality.

Architecture:
- BidirectionalTransformer: Full bidirectional attention over all
  audio tokens across all codebook levels.
- ConfidenceBasedDecoder: Iterative parallel decoding that unmasks
  tokens from most to least confident.
- SoundStorm: Full model that takes semantic tokens and generates
  multi-level codec tokens.

Reference: "SoundStorm: Efficient Parallel Audio Generation"
           Borsos et al., 2023 (https://arxiv.org/abs/2305.09636)
"""

from typing import Dict, Any, Optional, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

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
        return x + self.pe[:, :x.shape[1], :]


class BidirectionalTransformerBlock(nn.Module):
    """Bidirectional transformer block with pre-norm.

    Processes the full sequence with bidirectional (non-causal) attention,
    allowing each token to attend to all other tokens regardless of
    position. This is essential for the parallel decoding strategy.

    Args:
        d_model: Hidden dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
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

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, T, D).
            key_padding_mask: Optional padding mask.

        Returns:
            Output of shape (B, T, D).
        """
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class BidirectionalTransformer(NexusModule):
    """Bidirectional transformer for processing flattened audio tokens.

    Processes a sequence of audio codec tokens (flattened across time
    and codebook levels) with full bidirectional attention. The tokens
    are enriched with positional, codebook level, and mask embeddings.

    The flattening scheme interleaves codebook levels:
        [t0_level0, t0_level1, ..., t0_levelK, t1_level0, ...]

    This allows the model to jointly reason about all codebook levels
    at each time step.

    Args:
        config: Dictionary containing hyperparameters.
            - d_model (int): Hidden dimension. Default: 1024.
            - num_layers (int): Number of transformer layers. Default: 12.
            - num_heads (int): Number of attention heads. Default: 16.
            - num_codebooks (int): Number of codec codebook levels. Default: 8.
            - codebook_size (int): Entries per codebook. Default: 1024.
            - dropout (float): Dropout probability. Default: 0.1.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.d_model = config.get("d_model", 1024)
        self.num_layers = config.get("num_layers", 12)
        self.num_heads = config.get("num_heads", 16)
        self.num_codebooks = config.get("num_codebooks", 8)
        self.codebook_size = config.get("codebook_size", 1024)
        self.dropout = config.get("dropout", 0.1)

        # Per-codebook token embeddings
        self.token_embeds = nn.ModuleList([
            nn.Embedding(self.codebook_size + 1, self.d_model)  # +1 for mask token
            for _ in range(self.num_codebooks)
        ])

        # Codebook level embedding
        self.level_embed = nn.Embedding(self.num_codebooks, self.d_model)

        # Positional encoding (for temporal position)
        self.pos_enc = SinusoidalPositionalEncoding(self.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            BidirectionalTransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        ])

        self.output_norm = nn.LayerNorm(self.d_model)

        # Per-codebook output projections
        self.output_projs = nn.ModuleList([
            nn.Linear(self.d_model, self.codebook_size)
            for _ in range(self.num_codebooks)
        ])

        # Mask token ID (last index in extended vocabulary)
        self.mask_token_id = self.codebook_size

    def embed_tokens(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed codec tokens with level and positional information.

        Args:
            tokens: Codec token indices of shape (B, num_codebooks, T).
            mask: Boolean mask of shape (B, num_codebooks, T) where True
                indicates masked positions.

        Returns:
            Embedded sequence of shape (B, T * num_codebooks, d_model).
        """
        B, K, T = tokens.shape
        device = tokens.device

        # Replace masked positions with mask token
        if mask is not None:
            tokens = tokens.clone()
            tokens[mask] = self.mask_token_id

        # Embed each codebook level and interleave
        embeddings = []
        for t in range(T):
            for k in range(K):
                tok_emb = self.token_embeds[k](tokens[:, k, t])  # (B, D)
                lvl_emb = self.level_embed(
                    torch.full((B,), k, device=device, dtype=torch.long)
                )  # (B, D)
                embeddings.append(tok_emb + lvl_emb)

        # Stack: (B, T * K, D)
        sequence = torch.stack(embeddings, dim=1)

        # Add temporal positional encoding
        # Each time step has K tokens, so repeat the position embedding
        pos_indices = torch.arange(T, device=device).repeat_interleave(K).unsqueeze(0)
        pos_emb = self.pos_enc.pe[:, :T, :].repeat_interleave(K, dim=1)
        sequence = sequence + pos_emb[:, :sequence.shape[1], :]

        return sequence

    def forward(
        self,
        tokens: torch.Tensor,
        semantic_tokens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the bidirectional transformer.

        Args:
            tokens: Codec tokens of shape (B, num_codebooks, T).
            semantic_tokens: Optional semantic condition of shape (B, T_semantic, D)
                (pre-embedded from a semantic model like SoundStream/w2v-BERT).
            mask: Boolean mask of shape (B, num_codebooks, T) for masked positions.
            key_padding_mask: Optional padding mask.

        Returns:
            Dictionary with:
                - "logits": Per-codebook logits, list of (B, T, codebook_size).
                - "hidden_states": Final hidden states.
        """
        B, K, T = tokens.shape

        # Embed tokens
        sequence = self.embed_tokens(tokens, mask)

        # Optionally prepend semantic tokens as conditioning
        if semantic_tokens is not None:
            sequence = torch.cat([semantic_tokens, sequence], dim=1)

        # Process through transformer
        for layer in self.layers:
            sequence = layer(sequence, key_padding_mask)

        # Extract audio token region (skip semantic prefix if present)
        if semantic_tokens is not None:
            audio_hidden = sequence[:, semantic_tokens.shape[1]:, :]
        else:
            audio_hidden = sequence

        audio_hidden = self.output_norm(audio_hidden)

        # De-interleave and project to logits for each codebook level
        logits_list = []
        for k in range(K):
            # Extract every K-th token starting from position k
            level_hidden = audio_hidden[:, k::K, :]  # (B, T, D)
            logits = self.output_projs[k](level_hidden)  # (B, T, codebook_size)
            logits_list.append(logits)

        return {
            "logits": logits_list,
            "hidden_states": audio_hidden,
        }


class ConfidenceBasedDecoder:
    """Confidence-based parallel decoding (MaskGIT-style).

    Implements iterative parallel decoding where all tokens start masked,
    and in each iteration:
    1. Predict logits for all masked positions
    2. Compute confidence scores (max softmax probability)
    3. Unmask the most confident predictions
    4. Repeat with fewer masked positions

    The masking schedule determines what fraction of tokens to unmask
    at each iteration, following a cosine schedule from all-masked to
    all-unmasked.

    Reference: "SoundStorm" (https://arxiv.org/abs/2305.09636)
               "MaskGIT" (https://arxiv.org/abs/2202.04200)

    Args:
        num_iterations: Number of decoding iterations. Default: 8.
        temperature: Sampling temperature. Default: 1.0.
        schedule: Masking schedule type ('cosine', 'linear'). Default: 'cosine'.
    """

    def __init__(
        self,
        num_iterations: int = 8,
        temperature: float = 1.0,
        schedule: str = "cosine",
    ):
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.schedule = schedule

    def get_mask_ratio(self, iteration: int) -> float:
        """Compute the fraction of tokens that remain masked at each iteration.

        Uses a cosine or linear schedule to gradually unmask tokens.

        Args:
            iteration: Current iteration (0-indexed).

        Returns:
            Fraction of tokens to keep masked (1.0 = all masked, 0.0 = none).
        """
        progress = (iteration + 1) / self.num_iterations

        if self.schedule == "cosine":
            return math.cos(progress * math.pi / 2.0)
        elif self.schedule == "linear":
            return 1.0 - progress
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    @torch.no_grad()
    def decode(
        self,
        model: BidirectionalTransformer,
        initial_tokens: torch.Tensor,
        semantic_tokens: Optional[torch.Tensor] = None,
        codebook_order: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Iteratively decode all codec tokens using confidence-based unmasking.

        SoundStorm processes codebook levels in a coarse-to-fine order.
        Within each level, tokens are iteratively unmasked based on
        prediction confidence.

        Args:
            model: The BidirectionalTransformer model.
            initial_tokens: Starting tokens of shape (B, num_codebooks, T),
                typically all masked or with some levels pre-filled.
            semantic_tokens: Optional semantic conditioning of shape (B, T_s, D).
            codebook_order: Order in which to process codebook levels.
                Default: [0, 1, 2, ..., num_codebooks-1].

        Returns:
            Decoded codec tokens of shape (B, num_codebooks, T).
        """
        B, K, T = initial_tokens.shape
        device = initial_tokens.device

        if codebook_order is None:
            codebook_order = list(range(K))

        tokens = initial_tokens.clone()
        mask = torch.ones(B, K, T, dtype=torch.bool, device=device)

        # If some tokens are pre-filled (not mask_token_id), unmask them
        mask[tokens != model.mask_token_id] = False

        # Process each codebook level in order
        for level in codebook_order:
            # Get the number of masked tokens at this level
            level_mask = mask[:, level, :]  # (B, T)
            num_masked = level_mask.sum(dim=-1)  # (B,)

            if num_masked.max() == 0:
                continue

            # Iteratively unmask tokens at this level
            for iteration in range(self.num_iterations):
                # Forward pass with current tokens and mask
                output = model(tokens, semantic_tokens=semantic_tokens, mask=mask)
                level_logits = output["logits"][level]  # (B, T, codebook_size)

                # Apply temperature
                level_logits = level_logits / self.temperature

                # Compute confidence for masked positions
                probs = F.softmax(level_logits, dim=-1)
                confidence, predicted = probs.max(dim=-1)  # (B, T)

                # Only consider currently masked positions
                confidence[~level_mask] = float("inf")  # Already unmasked

                # Determine how many tokens to unmask
                target_ratio = self.get_mask_ratio(iteration)
                num_to_keep_masked = (target_ratio * num_masked.float()).long().clamp(min=0)

                # For each sample, unmask the most confident tokens
                for b in range(B):
                    if num_to_keep_masked[b] >= level_mask[b].sum():
                        continue

                    masked_indices = level_mask[b].nonzero(as_tuple=True)[0]
                    if len(masked_indices) == 0:
                        continue

                    masked_confidence = confidence[b, masked_indices]
                    num_to_unmask = len(masked_indices) - num_to_keep_masked[b]

                    if num_to_unmask <= 0:
                        continue

                    # Get indices of most confident masked tokens
                    _, top_indices = masked_confidence.topk(
                        min(num_to_unmask, len(masked_indices))
                    )
                    unmask_positions = masked_indices[top_indices]

                    # Sample tokens for unmasked positions
                    for pos in unmask_positions:
                        pos_probs = probs[b, pos, :]
                        sampled = torch.multinomial(pos_probs.unsqueeze(0), 1).squeeze()
                        tokens[b, level, pos] = sampled
                        mask[b, level, pos] = False

                # Update level mask
                level_mask = mask[:, level, :]

        return tokens


class SoundStorm(NexusModule):
    """SoundStorm: Efficient Parallel Audio Generation.

    Takes semantic tokens (from a speech representation model) and
    generates multi-level neural audio codec tokens using iterative
    parallel decoding. Achieves much faster generation than
    autoregressive approaches while maintaining high quality.

    The generation process:
    1. Receive semantic tokens as conditioning
    2. Initialize all codec tokens as masked
    3. For each codebook level (coarse to fine):
       a. Run bidirectional transformer on all tokens
       b. Predict tokens for masked positions
       c. Unmask most confident predictions
       d. Repeat for num_iterations
    4. Return fully decoded codec tokens

    Reference: "SoundStorm: Efficient Parallel Audio Generation"
               Borsos et al., 2023 (https://arxiv.org/abs/2305.09636)

    Args:
        config: Dictionary containing model hyperparameters.
            - d_model (int): Transformer hidden dimension. Default: 1024.
            - num_layers (int): Number of transformer layers. Default: 12.
            - num_heads (int): Number of attention heads. Default: 16.
            - num_codebooks (int): Number of codec levels. Default: 8.
            - codebook_size (int): Entries per codebook. Default: 1024.
            - num_iterations (int): Decoding iterations per level. Default: 8.
            - semantic_dim (int): Semantic token embedding dimension. Default: 1024.
            - dropout (float): Dropout probability. Default: 0.1.
            - temperature (float): Sampling temperature. Default: 1.0.
            - mask_schedule (str): Unmasking schedule ('cosine', 'linear'). Default: 'cosine'.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.d_model = config.get("d_model", 1024)
        self.num_codebooks = config.get("num_codebooks", 8)
        self.codebook_size = config.get("codebook_size", 1024)
        self.num_iterations = config.get("num_iterations", 8)
        self.semantic_dim = config.get("semantic_dim", 1024)
        self.temperature = config.get("temperature", 1.0)
        self.mask_schedule = config.get("mask_schedule", "cosine")

        # Semantic token projection
        self.semantic_proj = nn.Linear(self.semantic_dim, self.d_model)

        # Bidirectional transformer backbone
        self.transformer = BidirectionalTransformer(config)

        # Confidence-based decoder
        self.decoder = ConfidenceBasedDecoder(
            num_iterations=self.num_iterations,
            temperature=self.temperature,
            schedule=self.mask_schedule,
        )

    def forward(
        self,
        codec_tokens: torch.Tensor,
        semantic_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training with masked prediction.

        During training, random tokens are masked and the model predicts
        the masked token identities from the unmasked context.

        Args:
            codec_tokens: Ground truth codec tokens of shape (B, num_codebooks, T).
            semantic_tokens: Semantic condition embeddings of shape (B, T_semantic, semantic_dim).
            mask: Boolean mask of shape (B, num_codebooks, T).
                True = masked (to predict), False = unmasked (context).
                If None, a random mask is generated.

        Returns:
            Dictionary with:
                - "logits": Per-codebook logits, list of (B, T, codebook_size).
                - "loss": Masked prediction cross-entropy loss.
                - "accuracy": Prediction accuracy on masked tokens.
                - "hidden_states": Transformer hidden states.
        """
        B, K, T = codec_tokens.shape
        device = codec_tokens.device

        # Generate random mask if not provided
        if mask is None:
            mask = self._generate_training_mask(B, K, T, device)

        # Project semantic tokens
        semantic_emb = self.semantic_proj(semantic_tokens)

        # Forward through transformer
        output = self.transformer(
            tokens=codec_tokens,
            semantic_tokens=semantic_emb,
            mask=mask,
        )

        logits_list = output["logits"]

        # Compute loss only on masked positions
        total_loss = torch.tensor(0.0, device=device)
        total_correct = 0
        total_masked = 0

        for k in range(K):
            level_logits = logits_list[k]  # (B, T, codebook_size)
            level_targets = codec_tokens[:, k, :]  # (B, T)
            level_mask = mask[:, k, :]  # (B, T)

            if level_mask.any():
                # Masked positions only
                masked_logits = level_logits[level_mask]  # (N_masked, codebook_size)
                masked_targets = level_targets[level_mask]  # (N_masked,)

                level_loss = F.cross_entropy(masked_logits, masked_targets)
                total_loss = total_loss + level_loss

                # Accuracy
                predicted = masked_logits.argmax(dim=-1)
                total_correct += (predicted == masked_targets).sum().item()
                total_masked += masked_targets.numel()

        # Average over codebook levels
        total_loss = total_loss / K

        accuracy = total_correct / max(total_masked, 1)

        return {
            "logits": logits_list,
            "loss": total_loss,
            "accuracy": accuracy,
            "hidden_states": output["hidden_states"],
        }

    def _generate_training_mask(
        self,
        batch_size: int,
        num_codebooks: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate a random training mask with variable masking ratio.

        For each sample, randomly selects a masking ratio from a cosine
        schedule, then masks that fraction of tokens uniformly.

        Args:
            batch_size: Number of samples.
            num_codebooks: Number of codebook levels.
            seq_len: Sequence length.
            device: Target device.

        Returns:
            Boolean mask of shape (B, num_codebooks, seq_len).
        """
        mask = torch.zeros(
            batch_size, num_codebooks, seq_len,
            dtype=torch.bool, device=device,
        )

        for b in range(batch_size):
            # Sample a random masking ratio
            ratio = torch.rand(1, device=device).item()
            # Apply cosine schedule to get actual mask ratio
            mask_ratio = math.cos(ratio * math.pi / 2.0)

            num_to_mask = int(mask_ratio * seq_len)
            if num_to_mask > 0:
                for k in range(num_codebooks):
                    indices = torch.randperm(seq_len, device=device)[:num_to_mask]
                    mask[b, k, indices] = True

        return mask

    def compute_loss(
        self,
        codec_tokens: torch.Tensor,
        semantic_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the masked prediction training loss.

        Args:
            codec_tokens: Ground truth codec tokens of shape (B, num_codebooks, T).
            semantic_tokens: Semantic embeddings of shape (B, T_semantic, semantic_dim).
            mask: Optional mask (generated randomly if not provided).

        Returns:
            Dictionary with loss, accuracy, and logits.
        """
        return self.forward(codec_tokens, semantic_tokens, mask)

    @torch.no_grad()
    def generate(
        self,
        semantic_tokens: torch.Tensor,
        num_frames: int,
        temperature: Optional[float] = None,
        num_iterations: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate audio codec tokens from semantic tokens.

        Uses confidence-based parallel decoding to iteratively
        unmask all codec tokens.

        Args:
            semantic_tokens: Semantic condition of shape (B, T_semantic, semantic_dim).
            num_frames: Number of audio frames to generate.
            temperature: Sampling temperature (overrides default).
            num_iterations: Decoding iterations (overrides default).

        Returns:
            Generated codec tokens of shape (B, num_codebooks, num_frames).
        """
        B = semantic_tokens.shape[0]
        device = semantic_tokens.device

        if temperature is not None:
            self.decoder.temperature = temperature
        if num_iterations is not None:
            self.decoder.num_iterations = num_iterations

        # Project semantic tokens
        semantic_emb = self.semantic_proj(semantic_tokens)

        # Initialize all tokens as masked
        initial_tokens = torch.full(
            (B, self.num_codebooks, num_frames),
            self.transformer.mask_token_id,
            device=device,
            dtype=torch.long,
        )

        # Run confidence-based parallel decoding
        generated_tokens = self.decoder.decode(
            model=self.transformer,
            initial_tokens=initial_tokens,
            semantic_tokens=semantic_emb,
        )

        return generated_tokens

    @torch.no_grad()
    def generate_with_prompt(
        self,
        semantic_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        num_frames: int,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate audio codec tokens with an acoustic prompt for voice cloning.

        Pre-fills the beginning of the sequence with prompt tokens
        (unmasked) and generates the remaining tokens.

        Args:
            semantic_tokens: Semantic condition of shape (B, T_semantic, semantic_dim).
            prompt_tokens: Prompt codec tokens of shape (B, num_codebooks, T_prompt).
            num_frames: Total output frames (including prompt).
            temperature: Sampling temperature.

        Returns:
            Generated codec tokens of shape (B, num_codebooks, num_frames).
        """
        B = semantic_tokens.shape[0]
        device = semantic_tokens.device
        T_prompt = prompt_tokens.shape[2]

        if temperature is not None:
            self.decoder.temperature = temperature

        semantic_emb = self.semantic_proj(semantic_tokens)

        # Initialize: prompt tokens are unmasked, rest are masked
        initial_tokens = torch.full(
            (B, self.num_codebooks, num_frames),
            self.transformer.mask_token_id,
            device=device,
            dtype=torch.long,
        )
        initial_tokens[:, :, :T_prompt] = prompt_tokens

        generated_tokens = self.decoder.decode(
            model=self.transformer,
            initial_tokens=initial_tokens,
            semantic_tokens=semantic_emb,
        )

        return generated_tokens
