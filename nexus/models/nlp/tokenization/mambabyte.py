"""MambaByte: Byte-level Modeling with Mamba Architecture.

Reference:
    Wang, J., et al. "MambaByte: Token-free Selective State Space Model."
    COLM 2024. https://arxiv.org/abs/2401.13660

MambaByte applies the Mamba (selective state space model) architecture directly
to raw bytes without tokenization. This eliminates the need for a fixed vocabulary
and tokenizer while maintaining strong performance through Mamba's efficient
sequence modeling.

Key advantages:
- No tokenizer required (truly language-agnostic)
- Efficient long-range modeling via SSM
- Handles any byte sequence (text, code, binary data)
- Better scaling than byte-level transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import math

from nexus.core.base import NexusModule


@dataclass
class MambaByteConfig:
    """Configuration for MambaByte.

    Attributes:
        vocab_size: Byte vocabulary size (256 for raw bytes).
        hidden_size: Hidden dimension size.
        state_size: SSM state dimension.
        num_layers: Number of Mamba blocks.
        expand_factor: Expansion factor for intermediate dimension.
        conv_kernel_size: Kernel size for local convolution.
        use_bias: Whether to use bias in linear layers.
        dropout: Dropout probability.
    """
    vocab_size: int = 256  # 256 bytes
    hidden_size: int = 768
    state_size: int = 16
    num_layers: int = 12
    expand_factor: int = 2
    conv_kernel_size: int = 4
    use_bias: bool = False
    dropout: float = 0.1


class SelectiveSSM(nn.Module):
    """Selective State Space Model for byte-level sequences.

    Implements the core Mamba selective SSM with input-dependent parameters.

    Args:
        hidden_size: Hidden dimension.
        state_size: State dimension.
    """

    def __init__(self, hidden_size: int, state_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size

        # Input-dependent parameters
        self.x_proj = nn.Linear(hidden_size, state_size + state_size + hidden_size, bias=False)

        # SSM parameters (structured initialization)
        self.A_log = nn.Parameter(torch.randn(state_size))
        self.D = nn.Parameter(torch.ones(hidden_size))

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply selective SSM.

        Args:
            x: Input tensor, shape (batch, seq_len, hidden_size).

        Returns:
            Output tensor, shape (batch, seq_len, hidden_size).
        """
        batch_size, seq_len, _ = x.shape

        # Compute input-dependent SSM parameters
        x_proj = self.x_proj(x)  # (batch, seq_len, state_size + state_size + hidden_size)

        # Split into B, C, and delta
        B, C, delta = torch.split(
            x_proj,
            [self.state_size, self.state_size, self.hidden_size],
            dim=-1
        )

        # Discretize continuous parameters (simplified)
        A = -torch.exp(self.A_log)  # (state_size,)
        delta = F.softplus(delta)    # (batch, seq_len, hidden_size)

        # Selective scan (simplified implementation)
        # Full implementation would use parallel scan for efficiency
        h = torch.zeros(batch_size, self.state_size, self.hidden_size, device=x.device)
        outputs = []

        for t in range(seq_len):
            # Discretization
            deltaA = torch.exp(delta[:, t:t+1, :].unsqueeze(1) * A.unsqueeze(0).unsqueeze(-1))  # (batch, 1, state_size, hidden_size)
            deltaB = delta[:, t:t+1, :].unsqueeze(1) * B[:, t:t+1, :].unsqueeze(-1)  # (batch, 1, state_size, hidden_size)

            # State update
            h = deltaA.squeeze(1) * h + deltaB.squeeze(1) * x[:, t:t+1, :].unsqueeze(1)

            # Output
            y = torch.einsum('bsh,bs->bh', h, C[:, t, :])  # (batch, hidden_size)
            y = y + self.D * x[:, t, :]
            outputs.append(y)

        # Stack outputs
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_size)

        # Output projection
        y = self.out_proj(y)

        return y


class MambaBlock(nn.Module):
    """Mamba block with selective SSM and gating.

    Args:
        config: MambaByte configuration.
    """

    def __init__(self, config: MambaByteConfig):
        super().__init__()
        self.config = config

        inner_dim = config.hidden_size * config.expand_factor

        # Input projection and gating
        self.in_proj = nn.Linear(config.hidden_size, inner_dim * 2, bias=config.use_bias)

        # Local convolution for short-range dependencies
        self.conv1d = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size - 1,
            groups=inner_dim,
            bias=config.use_bias,
        )

        # Selective SSM
        self.ssm = SelectiveSSM(inner_dim, config.state_size)

        # Output projection
        self.out_proj = nn.Linear(inner_dim, config.hidden_size, bias=config.use_bias)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Normalization
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba block.

        Args:
            x: Input tensor, shape (batch, seq_len, hidden_size).

        Returns:
            Output tensor, shape (batch, seq_len, hidden_size).
        """
        residual = x

        # Normalize
        x = self.norm(x)

        # Input projection and split for gating
        x_proj = self.in_proj(x)  # (batch, seq_len, inner_dim * 2)
        x, gate = x_proj.chunk(2, dim=-1)  # Each (batch, seq_len, inner_dim)

        # Apply convolution (need to transpose for Conv1d)
        x = x.transpose(1, 2)  # (batch, inner_dim, seq_len)
        x = self.conv1d(x)[:, :, :x.shape[-1]]  # Trim padding
        x = x.transpose(1, 2)  # (batch, seq_len, inner_dim)

        # Activation
        x = F.silu(x)

        # Selective SSM
        x = self.ssm(x)

        # Gating
        x = x * F.silu(gate)

        # Output projection
        x = self.out_proj(x)
        x = self.dropout(x)

        # Residual connection
        x = x + residual

        return x


class MambaByte(NexusModule):
    """MambaByte: Byte-level language model using Mamba architecture.

    Args:
        config: MambaByte configuration.
    """

    def __init__(self, config: MambaByteConfig):
        super().__init__(config.__dict__)

        self.config = config

        # Byte embedding
        self.byte_embed = nn.Embedding(config.vocab_size, config.hidden_size)

        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(config.num_layers)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(config.hidden_size)

        # Output head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.byte_embed.weight

    def forward(
        self,
        byte_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through MambaByte.

        Args:
            byte_ids: Byte token IDs, shape (batch, seq_len).
            labels: Optional labels for language modeling loss.

        Returns:
            Dictionary with 'logits' and optionally 'loss'.
        """
        # Embed bytes
        x = self.byte_embed(byte_ids)  # (batch, seq_len, hidden_size)

        # Apply Mamba blocks
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.final_norm(x)

        # Compute logits
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        outputs = {'logits': logits}

        # Compute loss if labels provided
        if labels is not None:
            # Shift logits and labels for next-byte prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            outputs['loss'] = loss

        return outputs

    def generate(
        self,
        prompt_bytes: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Generate bytes autoregressively.

        Args:
            prompt_bytes: Prompt byte IDs, shape (batch, prompt_len).
            max_length: Maximum bytes to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated byte IDs, shape (batch, total_len).
        """
        self.requires_grad_(False)

        generated = prompt_bytes.clone()

        for _ in range(max_length):
            # Forward pass
            outputs = self.forward(generated)
            logits = outputs['logits']

            # Get last byte logits
            next_byte_logits = logits[:, -1, :] / temperature

            # Apply nucleus sampling if top_p < 1.0
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_byte_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_byte_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_byte_logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_byte], dim=1)

        return generated

    def get_compression_stats(self, text: str) -> Dict[str, float]:
        """Get compression statistics for byte-level modeling.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary with statistics.
        """
        byte_length = len(text.encode('utf-8'))
        char_length = len(text)

        return {
            'byte_length': byte_length,
            'char_length': char_length,
            'bytes_per_char': byte_length / char_length if char_length > 0 else 0,
            'vocab_size': self.config.vocab_size,
        }


def create_mambabyte(config: Optional[MambaByteConfig] = None) -> MambaByte:
    """Create a MambaByte model.

    Args:
        config: MambaByte configuration (uses defaults if None).

    Returns:
        MambaByte instance.
    """
    config = config or MambaByteConfig()
    return MambaByte(config)


def encode_text_to_bytes(text: str) -> torch.Tensor:
    """Encode text to byte tensor.

    Args:
        text: Input text.

    Returns:
        Byte tensor, shape (1, num_bytes).
    """
    byte_list = list(text.encode('utf-8'))
    return torch.tensor([byte_list], dtype=torch.long)


def decode_bytes_to_text(byte_ids: torch.Tensor) -> str:
    """Decode byte tensor to text.

    Args:
        byte_ids: Byte IDs, shape (batch, seq_len) or (seq_len,).

    Returns:
        Decoded text string.
    """
    if byte_ids.dim() == 2:
        byte_ids = byte_ids[0]  # Take first batch

    byte_list = byte_ids.cpu().tolist()
    return bytes(byte_list).decode('utf-8', errors='ignore')
