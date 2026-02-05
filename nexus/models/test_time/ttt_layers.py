"""
Test-Time Training (TTT) Layers.

TTT layers treat the hidden state itself as a machine learning model that is
trained at test time to reconstruct the input sequence. This self-supervised
learning at inference enables the model to adapt to the test distribution without
any labeled data.

Key innovations:
- Hidden state as a learnable model (e.g., linear model)
- Self-supervised test-time training via reconstruction
- Gradient descent updates during forward pass at inference
- Improves performance on distribution shifts
- Can be inserted into any transformer architecture

The TTT layer replaces standard attention with:
1. Use hidden state as a mini-model to predict current token
2. Update hidden state via gradient descent on reconstruction loss
3. Use updated hidden state for next token

This creates a sequence model that continuously adapts during inference.

Paper: "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
       Sun et al., 2024
       https://arxiv.org/abs/2407.04620

Key insight: The hidden state is not just a static representation, but an
active learner that trains itself on the fly during inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from nexus.core.base import NexusModule


class TTTLinearModel(nn.Module):
    """Linear model used as the learnable hidden state.

    This is the "mini-model" that the TTT layer trains at test time.

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation.

        Args:
            x: Input tensor (..., input_dim)

        Returns:
            Output tensor (..., output_dim)
        """
        return F.linear(x, self.weight, self.bias)

    def clone_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cloned parameters for test-time updates."""
        return self.weight.clone(), self.bias.clone()

    def set_params(self, weight: torch.Tensor, bias: torch.Tensor):
        """Set parameters from external tensors."""
        self.weight.data = weight
        self.bias.data = bias


class TTTLayer(NexusModule):
    """Test-Time Training Layer.

    Replaces standard attention with a mechanism that trains the hidden state
    at test time via self-supervised reconstruction.

    Args:
        config: Configuration dictionary with keys:
            - embed_dim (int): Model dimension
            - num_heads (int): Number of attention heads (for compatibility)
            - ttt_lr (float): Learning rate for test-time updates. Default 0.1
            - ttt_steps (int): Number of gradient steps per token. Default 1
            - reconstruction_loss (str): 'mse' or 'contrastive'. Default 'mse'
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 512)
        self.num_heads = config.get('num_heads', 8)
        self.ttt_lr = config.get('ttt_lr', 0.1)
        self.ttt_steps = config.get('ttt_steps', 1)
        self.reconstruction_loss_type = config.get('reconstruction_loss', 'mse')

        # Input projection
        self.input_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Mini-model (hidden state as a learnable model)
        self.mini_model = TTTLinearModel(self.embed_dim, self.embed_dim)

        # Output projection
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Layer normalization
        self.norm = nn.LayerNorm(self.embed_dim)

    def test_time_update(self,
                        model_weight: torch.Tensor,
                        model_bias: torch.Tensor,
                        context: torch.Tensor,
                        target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one step of test-time gradient descent.

        Args:
            model_weight: Current model weight (output_dim, input_dim)
            model_bias: Current model bias (output_dim,)
            context: Context features to predict from (B, context_len, embed_dim)
            target: Target features to predict (B, embed_dim)

        Returns:
            Updated (weight, bias)
        """
        # Enable gradients for test-time update
        model_weight = model_weight.clone().requires_grad_(True)
        model_bias = model_bias.clone().requires_grad_(True)

        # Forward pass with current model parameters
        # Aggregate context (mean pooling for simplicity)
        context_agg = context.mean(dim=1)  # (B, embed_dim)

        # Predict target
        prediction = F.linear(context_agg, model_weight, model_bias)

        # Compute reconstruction loss
        if self.reconstruction_loss_type == 'mse':
            loss = F.mse_loss(prediction, target)
        elif self.reconstruction_loss_type == 'contrastive':
            # Contrastive loss: maximize similarity with target, minimize with negatives
            pos_sim = F.cosine_similarity(prediction, target, dim=-1)
            loss = -pos_sim.mean()
        else:
            loss = F.mse_loss(prediction, target)

        # Gradient descent step
        grad_weight, grad_bias = torch.autograd.grad(
            loss,
            [model_weight, model_bias],
            create_graph=False  # Don't backprop through test-time updates during training
        )

        # Update parameters
        new_weight = model_weight - self.ttt_lr * grad_weight
        new_bias = model_bias - self.ttt_lr * grad_bias

        return new_weight.detach(), new_bias.detach()

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                is_training: bool = True) -> torch.Tensor:
        """Forward pass with test-time training.

        Args:
            x: Input sequence (B, seq_len, embed_dim)
            mask: Optional attention mask
            is_training: If False, perform test-time updates

        Returns:
            Output sequence (B, seq_len, embed_dim)
        """
        B, seq_len, _ = x.shape

        # Project input
        x_proj = self.input_proj(x)

        # During training: use mini-model without test-time updates
        if is_training:
            # Simple training mode: apply mini-model to mean-pooled context
            outputs = []
            for t in range(seq_len):
                if t == 0:
                    # No context for first token
                    context = x_proj[:, :1]
                else:
                    context = x_proj[:, :t]

                # Aggregate context
                context_agg = context.mean(dim=1)

                # Apply mini-model
                output = self.mini_model(context_agg)
                outputs.append(output)

            out = torch.stack(outputs, dim=1)

        else:
            # Test-time training mode: update mini-model for each token
            # Start with current mini-model parameters
            weight, bias = self.mini_model.clone_params()

            outputs = []
            for t in range(seq_len):
                if t == 0:
                    context = x_proj[:, :1]
                else:
                    context = x_proj[:, :t]

                target = x_proj[:, t]

                # Perform test-time gradient descent updates
                for _ in range(self.ttt_steps):
                    weight, bias = self.test_time_update(weight, bias, context, target)

                # Use updated model to compute output
                context_agg = context.mean(dim=1)
                output = F.linear(context_agg, weight, bias)
                outputs.append(output)

            out = torch.stack(outputs, dim=1)

        # Output projection and normalization
        out = self.output_proj(out)
        out = self.norm(out + x)  # Residual connection

        return out


class TTTBlock(NexusModule):
    """Full TTT transformer block with TTT layer and feedforward.

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 512)

        # TTT layer (replaces attention)
        self.ttt_layer = TTTLayer(config)

        # Feedforward network
        ff_dim = config.get('ff_dim', self.embed_dim * 4)
        self.ff_net = nn.Sequential(
            nn.Linear(self.embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, self.embed_dim)
        )

        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                is_training: bool = True) -> torch.Tensor:
        """Forward pass through TTT block.

        Args:
            x: Input tensor (B, seq_len, embed_dim)
            mask: Optional attention mask
            is_training: Training mode flag

        Returns:
            Output tensor (B, seq_len, embed_dim)
        """
        # TTT layer
        x = self.ttt_layer(x, mask, is_training)

        # Feedforward with residual
        ff_out = self.ff_net(x)
        x = self.norm(x + ff_out)

        return x


class TTTTransformer(NexusModule):
    """Transformer model using TTT layers.

    Can be used as a language model or sequence-to-sequence model that
    adapts at test time.

    Args:
        config: Configuration dictionary with keys:
            - vocab_size (int): Vocabulary size
            - embed_dim (int): Model dimension. Default 512
            - num_layers (int): Number of TTT blocks. Default 6
            - num_heads (int): Number of heads (for compatibility). Default 8
            - max_seq_len (int): Maximum sequence length. Default 2048
            - ttt_lr (float): Test-time learning rate. Default 0.1
            - ttt_steps (int): Test-time gradient steps. Default 1
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vocab_size = config['vocab_size']
        self.embed_dim = config.get('embed_dim', 512)
        self.num_layers = config.get('num_layers', 6)
        self.max_seq_len = config.get('max_seq_len', 2048)

        # Token embedding
        self.token_embed = nn.Embedding(self.vocab_size, self.embed_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_len, self.embed_dim) * 0.02)

        # TTT blocks
        self.blocks = nn.ModuleList([
            TTTBlock(config) for _ in range(self.num_layers)
        ])

        # Output head
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.output_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

        # Tie embeddings
        self.output_head.weight = self.token_embed.weight

    def forward(self,
                input_ids: torch.Tensor,
                is_training: bool = True) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Input token IDs (B, seq_len)
            is_training: If False, enable test-time training

        Returns:
            Logits (B, seq_len, vocab_size)
        """
        B, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embed(input_ids)

        # Add positional embeddings
        if seq_len <= self.max_seq_len:
            x = x + self.pos_embed[:, :seq_len]
        else:
            # Interpolate positional embeddings if needed
            pos_embed = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=seq_len,
                mode='linear'
            ).permute(0, 2, 1)
            x = x + pos_embed

        # Pass through TTT blocks
        for block in self.blocks:
            x = block(x, is_training=is_training)

        # Output projection
        x = self.output_norm(x)
        logits = self.output_head(x)

        return logits

    def generate(self,
                 input_ids: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 enable_ttt: bool = True) -> torch.Tensor:
        """Generate tokens autoregressively with optional test-time training.

        Args:
            input_ids: Initial token IDs (B, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (if not None)
            enable_ttt: Enable test-time training during generation

        Returns:
            Generated token IDs (B, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Get logits for next token
            with torch.set_grad_enabled(not enable_ttt):
                logits = self(input_ids, is_training=not enable_ttt)

            # Take last token logits
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


__all__ = [
    'TTTLinearModel',
    'TTTLayer',
    'TTTBlock',
    'TTTTransformer'
]
