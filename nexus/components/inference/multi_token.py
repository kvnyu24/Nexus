"""
Multi-Token Prediction heads for LLMs.

Enables predicting multiple future tokens simultaneously,
which can be used for faster inference and improved training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from nexus.core.base import NexusModule


class MultiTokenPredictionHead(NexusModule):
    """Multi-Token Prediction Head.

    Predicts multiple future tokens simultaneously using independent
    prediction heads. Can be used for:
    1. Auxiliary training objective (improves representations)
    2. Speculative decoding without draft model

    Reference: https://arxiv.org/abs/2404.19737 (Meta)

    Args:
        dim: Model dimension
        vocab_size: Vocabulary size
        num_future_tokens: Number of future tokens to predict
        shared_trunk: Whether heads share a trunk layer
        trunk_dim: Dimension of shared trunk (if used)
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        num_future_tokens: int = 4,
        shared_trunk: bool = False,
        trunk_dim: Optional[int] = None
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.num_future_tokens = num_future_tokens
        self.shared_trunk = shared_trunk

        if shared_trunk:
            trunk_dim = trunk_dim or dim
            self.trunk = nn.Linear(dim, trunk_dim)
            head_input_dim = trunk_dim
        else:
            self.trunk = None
            head_input_dim = dim

        # Independent heads for each future position
        self.heads = nn.ModuleList([
            nn.Linear(head_input_dim, vocab_size)
            for _ in range(num_future_tokens)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_all: bool = True
    ) -> torch.Tensor:
        """
        Predict multiple future tokens.

        Args:
            hidden_states: Model hidden states (batch, seq_len, dim)
            return_all: If True, return all predictions; else just first

        Returns:
            logits: Shape (batch, seq_len, num_future, vocab_size) if return_all
                   else (batch, seq_len, vocab_size)
        """
        if self.trunk is not None:
            hidden_states = self.trunk(hidden_states)

        if return_all:
            # Get predictions from all heads
            all_logits = [head(hidden_states) for head in self.heads]
            # Stack: (batch, seq, num_future, vocab)
            return torch.stack(all_logits, dim=2)
        else:
            # Just return first prediction (standard LM head)
            return self.heads[0](hidden_states)

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Compute multi-token prediction loss.

        Args:
            hidden_states: Model hidden states (batch, seq_len, dim)
            labels: Target token IDs (batch, seq_len)
            weights: Optional weights for each future position

        Returns:
            Combined loss for all future positions
        """
        batch_size, seq_len, _ = hidden_states.shape

        if weights is None:
            # Default: decreasing weights for further predictions
            weights = [1.0 / (i + 1) for i in range(self.num_future_tokens)]

        total_loss = 0.0
        total_weight = sum(weights)

        for i, (head, weight) in enumerate(zip(self.heads, weights)):
            if i >= seq_len - 1:
                break

            # Predictions for position i+1, i+2, etc.
            logits = head(hidden_states[:, :-i-1])  # (batch, seq-i-1, vocab)

            # Target is shifted by i+1
            targets = labels[:, i+1:]  # (batch, seq-i-1)

            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100
            )

            total_loss += weight * loss

        return total_loss / total_weight


class MedusaHead(NexusModule):
    """Medusa-style speculative heads.

    Uses multiple heads to propose tree-structured candidates,
    enabling efficient parallel verification.

    Reference: https://arxiv.org/abs/2401.10774

    Args:
        dim: Model dimension
        vocab_size: Vocabulary size
        num_heads: Number of Medusa heads
        num_layers_per_head: Depth of each head's MLP
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        num_heads: int = 4,
        num_layers_per_head: int = 1
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads

        # Each head is a small MLP
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            layers = []
            for j in range(num_layers_per_head):
                layers.extend([
                    nn.Linear(dim, dim),
                    nn.SiLU()
                ])
            layers.append(nn.Linear(dim, vocab_size))
            self.heads.append(nn.Sequential(*layers))

        # Residual connections for each head
        self.residual_weight = nn.Parameter(torch.ones(num_heads) * 0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        base_logits: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Generate predictions from all Medusa heads.

        Args:
            hidden_states: Final hidden states (batch, seq_len, dim)
            base_logits: Base model logits for residual connection

        Returns:
            List of logits from each head (each: batch, seq_len, vocab_size)
        """
        head_logits = []

        for i, head in enumerate(self.heads):
            logits = head(hidden_states)

            # Optional residual with base logits
            if base_logits is not None:
                w = torch.sigmoid(self.residual_weight[i])
                logits = w * logits + (1 - w) * base_logits

            head_logits.append(logits)

        return head_logits

    def generate_candidates(
        self,
        hidden_states: torch.Tensor,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tree-structured candidate sequences.

        Args:
            hidden_states: Hidden states for last position (batch, 1, dim)
            top_k: Number of top candidates per head

        Returns:
            candidates: Token candidates (batch, num_candidates, num_heads)
            scores: Candidate scores (batch, num_candidates)
        """
        head_logits = self.forward(hidden_states[:, -1:, :])

        # Get top-k from each head
        all_topk_tokens = []
        all_topk_probs = []

        for logits in head_logits:
            probs = F.softmax(logits[:, -1, :], dim=-1)
            topk_probs, topk_tokens = torch.topk(probs, top_k, dim=-1)
            all_topk_tokens.append(topk_tokens)
            all_topk_probs.append(topk_probs)

        # Stack for tree structure
        candidates = torch.stack(all_topk_tokens, dim=2)  # (batch, top_k, num_heads)
        scores = torch.stack(all_topk_probs, dim=2)  # (batch, top_k, num_heads)

        # Compute combined scores (product of probs)
        combined_scores = scores.prod(dim=-1)  # (batch, top_k)

        return candidates, combined_scores


class EAGLEHead(NexusModule):
    """EAGLE-style feature-level speculative decoding head.

    EAGLE uses feature-level draft rather than token-level,
    providing better acceptance rates.

    Reference: https://arxiv.org/abs/2401.15077

    Args:
        dim: Model dimension
        vocab_size: Vocabulary size
        num_speculation: Number of speculative positions
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        num_speculation: int = 5
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.num_speculation = num_speculation

        # Feature predictor (predicts next hidden state)
        self.feature_predictor = nn.Sequential(
            nn.Linear(dim + dim, dim),  # hidden + embedding
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        # LM head for final prediction
        self.lm_head = nn.Linear(dim, vocab_size)

        # Embedding for conditioning on predicted tokens
        self.token_embedding = nn.Embedding(vocab_size, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next hidden state and token.

        Args:
            hidden_states: Current hidden states
            input_embeds: Optional input embeddings

        Returns:
            next_hidden: Predicted next hidden state
            logits: Token logits
        """
        if input_embeds is None:
            input_embeds = torch.zeros_like(hidden_states)

        # Predict next feature
        combined = torch.cat([hidden_states, input_embeds], dim=-1)
        next_hidden = self.feature_predictor(combined)

        # Predict token
        logits = self.lm_head(next_hidden)

        return next_hidden, logits

    def speculate(
        self,
        hidden_states: torch.Tensor,
        num_tokens: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate speculative sequence.

        Args:
            hidden_states: Starting hidden state (batch, 1, dim)
            num_tokens: Number of tokens to speculate

        Returns:
            features: List of predicted features
            tokens: List of predicted token indices
        """
        num_tokens = num_tokens or self.num_speculation

        features = [hidden_states]
        tokens = []
        current_hidden = hidden_states

        for _ in range(num_tokens):
            # Get previous token embedding if available
            if tokens:
                prev_embed = self.token_embedding(tokens[-1])
            else:
                prev_embed = torch.zeros_like(current_hidden)

            # Predict next
            next_hidden, logits = self.forward(current_hidden, prev_embed)

            # Sample token
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

            features.append(next_hidden)
            tokens.append(next_token)
            current_hidden = next_hidden

        return features, tokens
