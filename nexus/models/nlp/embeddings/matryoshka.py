"""
Matryoshka Representation Learning: Nested Multi-Granularity Embeddings.

Paper: "Matryoshka Representation Learning"
       Kusupati et al., NeurIPS 2022
       https://arxiv.org/abs/2205.13147

Matryoshka Representation Learning (MRL) trains embeddings that can be truncated
to different dimensions while maintaining good performance. This enables flexible
trade-offs between accuracy and efficiency at deployment time, without retraining.

Key innovations:
- Nested embedding structure (like Russian dolls)
- Multi-granularity training: optimize all truncation levels simultaneously
- No performance loss at full dimension
- Graceful degradation at lower dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from nexus.core.base import NexusModule


class MatryoshkaEmbedding(NexusModule):
    """Matryoshka Embedding model with nested representations.

    Args:
        config: Configuration dictionary with keys:
            - input_dim (int): Input dimension
            - d_model (int): Full embedding dimension
            - nesting_dims (list): Nested dimensions to optimize, e.g. [32, 64, 128, 256, 512, 768]
            - encoder: Optional encoder network (e.g., transformer)
            - pooling (str): Pooling method 'mean', 'cls', or 'max'. Default 'mean'
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.input_dim = config.get('input_dim', 768)
        self.d_model = config['d_model']
        self.nesting_dims = config.get('nesting_dims', [32, 64, 128, 256, 512, 768])
        self.pooling = config.get('pooling', 'mean')

        # Validate nesting dimensions
        assert max(self.nesting_dims) <= self.d_model, \
            "All nesting dims must be <= d_model"
        assert sorted(self.nesting_dims) == self.nesting_dims, \
            "Nesting dims must be sorted"

        # Encoder (can be provided or use simple MLP)
        if 'encoder' in config:
            self.encoder = config['encoder']
        else:
            # Simple MLP encoder
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model)
            )

        # Normalization layer
        self.norm = nn.LayerNorm(self.d_model)

    def _pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool sequence of hidden states to single vector.

        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: Optional mask [batch, seq_len]

        Returns:
            Pooled embedding [batch, d_model]
        """
        if self.pooling == 'mean':
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return hidden_states.mean(dim=1)

        elif self.pooling == 'cls':
            # Use first token (CLS token)
            return hidden_states[:, 0]

        elif self.pooling == 'max':
            return hidden_states.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Produce full embedding.

        Args:
            x: Input tensor [batch, seq_len, input_dim] or [batch, input_dim]
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Full embedding [batch, d_model]
        """
        # Encode
        if x.dim() == 2:
            # Single vector input
            hidden = self.encoder(x)
        else:
            # Sequence input
            hidden = self.encoder(x)
            hidden = self._pool(hidden, attention_mask)

        # Normalize
        embedding = self.norm(hidden)

        return embedding

    def get_embedding(
        self,
        x: torch.Tensor,
        dim: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get embedding truncated to specific dimension.

        Args:
            x: Input tensor
            dim: Target dimension to truncate to
            attention_mask: Optional attention mask

        Returns:
            Truncated embedding [batch, dim]
        """
        # Get full embedding
        full_embedding = self.forward(x, attention_mask)

        # Truncate
        truncated = full_embedding[:, :dim]

        # Renormalize
        truncated = F.normalize(truncated, p=2, dim=-1)

        return truncated

    def get_all_embeddings(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[int, torch.Tensor]:
        """Get embeddings at all nesting dimensions.

        Args:
            x: Input tensor
            attention_mask: Optional attention mask

        Returns:
            Dictionary mapping dimension to embedding tensor
        """
        full_embedding = self.forward(x, attention_mask)

        embeddings = {}
        for dim in self.nesting_dims:
            embeddings[dim] = self.get_embedding(x, dim, attention_mask)

        return embeddings


class MatryoshkaLoss(NexusModule):
    """Multi-granularity loss for Matryoshka Representation Learning.

    Trains the model to maintain good performance at all nesting dimensions
    by computing loss at each truncation level.

    Args:
        config: Configuration dictionary with keys:
            - nesting_dims (list): Nested dimensions to optimize
            - loss_weights (list): Optional weights for each dimension. Default uniform
            - base_loss_fn (str): Base loss function 'contrastive' or 'triplet'. Default 'contrastive'
            - temperature (float): Temperature for contrastive loss. Default 0.07
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.nesting_dims = config['nesting_dims']
        self.loss_weights = config.get('loss_weights', [1.0] * len(self.nesting_dims))
        self.base_loss_fn = config.get('base_loss_fn', 'contrastive')
        self.temperature = config.get('temperature', 0.07)

        assert len(self.loss_weights) == len(self.nesting_dims), \
            "Must have one weight per nesting dimension"

    def _contrastive_loss(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss (InfoNCE-style).

        Args:
            embeddings_a: First set of embeddings [batch, dim]
            embeddings_b: Second set of embeddings [batch, dim]

        Returns:
            Loss scalar
        """
        # Normalize embeddings
        embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(embeddings_a, embeddings_b.T) / self.temperature

        # Labels: positive pairs are on diagonal
        batch_size = embeddings_a.size(0)
        labels = torch.arange(batch_size, device=embeddings_a.device)

        # Cross-entropy loss (both directions)
        loss_a = F.cross_entropy(similarity, labels)
        loss_b = F.cross_entropy(similarity.T, labels)

        loss = (loss_a + loss_b) / 2

        return loss

    def forward(
        self,
        full_embedding_a: torch.Tensor,
        full_embedding_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-granularity Matryoshka loss.

        Args:
            full_embedding_a: Full embeddings from first view [batch, d_model]
            full_embedding_b: Full embeddings from second view [batch, d_model]

        Returns:
            Dictionary with total loss and per-dimension losses
        """
        total_loss = 0.0
        losses = {}

        for dim, weight in zip(self.nesting_dims, self.loss_weights):
            # Truncate embeddings
            emb_a = full_embedding_a[:, :dim]
            emb_b = full_embedding_b[:, :dim]

            # Compute loss at this dimension
            if self.base_loss_fn == 'contrastive':
                loss = self._contrastive_loss(emb_a, emb_b)
            else:
                raise ValueError(f"Unknown loss function: {self.base_loss_fn}")

            losses[f'loss_dim_{dim}'] = loss
            total_loss += weight * loss

        losses['total_loss'] = total_loss

        return losses


# Convenience function
def create_matryoshka_model(
    d_model: int = 768,
    nesting_dims: Optional[List[int]] = None
) -> MatryoshkaEmbedding:
    """Create a Matryoshka embedding model.

    Args:
        d_model: Full embedding dimension
        nesting_dims: Nested dimensions to optimize

    Returns:
        MatryoshkaEmbedding model
    """
    if nesting_dims is None:
        # Default OpenAI-style nesting
        nesting_dims = [64, 128, 256, 512, 768, 1024, 1536][:d_model.bit_length()]
        nesting_dims = [d for d in nesting_dims if d <= d_model]

    config = {
        'd_model': d_model,
        'nesting_dims': nesting_dims
    }

    return MatryoshkaEmbedding(config)
