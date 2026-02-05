"""BGE-M3: Multi-Functionality Multi-Granularity Multi-Lingual Embeddings.

Reference:
    Chen, J., et al. "BGE M3-Embedding: Multi-Lingual, Multi-Functionality,
    Multi-Granularity Text Embeddings Through Self-Knowledge Distillation."
    BAAI, 2024. https://arxiv.org/abs/2402.03216

BGE-M3 unifies three types of retrieval in a single model:
1. Dense retrieval: Traditional dense embeddings
2. Sparse retrieval: Learned sparse representations (like SPLADE)
3. Multi-vector retrieval: ColBERT-style token-level matching

This unified approach achieves state-of-the-art results across multiple
retrieval benchmarks and supports 100+ languages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import math

from nexus.core.base import NexusModule


@dataclass
class BGEM3Config:
    """Configuration for BGE-M3 embeddings.

    Attributes:
        hidden_size: Hidden dimension size.
        vocab_size: Vocabulary size for sparse embeddings.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        max_length: Maximum sequence length.
        dense_dim: Dimension for dense embeddings.
        enable_dense: Whether to compute dense embeddings.
        enable_sparse: Whether to compute sparse embeddings.
        enable_colbert: Whether to compute ColBERT multi-vector embeddings.
        sparse_top_k: Top-k for sparse embedding activations.
        colbert_dim: Dimension for ColBERT token embeddings.
    """
    hidden_size: int = 768
    vocab_size: int = 30522
    num_layers: int = 12
    num_heads: int = 12
    max_length: int = 512
    dense_dim: int = 1024
    enable_dense: bool = True
    enable_sparse: bool = True
    enable_colbert: bool = True
    sparse_top_k: int = 100
    colbert_dim: int = 128


class DenseEmbeddingHead(nn.Module):
    """Dense embedding head for traditional dense retrieval.

    Args:
        hidden_size: Input hidden size.
        output_dim: Output embedding dimension.
    """

    def __init__(self, hidden_size: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract dense embedding from [CLS] token.

        Args:
            hidden_states: Transformer output, shape (batch, seq_len, hidden).

        Returns:
            Dense embeddings, shape (batch, output_dim).
        """
        # Use [CLS] token (first token)
        cls_embedding = hidden_states[:, 0, :]  # (batch, hidden)
        dense_emb = self.linear(cls_embedding)
        dense_emb = self.layer_norm(dense_emb)
        dense_emb = F.normalize(dense_emb, p=2, dim=-1)
        return dense_emb


class SparseEmbeddingHead(nn.Module):
    """Sparse embedding head for learned sparse retrieval (SPLADE-style).

    Args:
        hidden_size: Input hidden size.
        vocab_size: Vocabulary size.
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        top_k: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract sparse embedding with importance scores.

        Args:
            hidden_states: Transformer output, shape (batch, seq_len, hidden).
            attention_mask: Attention mask, shape (batch, seq_len).
            top_k: Number of top activations to keep.

        Returns:
            Tuple of (sparse_indices, sparse_values).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to vocabulary space
        logits = self.linear(hidden_states)  # (batch, seq_len, vocab)

        # Apply log(1 + ReLU(x)) activation (SPLADE)
        sparse_weights = torch.log1p(F.relu(logits))

        # Apply attention mask
        if attention_mask is not None:
            attention_mask_expanded = attention_mask.unsqueeze(-1)
            sparse_weights = sparse_weights * attention_mask_expanded

        # Max pooling over sequence length
        sparse_weights, _ = torch.max(sparse_weights, dim=1)  # (batch, vocab)

        # Keep top-k activations
        top_values, top_indices = torch.topk(sparse_weights, k=top_k, dim=-1)

        return top_indices, top_values


class ColBERTEmbeddingHead(nn.Module):
    """ColBERT multi-vector embedding head for token-level matching.

    Args:
        hidden_size: Input hidden size.
        output_dim: Output token embedding dimension.
    """

    def __init__(self, hidden_size: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract token-level embeddings for ColBERT matching.

        Args:
            hidden_states: Transformer output, shape (batch, seq_len, hidden).
            attention_mask: Attention mask, shape (batch, seq_len).

        Returns:
            Token embeddings, shape (batch, seq_len, output_dim).
        """
        # Project and normalize each token
        token_embeddings = self.linear(hidden_states)  # (batch, seq_len, output_dim)
        token_embeddings = F.normalize(token_embeddings, p=2, dim=-1)

        # Apply attention mask
        if attention_mask is not None:
            attention_mask_expanded = attention_mask.unsqueeze(-1)
            token_embeddings = token_embeddings * attention_mask_expanded

        return token_embeddings


class BGEM3Embedder(NexusModule):
    """BGE-M3 multi-functionality embedding model.

    Supports dense, sparse, and multi-vector (ColBERT) retrieval in one model.

    Args:
        config: BGE-M3 configuration.
        encoder: Optional pre-trained encoder (e.g., BERT).
    """

    def __init__(self, config: BGEM3Config, encoder: Optional[nn.Module] = None):
        super().__init__(config.__dict__)

        self.config = config

        # Encoder (transformer backbone)
        if encoder is None:
            # Create a simple transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_size * 4,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        else:
            self.encoder = encoder

        # Embedding heads
        if config.enable_dense:
            self.dense_head = DenseEmbeddingHead(config.hidden_size, config.dense_dim)

        if config.enable_sparse:
            self.sparse_head = SparseEmbeddingHead(config.hidden_size, config.vocab_size)

        if config.enable_colbert:
            self.colbert_head = ColBERTEmbeddingHead(config.hidden_size, config.colbert_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute embeddings for multiple retrieval modes.

        Args:
            input_ids: Input token IDs, shape (batch, seq_len).
            attention_mask: Attention mask, shape (batch, seq_len).
            return_dense: Whether to return dense embeddings.
            return_sparse: Whether to return sparse embeddings.
            return_colbert: Whether to return ColBERT embeddings.

        Returns:
            Dictionary with requested embedding types.
        """
        # Encode input
        hidden_states = self.encoder(input_ids)

        outputs = {}

        # Dense embeddings
        if return_dense and self.config.enable_dense:
            dense_emb = self.dense_head(hidden_states)
            outputs['dense_embeddings'] = dense_emb

        # Sparse embeddings
        if return_sparse and self.config.enable_sparse:
            sparse_indices, sparse_values = self.sparse_head(
                hidden_states,
                attention_mask,
                top_k=self.config.sparse_top_k
            )
            outputs['sparse_indices'] = sparse_indices
            outputs['sparse_values'] = sparse_values

        # ColBERT embeddings
        if return_colbert and self.config.enable_colbert:
            colbert_emb = self.colbert_head(hidden_states, attention_mask)
            outputs['colbert_embeddings'] = colbert_emb

        return outputs

    def compute_similarity(
        self,
        query_output: Dict[str, torch.Tensor],
        doc_output: Dict[str, torch.Tensor],
        mode: str = "hybrid",
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """Compute similarity between query and document embeddings.

        Args:
            query_output: Query embeddings from forward().
            doc_output: Document embeddings from forward().
            mode: Similarity mode:
                - "dense": Only dense similarity
                - "sparse": Only sparse similarity
                - "colbert": Only ColBERT similarity
                - "hybrid": Weighted combination
            weights: Weights for hybrid mode (default: equal weights).

        Returns:
            Similarity scores, shape (batch,).
        """
        if weights is None:
            weights = {"dense": 0.4, "sparse": 0.3, "colbert": 0.3}

        scores = []

        # Dense similarity (cosine)
        if mode in ["dense", "hybrid"] and 'dense_embeddings' in query_output:
            dense_sim = (query_output['dense_embeddings'] * doc_output['dense_embeddings']).sum(dim=-1)
            if mode == "hybrid":
                scores.append(weights.get("dense", 0.0) * dense_sim)
            else:
                return dense_sim

        # Sparse similarity (dot product of sparse vectors)
        if mode in ["sparse", "hybrid"] and 'sparse_indices' in query_output:
            # Reconstruct sparse vectors and compute dot product
            batch_size = query_output['sparse_indices'].shape[0]
            sparse_sim = torch.zeros(batch_size, device=query_output['sparse_indices'].device)

            for i in range(batch_size):
                q_indices = query_output['sparse_indices'][i]
                q_values = query_output['sparse_values'][i]
                d_indices = doc_output['sparse_indices'][i]
                d_values = doc_output['sparse_values'][i]

                # Find common indices
                common_mask = torch.isin(q_indices, d_indices)
                common_indices = q_indices[common_mask]

                # Sum values for common indices
                for idx in common_indices:
                    q_val = q_values[q_indices == idx]
                    d_val = d_values[d_indices == idx]
                    sparse_sim[i] += (q_val * d_val).sum()

            if mode == "hybrid":
                scores.append(weights.get("sparse", 0.0) * sparse_sim)
            else:
                return sparse_sim

        # ColBERT similarity (MaxSim)
        if mode in ["colbert", "hybrid"] and 'colbert_embeddings' in query_output:
            q_emb = query_output['colbert_embeddings']  # (batch, q_len, dim)
            d_emb = doc_output['colbert_embeddings']    # (batch, d_len, dim)

            # Compute pairwise similarities
            similarities = torch.bmm(q_emb, d_emb.transpose(1, 2))  # (batch, q_len, d_len)

            # MaxSim: for each query token, find max similarity with doc tokens
            maxsim_scores = similarities.max(dim=-1)[0]  # (batch, q_len)
            colbert_sim = maxsim_scores.sum(dim=-1)  # (batch,)

            if mode == "hybrid":
                scores.append(weights.get("colbert", 0.0) * colbert_sim)
            else:
                return colbert_sim

        # Combine scores for hybrid mode
        if mode == "hybrid":
            return sum(scores)

        raise ValueError(f"No valid similarity mode or missing embeddings")


def create_bge_m3_embedder(
    config: Optional[BGEM3Config] = None,
    pretrained_encoder: Optional[nn.Module] = None
) -> BGEM3Embedder:
    """Create a BGE-M3 embedder.

    Args:
        config: Configuration (uses defaults if None).
        pretrained_encoder: Optional pretrained encoder.

    Returns:
        BGEM3Embedder instance.
    """
    config = config or BGEM3Config()
    return BGEM3Embedder(config, pretrained_encoder)
