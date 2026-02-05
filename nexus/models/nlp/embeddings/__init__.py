"""Advanced embedding models for NLP."""

from .matryoshka import MatryoshkaEmbedding, MatryoshkaLoss, create_matryoshka_model
from .bge_m3 import (
    BGEM3Config,
    BGEM3Embedder,
    DenseEmbeddingHead,
    SparseEmbeddingHead,
    ColBERTEmbeddingHead,
    create_bge_m3_embedder,
)

__all__ = [
    # Matryoshka
    'MatryoshkaEmbedding',
    'MatryoshkaLoss',
    'create_matryoshka_model',
    # BGE-M3
    'BGEM3Config',
    'BGEM3Embedder',
    'DenseEmbeddingHead',
    'SparseEmbeddingHead',
    'ColBERTEmbeddingHead',
    'create_bge_m3_embedder',
]
