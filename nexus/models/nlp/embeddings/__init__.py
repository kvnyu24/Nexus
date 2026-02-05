"""Advanced embedding models for NLP."""

from .matryoshka import MatryoshkaEmbedding, MatryoshkaLoss, create_matryoshka_model

__all__ = [
    'MatryoshkaEmbedding',
    'MatryoshkaLoss',
    'create_matryoshka_model',
]
