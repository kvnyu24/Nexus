"""
Inference optimization components for LLMs.

Includes speculative decoding, multi-token prediction, and KV cache management.
"""
from .speculative import (
    SpeculativeDecoder,
    NGramSpeculator
)
from .multi_token import (
    MultiTokenPredictionHead,
    MedusaHead,
    EAGLEHead
)
from .kv_cache import (
    KVCache,
    PagedKVCache,
    StaticKVCache
)

__all__ = [
    'SpeculativeDecoder',
    'NGramSpeculator',
    'MultiTokenPredictionHead',
    'MedusaHead',
    'EAGLEHead',
    'KVCache',
    'PagedKVCache',
    'StaticKVCache'
]
