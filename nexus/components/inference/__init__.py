"""
Inference optimization components for LLMs.

Includes speculative decoding, multi-token prediction, KV cache management,
prefix caching, and continuous batching.
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
    StaticKVCache,
    QuantizedKVCache
)
from .prefix_cache import (
    PrefixCache,
    RadixPrefixCache
)
from .continuous_batching import (
    ContinuousBatcher,
    IterationLevelBatcher,
    GenerationRequest,
    RequestStatus,
    BatchState
)

__all__ = [
    # Speculative decoding
    'SpeculativeDecoder',
    'NGramSpeculator',
    # Multi-token prediction
    'MultiTokenPredictionHead',
    'MedusaHead',
    'EAGLEHead',
    # KV cache
    'KVCache',
    'PagedKVCache',
    'StaticKVCache',
    'QuantizedKVCache',
    # Prefix caching
    'PrefixCache',
    'RadixPrefixCache',
    # Continuous batching
    'ContinuousBatcher',
    'IterationLevelBatcher',
    'GenerationRequest',
    'RequestStatus',
    'BatchState'
]
