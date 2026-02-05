"""
Hybrid architecture models combining SSM, attention, and MoE components.

These models interleave different sequence modeling mechanisms (recurrence,
attention, convolution) to achieve the strengths of each approach:
- Griffin: Gated linear recurrence + local multi-query attention
- Hyena: Long convolutions with data-controlled gating
- Jamba: Transformer attention + Mamba SSM + Mixture-of-Experts
- Based: Linear attention + sliding window for extreme throughput
- StripedHyena: Attention-Hyena hybrid for 128K context
- Zamba: Mamba backbone with shared attention blocks
- GoldFinch: RWKV-Transformer hybrid with extreme KV cache compression
- RecurrentGemma: Griffin-based open language model
- Hawk: Pure gated linear recurrence model
"""
from .griffin import (
    RealGatedLinearRecurrentUnit,
    GatedLinearRecurrence,
    LocalMultiQueryAttention,
    GriffinBlock,
    GriffinModel
)

from .hyena import (
    ImplicitFilter,
    HyenaOperator,
    HyenaBlock,
    HyenaModel
)

from .jamba import (
    JambaMambaLayer,
    JambaAttentionLayer,
    JambaMoELayer,
    JambaBlock,
    JambaModel
)

from .based import (
    TaylorLinearAttention,
    SlidingWindowAttention,
    BasedBlock,
    BasedModel
)

from .striped_hyena import (
    HyenaFilter,
    HyenaOperator as StripedHyenaOperator,
    AttentionBlock,
    StripedHyenaBlock,
    StripedHyenaModel
)

from .zamba import (
    MambaBlock as ZambaMambaBlock,
    SharedAttentionBlock,
    ZambaBlock,
    ZambaModel
)

from .goldfinch import (
    RWKVTimeMixing,
    SparseAttention,
    GoldFinchBlock,
    GoldFinchModel
)

from .recurrent_gemma import (
    RMSNorm,
    RGLRU as RecurrentGemmaRGLRU,
    LocalSlidingWindowAttention,
    GeGLU,
    RecurrentGemmaBlock,
    RecurrentGemmaModel
)

from .hawk import (
    RGLRU as HawkRGLRU,
    TemporalConvolution,
    SwiGLU,
    HawkBlock,
    HawkModel
)

__all__ = [
    # Griffin
    'RealGatedLinearRecurrentUnit',
    'GatedLinearRecurrence',
    'LocalMultiQueryAttention',
    'GriffinBlock',
    'GriffinModel',
    # Hyena
    'ImplicitFilter',
    'HyenaOperator',
    'HyenaBlock',
    'HyenaModel',
    # Jamba
    'JambaMambaLayer',
    'JambaAttentionLayer',
    'JambaMoELayer',
    'JambaBlock',
    'JambaModel',
    # Based
    'TaylorLinearAttention',
    'SlidingWindowAttention',
    'BasedBlock',
    'BasedModel',
    # StripedHyena
    'HyenaFilter',
    'StripedHyenaOperator',
    'AttentionBlock',
    'StripedHyenaBlock',
    'StripedHyenaModel',
    # Zamba
    'ZambaMambaBlock',
    'SharedAttentionBlock',
    'ZambaBlock',
    'ZambaModel',
    # GoldFinch
    'RWKVTimeMixing',
    'SparseAttention',
    'GoldFinchBlock',
    'GoldFinchModel',
    # RecurrentGemma
    'RMSNorm',
    'RecurrentGemmaRGLRU',
    'LocalSlidingWindowAttention',
    'GeGLU',
    'RecurrentGemmaBlock',
    'RecurrentGemmaModel',
    # Hawk
    'HawkRGLRU',
    'TemporalConvolution',
    'SwiGLU',
    'HawkBlock',
    'HawkModel',
]
