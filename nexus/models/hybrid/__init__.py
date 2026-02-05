"""
Hybrid architecture models combining SSM, attention, and MoE components.

These models interleave different sequence modeling mechanisms (recurrence,
attention, convolution) to achieve the strengths of each approach:
- Griffin: Gated linear recurrence + local multi-query attention
- Hyena: Long convolutions with data-controlled gating
- Jamba: Transformer attention + Mamba SSM + Mixture-of-Experts
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
]
