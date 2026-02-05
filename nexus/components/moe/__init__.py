"""
Mixture of Experts (MoE) components.

Provides routing mechanisms and expert layers for sparse MoE architectures.
"""
from .router import (
    ExpertRouter,
    LoadBalancingLoss,
    LossFreeBalancing,
    ExpertChoiceRouter
)
from .expert import (
    ExpertLayer,
    SharedExpert,
    MoELayer,
    SparseMoE
)
from .mixture_of_agents import (
    MoALayer,
    MixtureOfAgents,
    SimpleMoA
)
from .deepseek_moe import (
    FineGrainedExpert,
    DeepSeekMoELayer,
    DeepSeekMoE
)
from .switch_transformer import (
    SwitchRouter,
    SwitchFFN,
    SwitchTransformerLayer,
    SwitchTransformer
)
from .switch_all import (
    SwitchAllLayer,
    SwitchAll,
    SwitchAllConfig
)

__all__ = [
    # Base MoE components
    'ExpertRouter',
    'LoadBalancingLoss',
    'LossFreeBalancing',
    'ExpertChoiceRouter',
    'ExpertLayer',
    'SharedExpert',
    'MoELayer',
    'SparseMoE',
    # Mixture-of-Agents (Multi-LLM collaboration)
    'MoALayer',
    'MixtureOfAgents',
    'SimpleMoA',
    # DeepSeek MoE (Shared + routed experts)
    'FineGrainedExpert',
    'DeepSeekMoELayer',
    'DeepSeekMoE',
    # Switch Transformer (Top-1 routing)
    'SwitchRouter',
    'SwitchFFN',
    'SwitchTransformerLayer',
    'SwitchTransformer',
    # SwitchAll (Full MoE attention + FFN)
    'SwitchAllLayer',
    'SwitchAll',
    'SwitchAllConfig',
]
