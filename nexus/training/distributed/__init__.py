"""Nexus distributed training module.

Provides advanced distributed training strategies including:
- FSDP2: Next-generation fully sharded data parallelism
- ZeRO++: 4x communication reduction for distributed training
- Context Parallelism: Sequence-length parallelism for long contexts

Example:
    >>> # FSDP2
    >>> from nexus.training.distributed import FSDP2Config, wrap_model_fsdp2
    >>> config = FSDP2Config(sharding_strategy="full")
    >>> model = wrap_model_fsdp2(model, config)
    >>>
    >>> # ZeRO++
    >>> from nexus.training.distributed import ZeroPlusPlusConfig, ZeroPlusPlusOptimizer
    >>> config = ZeroPlusPlusConfig(quantize_weights=True)
    >>> optimizer = ZeroPlusPlusOptimizer(
    ...     model.parameters(),
    ...     base_optimizer=torch.optim.AdamW,
    ...     config=config,
    ... )
    >>>
    >>> # Context Parallelism
    >>> from nexus.training.distributed import init_context_parallel_group, ContextParallelAttention
    >>> cp_group = init_context_parallel_group(cp_size=4)
    >>> attention = ContextParallelAttention(hidden_size=2048, num_heads=16, cp_group=cp_group)
"""

from .fsdp2 import (
    FSDP2Config,
    FSDP2Wrapper,
    wrap_model_fsdp2,
    apply_activation_checkpointing,
    save_fsdp2_checkpoint,
    load_fsdp2_checkpoint,
)

from .zero_plusplus import (
    ZeroPlusPlusConfig,
    ZeroPlusPlusOptimizer,
    QuantizedAllGather,
    QuantizedReduceScatter,
    HierarchicalPartitioner,
    quantize_block_wise,
    dequantize_block_wise,
    estimate_zero_plusplus_savings,
)

from .context_parallelism import (
    ContextParallelGroup,
    ContextParallelAttention,
    RingAttentionCommunicator,
    init_context_parallel_group,
    estimate_context_parallel_memory_savings,
)


__all__ = [
    # FSDP2
    "FSDP2Config",
    "FSDP2Wrapper",
    "wrap_model_fsdp2",
    "apply_activation_checkpointing",
    "save_fsdp2_checkpoint",
    "load_fsdp2_checkpoint",

    # ZeRO++
    "ZeroPlusPlusConfig",
    "ZeroPlusPlusOptimizer",
    "QuantizedAllGather",
    "QuantizedReduceScatter",
    "HierarchicalPartitioner",
    "quantize_block_wise",
    "dequantize_block_wise",
    "estimate_zero_plusplus_savings",

    # Context Parallelism
    "ContextParallelGroup",
    "ContextParallelAttention",
    "RingAttentionCommunicator",
    "init_context_parallel_group",
    "estimate_context_parallel_memory_savings",
]
