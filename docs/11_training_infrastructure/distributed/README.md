# Distributed Training

Parallelism strategies for training across multiple GPUs/nodes.

## Overview

Distributed training enables:
- Training models too large for a single GPU
- Faster training through parallelism
- Efficient use of compute clusters

## Parallelism Strategies

| Strategy | Partitions | Communication | Best For |
|----------|-----------|---------------|----------|
| **Data Parallel (DDP)** | Batch | Gradients | Small models, many GPUs |
| **FSDP2** | Model, Optimizer, Gradients | Weights, Gradients | Large models (>1B params) |
| **Tensor Parallel** | Layers | Activations | Very large layers |
| **Pipeline Parallel** | Layers | Activations | Very deep models |
| **Context Parallel** | Sequence | KV cache | Long contexts (>32K tokens) |
| **ZeRO++** | With quantization | Quantized comm | Communication-bound |

## Quick Start

### FSDP2 (Fully Sharded Data Parallel)

```python
from nexus.training.distributed import wrap_model_fsdp2, FSDP2Config

config = FSDP2Config(
    sharding_strategy="full",  # ZERO-3 style
    mixed_precision=True,
    activation_checkpointing=True,
)

model = wrap_model_fsdp2(model, config)
```

### ZeRO++ (4x Communication Reduction)

```python
from nexus.training.distributed import ZeroPlusPlusOptimizer, ZeroPlusPlusConfig

config = ZeroPlusPlusConfig(
    quantize_weights=True,      # Quantized all-gather
    quantize_gradients=True,     # Quantized reduce-scatter
    hierarchical_partition=True,  # Per-node optimization
)

optimizer = ZeroPlusPlusOptimizer(
    model.parameters(),
    base_optimizer=torch.optim.AdamW,
    config=config,
    lr=1e-4,
)
```

### Context Parallelism (Long Context)

```python
from nexus.training.distributed import (
    init_context_parallel_group,
    ContextParallelAttention,
)

# Initialize 4-way sequence parallelism
cp_group = init_context_parallel_group(cp_size=4)

# Use in attention layers
attention = ContextParallelAttention(
    hidden_size=2048,
    num_heads=16,
    cp_group=cp_group,
)
```

## Detailed Documentation

- [FSDP2](fsdp2.md) - Next-generation fully sharded data parallelism
- [ZeRO++](zero_plusplus.md) - 4x communication reduction
- [Context Parallelism](context_parallelism.md) - Sequence-length parallelism

## Memory Scaling

### FSDP2 Memory Per GPU (7B model)

| GPUs | Params/GPU | Memory/GPU | Total Memory |
|------|------------|------------|--------------|
| 1 | 7B | 84 GB | 84 GB |
| 2 | 3.5B | 42 GB | 84 GB |
| 4 | 1.75B | 21 GB | 84 GB |
| 8 | 875M | 10.5 GB | 84 GB |

### Context Parallelism (Seq Length Scaling)

| Seq Length | GPUs (CP) | Memory/GPU | Max Seq Length |
|------------|-----------|------------|----------------|
| 32K | 1 | 24 GB | 32K |
| 32K | 4 | 6 GB | 128K |
| 32K | 8 | 3 GB | 256K |
| 32K | 16 | 1.5 GB | 512K |

## References

See individual strategy documentation for detailed references.
