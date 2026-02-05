# Gradient Methods

Memory-efficient gradient computation techniques.

## Overview

Gradient methods reduce memory usage during training:
- Activation checkpointing (recompute instead of store)
- Activation offloading (move to CPU)
- Selective application for optimal tradeoffs

## Technique Comparison

| Method | Memory Savings | Compute Overhead | Implementation |
|--------|----------------|------------------|----------------|
| **Full Checkpointing** | 90% | +100% | Simple |
| **Selective Checkpointing** | 50-70% | +30% | Flexible |
| **Activation Offloading** | 50-90% | +20% (transfer) | Complex |
| **Combination** | 95% | +50% | Best |

## Quick Start

### Selective Checkpointing

```python
from nexus.training.gradient_methods import (
    SelectiveCheckpointConfig,
    apply_selective_checkpointing,
    CheckpointPolicy,
)

# Checkpoint only heavy operations (attention, large linears)
config = SelectiveCheckpointConfig(policy=CheckpointPolicy.HEAVY_OPS)
model = apply_selective_checkpointing(model, config)

# Or alternate layers
config = SelectiveCheckpointConfig(policy=CheckpointPolicy.ALTERNATE)
model = apply_selective_checkpointing(model, config)
```

### Activation Offloading

```python
from nexus.training.gradient_methods import ActivationOffloader

offloader = ActivationOffloader(
    enabled=True,
    offload_threshold_mb=10.0,  # Offload activations > 10 MB
)

# Use as context manager
with offloader.offload_context():
    output = model(input)
    loss = criterion(output, target)
    loss.backward()  # Activations auto-prefetched
```

## Detailed Documentation

- [Selective Checkpointing](selective_checkpointing.md) - Fine-grained activation checkpointing
- [Activation Offloading](activation_offloading.md) - CPU offloading with async prefetch

## Memory Savings Analysis

### GPT-3 Style Model (175B parameters)

| Technique | Activation Memory | Compute Overhead |
|-----------|-------------------|------------------|
| None | 180 GB | 0% |
| Selective (Heavy Ops) | 72 GB (60% savings) | +30% |
| Full Checkpointing | 18 GB (90% savings) | +100% |
| Offloading (50%) | 90 GB (50% savings) | +20% |
| Selective + Offloading | 36 GB (80% savings) | +50% |

### Batch Size Impact

With freed memory, increase batch size:

| Method | Memory (GB) | Batch Size | Effective Throughput |
|--------|-------------|------------|---------------------|
| None | 180 | 1 | 1.0x |
| Selective | 72 | 2 | 1.5x (2 / 1.3) |
| Full Checkpoint | 18 | 8 | 4.0x (8 / 2.0) |

## References

See individual method documentation for detailed references.
