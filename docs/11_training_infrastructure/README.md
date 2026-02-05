# Training Infrastructure

Comprehensive documentation for efficient training infrastructure components in Nexus.

## Overview

Modern deep learning requires sophisticated training infrastructure to handle:
- Large-scale models (billions of parameters)
- Long training runs (weeks to months)
- Limited hardware resources
- Communication bottlenecks in distributed settings
- Memory constraints

This documentation covers state-of-the-art techniques for optimizing training efficiency, reducing memory usage, and accelerating convergence.

## Categories

### 1. [Optimizers](optimizers/)
Advanced optimization algorithms that improve convergence speed and reduce hyperparameter sensitivity:
- **Lion**: Evolved sign momentum with 50% memory reduction vs AdamW
- **Sophia**: Second-order optimizer with diagonal Hessian estimation
- **Prodigy**: Learning-rate-free adaptive optimization
- **Schedule-Free AdamW**: Eliminates need for LR schedules
- **SOAP**: Shampoo with Adam preconditioning for better conditioning
- **Muon**: Momentum + orthogonalization for transformer training

### 2. [LR Schedules](schedules/)
Learning rate scheduling strategies for stable and efficient training:
- **Cosine Annealing with Restarts (SGDR)**: Periodic warm restarts for exploration
- **Warmup-Stable-Decay (WSD)**: Three-phase schedule without predefined total steps
- **Linear Warmup + Cosine Decay**: Standard approach for transformer training

### 3. [Mixed Precision](mixed_precision/)
Low-precision training techniques for memory and compute efficiency:
- **FP8 Training**: 8-bit floating point with dynamic scaling
- **MXFP8**: Microscaling FP8 with block-level scaling
- **FP4/MXFP4**: 4-bit training for extreme memory reduction

### 4. [Distributed Training](distributed/)
Parallelism strategies for training across multiple GPUs/nodes:
- **FSDP2**: Next-generation fully sharded data parallelism
- **ZeRO++**: 4x communication reduction via quantized communication
- **Context Parallelism**: Sequence-length parallelism for long contexts (>1M tokens)

### 5. [Loss Functions](losses/)
Specialized loss functions for contrastive learning and self-supervised training:
- **InfoNCE**: Contrastive loss for representation learning
- **SigLIP Loss**: Sigmoid-based contrastive loss (more scalable than softmax)
- **VICReg**: Variance-invariance-covariance regularization

### 6. [Gradient Methods](gradient_methods/)
Memory-efficient gradient computation techniques:
- **Selective Checkpointing**: Fine-grained activation checkpointing
- **Activation Offloading**: CPU offloading with async prefetch

## Quick Start

### Basic Training Loop

```python
import torch
from nexus.training.optimizers import Lion
from nexus.training.schedulers import WSDScheduler

# Initialize optimizer
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1.0)

# Initialize scheduler
scheduler = WSDScheduler(
    optimizer,
    warmup_steps=1000,
    stable_steps=50000,
    decay_steps=10000,
    peak_lr=1e-3,
)

# Training loop
for step, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

### Mixed Precision Training

```python
from nexus.training.mixed_precision import FP8Linear, convert_to_fp8

# Convert model to FP8
model = convert_to_fp8(model, fp8_format=FP8Format.E4M3)

# Or use FP8 layers directly
layer = FP8Linear(768, 3072, fp8_format=FP8Format.E4M3)
```

### Distributed Training

```python
from nexus.training.distributed import wrap_model_fsdp2, FSDP2Config

# Configure FSDP2
config = FSDP2Config(
    sharding_strategy="full",
    mixed_precision=True,
    activation_checkpointing=True,
)

# Wrap model
model = wrap_model_fsdp2(model, config)
```

### Memory-Efficient Gradient Computation

```python
from nexus.training.gradient_methods import SelectiveCheckpointConfig, apply_selective_checkpointing

# Configure selective checkpointing
config = SelectiveCheckpointConfig(policy="heavy_ops")

# Apply to model
model = apply_selective_checkpointing(model, config)
```

## Performance Comparison

| Technique | Memory Savings | Compute Overhead | Quality Impact |
|-----------|----------------|------------------|----------------|
| Lion (vs AdamW) | 50% optimizer states | None | Equal/Better |
| Sophia (vs Adam) | None | +20% (Hessian) | 2x faster convergence |
| Schedule-Free | None | None | Equal |
| FP8 Training | 50% weights | None | <1% degradation |
| MXFP8 | 50% weights | None | <0.5% degradation |
| FSDP2 | Linear w/ GPUs | Minimal | None |
| ZeRO++ | 4x comm reduction | None | None |
| Context Parallelism | Linear w/ seq_len | Ring comm | None |
| Selective Checkpointing | 50-90% activations | +30% recompute | None |

## Best Practices

### Optimizer Selection
- **Large-scale training**: Use Lion (memory efficient) or Sophia (faster convergence)
- **Hyperparameter-free**: Use Prodigy (no LR tuning required)
- **Transformer training**: Use Muon (orthogonalization helps)
- **Stable training**: Use Schedule-Free AdamW (no schedule needed)

### Mixed Precision
- **H100/A100 GPUs**: Use FP8 for maximum efficiency
- **Memory-critical**: Use MXFP8 (better accuracy than standard FP8)
- **Extreme memory constraints**: Use FP4/MXFP4 (8x reduction)

### Distributed Strategy
- **Multi-node training**: Use FSDP2 with ZeRO++ for comm efficiency
- **Long context**: Use Context Parallelism (sequence sharding)
- **Hybrid approach**: Combine FSDP2 + Context Parallelism

### Memory Optimization
1. Start with FSDP2 (model sharding)
2. Add selective checkpointing (activation reduction)
3. Use mixed precision (FP8/MXFP8)
4. If still OOM, use activation offloading

## Implementation Details

All implementations follow these principles:
- **PyTorch native**: Built on standard PyTorch APIs
- **Distributed-friendly**: Work with DDP, FSDP, and other parallelism
- **Type-safe**: Full type annotations
- **Well-documented**: Extensive docstrings and examples
- **Production-ready**: Used in real large-scale training

## References

See individual documentation pages for detailed references to original papers and implementations.

## Contributing

When adding new training infrastructure:
1. Add implementation to `nexus/training/`
2. Create documentation with all sections (overview, math, implementation, experiments)
3. Include usage examples and benchmarks
4. Add tests for correctness and performance

## Support

For questions or issues related to training infrastructure, please open an issue on GitHub or consult the individual component documentation.
