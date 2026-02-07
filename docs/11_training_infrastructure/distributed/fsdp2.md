# FSDP2: Next-Generation Fully Sharded Data Parallel

## Overview

FSDP2 is PyTorch's improved fully sharded data parallelism, providing better performance, easier debugging, and more flexible memory management than FSDP1. Enables training models that don't fit on a single GPU by sharding parameters, gradients, and optimizer states across GPUs.

## Key Improvements Over FSDP1

1. **Better Communication Overlap**: Computation and communication pipelined more efficiently
2. **Improved Checkpointing**: Seamless integration with activation checkpointing
3. **Heterogeneous Sharding**: Mix different sharding strategies in one model
4. **Cleaner API**: Easier to use and debug
5. **Better Profiling**: More visibility into performance bottlenecks

## Sharding Strategies

### FULL_SHARD (ZeRO-3 Style)
- Shards parameters, gradients, AND optimizer states
- Maximum memory savings
- All-gather on forward, reduce-scatter on backward
- **Best for**: Largest models

### SHARD_GRAD_OP (ZeRO-2 Style)
- Shards gradients and optimizer states only
- Parameters replicated
- Less communication than FULL_SHARD
- **Best for**: Medium models, fast interconnect

### HYBRID_SHARD
- Shard within nodes, replicate across nodes
- Leverages fast intra-node communication
- **Best for**: Multi-node training

### NO_SHARD (DDP Style)
- Replicate everything
- Minimal communication (only gradient reduce)
- **Best for**: Small models, many GPUs

## Implementation

### Basic Setup

```python
from nexus.training.distributed.fsdp2 import wrap_model_fsdp2, FSDP2Config

config = FSDP2Config(
    sharding_strategy="full",  # "full", "shard_grad_op", "hybrid_full", "no_shard"
    mixed_precision=True,
    compute_dtype=torch.bfloat16,
    cpu_offload=False,
    activation_checkpointing=False,
    auto_wrap_policy="transformer"  # Automatically wrap transformer blocks
)

model = MyTransformer()
model = wrap_model_fsdp2(model, config)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop (same as non-FSDP!)
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### With Activation Checkpointing

```python
from nexus.training.distributed.fsdp2 import apply_activation_checkpointing

model = wrap_model_fsdp2(model, config)
apply_activation_checkpointing(model)  # Checkpoint transformer blocks

# Now training uses less memory (recomputes activations in backward)
```

### Saving and Loading Checkpoints

```python
from nexus.training.distributed.fsdp2 import save_fsdp2_checkpoint, load_fsdp2_checkpoint

# Save (only rank 0 writes to disk)
save_fsdp2_checkpoint(model, optimizer, "checkpoint.pt", rank=0)

# Load (all ranks load)
load_fsdp2_checkpoint(model, optimizer, "checkpoint.pt")
```

## Memory Breakdown

For model with N parameters across P GPUs:

| Strategy | Parameters | Gradients | Optimizer |
|----------|-----------|-----------|-----------|
| **DDP** | N | N | N |
| **FSDP (SHARD_GRAD_OP)** | N | N/P | N/P |
| **FSDP (FULL_SHARD)** | N/P | N/P | N/P |

**Example**: 10B param model, 8 GPUs, FULL_SHARD
- Per GPU: 1.25B params (vs 10B with DDP)
- **8x memory reduction**

## Performance Optimization

### Communication-Computation Overlap

```python
config = FSDP2Config(
    sharding_strategy="full",
    backward_prefetch="backward_pre",  # Prefetch next layer's params
    forward_prefetch=True,  # Prefetch during forward too
    limit_all_gathers=True  # Limit concurrent all-gathers (memory vs speed)
)
```

### Mixed Precision

```python
config = FSDP2Config(
    mixed_precision=True,
    compute_dtype=torch.bfloat16,  # Compute in BF16
    param_dtype=torch.float32,  # Store params in FP32
    reduce_dtype=torch.float32  # Reduce gradients in FP32
)
```

## Auto-Wrapping Policies

### Transformer Policy

```python
config = FSDP2Config(
    auto_wrap_policy="transformer",  # Wraps TransformerBlock-like modules
    sharding_strategy="full"
)
```

### Size-Based Policy

```python
config = FSDP2Config(
    auto_wrap_policy="size_based",
    auto_wrap_min_params=100_000_000,  # Wrap modules with >100M params
    sharding_strategy="full"
)
```

### Manual Wrapping

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Wrap specific submodules
model.layer1 = FSDP(model.layer1, ...)
model.layer2 = FSDP(model.layer2, ...)
```

## Debugging Tips

### Check Memory Usage

```python
from nexus.training.distributed.fsdp2 import FSDP2Wrapper

wrapper = FSDP2Wrapper(config)
model = wrapper.wrap_model(model)

# Get memory stats
stats = wrapper.get_memory_stats(model)
print(f"Allocated: {stats['allocated_gb']:.2f} GB")
print(f"Max Allocated: {stats['max_allocated_gb']:.2f} GB")
```

### Verify Sharding

```python
# Check if parameters are sharded
for name, param in model.named_parameters():
    print(f"{name}: shape {param.shape}, device {param.device}")
```

### Profile Communication

```python
with torch.profiler.profile() as prof:
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Common Issues

### Out of Memory
1. Use FULL_SHARD (not SHARD_GRAD_OP)
2. Enable CPU offload: `cpu_offload=True`
3. Add activation checkpointing
4. Reduce batch size or sequence length

### Slow Training
1. Check `backward_prefetch="backward_pre"`
2. Verify mixed precision enabled
3. Use `limit_all_gathers=False` if memory allows
4. Check network bandwidth between nodes

### Incorrect Results
1. Verify `sync_module_states=True` (syncs params at start)
2. Check mixed precision settings
3. Ensure gradient clipping done AFTER unscaling

## Performance Expectations

**Scalability** (training time):
- 8 GPUs: ~7x speedup vs 1 GPU (FULL_SHARD)
- 64 GPUs: ~50-55x speedup

**Memory Efficiency**:
- FULL_SHARD: ~N/P parameters per GPU
- With activation checkpointing: ~sqrt(L) memory for L layers

## References

**PyTorch FSDP2: Rethinking Fully Sharded Data Parallelism**  
PyTorch Team, 2024  
https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

**Implementation**: `nexus/training/distributed/fsdp2.py`
