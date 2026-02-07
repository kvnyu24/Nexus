# Selective Activation Checkpointing: Fine-Grained Memory Optimization

## Overview

Selective activation checkpointing (Korthikanti et al., 2023) provides fine-grained control over which operations are checkpointed, optimizing the memory-compute tradeoff. Unlike blanket checkpointing, selectively checkpoints only memory-intensive operations.

## Memory-Compute Tradeoff

### Standard Training
**Memory**: Store all activations for backward pass  
**Compute**: One forward, one backward

### Full Checkpointing
**Memory**: Store only selected activations, recompute rest  
**Compute**: One forward, one backward, one recomputation pass  
**Tradeoff**: 2x compute for N-fold memory reduction

### Selective Checkpointing
**Memory**: Store cheap activations, checkpoint expensive ones  
**Compute**: Minimal recomputation (only heavy ops)  
**Tradeoff**: <1.5x compute for significant memory reduction

## Checkpointing Policies

### 1. NONE
No checkpointing (baseline).

### 2. ALL
Checkpoint every operation (maximum memory savings, maximum recomputation).

### 3. HEAVY_OPS
Checkpoint only memory-intensive operations:
- Multi-head attention
- Large linear layers
- Large convolutions

**Best for**: Transformer models (attention is memory bottleneck)

### 4. ALTERNATE
Checkpoint every other layer.

**Best for**: Balanced memory/compute tradeoff

### 5. AUTO
Automatically checkpoint based on operation size (>1M parameters).

**Best for**: Unknown architecture, let system decide

### 6. CUSTOM
User-defined policy function.

**Best for**: Expert users with specific requirements

## Implementation

### Basic Usage

```python
from nexus.training.gradient_methods import SelectiveCheckpoint, SelectiveCheckpointConfig, CheckpointPolicy

# Configure policy
config = SelectiveCheckpointConfig(
    policy=CheckpointPolicy.HEAVY_OPS,
    preserve_rng_state=True,
    use_reentrant=False  # PyTorch 2.0+ non-reentrant mode
)

# Create checkpoint wrapper
checkpoint_fn = SelectiveCheckpoint(config)

# Apply to module
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention(...)
        self.ffn = FeedForward(...)
    
    def forward(self, x):
        # Checkpoint attention (heavy operation)
        x = checkpoint_fn(self.attention, x)
        
        # No checkpoint for FFN (cheap operation)
        x = self.ffn(x)
        
        return x
```

### Apply to Entire Model

```python
from nexus.training.gradient_methods import apply_selective_checkpointing

config = SelectiveCheckpointConfig(policy=CheckpointPolicy.HEAVY_OPS)

model = MyTransformer()
model = apply_selective_checkpointing(model, config)

# Now all heavy ops automatically checkpointed
```

### Custom Policy

```python
def my_policy(module):
    # Checkpoint attention and large linear layers
    if isinstance(module, MultiHeadAttention):
        return True
    if isinstance(module, nn.Linear) and module.in_features > 1024:
        return True
    return False

config = SelectiveCheckpointConfig(
    policy=CheckpointPolicy.CUSTOM,
    custom_policy=my_policy
)

checkpoint_fn = SelectiveCheckpoint(config)
```

## Memory Savings Estimation

```python
from nexus.training.gradient_methods import estimate_checkpointing_memory_savings

savings = estimate_checkpointing_memory_savings(
    activation_memory_mb=100,  # Memory per layer
    num_checkpointed_layers=24,  # Layers to checkpoint
    total_layers=32  # Total layers
)

print(f"Without checkpointing: {savings['without_checkpointing_mb']:.0f} MB")
print(f"With checkpointing: {savings['with_checkpointing_mb']:.0f} MB")
print(f"Savings: {savings['savings_mb']:.0f} MB ({savings['savings_percent']:.1f}%)")
```

## PyTorch 2.0+ Non-Reentrant Mode

**Recommended**: Use `use_reentrant=False` (PyTorch 2.0+)

**Advantages**:
1. **Better memory efficiency**: No need to save RNG state
2. **Cleaner backward**: Proper gradient flow
3. **Fewer bugs**: Simpler implementation

```python
config = SelectiveCheckpointConfig(
    policy=CheckpointPolicy.HEAVY_OPS,
    use_reentrant=False  # Use new non-reentrant API
)
```

## Typical Configurations

### GPT-Style Transformer

```python
# Checkpoint attention, not FFN
config = SelectiveCheckpointConfig(
    policy=CheckpointPolicy.HEAVY_OPS,
    heavy_ops=['MultiheadAttention', 'Attention', 'SelfAttention']
)
```

**Result**: ~40% memory reduction, <10% compute overhead

### Vision Transformer

```python
# Checkpoint transformer blocks
config = SelectiveCheckpointConfig(
    policy=CheckpointPolicy.ALTERNATE  # Every other block
)
```

**Result**: ~50% memory reduction, ~50% compute overhead

### Large CNN

```python
# Checkpoint large convolutions
config = SelectiveCheckpointConfig(
    policy=CheckpointPolicy.AUTO,  # Auto-detect large layers
    auto_wrap_min_params=1_000_000
)
```

## Performance Characteristics

| Policy | Memory Savings | Compute Overhead | Use Case |
|--------|----------------|------------------|----------|
| **NONE** | 0% | 0% | Baseline |
| **HEAVY_OPS** | 30-50% | 5-15% | Transformers |
| **ALTERNATE** | ~50% | ~50% | Balanced |
| **ALL** | 70-90% | 100% | Maximum memory savings |

## Best Practices

### 1. Checkpoint Heavy Operations
- Multi-head attention
- Large matmuls
- Expensive activations

### 2. Don't Checkpoint Cheap Operations
- Element-wise operations
- Normalization layers
- Small linear layers

### 3. Profile Before Deciding
```python
# Measure memory and time
import torch.autograd.profiler as profiler

with profiler.profile(use_cuda=True) as prof:
    loss = model(input)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_memory_usage"))
```

### 4. Combine with Other Techniques
- Gradient accumulation
- Mixed precision (AMP)
- Distributed training (FSDP)

## Debugging

### Check What's Checkpointed

```python
checkpoint_fn = SelectiveCheckpoint(config)

for name, module in model.named_modules():
    is_checkpointed = checkpoint_fn.should_checkpoint(module)
    if is_checkpointed:
        print(f"✓ Checkpointing: {name}")
    else:
        print(f"✗ Not checkpointing: {name}")
```

### Measure Actual Savings

```python
import torch

# Without checkpointing
torch.cuda.reset_peak_memory_stats()
loss = model_no_checkpoint(input)
loss.backward()
memory_no_cp = torch.cuda.max_memory_allocated() / 1e9

# With checkpointing
torch.cuda.reset_peak_memory_stats()
loss = model_with_checkpoint(input)
loss.backward()
memory_with_cp = torch.cuda.max_memory_allocated() / 1e9

print(f"Memory savings: {(1 - memory_with_cp/memory_no_cp)*100:.1f}%")
```

## Limitations

1. **Recomputation Cost**: Always adds some overhead
2. **RNG State**: May need special handling for dropout
3. **Debugging**: Harder to debug (activations not stored)

## When to Use

**Use selective checkpointing when**:
- Training large models (>1B params)
- Memory is bottleneck
- Can tolerate small compute overhead

**Don't use when**:
- Small models (overhead not worth it)
- Memory is plentiful
- Compute is critical bottleneck

## References

**Reducing Activation Recomputation in Large Transformer Models**  
Korthikanti et al., 2023

**PyTorch Activation Checkpointing**  
https://pytorch.org/docs/stable/checkpoint.html

**Implementation**: `nexus/training/gradient_methods.py` (SelectiveCheckpoint class)
