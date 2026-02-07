# Context Parallelism: Sequence-Length Parallelism for Long Context Training

## Overview

Context Parallelism (CP) enables training models with extremely long contexts (>1M tokens) by partitioning the sequence dimension across GPUs. Different from tensor parallelism (partitions model) and data parallelism (partitions batch).

## Key Concept

**Traditional Parallelism**:
- Data Parallel: Split batch across GPUs
- Tensor Parallel: Split model weights across GPUs
- Pipeline Parallel: Split layers across GPUs

**Context Parallel**: Split sequence length across GPUs

**Example**: 1M token sequence, 8 GPUs
- Each GPU: 125K tokens
- Memory: 8x reduction for sequence-dependent activations
- Communication: Ring-based KV passing

## Mathematical Foundation

### Ring Attention

For sequence split across P GPUs, each GPU $i$ holds:
- Query $Q_i$: local segment
- Key $K_i$, Value $V_i$: local segment

**Algorithm**:
```
For step = 0 to P-1:
    # Compute attention with current KV
    scores = Q_i @ K_current.T / sqrt(d)
    attn = softmax(scores) @ V_current
    
    # Accumulate with proper normalization
    output += attn
    
    # Ring communication: pass KV to next GPU
    K_current = receive_from_previous_gpu()
    V_current = receive_from_previous_gpu()
```

**Key insight**: Each GPU processes all KV blocks through ring communication, computing full attention without storing entire sequence.

### Memory Scaling

**Without CP** (single GPU):
- QKV memory: $B \\times L \\times 3H$
- Attention scores: $B \\times L \\times L$
- **Total**: $O(L^2)$ in sequence length

**With CP** (P GPUs):
- QKV memory per GPU: $B \\times (L/P) \\times 3H$
- Attention scores per GPU: $B \\times (L/P) \\times (L/P)$
- **Total per GPU**: $O((L/P)^2)$

**Effective max sequence length**: $L_{\\text{max}} \\times \\sqrt{P}$

## Implementation

### Basic Setup

```python
from nexus.training.distributed.context_parallelism import init_context_parallel_group, ContextParallelAttention

# Initialize CP group (4-way sequence parallelism)
cp_group = init_context_parallel_group(cp_size=4)

# Use in attention layer
attention = ContextParallelAttention(
    hidden_size=4096,
    num_heads=32,
    cp_group=cp_group,
    causal=True  # For autoregressive models
)

# Input: local sequence shard (batch, seq_len_local, hidden_size)
output = attention(x_local)  # Computes full attention via ring communication
```

### Complete Training Example

```python
import torch
import torch.distributed as dist
from nexus.training.distributed.context_parallelism import init_context_parallel_group

# Initialize distributed
dist.init_process_group(backend='nccl')

# 8 GPUs total: 4-way CP, 2-way DP
cp_group = init_context_parallel_group(cp_size=4)

# Model with CP attention
model = TransformerWithCP(cp_group=cp_group)

# Dataset: long sequences
dataset = LongSequenceDataset(seq_len=1_000_000)

# Partition sequences across CP group
for batch in dataloader:
    # Each GPU gets seq_len/4 = 250K tokens
    local_seq = cp_group.partition_sequence(batch['input_ids'], seq_dim=1)
    
    # Forward: ring attention computes full attention
    loss = model(local_seq)
    
    # Backward: standard (gradients automatically synced)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Memory Estimation

```python
from nexus.training.distributed.context_parallelism import estimate_context_parallel_memory_savings

savings = estimate_context_parallel_memory_savings(
    seq_len=1_000_000,
    hidden_size=4096,
    num_layers=32,
    batch_size=1,
    cp_size=8
)

print(f"Without CP: {savings['without_cp_gb']:.1f} GB")
print(f"With CP:    {savings['with_cp_gb']:.1f} GB")
print(f"Savings:    {savings['savings_gb']:.1f} GB")
print(f"Reduction:  {savings['reduction_factor']:.1f}x")
print(f"Max seq len (theoretical): {savings['max_seq_len_with_cp']:,} tokens")
```

## Performance Characteristics

**Memory Scaling**: $O((L/P)^2)$ per GPU  
**Communication**: $O(P)$ ring passes per attention layer  
**Compute**: Same total FLOPs as single-GPU  
**Efficiency**: 70-85% (communication overhead)

**Optimal Use Case**:
- Very long sequences (>100K tokens)
- Sufficient GPUs for CP (4-8 way typical)
- Fast interconnect (NVLink, InfiniBand)

## Causal Masking

For autoregressive models, causal masking is handled automatically:

```python
attention = ContextParallelAttention(
    hidden_size=4096,
    num_heads=32,
    cp_group=cp_group,
    causal=True  # Automatically applies causal mask across ring
)
```

**Implementation**: Masks future blocks entirely, within-block uses standard causal mask.

## Combining with Other Parallelism

### CP + DP (Common)
```
8 GPUs: 4-way CP, 2-way DP
- CP: Split sequence 4 ways
- DP: Replicate model 2 ways
- Max seq len: 4x longer
- Throughput: 2x batch size
```

### CP + TP (Advanced)
```
16 GPUs: 4-way CP, 4-way TP
- CP: Split sequence
- TP: Split model weights
- Enables: Very long sequences + very large models
```

### 3D Parallelism: CP + TP + DP
```
64 GPUs: 8-way CP, 4-way TP, 2-way DP
- CP: 8x longer sequences
- TP: 4x larger models
- DP: 2x throughput
```

## Limitations

**Sequence Length**: Must be divisible by cp_size
```python
assert seq_len % cp_size == 0, "Sequence length must divide evenly"
```

**Communication Bottleneck**: P ring passes per layer adds overhead
- **Mitigation**: Use with fast interconnect
- **Mitigation**: Combine with Flash Attention for efficiency

**Not All Operations Parallelizable**: Only attention benefits
- Embeddings: Replicated or model-parallel
- FFN: Replicated or tensor-parallel
- Only attention memory scales with $O((L/P)^2)$

## When to Use

**Best for**:
- Long context LLMs (>100K tokens)
- Document understanding, book-level comprehension
- RAG systems with huge context windows
- Genomic sequences, long time-series

**Not needed for**:
- Standard sequences (<8K tokens) - fits on single GPU
- Small batch sizes - other parallelism more important
- Models where attention is not memory bottleneck

## Debugging

**Check Sequence Partitioning**:
```python
# Verify each GPU has correct shard
local_seq = cp_group.partition_sequence(full_seq, seq_dim=1)
print(f"GPU {cp_group.rank}: seq shape {local_seq.shape}")

# Should be: (batch, seq_len/cp_size, hidden)
```

**Verify Ring Communication**:
```python
# Check KV passing in ring attention
# Set breakpoint in RingAttentionCommunicator.ring_forward()
# Verify send/recv operations
```

**Profile Communication Overhead**:
```python
with torch.profiler.profile() as prof:
    output = attention(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
# Look for send/recv/wait operations
```

## References

**Ring Attention with Blockwise Transformers for Near-Infinite Context**  
Liu et al., 2023  
https://arxiv.org/abs/2310.01889

**DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models**  
Jacobs et al., Microsoft, 2024

**Implementation**: `nexus/training/distributed/context_parallelism.py`
