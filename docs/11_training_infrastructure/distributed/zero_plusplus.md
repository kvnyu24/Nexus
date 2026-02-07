# ZeRO++: 4x Communication Reduction for Distributed Training

## Overview

ZeRO++ (Wang et al., 2023) extends Microsoft's ZeRO optimizer with three key optimizations that reduce communication by up to 4x:
1. **qwZ**: Quantized weight communication (INT8/FP8 all-gather)
2. **hpZ**: Hierarchical partitioning (partition within/across nodes differently)
3. **qgZ**: Quantized gradient communication (INT8 reduce-scatter)

## Three Optimizations

### 1. Quantized Weight Communication (qwZ)

**Problem**: All-gather weights in FP32 = expensive

**Solution**: Quantize to INT8 during all-gather, dequantize on arrival

```
Normal all-gather: 4 bytes/param
Quantized (INT8):  1 byte/param + small scales
Savings: 4x
```

**Block-wise quantization** (typically 64-128 elements per block):
$$\text{scale}_i = \frac{\max|\text{block}_i|}{127}$$
$$\text{quant}_i = \text{round}(\text{block}_i / \text{scale}_i)$$

### 2. Hierarchical Partitioning (hpZ)

**Problem**: Cross-node communication slower than intra-node

**Solution**: 
- **Primary partition**: Across all GPUs (for parallelism)
- **Secondary partition**: Replicate within nodes (fast access)

**Effect**: Reduces cross-node all-gathers significantly

### 3. Quantized Gradient Communication (qgZ)

**Problem**: Reduce-scatter gradients in FP32 = expensive

**Solution**: Quantize to INT8, use error feedback to preserve accuracy

```python
# Quantize
quantized_grad = quantize(grad)

# Reduce
reduced = all_reduce(quantized_grad)

# Dequantize
deq_grad = dequantize(reduced)

# Error feedback (crucial!)
error = grad - deq_grad
carry_over_to_next_step(error)
```

## Implementation

### Basic Usage

```python
from nexus.training.distributed.zero_plusplus import ZeroPlusPlusOptimizer, ZeroPlusPlusConfig

config = ZeroPlusPlusConfig(
    quantize_weights=True,  # qwZ
    quantize_gradients=True,  # qgZ
    hierarchical_partition=True,  # hpZ
    weight_quantization_bits=8,
    gradient_quantization_bits=8,
    block_size=64,
    use_error_feedback=True  # Essential for accuracy!
)

optimizer = ZeroPlusPlusOptimizer(
    model.parameters(),
    base_optimizer=torch.optim.AdamW,
    config=config,
    lr=1e-4,
    weight_decay=0.1
)

# Training (same as any optimizer)
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Configuration Options

**Quantization Bits**: 4, 6, or 8
- **8-bit**: Best accuracy, 4x reduction
- **6-bit**: Good accuracy, 5.3x reduction
- **4-bit**: Lower accuracy, 8x reduction
- **Recommended**: 8-bit for gradients, 6-bit for weights

**Block Size**: 64 or 128
- Larger: Less overhead, slightly lower accuracy
- Smaller: More overhead, better accuracy
- **Recommended**: 64

**Error Feedback**: Always enable for gradients
- Accumulates quantization error
- Applies correction next step
- Critical for maintaining accuracy

## Communication Reduction

### Baseline ZeRO-3 (per training step)

**Forward**:
- All-gather parameters: N params × 4 bytes = 4N bytes

**Backward**:
- Reduce-scatter gradients: N params × 4 bytes = 4N bytes

**Total**: 8N bytes per step

### With ZeRO++

**Forward (qwZ, 8-bit)**:
- All-gather quantized params: N params × 1 byte = N bytes
- All-gather scales: N/64 × 2 bytes ≈ N/32 bytes
- **Total**: ~1.03N bytes (3.9x reduction)

**Backward (qgZ, 8-bit)**:
- Reduce-scatter quantized grads: N bytes
- **Total**: ~1.03N bytes (3.9x reduction)

**Combined**: ~2.06N bytes per step

**Overall Reduction**: 8N / 2.06N ≈ **3.9x**

With hpZ (hierarchical partitioning), further reduces cross-node traffic.

## Memory vs Communication Trade-off

| Config | Memory | Communication | Accuracy |
|--------|--------|---------------|----------|
| **ZeRO-3** | Baseline | 8N bytes/step | 100% |
| **ZeRO++ (8-bit)** | +2% | 2.1N bytes/step | 99.5% |
| **ZeRO++ (6-bit)** | +3% | 1.6N bytes/step | 98.5% |
| **ZeRO++ (4-bit)** | +5% | 1.2N bytes/step | 95-97% |

**Memory overhead**: Quantization buffers and scales

## Performance

### Training Time Reduction

**Measured on 64 GPUs** (30B param model):
- ZeRO-3: 100%
- ZeRO++ (8-bit): 65% (1.54x faster)
- ZeRO++ (6-bit): 58% (1.72x faster)

**Speedup increases with**:
- More GPUs (communication becomes bottleneck)
- Slower network (communication cost higher)
- Larger models (more data to transfer)

### Accuracy

**GPT-style models**:
- 8-bit: <0.1% perplexity increase
- 6-bit: 0.2-0.5% perplexity increase
- 4-bit: 1-2% perplexity increase

**With error feedback**: Nearly perfect accuracy even at 8-bit

## Advanced Usage

### Estimate Savings

```python
from nexus.training.distributed.zero_plusplus import estimate_zero_plusplus_savings

savings = estimate_zero_plusplus_savings(
    model_params=70_000_000_000,  # 70B params
    world_size=64,
    config=config
)

print(f"Baseline: {savings['baseline_gb']:.1f} GB")
print(f"ZeRO++:   {savings['zero_plusplus_gb']:.1f} GB")
print(f"Reduction: {savings['reduction_factor']:.2f}x")
print(f"Savings:   {savings['savings_percent']:.1f}%")
```

### Hierarchical Groups

For multi-node training, set secondary group size to GPUs per node:

```python
config = ZeroPlusPlusConfig(
    hierarchical_partition=True,
    secondary_group_size=8  # 8 GPUs per node
)
```

This replicates data within nodes for fast access, shards across nodes for parallelism.

## Comparison to Other Methods

### ZeRO++ vs Standard ZeRO-3

| Aspect | ZeRO-3 | ZeRO++ |
|--------|--------|---------|
| **Communication** | 8N bytes/step | 2N bytes/step |
| **Memory** | Baseline | +2-5% |
| **Accuracy** | 100% | 99.5-99.9% |
| **Implementation** | Simple | Moderate complexity |

### ZeRO++ vs FSDP

FSDP is PyTorch's implementation of ZeRO-3. ZeRO++ can be thought of as "optimized ZeRO-3":
- Quantized communication
- Hierarchical partitioning
- Error feedback

**Use ZeRO++**: When communication is bottleneck (many GPUs, slow network)

## When to Use

**Best for**:
- Very large models (>10B params)
- Many GPUs (>16)
- Slow inter-node communication
- Long training runs (amortizes implementation complexity)

**Not necessary for**:
- Small models (<1B params)
- Few GPUs (<8)
- Fast networks (NVLink/InfiniBand with high bandwidth)
- Short experiments

## Troubleshooting

### Accuracy Degradation

1. **Enable error feedback**: `use_error_feedback=True` (essential!)
2. **Increase quantization bits**: Try 8-bit instead of 6-bit
3. **Larger block size**: Reduces quantization error
4. **Check gradient clipping**: Clip after unscaling quantized gradients

### No Speedup

1. **Verify fast intra-node network**: hpZ requires fast local communication
2. **Check quantization overhead**: May outweigh benefits on very fast networks
3. **Profile communication**: Ensure communication is actually the bottleneck

## References

**ZeRO++: Extremely Efficient Collective Communication for Giant Model Training**  
Guanhua Wang et al., Microsoft, 2023  
https://arxiv.org/abs/2306.10209

**Implementation**: `nexus/training/distributed/zero_plusplus.py`
