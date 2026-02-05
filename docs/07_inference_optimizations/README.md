# Inference Optimizations for Large Language Models

This directory contains comprehensive documentation on state-of-the-art inference optimization techniques for LLMs. These techniques are essential for deploying models in production, reducing latency, increasing throughput, and managing memory efficiently.

## Overview

LLM inference is fundamentally constrained by three bottlenecks:

1. **Memory Bandwidth** - Moving weights and activations from memory to compute units
2. **Compute Utilization** - Keeping GPU cores busy during sequential generation
3. **Memory Capacity** - Storing KV caches and model weights within limited VRAM

Modern inference optimizations target these bottlenecks through various approaches:

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Inference Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Prefill Phase          Decode Phase                        │
│  ┌──────────┐          ┌──────────────────┐               │
│  │  Prompt  │   KV     │  Autoregressive  │               │
│  │Processing│  Cache   │    Generation    │               │
│  └──────────┘   ↓↓↓    └──────────────────┘               │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Optimization Layers                    │    │
│  ├────────────────────────────────────────────────────┤    │
│  │ Memory:    KV Cache | Quantization | Paging        │    │
│  │ Compute:   Speculative | Multi-Token | Parallel    │    │
│  │ Batching:  Continuous | Iteration-Level            │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Optimization Techniques

### Memory Optimizations

These techniques reduce memory footprint and improve memory bandwidth utilization:

| Technique | Memory Savings | Throughput Impact | Latency Impact |
|-----------|----------------|-------------------|----------------|
| **[KV Cache](01_kv_cache.md)** | Baseline | Baseline | Baseline |
| **[PagedAttention](02_paged_attention.md)** | 3-4x | +20-30% | ~0% |
| **[Quantized KV Cache](03_quantized_kv_cache.md)** | 2-4x | -5-10% | +5-10% |
| **[Prefix Caching](04_prefix_caching.md)** | Variable | +2-10x | -50-90% |

### Speculative Decoding Methods

These techniques predict multiple tokens in advance, then verify them in parallel:

| Technique | Speedup | Extra Memory | Model Requirements |
|-----------|---------|--------------|-------------------|
| **[Speculative Decoding](05_speculative_decoding.md)** | 2-3x | Draft model | Separate draft model |
| **[Medusa](06_medusa.md)** | 2-3x | Small heads | Fine-tuned heads |
| **[EAGLE-3](07_eagle.md)** | 2.5-4x | Small heads | Fine-tuned heads |
| **[Lookahead Decoding](08_lookahead_decoding.md)** | 1.5-2x | N-gram pool | None |

### Multi-Token Prediction

Predicting multiple future tokens simultaneously:

| Technique | Training Required | Inference Speedup | Quality Impact |
|-----------|-------------------|-------------------|----------------|
| **[Multi-Token Prediction](09_multi_token_prediction.md)** | Yes (from scratch) | 2-3x | Improved |

### Batching Strategies

Efficiently processing multiple requests simultaneously:

| Technique | Throughput Gain | Latency Impact | Complexity |
|-----------|-----------------|----------------|------------|
| **[Continuous Batching](10_continuous_batching.md)** | 5-10x | Minimal | Medium |
| **[Iteration-Level Batching](10_continuous_batching.md#iteration-level)** | 8-15x | Low | High |

## Performance Comparison

### Latency vs Throughput Trade-offs

```
Throughput (tokens/sec)
    │
    │                            ● Continuous Batching
    │                          ● Iteration-Level
    │
    │              ● EAGLE + Batching
    │            ● Medusa + Batching
    │
    │        ● Speculative Decoding
    │      ● EAGLE
    │    ● Medusa
    │  ● Lookahead
    │
    │● Baseline Autoregressive
    └────────────────────────────────────────► Latency (ms)
```

### Memory Usage Comparison

For a 7B model, 2048 sequence length, batch size 32:

```
Memory (GB)
    │
 80 │██████████████████████  Baseline (FP16)
    │
 40 │███████████  PagedAttention + Prefix Cache
    │
 30 │████████  + Quantized KV (INT8)
    │
 20 │█████  + Weight Quantization
    │
 10 │██  Optimal Stack
    │
  0 └────────────────────────────────────────►
```

## Combining Techniques

The most powerful approach is to **stack multiple optimizations**. Here are recommended combinations:

### 1. Production Serving Stack

**Goal**: Maximum throughput at reasonable latency

```python
from nexus.components.inference import (
    PagedKVCache,
    ContinuousBatcher,
    QuantizedKVCache
)

# Memory management
kv_cache = PagedKVCache(
    num_layers=32,
    num_heads=32,
    head_dim=128,
    block_size=16,
    num_blocks=2048
)

# Request batching
batcher = ContinuousBatcher(
    max_batch_size=128,
    max_seq_len=2048,
    kv_cache=kv_cache,
    scheduling_policy='priority'
)

# Memory compression
quantized_cache = QuantizedKVCache(
    num_layers=32,
    quant_type='int8'  # 2x memory reduction
)

# Expected: 10-20x throughput, <10% latency increase
```

### 2. Low-Latency Stack

**Goal**: Minimize time-to-first-token and generation latency

```python
from nexus.components.inference import (
    EAGLEDecoder,
    StaticKVCache,
    PrefixCache
)

# Fast speculative decoding
eagle_decoder = EAGLEDecoder(
    target_model=model,
    hidden_dim=4096,
    vocab_size=32000,
    tree_width=10,
    tree_depth=4
)

# Prefix caching for common prompts
prefix_cache = PrefixCache(
    max_entries=1000,
    eviction_policy='lru'
)

# Expected: 2-4x speedup, 50-90% TTFT reduction with prefix hits
```

### 3. Memory-Constrained Stack

**Goal**: Fit maximum batch size in limited VRAM

```python
from nexus.components.inference import (
    QuantizedKVCache,
    PagedKVCache,
    ContinuousBatcher
)

# Aggressive quantization
kv_cache = QuantizedKVCache(
    num_layers=32,
    quant_type='int4',  # 4x compression
    group_size=64
)

# Paging for fragmentation reduction
paged_cache = PagedKVCache(
    block_size=16,
    num_blocks=4096
)

# Dynamic batching
batcher = ContinuousBatcher(
    max_batch_size=256,  # 4x larger than baseline
    kv_cache=paged_cache
)

# Expected: 4x memory reduction, 3-4x throughput increase
```

### 4. Maximum Performance Stack

**Goal**: Best possible throughput and latency (research/experimentation)

```python
from nexus.components.inference import (
    EAGLEDecoder,
    PagedKVCache,
    IterationLevelBatcher,
    QuantizedKVCache,
    RadixPrefixCache
)

# All optimizations combined
eagle = EAGLEDecoder(target_model=model, ...)
paged_cache = PagedKVCache(...)
batcher = IterationLevelBatcher(...)
quant_cache = QuantizedKVCache(quant_type='int8', ...)
prefix_cache = RadixPrefixCache(...)

# Expected: 20-50x throughput vs baseline, 3-5x latency reduction
```

## Quick Start Guide

### 1. Identify Your Bottleneck

Run profiling to understand your constraints:

```python
from nexus.utils.profiling import InferenceProfiler

profiler = InferenceProfiler(model)
stats = profiler.profile(
    batch_size=32,
    seq_len=512,
    num_steps=100
)

print(f"Memory utilization: {stats.memory_util:.1%}")
print(f"Compute utilization: {stats.compute_util:.1%}")
print(f"Bandwidth utilization: {stats.bandwidth_util:.1%}")
```

### 2. Select Optimizations

Based on bottlenecks:

- **Memory-bound** (>90% memory, <50% compute) → KV Cache optimizations
- **Compute-bound** (<50% memory, >90% compute) → Speculative decoding
- **Low batch size** → Continuous batching
- **Long sequences** → PagedAttention + Prefix caching
- **High throughput needed** → Combine batching + memory optimizations

### 3. Implement and Benchmark

```python
# Baseline
baseline_throughput = benchmark(model, standard_inference)

# With optimization
optimized_throughput = benchmark(model, optimized_inference)

speedup = optimized_throughput / baseline_throughput
print(f"Speedup: {speedup:.2f}x")
```

## Implementation Details

All techniques are implemented in `/nexus/components/inference/`:

```
nexus/components/inference/
├── kv_cache.py              # KV Cache, PagedKVCache, QuantizedKVCache
├── prefix_cache.py          # Prefix caching with radix trees
├── speculative.py           # Speculative decoding
├── medusa.py                # Medusa multi-head decoding
├── eagle.py                 # EAGLE speculative decoding
├── lookahead.py             # Lookahead decoding
├── multi_token.py           # Multi-token prediction heads
└── continuous_batching.py   # Continuous and iteration-level batching
```

## Detailed Documentation

1. **[KV Cache Management](01_kv_cache.md)** - Understanding and optimizing the core memory structure
2. **[PagedAttention](02_paged_attention.md)** - OS-style virtual memory for KV caches
3. **[Quantized KV Cache](03_quantized_kv_cache.md)** - INT8/INT4/FP8 compression of cached values
4. **[Prefix Caching](04_prefix_caching.md)** - Reusing computations for common prefixes
5. **[Speculative Decoding](05_speculative_decoding.md)** - Draft model speculation
6. **[Medusa Decoding](06_medusa.md)** - Tree-based multi-head speculation
7. **[EAGLE Decoding](07_eagle.md)** - Feature-level speculation with dynamic trees
8. **[Lookahead Decoding](08_lookahead_decoding.md)** - Jacobi iteration for parallel generation
9. **[Multi-Token Prediction](09_multi_token_prediction.md)** - Predicting multiple tokens simultaneously
10. **[Continuous Batching](10_continuous_batching.md)** - Dynamic batching for throughput

## Research Papers

Essential papers for understanding these techniques:

### Memory Optimizations
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM)
- [KV Cache Quantization](https://arxiv.org/abs/2402.02750)
- [RadixAttention: Automatic Prefix Caching for LLMs](https://arxiv.org/abs/2312.07104) (SGLang)

### Speculative Decoding
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Medusa: Simple LLM Inference Acceleration Framework](https://arxiv.org/abs/2401.10774)
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [EAGLE-2: Faster Inference with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858)
- [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057)

### Multi-Token Prediction
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) (Meta)

### Batching
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) (Continuous Batching)
- [Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)

## Benchmarking Tools

```python
from nexus.benchmarks.inference import InferenceBenchmark

benchmark = InferenceBenchmark(
    model_name="llama-7b",
    techniques=["baseline", "paged", "speculative", "continuous_batch"],
    batch_sizes=[1, 8, 32, 128],
    sequence_lengths=[512, 1024, 2048],
    num_trials=10
)

results = benchmark.run()
benchmark.plot_comparison(save_path="results.png")
benchmark.save_report("benchmark_report.md")
```

## Contributing

When adding new optimization techniques:

1. Implement in `/nexus/components/inference/`
2. Add comprehensive documentation following the template
3. Include theoretical analysis and complexity bounds
4. Provide working code examples
5. Add benchmarks comparing to baseline and other techniques
6. Document integration with existing optimizations

## FAQ

**Q: Which optimization gives the best speedup?**
A: It depends on your workload. Continuous batching gives the highest throughput gains (5-10x), while speculative decoding gives the best single-sequence latency (2-3x).

**Q: Can I combine all optimizations?**
A: Yes, but with diminishing returns. The recommended maximum stack is: PagedAttention + Quantized KV + Speculative Decoding + Continuous Batching.

**Q: Do these work with quantized models?**
A: Yes. Weight quantization (INT8/INT4) is orthogonal to these inference optimizations and can be combined.

**Q: What about Flash Attention?**
A: Flash Attention is a kernel-level optimization that reduces memory I/O. It's complementary to these techniques and can be combined for additional speedup.

**Q: Production deployment recommendations?**
A: Start with PagedAttention + Continuous Batching. This gives 5-10x throughput with minimal complexity. Add speculative decoding if latency is critical.

## License

Part of the Nexus framework. See LICENSE for details.
