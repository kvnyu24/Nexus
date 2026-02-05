# Inference Optimizations Documentation - Overview

This directory contains comprehensive documentation on state-of-the-art LLM inference optimization techniques. All techniques are implemented in `/nexus/components/inference/`.

## Documentation Structure

### Core Documentation Files

#### ✅ Completed

1. **[README.md](README.md)** - Complete optimization landscape
   - Overview of all techniques
   - Performance comparison tables
   - Recommended stacks for different use cases
   - Quick start guide
   - Benchmarking tools

2. **[01_kv_cache.md](01_kv_cache.md)** - KV Cache Management
   - Foundation of efficient inference
   - Latency: 100x+ speedup over naive generation
   - Memory: Trades O(T²) attention for O(L×T×d) cache
   - Implementation: `nexus/components/inference/kv_cache.py`

3. **[05_speculative_decoding.md](05_speculative_decoding.md)** - Speculative Decoding
   - 2-3x speedup with draft model
   - Mathematically proven identical distribution
   - Implementation: `nexus/components/inference/speculative.py`

4. **[10_continuous_batching.md](10_continuous_batching.md)** - Continuous Batching
   - 5-10x throughput improvement
   - Dynamic batch composition
   - GPU utilization: 95%+ (vs 40-60% static)
   - Implementation: `nexus/components/inference/continuous_batching.py`

### Implementation Files Reference

All implementations are in `/nexus/components/inference/`:

```
nexus/components/inference/
├── __init__.py              # Exports all optimization classes
├── kv_cache.py              # KVCache, PagedKVCache, QuantizedKVCache
├── prefix_cache.py          # PrefixCache, RadixPrefixCache
├── speculative.py           # SpeculativeDecoder, NGramSpeculator
├── medusa.py                # MedusaFFNHead, MedusaDecoder
├── eagle.py                 # EAGLEDraftHead, EAGLEDecoder
├── lookahead.py             # NGramPool, LookaheadDecoder
├── multi_token.py           # MultiTokenPredictionHead
└── continuous_batching.py   # ContinuousBatcher, IterationLevelBatcher
```

## Quick Reference Guide

### Memory Optimizations

| Technique | Memory Reduction | Quality Impact | Docs | Implementation |
|-----------|-----------------|----------------|------|----------------|
| **KV Cache** | Baseline | None | [01_kv_cache.md](01_kv_cache.md) | `kv_cache.py::KVCache` |
| **PagedAttention** | 3-4x | None | README.md | `kv_cache.py::PagedKVCache` |
| **Quantized KV** | 2-4x | Minimal | README.md | `kv_cache.py::QuantizedKVCache` |
| **Prefix Caching** | Variable | None | README.md | `prefix_cache.py` |

### Latency Optimizations

| Technique | Speedup | Extra Memory | Quality | Docs | Implementation |
|-----------|---------|--------------|---------|------|----------------|
| **Speculative Decoding** | 2-3x | Draft model | Identical | [05_speculative_decoding.md](05_speculative_decoding.md) | `speculative.py` |
| **EAGLE** | 2.5-4x | Small heads | Identical | README.md | `eagle.py` |
| **Medusa** | 2-3x | Small heads | Identical | README.md | `medusa.py` |
| **Lookahead** | 1.5-2x | N-gram pool | Identical | README.md | `lookahead.py` |

### Throughput Optimizations

| Technique | Throughput Gain | Latency Impact | Docs | Implementation |
|-----------|-----------------|----------------|------|----------------|
| **Continuous Batching** | 5-10x | Reduced | [10_continuous_batching.md](10_continuous_batching.md) | `continuous_batching.py` |
| **Iteration-Level Batching** | 8-15x | Low | [10_continuous_batching.md](10_continuous_batching.md) | `continuous_batching.py::IterationLevelBatcher` |

## Usage Examples

### Basic KV Cache

```python
from nexus.components.inference import KVCache

# Initialize cache
cache = KVCache(
    num_layers=32,
    max_batch_size=8,
    max_seq_len=2048,
    num_heads=32,
    head_dim=128
)

# Generate with cache
outputs = model(input_ids, cache=cache)
```

### Speculative Decoding

```python
from nexus.components.inference import SpeculativeDecoder

# Create decoder
decoder = SpeculativeDecoder(
    target_model=llama_7b,
    draft_model=llama_68m,
    num_speculative_tokens=5
)

# Generate (2-3x faster!)
output = decoder.generate(input_ids, max_new_tokens=100)
```

### Continuous Batching

```python
from nexus.components.inference import ContinuousBatcher

# Create batcher
batcher = ContinuousBatcher(
    max_batch_size=128,
    max_seq_len=2048,
    scheduling_policy='priority'
)

# Add requests
for input_ids in requests:
    batcher.add_request(input_ids, max_new_tokens=100)

# Generate with continuous batching (5-10x throughput!)
while batcher.has_active_requests():
    batch = batcher.prepare_batch()
    logits = model(batch.input_ids, batch.attention_mask)
    next_tokens = sample(logits)
    completed = batcher.step(next_tokens)
```

### Combined Stack (Maximum Performance)

```python
from nexus.components.inference import (
    PagedKVCache,
    SpeculativeDecoder,
    ContinuousBatcher
)

# Memory management
cache = PagedKVCache(
    num_layers=32,
    block_size=16,
    num_blocks=2048
)

# Latency optimization
speculative = SpeculativeDecoder(
    target_model=model,
    draft_model=draft,
    num_speculative_tokens=5
)

# Throughput optimization
batcher = ContinuousBatcher(
    max_batch_size=128,
    kv_cache=cache
)

# Expected: 20-50x overall improvement!
```

## Performance Summary

### Single Sequence (Latency-Focused)

```
Baseline:                                   1.0x  ████
+ KV Cache:                               100.0x  (essential)
+ Speculative Decoding:                     2.5x  ██████
+ EAGLE (instead of Speculative):           3.5x  ████████
───────────────────────────────────────────────────────────
Net speedup with EAGLE + KV Cache:        350.0x
```

### Batch Processing (Throughput-Focused)

```
Baseline (batch=8):                        1.0x  ████
+ KV Cache:                               10.0x  (enables batching)
+ Continuous Batching:                     6.0x  ████████████████
+ PagedAttention:                          1.3x  █████
───────────────────────────────────────────────────────
Net throughput improvement:               78.0x
```

## Getting Started

### 1. Start with KV Cache

Every inference optimization builds on KV caching. Start here:

```python
# Read the documentation
open("01_kv_cache.md")

# Try the basic implementation
from nexus.components.inference import KVCache
cache = KVCache(...)
```

### 2. Choose Your Optimization Path

**Path A: Low Latency (Single User)**
- [x] KV Cache (essential)
- [ ] Speculative Decoding → [05_speculative_decoding.md](05_speculative_decoding.md)
- [ ] EAGLE (better than speculative) → README.md
- [ ] Prefix Caching (for repeated prompts) → README.md

**Path B: High Throughput (Many Users)**
- [x] KV Cache (essential)
- [ ] Continuous Batching → [10_continuous_batching.md](10_continuous_batching.md)
- [ ] PagedAttention → README.md
- [ ] Quantized KV Cache → README.md

**Path C: Memory-Constrained**
- [x] KV Cache (essential)
- [ ] PagedAttention → README.md
- [ ] Quantized KV Cache (INT8/INT4) → README.md
- [ ] Continuous Batching → [10_continuous_batching.md](10_continuous_batching.md)

### 3. Benchmark Your Setup

```python
from nexus.benchmarks.inference import InferenceBenchmark

benchmark = InferenceBenchmark(
    model_name="llama-7b",
    techniques=["baseline", "kv_cache", "speculative", "continuous_batch"],
    batch_sizes=[1, 8, 32],
    sequence_lengths=[512, 1024, 2048]
)

results = benchmark.run()
benchmark.plot_comparison(save_path="results.png")
```

## Documentation Standards

Each optimization technique document includes:

1. **Overview & Motivation**: What problem does it solve?
2. **Theoretical Background**: How does it work?
3. **Mathematical Formulation**: Complexity analysis and speedup bounds
4. **High-Level Intuition**: Visual diagrams and examples
5. **Implementation Details**: System-level considerations
6. **Code Walkthrough**: Detailed code examples from nexus/inference/
7. **Optimization Tricks**: Best practices and advanced techniques
8. **Experiments & Results**: Benchmarks with real numbers
9. **Common Pitfalls**: What to avoid
10. **References**: Papers, blogs, related docs

## Implementation Quality

All implementations in `nexus/components/inference/` include:

- ✅ Complete docstrings with parameter descriptions
- ✅ Type hints for all functions
- ✅ Working code examples in documentation
- ✅ Mathematical correctness
- ✅ Efficient CUDA operations where applicable
- ✅ Integration with existing nexus components

## Additional Resources

### Papers (Essential Reading)

1. **KV Cache**:
   - Attention Is All You Need (Vaswani et al., 2017)

2. **Speculative Decoding**:
   - Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2022)
   - EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (2024)
   - Medusa: Simple LLM Inference Acceleration Framework (2024)

3. **Batching**:
   - Orca: A Distributed Serving System (Yu et al., 2022)
   - vLLM: PagedAttention (Kwon et al., 2023)

4. **Memory**:
   - Flash Attention (Dao et al., 2022)
   - Multi-Query Attention (Shazeer, 2019)

### Code References

- **HuggingFace Transformers**: `transformers/cache_utils.py`, `generation/utils.py`
- **vLLM**: Complete production serving system
- **TensorRT-LLM**: NVIDIA's optimized inference library
- **Text Generation Inference (TGI)**: HuggingFace's serving solution

### Benchmarking Tools

```bash
# Run comprehensive benchmarks
python -m nexus.benchmarks.inference \
    --model llama-7b \
    --techniques all \
    --output results/

# Compare specific techniques
python -m nexus.benchmarks.compare \
    --baseline standard \
    --optimized speculative,continuous_batch \
    --metrics latency,throughput,memory
```

## Contributing

When adding new optimization documentation:

1. Follow the standard template (see existing docs)
2. Include working code examples from nexus/inference/
3. Add performance benchmarks with real numbers
4. Provide mathematical analysis where applicable
5. Document integration with other techniques
6. Add to this overview file

## FAQ

**Q: Where do I start?**
A: Read [01_kv_cache.md](01_kv_cache.md) - it's the foundation for everything else.

**Q: Which optimization gives the best speedup?**
A: For latency: Speculative/EAGLE (2-4x). For throughput: Continuous Batching (5-10x).

**Q: Can I combine multiple optimizations?**
A: Yes! Most techniques are orthogonal. See README.md for recommended stacks.

**Q: What about Flash Attention?**
A: Flash Attention is a kernel-level optimization (complementary to these techniques).

**Q: Do these work with quantized models?**
A: Yes. Weight quantization is orthogonal to inference optimizations.

## Status

### Documentation Coverage

- ✅ README.md - Complete optimization landscape
- ✅ 01_kv_cache.md - Complete with full implementation details
- ✅ 05_speculative_decoding.md - Complete with mathematical analysis
- ✅ 10_continuous_batching.md - Complete with system design
- ⏳ 02_paged_attention.md - Planned (implementation exists)
- ⏳ 03_quantized_kv_cache.md - Planned (implementation exists)
- ⏳ 04_prefix_caching.md - Planned (implementation exists)
- ⏳ 06_medusa.md - Planned (implementation exists)
- ⏳ 07_eagle.md - Planned (implementation exists)
- ⏳ 08_lookahead_decoding.md - Planned (implementation exists)
- ⏳ 09_multi_token_prediction.md - Planned (implementation exists)

### Implementation Coverage

- ✅ All core techniques implemented in `nexus/components/inference/`
- ✅ Working code examples in all completed documentation
- ✅ Integration tests passing
- ✅ Benchmarking framework available

## License

Part of the Nexus framework. See LICENSE for details.
