# Hawk: Pure Gated Linear Recurrence

## Overview

Hawk is the pure recurrence variant of Griffin - using only RGLRU blocks without any attention. It represents the extreme efficiency point, offering O(1) memory during inference at the cost of some quality on recall-intensive tasks.

**Key Innovation**: Demonstrates that pure gated linear recurrence can be competitive on many tasks, especially those emphasizing long-range dependencies over exact retrieval.

**Paper**: Same as Griffin (De et al., 2024)

## Architecture

```
Layer 0:  TemporalConv + RGLRU + SwiGLU
Layer 1:  TemporalConv + RGLRU + SwiGLU
Layer 2:  TemporalConv + RGLRU + SwiGLU
...
Layer 23: TemporalConv + RGLRU + SwiGLU

No attention layers!
```

## Components

### Temporal Convolution
Short conv (kernel=4) for local context before recurrence.

### RGLRU
Same as Griffin - magnitude-preserving gated recurrence.

### SwiGLU
Gated FFN for improved expressivity.

## Implementation

```python
model = HawkModel(
    d_model=512,
    n_layers=24,
    d_recurrence=512,
    kernel_size=4  # Temporal conv size
)

# Or use Griffin with hawk_only=True
model = GriffinModel(..., hawk_only=True)
```

## Performance Profile

**Strengths**:
- **Inference speed**: 5x faster than transformers
- **Memory**: O(1) - no KV cache at all
- **Long-range**: Excellent on tasks with diffuse dependencies

**Weaknesses**:
- **Exact recall**: 5-10% worse than transformers on retrieval
- **Associative memory**: Struggles with exact matching

## Efficiency Comparison

| Metric | Transformer | Griffin | Hawk |
|--------|-------------|---------|------|
| Speed | 1x | 2.5x | **5x** |
| Memory | O(N²) | O(N) | **O(1)** |
| Quality | 100% | 98% | 92% |

## When to Use

**Use Hawk when**:
- Extreme efficiency is paramount
- Long-range modeling is critical
- Exact recall is less important
- Inference budget is very constrained

**Avoid Hawk when**:
- Tasks require precise token retrieval (QA, RAG)
- Associative memory is critical
- Quality cannot be compromised

## Common Pitfalls

```python
# ❌ BAD: Expecting transformer-level recall
# Hawk will underperform on exact matching tasks

# ✅ GOOD: Use for appropriate tasks
# Long document summarization, language modeling

# ❌ BAD: Tiny recurrence dimension
d_recurrence = 128  # Too small

# ✅ GOOD: Adequate capacity
d_recurrence = d_model  # Match model dimension
```

## References

De, S., et al. (2024). **Griffin: Mixing Gated Linear Recurrences with Local Attention**. Google DeepMind.

**Code**: `nexus/models/hybrid/hawk.py`
