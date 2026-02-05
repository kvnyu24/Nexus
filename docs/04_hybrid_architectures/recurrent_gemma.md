# RecurrentGemma: Griffin-Based Open Language Model

## Overview

RecurrentGemma is Google's open-source Griffin-based language model combining RGLRU recurrence with local sliding window attention. It demonstrates that hybrid recurrence-attention can match transformer quality with better efficiency.

**Key Innovation**: Production-quality open implementation of Griffin architecture with additional optimizations (RMSNorm, GeGLU, RoPE).

**Paper**: Google DeepMind (2024)

## Architecture

Similar to Griffin but with Gemma-specific improvements:
- **RGLRU**: Gated linear recurrence (core)
- **Local Sliding Window**: Multi-query attention (window=2048)
- **RMSNorm**: Instead of LayerNorm
- **GeGLU**: Instead of standard FFN

## Pattern

```python
# Every 3rd layer is attention, rest are recurrence
Layer 0:  RGLRU + GeGLU
Layer 1:  RGLRU + GeGLU
Layer 2:  LocalAttn + GeGLU  ←─ Every 3 layers
Layer 3:  RGLRU + GeGLU
Layer 4:  RGLRU + GeGLU
Layer 5:  LocalAttn + GeGLU  ←─ Every 3 layers
...
```

## Implementation

```python
model = RecurrentGemmaModel(
    d_model=2048,
    n_layers=26,
    num_heads=16,
    window_size=2048,
    recurrence_every_n=3  # Recurrence every 3 layers
)
```

## Optimizations

### RMSNorm vs LayerNorm

```python
# Faster, simpler normalization
def rmsnorm(x):
    rms = sqrt(mean(x²) + eps)
    return weight * x / rms

# vs LayerNorm:
def layernorm(x):
    mean_x = mean(x)
    std_x = std(x)
    return weight * (x - mean_x) / std_x + bias

# RMSNorm: 2 ops (mean of squares, division)
# LayerNorm: 4 ops (mean, subtract, std, division)
```

### GeGLU Activation

```python
# Gated activation for FFN
def geglu(x):
    gate, value = split(W_gate(x), W_up(x))
    return W_down(gelu(gate) * value)

# Better than standard GELU FFN
```

## Performance

**Efficiency gains over Transformer**:
- Training: 1.5x faster
- Inference: 2.5x faster
- KV cache: 10x smaller (window + MQA)

**Quality**: Matches transformer on most benchmarks

## Common Pitfalls

```python
# ❌ BAD: Window too small
window_size = 128  # Can't capture enough context

# ✅ GOOD: Larger window
window_size = 2048  # Standard for RecurrentGemma

# ❌ BAD: Too much recurrence
recurrence_every_n = 10  # Quality drops

# ✅ GOOD: Balanced ratio
recurrence_every_n = 3  # 2:1 recurrence:attention
```

## References

Google DeepMind (2024). **RecurrentGemma: Moving Past Transformers for Efficient Open Language Models**. arXiv:2404.07839.

**Code**: `nexus/models/hybrid/recurrent_gemma.py`
**Official**: [google-deepmind/recurrentgemma](https://github.com/google-deepmind/recurrentgemma)
