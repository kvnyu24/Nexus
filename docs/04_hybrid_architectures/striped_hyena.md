# StripedHyena: Attention-Hyena Hybrid for 128K Context

## Overview

StripedHyena alternates between Hyena operators (long convolutions) and standard attention blocks in a "striped" pattern, achieving 128K+ token context length with strong efficiency. The alternating pattern combines Hyena's O(N log N) convolution efficiency with attention's precise retrieval capabilities.

**Key Innovation**: Regular alternation of Hyena and attention layers, allowing Hyena to handle bulk processing while attention provides precision at intervals.

**Paper**: Based on Hyena Hierarchy (Poli et al., 2023)

## Architecture Pattern

```
Layer 0:  Hyena
Layer 1:  Hyena
Layer 2:  Hyena
Layer 3:  Attention  ←─ Every 4th layer
Layer 4:  Hyena
Layer 5:  Hyena
Layer 6:  Hyena
Layer 7:  Attention  ←─ Every 4th layer
...
```

**Typical ratio**: 3:1 or 7:1 (Hyena:Attention)

## Implementation

```python
model = StripedHyenaModel(
    d_model=1024,
    n_layers=32,
    num_heads=8,
    seq_len=131072,      # 128K context!
    hyena_order=2,
    attention_every_n=4  # Attention every 4 layers
)
```

## Performance

| Context | Transformer | StripedHyena |
|---------|-------------|--------------|
| 8K      | 100%        | 150% throughput |
| 32K     | OOM         | 100% (baseline) |
| 128K    | OOM         | 100% |

**Key benefit**: Enables very long contexts (128K+) that transformers cannot handle.

## Common Pitfalls

```python
# ❌ BAD: Too frequent attention
attention_every_n = 2  # Overhead too high

# ✅ GOOD: Sparse attention
attention_every_n = 4  # Good balance

# ❌ BAD: All Hyena
attention_every_n = 0  # Quality drops

# ✅ GOOD: Some attention layers
attention_every_n = 4-8  # Maintains quality
```

## References

Poli, M., et al. (2023). **Hyena Hierarchy**. Together AI.

**Code**: `nexus/models/hybrid/striped_hyena.py`
