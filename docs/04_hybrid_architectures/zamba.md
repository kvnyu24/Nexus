# Zamba: Mamba Backbone with Shared Attention

## Overview

Zamba uses Mamba as the primary sequence modeling mechanism with strategically placed shared attention blocks. The key innovation is parameter sharing across attention layers, reducing model size while maintaining quality.

**Key Innovation**: Shared attention module reused across multiple positions, amortizing attention cost while keeping Mamba as efficient backbone.

**Paper**: Zyphra AI (2024)

## Architecture Design

```
Layer 0:  Mamba + Dense FFN
Layer 1:  Mamba + Dense FFN
Layer 2:  Mamba + Dense FFN
Layer 3:  Mamba + Dense FFN
Layer 4:  Mamba + Dense FFN
Layer 5:  Shared Attention + Dense FFN  ←─ Every 6 layers
Layer 6:  Mamba + Dense FFN
...
Layer 11: Shared Attention + Dense FFN  ←─ Same attention module!
```

## Implementation

```python
# Create shared attention (reused)
shared_attention = SharedAttentionBlock(d_model, num_heads)

# Build layers
layers = []
for i in range(n_layers):
    if (i + 1) % 6 == 0:
        # Use shared attention
        layers.append(ZambaBlock(
            block_type='attention',
            shared_attention=shared_attention
        ))
    else:
        # Use Mamba
        layers.append(ZambaBlock(block_type='mamba'))
```

## Benefits

**Parameter efficiency**:
```
Standard (independent attention):
    32 layers × attention params = 32 × 100M = 3.2B params

Zamba (shared attention):
    1 × attention params + 32 × mamba = 100M + 2B = 2.1B params
    
Savings: ~35% parameter reduction
```

## Tradeoffs

**Pros**:
- Smaller model size
- Fewer attention parameters to tune
- Consistent attention behavior

**Cons**:
- Less layer-specific adaptation
- Attention can't specialize per layer

## Common Pitfalls

```python
# ❌ BAD: Sharing too frequently
attention_every_n = 2  # Too much reuse

# ✅ GOOD: Sparse reuse
attention_every_n = 6  # Let Mamba dominate

# ❌ BAD: Multiple shared attention modules
shared_attn_1 = ...
shared_attn_2 = ...  # Defeats the purpose!

# ✅ GOOD: Single shared module
shared_attn = ...  # Reuse everywhere
```

## References

Zyphra AI (2024). **Zamba: A Compact 7B SSM Hybrid Model**.

**Code**: `nexus/models/hybrid/zamba.py`
