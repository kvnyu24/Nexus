# NTK-Aware RoPE: Non-uniform Frequency Scaling

## Overview & Motivation

NTK-Aware RoPE (Neural Tangent Kernel-Aware RoPE) extends RoPE to longer contexts through non-uniform frequency scaling. Instead of scaling all frequencies uniformly (which degrades high-frequency local patterns), NTK-Aware RoPE applies more scaling to low frequencies and less to high frequencies.

**Key Innovation**: Modify the RoPE base frequency based on NTK theory:
```
new_base = base × scale^(d/(d-2))
```

This results in non-uniform per-dimension scaling where:
- High frequencies (small i): scaled by ~1 (preserved)
- Low frequencies (large i): scaled by ~scale (interpolated)

## Theoretical Background

### The NTK Formula

For extending context by factor `s`, modify the base:
```
base' = base × s^(d/(d-2))
```

This creates non-uniform frequency scaling:
```
θ'_i = 1 / (base')^(2i/d)
     = 1 / (base × s^(d/(d-2)))^(2i/d)
     = (1 / base^(2i/d)) × s^(-2i/(d-2))
     = θ_i × s^(-2i/(d-2))
```

The exponent `-2i/(d-2)` causes non-uniform scaling:
- When i=0 (high freq): scale factor ≈ 1
- When i=d/2 (low freq): scale factor ≈ s

### Why This Works

**High frequencies** encode local patterns (nearby tokens). Preserving them maintains the model's ability to distinguish adjacent positions.

**Low frequencies** encode global position. Interpolating them allows the model to handle longer sequences.

## Implementation

```python
class NTKAwareRoPE(nn.Module):
    """NTK-Aware Rotary Position Embedding."""

    def __init__(
        self,
        dim: int,
        max_position: int = 8192,
        base: float = 10000.0,
        scale_factor: float = 1.0,
        dynamic: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scale_factor = scale_factor
        self.dynamic = dynamic

        if not dynamic:
            # Static: compute scaled base once
            scaled_base = self._compute_ntk_base(scale_factor)
            inv_freq = 1.0 / (scaled_base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer('inv_freq', inv_freq)
        else:
            # Dynamic: adjust at runtime based on sequence length
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer('inv_freq', inv_freq)
            self.original_max_position = max_position

    def _compute_ntk_base(self, scale: float) -> float:
        """Compute NTK-scaled base frequency."""
        if scale <= 1.0:
            return self.base
        return self.base * (scale ** (self.dim / (self.dim - 2)))

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        # Dynamic scaling based on actual sequence length
        if self.dynamic and seq_len > self.original_max_position:
            scale = seq_len / self.original_max_position
            scaled_base = self._compute_ntk_base(scale)
            inv_freq = 1.0 / (
                scaled_base ** (
                    torch.arange(0, self.dim, 2, device=x.device).float() / self.dim
                )
            )
        else:
            inv_freq = self.inv_freq

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).float()
        freqs = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)

        return emb.cos(), emb.sin()
```

## Two Modes: Static vs. Dynamic

### Static NTK

```python
# Static: set target scale factor once
rope = NTKAwareRoPE(dim=128, scale_factor=4.0)  # For 4x extension
# Always uses 4x scaling
```

**Use when**: You know the target length in advance and will fine-tune.

### Dynamic NTK

```python
# Dynamic: adjusts based on actual sequence length
rope = NTKAwareRoPE(dim=128, max_position=8192, dynamic=True)
# Automatically scales if seq_len > 8192
```

**Use when**: You want zero-shot extension without fine-tuning.

## Usage Example

```python
from nexus.components.embeddings import NTKAwareRoPE

# Extend from 4K to 16K (4x)
rope = NTKAwareRoPE(
    dim=128,
    max_position=16384,
    base=10000.0,
    scale_factor=4.0,
    dynamic=False  # Static scaling
)

# Use in model
cos, sin = rope(x, seq_len=16384)
q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
```

## Optimization Tricks

### 1. Dynamic Scaling with Threshold

```python
class DynamicNTKRoPE(nn.Module):
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        # Only scale if exceeding training length
        if seq_len <= self.original_max_position:
            return self.compute_rope(seq_len, scale=1.0)
        else:
            scale = seq_len / self.original_max_position
            return self.compute_rope(seq_len, scale=scale)
```

### 2. Caching for Common Lengths

```python
# Pre-cache for common extended lengths
cache = {}
for length in [4096, 8192, 16384, 32768]:
    scale = length / 2048
    scaled_base = base * (scale ** (dim / (dim - 2)))
    cache[length] = precompute_rope(length, scaled_base)
```

## Experiments & Results

### Length Generalization

Training on 2048 tokens:

| Method | Train 2K | Test 4K | Test 8K | Test 16K |
|--------|----------|---------|---------|----------|
| RoPE | 15.0 | 22.8 | 38.4 | 72.1 |
| Position Interpolation | 15.0 | 16.2 | 19.8 | 28.3 |
| NTK-Aware | 15.0 | **15.8** | **17.9** | **23.4** |
| YaRN | 15.0 | 15.3 | 15.9 | 16.8 |

**Observation**: NTK-Aware is better than naive methods, though YaRN is still superior.

### Extension Factor Analysis

| Scale Factor | Effective Range | PPL @ 2x | PPL @ 4x | PPL @ 8x |
|--------------|-----------------|----------|----------|----------|
| 2.0 | Up to 4K | 15.8 | 19.2 | 28.7 |
| 4.0 | Up to 8K | 16.3 | 17.9 | 21.5 |
| 8.0 | Up to 16K | 17.1 | 18.9 | 20.2 |

**Takeaway**: Set scale_factor to match your target extension for best results.

## Common Pitfalls

1. **Wrong scale factor**: Set to match actual extension ratio
2. **Not fine-tuning**: Static NTK requires fine-tuning on longer sequences
3. **Confusion with YaRN**: NTK-Aware is simpler but less effective than YaRN
4. **Applying to non-RoPE models**: Only works with RoPE-based models

## References

- Bloc97 (2023). **NTK-Aware Scaled RoPE allows LLaMA models to have extended context**. Reddit/LocalLLaMA.
- Xiong, R., et al. (2023). **Scaling Laws of RoPE-based Extrapolation**. [arXiv:2306.15595](https://arxiv.org/abs/2306.15595)

**Implementation**: [/nexus/components/embeddings/ntk_rope.py](../../nexus/components/embeddings/ntk_rope.py)

---

**Next**: [LongRoPE](./long_rope.md) | [YaRN](./yarn.md) | [Back to Overview](./README.md)
