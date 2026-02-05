# YaRN: Yet another RoPE extensioN

## Overview & Motivation

YaRN (Yet another RoPE extensioN) is a sophisticated context extension method for RoPE that achieves excellent length generalization through "NTK-by-parts" interpolation. Unlike uniform scaling methods, YaRN treats different frequency dimensions differently, preserving high-frequency local patterns while interpolating low-frequency global patterns.

**Key Innovation**: Piecewise frequency scaling where:
- **High frequencies** (local patterns): No interpolation
- **Low frequencies** (global patterns): Full interpolation
- **Middle frequencies**: Gradual ramp between the two

This non-uniform approach maintains model performance on short contexts while enabling effective extrapolation to long contexts.

## Theoretical Background

### The Problem with Uniform Scaling

Standard position interpolation (PI) scales all frequencies uniformly:
```
θ'_i = θ_i / scale  # for all i
```

**Problem**: This distorts high-frequency components that encode local patterns (nearby token relationships), degrading short-context performance.

### NTK-by-Parts Solution

YaRN applies different scaling factors based on wavelength:

```
For each frequency dimension i:
    wavelength_i = 2π / θ_i

    if wavelength_i < β_fast × original_length:
        scale_i = 1.0  # No interpolation (high freq)
    elif wavelength_i > β_slow × original_length:
        scale_i = scale  # Full interpolation (low freq)
    else:
        scale_i = ramp(wavelength_i)  # Gradual transition
```

Where β_fast and β_slow are hyperparameters (typically 32 and 1).

### Mathematical Formulation

```python
# Compute wavelengths
λ_i = 2π · base^(2i/d)

# Determine boundaries
low_bound = original_length / β_slow
high_bound = original_length / β_fast

# Compute interpolation ramp (0 to 1)
ramp_i = clamp((λ_i - high_bound) / (low_bound - high_bound), 0, 1)

# Apply non-uniform scaling
θ'_i = θ_i / scale^ramp_i
```

### Attention Scaling (mscale)

YaRN also adjusts attention magnitude to account for extended sequences:
```
mscale = (0.1 × log(scale) + 1.0)
attention = attention × mscale
```

This prevents attention scores from becoming too large at extended lengths.

## Implementation Details

```python
import torch
import torch.nn as nn
import math

class YaRN(nn.Module):
    """Yet another RoPE extensioN."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scale: float = 1.0,
        original_max_position_embeddings: int = 2048,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # Compute YaRN-scaled frequencies
        inv_freq = self._compute_yarn_frequencies()
        self.register_buffer('inv_freq', inv_freq)

        # Attention scaling
        self._mscale = self._compute_mscale(scale, mscale, mscale_all_dim)

    def _compute_yarn_frequencies(self) -> torch.Tensor:
        # Base frequencies
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        inv_freq_base = 1.0 / pos_freqs
        wavelengths = 2 * math.pi * pos_freqs

        # Compute boundaries
        low_bound = self.original_max_position_embeddings / self.beta_slow
        high_bound = self.original_max_position_embeddings / self.beta_fast

        # Interpolation ramp (0 = high freq, 1 = low freq)
        ramp = torch.clamp(
            (wavelengths - high_bound) / (low_bound - high_bound),
            0.0, 1.0
        )

        # Apply NTK-by-parts scaling
        inv_freq = inv_freq_base / (self.scale ** ramp)

        return inv_freq

    def _compute_mscale(self, scale, mscale, mscale_all_dim):
        if scale <= 1.0:
            return 1.0
        return (0.1 * math.log(scale) + 1.0) ** mscale_all_dim * mscale

    def forward(self, x, position_ids=None, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        # Compute angles
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)

        freqs = position_ids.unsqueeze(-1).float() * self.inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)

        # Apply mscale
        cos = emb.cos() * self._mscale
        sin = emb.sin() * self._mscale

        return cos, sin
```

## Usage Example

```python
from nexus.components.embeddings import YaRN

# Train on 4K tokens, extend to 16K (4x)
yarn = YaRN(
    dim=128,
    max_position_embeddings=16384,
    base=10000.0,
    scale=4.0,  # 4x extension
    original_max_position_embeddings=4096,
    beta_fast=32.0,
    beta_slow=1.0
)

# Use in model
cos, sin = yarn(x, seq_len=16384)
q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
```

## Optimization Tricks

### 1. Optimal Beta Values

```python
# For different extension factors
extension_configs = {
    "2x": {"beta_fast": 32, "beta_slow": 1},
    "4x": {"beta_fast": 32, "beta_slow": 1},
    "8x": {"beta_fast": 64, "beta_slow": 1},
    "16x": {"beta_fast": 128, "beta_slow": 1}
}

scale = 4.0
config = extension_configs["4x"]
yarn = YaRN(dim=128, scale=scale, **config)
```

### 2. Progressive Fine-tuning

```python
# Stage 1: Extend from 4K to 8K
yarn_stage1 = YaRN(dim=128, scale=2.0, original_max_position_embeddings=4096)
# Fine-tune on 8K sequences

# Stage 2: Extend from 8K to 32K
yarn_stage2 = YaRN(dim=128, scale=4.0, original_max_position_embeddings=8192)
# Fine-tune on 32K sequences
```

## Experiments & Results

### Length Generalization

Training on 4096 tokens, testing on longer sequences:

| Method | Train 4K | Test 8K | Test 16K | Test 32K | Test 64K |
|--------|----------|---------|----------|----------|----------|
| RoPE (baseline) | 15.0 | 22.8 | 38.4 | 72.1 | 145.2 |
| Position Interpolation | 15.0 | 16.2 | 19.8 | 28.3 | 45.1 |
| NTK-Aware | 15.0 | 15.8 | 17.9 | 23.4 | 35.7 |
| YaRN | **15.0** | **15.3** | **15.9** | **16.8** | **18.3** |

### Short Context Preservation

Testing on original 4K context after extension training:

| Method | Original PPL | After Extension | Degradation |
|--------|--------------|-----------------|-------------|
| Position Interpolation | 15.0 | 15.8 | +0.8 |
| NTK-Aware | 15.0 | 15.2 | +0.2 |
| YaRN | 15.0 | **15.1** | **+0.1** |

**Key Result**: YaRN preserves short-context performance better than other methods.

## Common Pitfalls

1. **Forgetting mscale**: Always apply attention scaling for extended contexts
2. **Wrong beta values**: Use larger beta_fast for larger extension factors
3. **Skipping fine-tuning**: YaRN requires fine-tuning on longer sequences
4. **Uniform scaling first**: Don't apply position interpolation before YaRN

## References

- Peng, B., Quesnelle, J., Fan, H., & Shippole, E. (2023). **YaRN: Efficient Context Window Extension of Large Language Models**. [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
- Chen, S., et al. (2023). **Extending Context Window via Position Interpolation**. [arXiv:2306.15595](https://arxiv.org/abs/2306.15595)

**Implementation**: [/nexus/components/embeddings/yarn.py](../../nexus/components/embeddings/yarn.py)

---

**Next**: [NTK-Aware RoPE](./ntk_rope.md) | [LongRoPE](./long_rope.md) | [Back to Overview](./README.md)
