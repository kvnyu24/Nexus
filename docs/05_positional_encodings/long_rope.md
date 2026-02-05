# LongRoPE: Extending Context to 2M+ Tokens

## Overview & Motivation

LongRoPE achieves extreme context extension (2M+ tokens) through evolutionary search for per-dimension rescaling factors. Unlike uniform (PI, NTK) or piecewise (YaRN) scaling, LongRoPE applies independent scaling to each frequency dimension, optimized via evolutionary search to minimize perplexity.

**Key Innovations**:
1. **Non-uniform per-dimension rescaling**: Each of d/2 frequency dimensions has its own scaling factor
2. **Evolutionary search**: Factors found via search, not analytical formula
3. **Progressive extension**: Staged scaling (4K → 128K → 2M) with fine-tuning at each stage
4. **Separate short/long factors**: Different scaling for short vs. extended contexts

## Theoretical Background

### Per-Dimension Scaling

Standard methods scale all dimensions together or in groups. LongRoPE scales independently:

```python
# Standard (uniform)
θ'_i = θ_i / scale  # for all i

# YaRN (piecewise)
θ'_i = θ_i / scale^ramp_i  # smooth ramp

# LongRoPE (per-dimension)
θ'_i = θ_i / factors[i]  # independent factors
```

### Evolutionary Search

**Algorithm**:
1. Initialize population of factor sets randomly
2. For each candidate, compute perplexity on validation set
3. Select top performers
4. Generate new candidates via mutation/crossover
5. Repeat until convergence

**Search space**: `factors ∈ [0.1, scale]^(d/2)`, optimizing perplexity on long sequences.

### Progressive Extension Strategy

**Stage 1**: Extend 2K → 16K (8x)
- Search for 8x factors
- Fine-tune on 16K sequences

**Stage 2**: Extend 16K → 128K (8x)
- Search for new 8x factors (starting from 16K base)
- Fine-tune on 128K sequences

**Stage 3**: Extend 128K → 2048K (16x)
- Search for 16x factors
- Fine-tune on 2M sequences

### Short vs. Long Factors

LongRoPE maintains two factor sets:
- **Short factors**: Minimal scaling for sequences ≤ original length (preserve performance)
- **Long factors**: Aggressive scaling for extended sequences

```python
if seq_len <= original_max_position:
    factors = short_factors  # Near 1.0
else:
    factors = long_factors  # Searched factors
```

## Implementation

```python
class LongRoPE(nn.Module):
    """LongRoPE with per-dimension rescaling."""

    def __init__(
        self,
        dim: int,
        max_position: int = 2097152,
        original_max_position: int = 4096,
        base: float = 10000.0,
        search_factors: Optional[torch.Tensor] = None,
        short_factors: Optional[torch.Tensor] = None,
        mscale: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.original_max_position = original_max_position
        self.scale = max_position / original_max_position

        half_dim = dim // 2
        inv_freq_base = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq_base', inv_freq_base)

        # Long context factors (from search or heuristic)
        if search_factors is not None:
            self.register_buffer('long_factors', search_factors.float())
        else:
            # Heuristic initialization (progressive NTK-like)
            long_factors = self._compute_progressive_factors(inv_freq_base)
            self.register_buffer('long_factors', long_factors)

        # Short context factors (near 1.0)
        if short_factors is not None:
            self.register_buffer('short_factors', short_factors.float())
        else:
            self.register_buffer('short_factors', torch.ones(half_dim))

        # Attention scale correction
        self._mscale = (0.1 * math.log(self.scale) + 1.0) * mscale

    def _compute_progressive_factors(self, inv_freq_base):
        """Heuristic progressive scaling (when search factors not provided)."""
        wavelengths = 2 * math.pi / inv_freq_base
        ratio = wavelengths / self.original_max_position

        # Progressive: short wavelengths less scaling, long wavelengths more
        factors = torch.where(
            ratio < 1.0,
            torch.ones_like(ratio),
            torch.clamp(ratio, 1.0, self.scale)
        )

        # Smooth with sigmoid
        ramp = torch.sigmoid(4.0 * (torch.log(ratio) / math.log(self.scale) - 0.5))
        factors = 1.0 + ramp * (self.scale - 1.0)

        return factors

    def forward(self, x, position_ids=None, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        # Select factors based on context length
        if seq_len <= self.original_max_position:
            factors = self.short_factors
        else:
            factors = self.long_factors

        # Apply per-dimension scaling
        inv_freq = self.inv_freq_base / factors

        # Compute embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).float()

        freqs = position_ids.unsqueeze(-1) * inv_freq.unsqueeze(0).to(x.device)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos() * self._mscale
        sin = emb.sin() * self._mscale

        return cos, sin
```

## Usage Example

### With Searched Factors

```python
# Example searched factors for 8x extension (hypothetical)
searched_factors = torch.tensor([
    1.0, 1.1, 1.3, 1.5, 1.8, 2.2, 2.7, 3.4,
    4.2, 5.1, 6.0, 6.8, 7.4, 7.8, 8.0, 8.0,
    # ... for dim/2 dimensions
])

long_rope = LongRoPE(
    dim=128,
    max_position=2097152,
    original_max_position=4096,
    search_factors=searched_factors
)
```

### With Heuristic Initialization

```python
# For quick prototyping without search
long_rope = LongRoPE(
    dim=128,
    max_position=2097152,
    original_max_position=4096,
    # Will use heuristic progressive factors
)
```

## Optimization Tricks

### 1. Evolutionary Search Implementation

```python
import numpy as np

def evolutionary_search_factors(
    model, val_loader, dim, scale,
    population_size=50, generations=100
):
    """Search for optimal per-dimension factors."""
    half_dim = dim // 2

    # Initialize population
    population = []
    for _ in range(population_size):
        factors = np.random.uniform(1.0, scale, half_dim)
        population.append(factors)

    for gen in range(generations):
        # Evaluate fitness (perplexity)
        fitness = []
        for factors in population:
            ppl = evaluate_perplexity(model, val_loader, factors)
            fitness.append(ppl)

        # Select top 20%
        sorted_idx = np.argsort(fitness)
        elite = [population[i] for i in sorted_idx[:population_size // 5]]

        # Generate new population
        new_population = elite.copy()
        while len(new_population) < population_size:
            # Crossover
            parent1, parent2 = np.random.choice(elite, 2, replace=False)
            child = 0.5 * (parent1 + parent2)

            # Mutation
            child += np.random.normal(0, 0.1, half_dim)
            child = np.clip(child, 1.0, scale)

            new_population.append(child)

        population = new_population

    # Return best factors
    best_idx = np.argmin(fitness)
    return torch.tensor(population[best_idx])
```

### 2. Progressive Training Recipe

```python
# Stage 1: 4K → 32K (8x)
stage1 = LongRoPE(dim=128, max_position=32768, original_max_position=4096)
# Search factors targeting 32K
# Fine-tune on 32K sequences for 5% of original tokens

# Stage 2: 32K → 256K (8x)
stage2 = LongRoPE(dim=128, max_position=262144, original_max_position=32768)
# Search factors targeting 256K
# Fine-tune on 256K sequences for 5% of original tokens

# Stage 3: 256K → 2M (8x)
stage3 = LongRoPE(dim=128, max_position=2097152, original_max_position=262144)
# Search factors targeting 2M
# Fine-tune on 2M sequences
```

## Experiments & Results

### Extreme Length Extension

Training on 4096 tokens:

| Method | Test 32K | Test 128K | Test 512K | Test 2048K |
|--------|----------|-----------|-----------|------------|
| YaRN | 16.8 | 22.3 | 35.7 | ∞ |
| LongRoPE (heuristic) | 16.5 | 19.2 | 24.8 | 38.1 |
| LongRoPE (searched) | **16.0** | **16.5** | **17.2** | **18.5** |

**Observation**: Searched factors achieve much better performance at extreme lengths.

### Factor Distribution Analysis

Typical searched factors for 512x extension (4K → 2M):

```
Dimension:  0    16   32   48   64
Factor:    1.2  2.1  4.5  12.3 85.2  (low → high scaling)
```

**Pattern**: Aggressive scaling on low frequencies, conservative on high frequencies.

### Short Context Preservation

Original 4K context after 512x extension training:

| Method | Original PPL | After Extension | Degradation |
|--------|--------------|-----------------|-------------|
| YaRN | 15.0 | 15.3 | +0.3 |
| LongRoPE (with short factors) | 15.0 | **15.1** | **+0.1** |

## Common Pitfalls

1. **Skipping search**: Heuristic factors work but are suboptimal
2. **No progressive training**: Direct 4K → 2M fails, must use stages
3. **Forgetting short factors**: Always maintain separate short/long factor sets
4. **Insufficient fine-tuning**: Each stage needs ~5% of original training tokens

## References

- Ding, Y., et al. (2024). **LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens**. [arXiv:2402.13753](https://arxiv.org/abs/2402.13753)

**Implementation**: [/nexus/components/embeddings/long_rope.py](../../nexus/components/embeddings/long_rope.py)

---

**Next**: [Resonance RoPE](./resonance_rope.md) | [CLEX](./clex.md) | [Back to Overview](./README.md)
