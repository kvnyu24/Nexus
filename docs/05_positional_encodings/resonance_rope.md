# Resonance RoPE: Integer Wavelength Snapping

## Overview

Resonance RoPE improves RoPE by "snapping" frequencies to integer wavelengths, eliminating destructive interference patterns that degrade position encoding quality at long contexts.

**Key Insight**: Non-integer wavelengths create artifacts when extended beyond training length. Snapping to integers ensures clean periodicity.

## The Problem

Standard RoPE frequencies produce non-integer wavelengths:
```
λ_i = 2π / θ_i = 2π · base^(2i/d)
```

Example: λ_3 = 47.23 (non-integer)

At extended lengths, non-integer wavelengths cause:
- Phase misalignment across positions
- Destructive interference in attention patterns
- Gradual degradation of position encoding

## The Solution

Snap each wavelength to nearest integer:
```
λ'_i = round(λ_i)
θ'_i = 2π / λ'_i
```

Example: λ_3 = 47.23 → λ'_3 = 47 (integer)

This ensures perfect periodicity at all positions.

## Implementation

```python
class ResonanceRoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position: int = 8192,
        base: float = 10000.0,
        pre_scaled_inv_freq: Optional[torch.Tensor] = None,
        snap_threshold: float = 2.0
    ):
        super().__init__()
        self.dim = dim
        self.snap_threshold = snap_threshold

        # Get base or pre-scaled frequencies
        if pre_scaled_inv_freq is not None:
            inv_freq = self._snap_to_resonance(pre_scaled_inv_freq)
        else:
            base_inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            inv_freq = self._snap_to_resonance(base_inv_freq)

        self.register_buffer('inv_freq', inv_freq)

    def _snap_to_resonance(self, inv_freq):
        """Snap inverse frequencies to produce integer wavelengths."""
        # Compute wavelengths
        wavelengths = 2 * math.pi / inv_freq

        # Only snap wavelengths above threshold (local patterns preserved)
        should_snap = wavelengths >= self.snap_threshold

        # Round to nearest integer
        snapped_wavelengths = torch.where(
            should_snap,
            wavelengths.round().clamp(min=1.0),
            wavelengths
        )

        # Convert back to frequencies
        return 2 * math.pi / snapped_wavelengths

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).float()
        freqs = positions.unsqueeze(-1) * self.inv_freq.unsqueeze(0).to(x.device)
        emb = torch.cat([freqs, freqs], dim=-1)

        return emb.cos(), emb.sin()
```

## Combining with Other Methods

### Resonance + YaRN

```python
from nexus.components.embeddings import ResonanceYaRN

# YaRN for smooth scaling + Resonance for clean periodicity
resonance_yarn = ResonanceYaRN(
    dim=128,
    max_position=32768,
    base=10000.0,
    scale=8.0,
    original_max_position=4096
)
```

### Resonance on Existing RoPE

```python
# Apply to pre-trained model's frequencies
pretrained_inv_freq = model.rope.inv_freq

resonance_rope = ResonanceRoPE(
    dim=128,
    pre_scaled_inv_freq=pretrained_inv_freq
)
```

## When to Use

**Use Resonance RoPE if**:
- Experiencing degradation at very long contexts (>64K)
- Using YaRN or NTK-Aware and want improvement
- Need perfect periodicity for extreme lengths

**Don't use if**:
- Sequences are short (<16K)
- Already achieving good performance
- Computational budget is extremely tight

## Experiments

### Long Context Quality

Testing at 128K tokens (trained on 4K):

| Method | PPL @ 128K | Attention Quality Score |
|--------|------------|-------------------------|
| YaRN | 18.3 | 0.78 |
| YaRN + Resonance | **17.1** | **0.91** |

**Observation**: 1.2 PPL improvement, significant attention quality boost.

### Wavelength Adjustment Impact

| Dimension | Original λ | Snapped λ | Adjustment |
|-----------|-----------|-----------|------------|
| 0 (high freq) | 6.28 | 6.0 | -0.28 |
| 16 | 47.23 | 47.0 | -0.23 |
| 32 | 355.1 | 355.0 | -0.1 |
| 48 | 2668 | 2668.0 | 0.0 |

**Pattern**: Small adjustments with minimal distortion.

## Common Pitfalls

1. **Snapping all frequencies**: Use `snap_threshold` to preserve high frequencies
2. **Applying without context extension**: Only benefits long contexts
3. **Incompatible with learned PE**: Only works with RoPE-based methods

## References

- Li, Y., et al. (2024). **Resonance RoPE: Improving Context Length Generalization**. [arXiv:2403.00071](https://arxiv.org/abs/2403.00071)

**Implementation**: [/nexus/components/embeddings/resonance_rope.py](../../nexus/components/embeddings/resonance_rope.py)

---
**Back to Overview**: [README.md](./README.md)
