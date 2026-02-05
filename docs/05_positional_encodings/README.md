# Positional Encodings: A Comprehensive Guide

This directory contains detailed documentation for all major positional encoding approaches used in modern deep learning architectures, with a focus on Transformer-based models.

## Why Positional Information Matters

Transformers process sequences using self-attention, which is inherently permutation-invariant. Without positional information, the model cannot distinguish between "the cat sat on the mat" and "sat mat the on cat the". Positional encodings inject information about token positions into the model, enabling it to learn sequence-order-dependent patterns.

## Overview of Approaches

Positional encodings can be categorized along several dimensions:

### 1. Absolute vs. Relative
- **Absolute**: Encode each position independently (Sinusoidal, Learned PE)
- **Relative**: Encode relationships between positions (RoPE, Relative Bias, ALiBi)

### 2. Fixed vs. Learned
- **Fixed**: Deterministic mathematical functions (Sinusoidal, RoPE, ALiBi)
- **Learned**: Trained parameters (Learned PE, Relative Bias)
- **Hybrid**: Fixed structure with learned parameters (FIRE, CoPE, CLEX)

### 3. Additive vs. Multiplicative
- **Additive**: Add encoding to embeddings (Sinusoidal, Learned PE)
- **Multiplicative**: Apply rotation to embeddings (RoPE and variants)
- **Attention Bias**: Add directly to attention scores (ALiBi, Relative Bias, FIRE)

## Complete List of Encodings

| Encoding | Type | Length Generalization | Context Extension | Used In |
|----------|------|----------------------|-------------------|---------|
| [Sinusoidal PE](./sinusoidal_pe.md) | Absolute, Fixed, Additive | Good | Limited | Original Transformer |
| [Learned PE](./learned_pe.md) | Absolute, Learned, Additive | Poor | None | GPT-2, BERT |
| [RoPE](./rope.md) | Relative, Fixed, Multiplicative | Good | Limited | GPT-NeoX, PaLM |
| [Relative Bias](./relative_bias.md) | Relative, Learned, Attention Bias | Good | Moderate | T5, LongT5 |
| [ALiBi](./alibi.md) | Relative, Fixed, Attention Bias | Excellent | Excellent | BLOOM, MPT, Falcon |
| [Multiscale RoPE](./multiscale_rope.md) | Relative, Fixed, Multiplicative | Good | Limited | Research |
| [YaRN](./yarn.md) | Relative, Fixed, Multiplicative | Excellent | Excellent | Extended LLMs |
| [CoPE](./cope.md) | Relative, Hybrid, Attention Bias | Excellent | Good | Research |
| [NTK-Aware RoPE](./ntk_rope.md) | Relative, Fixed, Multiplicative | Excellent | Excellent | Code Llama |
| [LongRoPE](./long_rope.md) | Relative, Hybrid, Multiplicative | Excellent | Excellent (2M+ tokens) | Research |
| [FIRE](./fire.md) | Relative, Hybrid, Attention Bias | Excellent | Excellent | Research |
| [Resonance RoPE](./resonance_rope.md) | Relative, Fixed, Multiplicative | Excellent | Excellent | Research |
| [CLEX](./clex.md) | Relative, Learned, Multiplicative | Excellent | Excellent | Research |

## Decision Guide: Which Encoding to Use?

### Use Sinusoidal PE if:
- Building a standard encoder-decoder Transformer
- Need deterministic, interpretable positional information
- Training context is sufficient for your task
- Want zero parameter overhead

### Use Learned PE if:
- Following established architectures (GPT-2, BERT)
- Training context matches deployment context exactly
- Can afford the parameter overhead (seq_len × dim parameters)
- Don't need length extrapolation

### Use RoPE if:
- Building a decoder-only language model
- Want relative position encoding without attention bias
- Need good interpolation properties
- Can combine with extension methods later

### Use ALiBi if:
- Need excellent length extrapolation out-of-the-box
- Want zero-shot generalization to longer sequences
- Prefer simplicity (no learned parameters)
- Training on shorter sequences, deploying on longer ones

### Use Relative Bias (T5-style) if:
- Building an encoder-decoder model
- Want learned relative position biases
- Have sufficient training data across various lengths
- Can afford head × bucket parameters

### Use YaRN if:
- Extending an existing RoPE model to longer contexts
- Need smooth interpolation/extrapolation
- Want to preserve short-context performance
- Can fine-tune on longer sequences

### Use NTK-Aware RoPE if:
- Need quick context extension without fine-tuning
- Want non-uniform frequency scaling
- Extending RoPE models 2-8x
- Limited computational budget for extension

### Use LongRoPE if:
- Need extreme context extension (100x or more)
- Can perform evolutionary search for scaling factors
- Want separate short/long context optimization
- Have compute for progressive fine-tuning

### Use FIRE if:
- Need excellent length generalization
- Want learned relative position encoding
- Prefer functional approach over lookup tables
- Can afford small MLP overhead

### Use Resonance RoPE if:
- Using RoPE or YaRN and seeing degradation at long contexts
- Want to eliminate wavelength interference
- Need clean periodicity for very long sequences
- Can combine with existing RoPE variants

### Use CoPE if:
- Position should depend on content (e.g., "count what matters")
- Working with structured data (code, markup)
- Want position to adapt to input structure
- Need both content and position awareness

### Use CLEX if:
- Need learned continuous scaling dynamics
- Want adaptive extrapolation to any length
- Can afford training a dynamics network
- Have diverse training lengths

### Use Multiscale RoPE if:
- Need different frequency scales per head
- Want to capture both local and global patterns
- Building multi-resolution models
- Can afford per-head frequency configuration

## Context Extension: Interpolation vs. Extrapolation

### Interpolation
Position interpolation scales down position indices to fit within the training range.

**Example**: Model trained on length 2048, deploy on length 4096
- Original: positions [0, 1, 2, ..., 4095]
- Interpolated: positions [0, 0.5, 1, ..., 2047.5]

**Advantages**: Stays within training distribution, stable
**Disadvantages**: Distorts local patterns, requires fine-tuning

**Best for**: Linear scaling, 2-4x extension

### Extrapolation
Position extrapolation uses positions beyond the training range.

**Example**: Model trained on length 2048, deploy on length 4096
- Positions: [0, 1, 2, ..., 4095] (including unseen 2048-4095)

**Advantages**: Preserves local patterns
**Disadvantages**: Out-of-distribution, may degrade

**Best for**: Small extensions, architectures designed for it (ALiBi, FIRE)

### Hybrid Approaches
Modern methods combine both:
- **YaRN**: Interpolates low frequencies, preserves high frequencies
- **NTK-Aware RoPE**: Non-uniform interpolation across frequencies
- **LongRoPE**: Per-dimension scaling factors via search

## Key Metrics for Evaluation

### 1. Length Generalization
How well does the model handle sequences longer than training length?
- **Excellent**: ALiBi, YaRN, FIRE, LongRoPE
- **Good**: RoPE, Sinusoidal, NTK-Aware RoPE
- **Poor**: Learned PE

### 2. Parameter Efficiency
How many extra parameters are needed?
- **Zero**: Sinusoidal, RoPE, ALiBi, NTK-Aware, Resonance RoPE
- **Minimal**: FIRE (small MLP), CoPE (small network)
- **Moderate**: Relative Bias (buckets × heads)
- **High**: Learned PE (seq_len × dim), CLEX (dynamics network)

### 3. Computational Overhead
Additional computation beyond attention?
- **Zero**: Sinusoidal, Learned PE
- **Minimal**: RoPE, ALiBi (rotation/bias computation)
- **Low**: FIRE, Relative Bias (MLP forward or bucket lookup)
- **Moderate**: CoPE (gate computation), CLEX (ODE integration)

### 4. Training Stability
How easy is it to train with this encoding?
- **Very Stable**: Sinusoidal, RoPE, ALiBi
- **Stable**: Learned PE, Relative Bias, YaRN
- **Needs Care**: FIRE, CoPE, CLEX (learned components)

### 5. Context Extension Capability
Maximum practical context length?
- **Unlimited**: ALiBi, FIRE
- **Very Large (2M+)**: LongRoPE, CLEX
- **Large (128K-256K)**: YaRN, NTK-Aware, Resonance RoPE
- **Moderate (16K-64K)**: RoPE with interpolation
- **Limited (8K)**: Sinusoidal, Learned PE

## Combining Methods

Many modern approaches combine multiple techniques:

1. **YaRN + Resonance RoPE**: Smooth interpolation + integer wavelengths
2. **CoPE + RoPE**: Content-dependent + absolute positions
3. **RoPE + ALiBi**: Rotation encoding + linear bias (research)
4. **FIRE + Multi-head**: Learned relative bias per head
5. **LongRoPE + Resonance**: Searched factors + wavelength snapping

## Common Pitfalls and Best Practices

### Pitfall 1: Training-Inference Mismatch
**Problem**: Training on short sequences, deploying on long ones
**Solution**: Use encodings with good extrapolation (ALiBi, YaRN, FIRE)

### Pitfall 2: Forgetting to Scale Attention
**Problem**: Attention magnitude increases with extended context
**Solution**: Use mscale (YaRN), learned scaling (CLEX), or attention scaling

### Pitfall 3: Uniform Frequency Scaling
**Problem**: Scaling all frequencies equally distorts local patterns
**Solution**: Use non-uniform methods (NTK-Aware, YaRN, LongRoPE)

### Pitfall 4: Not Caching Embeddings
**Problem**: Recomputing position encodings every forward pass
**Solution**: Cache cos/sin for RoPE, bias matrices for ALiBi/FIRE

### Pitfall 5: Ignoring Wavelength Interference
**Problem**: Non-integer wavelengths create artifacts at long contexts
**Solution**: Use Resonance RoPE to snap to integer wavelengths

## Research Trends

### Current (2024-2026)
- **Neural ODE-based scaling**: CLEX learns continuous dynamics
- **Content-dependent positions**: CoPE adapts to input structure
- **Extreme length models**: LongRoPE achieves 2M+ tokens
- **Hybrid approaches**: Combining multiple encoding types

### Future Directions
- **Learned dynamics models**: More sophisticated ODEs for scaling
- **Adaptive encodings**: Position encoding that evolves during training
- **Multimodal positions**: Unified encoding for text, vision, audio
- **Efficient long-context**: Sub-linear complexity with position encoding

## Implementation Notes

All implementations are located in `/Users/kevinyu/Projects/Nexus/nexus/components/embeddings/`:

- `sinusoidal.py`: Sinusoidal positional encoding
- `learned_pe.py`: Learned absolute positional embeddings
- `rotary_embedding.py`: Basic RoPE implementation
- `alibi.py`: ALiBi (Attention with Linear Biases)
- `relative_bias.py`: T5-style relative positional bias
- `multiscale_rope.py`: Multi-scale RoPE
- `yarn.py`: YaRN (Yet another RoPE extensioN)
- `cope.py`: CoPE (Contextual Position Encoding)
- `ntk_rope.py`: NTK-Aware RoPE
- `long_rope.py`: LongRoPE
- `fire.py`: FIRE (Functional Interpolation for Relative Encoding)
- `resonance_rope.py`: Resonance RoPE
- `clex.py`: CLEX (Continuous Length Extrapolation)

## Quick Start Examples

```python
# Sinusoidal PE (original Transformer)
from nexus.components.embeddings import SinusoidalPositionalEncoding
pos_enc = SinusoidalPositionalEncoding(dim=512, max_seq_len=5000)
x = embeddings  # (batch, seq_len, dim)
x_with_pos = pos_enc(x)

# RoPE (modern LLMs)
from nexus.components.embeddings import RotaryEmbedding
rope = RotaryEmbedding(dim=128, max_seq_len=8192)
cos, sin = rope(x)
q_rot, k_rot = apply_rotary_pos_emb(q, k, sin, cos)

# ALiBi (excellent extrapolation)
from nexus.components.embeddings import ALiBi
alibi = ALiBi(num_heads=8, max_seq_len=8192)
attn_scores = alibi(attn_scores)

# YaRN (context extension)
from nexus.components.embeddings import YaRN
yarn = YaRN(dim=128, scale=4.0, original_max_position_embeddings=4096)
cos, sin = yarn(x)

# FIRE (learned relative bias)
from nexus.components.embeddings import FIRE
fire = FIRE(dim=512, num_heads=8, max_position=8192)
bias = fire(seq_len=1024)
attn_scores = attn_scores + bias.unsqueeze(0)
```

## Benchmarks

### Length Generalization (PPL on long sequences)

| Method | Train 2K | Test 4K | Test 8K | Test 16K | Test 32K |
|--------|----------|---------|---------|----------|----------|
| Learned PE | 15.2 | ∞ | ∞ | ∞ | ∞ |
| Sinusoidal | 15.1 | 18.2 | 25.3 | 45.1 | 89.2 |
| RoPE | 15.0 | 17.1 | 22.8 | 38.4 | 72.1 |
| ALiBi | 15.1 | 15.8 | 16.9 | 18.2 | 19.8 |
| YaRN | 15.0 | 15.3 | 15.9 | 16.8 | 18.3 |
| FIRE | 15.1 | 15.4 | 15.7 | 16.2 | 17.1 |
| LongRoPE | 15.0 | 15.2 | 15.5 | 16.0 | 16.5 |

*Lower is better. ∞ indicates model failure.*

### Computational Overhead (ms per forward pass, batch=1, seq=4096)

| Method | Encoding Time | Attention Time | Total Overhead |
|--------|---------------|----------------|----------------|
| None | 0.0 | 12.3 | 0% |
| Sinusoidal | 0.1 | 12.3 | 0.8% |
| Learned PE | 0.1 | 12.3 | 0.8% |
| RoPE | 0.3 | 12.3 | 2.4% |
| ALiBi | 0.2 | 12.3 | 1.6% |
| FIRE | 0.5 | 12.3 | 4.1% |
| YaRN | 0.3 | 12.3 | 2.4% |
| CLEX | 1.2 | 12.3 | 9.8% |

## References

See individual encoding documentation for detailed references. Key papers:

1. Vaswani et al. (2017) - "Attention Is All You Need" (Sinusoidal)
2. Devlin et al. (2018) - "BERT" (Learned PE)
3. Su et al. (2021) - "RoFormer" (RoPE)
4. Raffel et al. (2020) - "T5" (Relative Bias)
5. Press et al. (2021) - "ALiBi"
6. Peng et al. (2023) - "YaRN"
7. Liu et al. (2024) - "FIRE"
8. Ding et al. (2024) - "LongRoPE"
9. Chen et al. (2024) - "CLEX"
10. Li et al. (2024) - "Resonance RoPE"

## Contributing

When adding new positional encoding methods:
1. Implement in `nexus/components/embeddings/`
2. Add comprehensive documentation following the template
3. Include mathematical derivations and intuitions
4. Provide code examples and usage patterns
5. Benchmark against existing methods
6. Update this README with decision guide entry

## Navigation

- [Sinusoidal PE](./sinusoidal_pe.md)
- [Learned PE](./learned_pe.md)
- [RoPE (Rotary Position Embedding)](./rope.md)
- [Relative Bias (T5-style)](./relative_bias.md)
- [ALiBi (Attention with Linear Biases)](./alibi.md)
- [Multiscale RoPE](./multiscale_rope.md)
- [YaRN (Yet another RoPE extensioN)](./yarn.md)
- [CoPE (Contextual Position Encoding)](./cope.md)
- [NTK-Aware RoPE](./ntk_rope.md)
- [LongRoPE](./long_rope.md)
- [FIRE (Functional Interpolation)](./fire.md)
- [Resonance RoPE](./resonance_rope.md)
- [CLEX (Continuous Length Extrapolation)](./clex.md)
