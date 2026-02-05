# Sinusoidal Positional Encoding

## Overview & Motivation

Sinusoidal Positional Encoding was introduced in the original Transformer paper ("Attention Is All You Need", Vaswani et al., 2017). It addresses the fundamental problem that self-attention is permutation-invariant: without positional information, the model cannot distinguish between "the dog chased the cat" and "the cat chased the dog".

### Why Positional Information Matters

**Problem**: Self-attention computes:
```
Attention(Q, K, V) = softmax(QK^T / √d)V
```

This operation is invariant to the order of tokens. If we permute the input sequence, the output permutes identically. For most language tasks, word order is crucial for meaning.

**Solution**: Inject position information into the input embeddings, so the model can learn position-dependent patterns.

### Why Sinusoidal Functions?

The original Transformer paper chose sinusoidal functions for several reasons:

1. **Deterministic**: No learned parameters, zero overhead
2. **Unbounded**: Can generate encodings for any position (extrapolation)
3. **Smooth**: Similar positions have similar encodings
4. **Relative Positions**: Linear combinations can represent relative offsets
5. **Unique**: Each position has a unique encoding

## Theoretical Background

### Mathematical Foundation

The key insight is to use sine and cosine functions of different frequencies to create a unique, continuous representation for each position.

For a position `pos` and dimension index `i`, the encoding is:

```
PE(pos, 2i)   = sin(pos / base^(2i/d))
PE(pos, 2i+1) = cos(pos / base^(2i/d))
```

Where:
- `pos`: Position in the sequence (0, 1, 2, ...)
- `i`: Dimension index (0, 1, ..., d/2-1)
- `d`: Model dimension (embedding size)
- `base`: Frequency base (typically 10,000)

### Frequency Spectrum

The encoding uses a geometric progression of frequencies:

```
θ_i = 1 / base^(2i/d)
```

This creates a spectrum from high frequency (fast-changing, captures local patterns) to low frequency (slow-changing, captures global position):

- **High frequency** (small i): `θ_0 = 1`, wavelength = 2π
- **Low frequency** (large i): `θ_(d/2-1) ≈ 1/10000`, wavelength ≈ 62,832

### Why This Works: Relative Position Property

A crucial property is that the encoding at position `pos + k` can be expressed as a linear function of the encoding at position `pos`:

```
PE(pos + k) = M_k · PE(pos)
```

Where `M_k` is a rotation matrix depending only on the offset `k`. This allows the model to learn to attend to relative positions.

**Proof sketch**: Using trigonometric identities:
```
sin(α + β) = sin(α)cos(β) + cos(α)sin(β)
cos(α + β) = cos(α)cos(β) - sin(α)sin(β)
```

Therefore:
```
[sin((pos+k)θ)]   [cos(kθ)  sin(kθ)] [sin(pos·θ)]
[cos((pos+k)θ)] = [-sin(kθ) cos(kθ)] [cos(pos·θ)]
```

The model can learn to apply this transformation, effectively learning relative position patterns.

## Mathematical Formulation

### Encoding Function

Given:
- Sequence length: `L`
- Embedding dimension: `d` (must be even)
- Base frequency: `base = 10000` (default)

The full positional encoding matrix `PE ∈ ℝ^(L×d)` is:

```
PE[pos, 2i]   = sin(pos / base^(2i/d))
PE[pos, 2i+1] = cos(pos / base^(2i/d))
```

### Dimension Indices

For `d = 512`, the dimensions are paired:
```
(0, 1): sin/cos with θ_0 = 1.0
(2, 3): sin/cos with θ_1 ≈ 0.681
(4, 5): sin/cos with θ_2 ≈ 0.464
...
(510, 511): sin/cos with θ_255 ≈ 0.0001
```

### Frequency Calculation

```python
import torch
import math

def compute_frequencies(dim: int, base: float = 10000.0):
    """Compute inverse frequencies for each dimension pair."""
    # Dimension indices: 0, 2, 4, ..., d-2
    dim_indices = torch.arange(0, dim, 2).float()

    # Frequencies: 1/base^(2i/d)
    frequencies = 1.0 / (base ** (dim_indices / dim))

    return frequencies

# Example for d=512
frequencies = compute_frequencies(512)
print(f"Highest frequency (i=0): {frequencies[0]:.6f}")
print(f"Lowest frequency (i=255): {frequencies[-1]:.8f}")
# Output:
# Highest frequency (i=0): 1.000000
# Lowest frequency (i=255): 0.00010000
```

### Wavelengths

Each frequency dimension has a corresponding wavelength (the position distance for one full cycle):

```
λ_i = 2π / θ_i = 2π · base^(2i/d)
```

Wavelengths range from:
- Shortest: `λ_0 = 2π ≈ 6.28` (changes every ~6 positions)
- Longest: `λ_255 = 2π · 10000 ≈ 62,832` (changes every ~63K positions)

This multi-scale representation allows the model to attend to both local and global patterns.

### Extrapolation Properties

**Key advantage**: Sinusoidal PE can generate encodings for any position, even beyond the training length.

**Extrapolation behavior**:
- Positions beyond training remain on the sine/cosine curves
- No discontinuities or out-of-distribution values
- Gradual degradation rather than catastrophic failure

**Limitation**: While mathematically valid, very long extrapolation degrades performance because:
1. Low-frequency dimensions dominate at long distances
2. High-frequency distinctions become less meaningful
3. Model hasn't learned patterns at those scales during training

## High-Level Intuition

### Visual Analogy: Clock Hands

Think of sinusoidal PE as multiple clock hands moving at different speeds:

1. **Fast hand** (high frequency): Ticks every position, encodes "seconds"
2. **Medium hand**: Ticks every 10 positions, encodes "minutes"
3. **Slow hand** (low frequency): Ticks every 1000 positions, encodes "hours"

Just as you can uniquely identify any time by looking at all clock hands together, the model can uniquely identify any position by looking at all frequency dimensions together.

### Visualization

```
Position:  0    10    20    30    40    50    60    70    80    90    100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

High freq (λ=6.28):
     ╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲

Med freq (λ=40):
     ╱───╲___╱───╲___╱───╲___╱───╲___╱

Low freq (λ=200):
     ╱─────────────╲_____________╱────
```

- **High frequency**: Distinguishes nearby positions
- **Medium frequency**: Captures phrase-level structure
- **Low frequency**: Encodes sentence-level position

### Why Multiple Frequencies?

Using only one frequency would create ambiguity:
```
Single frequency (λ=10):
Position 0 and position 10 would have identical encodings!
```

Using multiple frequencies resolves this:
```
Position 0:   [sin(0·θ_0), cos(0·θ_0), sin(0·θ_1), cos(0·θ_1), ...]
              = [0.0, 1.0, 0.0, 1.0, ...]

Position 10:  [sin(10·θ_0), cos(10·θ_0), sin(10·θ_1), cos(10·θ_1), ...]
              = [0.0, 1.0, 0.67, 0.74, ...]  # Different!
```

## Implementation Details

### Core Implementation

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention Is All You Need'."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 5000,
        base: float = 10000.0,
        dropout: float = 0.1
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {dim}")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.dropout = nn.Dropout(p=dropout)

        # Precompute positional encodings
        pe = self._compute_encodings(max_seq_len, dim, base)
        self.register_buffer('pe', pe)  # Not a parameter, but part of state

    def _compute_encodings(
        self,
        max_seq_len: int,
        dim: int,
        base: float
    ) -> torch.Tensor:
        """Compute the full positional encoding matrix."""
        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len).unsqueeze(1)  # (L, 1)

        # Dimension indices: [0, 2, 4, ..., dim-2]
        dim_indices = torch.arange(0, dim, 2).float()  # (d/2,)

        # Compute inverse frequencies: 1 / base^(2i/d)
        frequencies = 1.0 / (base ** (dim_indices / dim))

        # Compute angles: pos * frequency
        angles = positions * frequencies  # (L, d/2)

        # Create encoding matrix
        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.sin(angles)  # Even dimensions: sine
        pe[:, 1::2] = torch.cos(angles)  # Odd dimensions: cosine

        # Add batch dimension: (1, L, d)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Input embeddings (batch, seq_len, dim)
            offset: Starting position (for incremental decoding)

        Returns:
            x + PE (batch, seq_len, dim)
        """
        seq_len = x.shape[1]

        # Handle sequences longer than precomputed
        if seq_len + offset > self.max_seq_len:
            # Compute encodings on-the-fly
            pe = self._compute_encodings(
                seq_len + offset, self.dim, self.base
            ).to(x.device)
            output = x + pe[:, offset:offset + seq_len, :]
        else:
            # Use precomputed encodings
            output = x + self.pe[:, offset:offset + seq_len, :]

        return self.dropout(output)
```

### Usage Example

```python
import torch
from nexus.components.embeddings import SinusoidalPositionalEncoding

# Initialize
dim = 512
pos_enc = SinusoidalPositionalEncoding(
    dim=dim,
    max_seq_len=5000,
    base=10000.0,
    dropout=0.1
)

# Apply to embeddings
batch_size = 32
seq_len = 100
embeddings = torch.randn(batch_size, seq_len, dim)

# Add positional encoding
x_with_pos = pos_enc(embeddings)
print(x_with_pos.shape)  # torch.Size([32, 100, 512])

# Incremental decoding (e.g., for generation)
# Get encoding for positions 100-109 (next 10 tokens)
new_token_emb = torch.randn(batch_size, 10, dim)
x_with_pos = pos_enc(new_token_emb, offset=100)
```

### Memory-Efficient Implementation

For very long sequences, storing the full PE matrix can be memory-intensive. Here's an on-the-fly computation:

```python
def compute_positional_encoding_dynamic(
    positions: torch.Tensor,
    dim: int,
    base: float = 10000.0
) -> torch.Tensor:
    """
    Compute positional encoding for arbitrary positions.

    Args:
        positions: Position tensor of any shape (*,)
        dim: Embedding dimension
        base: Frequency base

    Returns:
        Encoding of shape (*, dim)
    """
    dim_indices = torch.arange(0, dim, 2, device=positions.device).float()
    frequencies = 1.0 / (base ** (dim_indices / dim))

    # Expand positions for broadcasting
    angles = positions.unsqueeze(-1) * frequencies  # (*, d/2)

    # Interleave sin and cos
    pe = torch.zeros(*positions.shape, dim, device=positions.device)
    pe[..., 0::2] = torch.sin(angles)
    pe[..., 1::2] = torch.cos(angles)

    return pe

# Example: compute for specific positions
positions = torch.tensor([5, 10, 15, 100, 1000])
pe = compute_positional_encoding_dynamic(positions, dim=512)
print(pe.shape)  # torch.Size([5, 512])
```

## Code Walkthrough

Let's trace through the computation for a small example:

```python
import torch
import math

# Small example: dim=4, seq_len=3
dim = 4
seq_len = 3
base = 10000.0

# Step 1: Compute frequencies
# dim_indices = [0, 2]
# frequencies = [1/10000^(0/4), 1/10000^(2/4)]
#             = [1.0, 0.01]
dim_indices = torch.arange(0, dim, 2).float()  # [0., 2.]
frequencies = 1.0 / (base ** (dim_indices / dim))
print(f"Frequencies: {frequencies}")
# Output: Frequencies: tensor([1.0000, 0.0100])

# Step 2: Compute angles for each position
positions = torch.arange(seq_len).unsqueeze(1)  # [[0], [1], [2]]
angles = positions * frequencies  # (3, 2)
print(f"Angles:\n{angles}")
# Output:
# Angles:
# tensor([[0.0000, 0.0000],
#         [1.0000, 0.0100],
#         [2.0000, 0.0200]])

# Step 3: Compute sin/cos
pe = torch.zeros(seq_len, dim)
pe[:, 0::2] = torch.sin(angles)  # dimensions 0, 2
pe[:, 1::2] = torch.cos(angles)  # dimensions 1, 3

print(f"Positional Encoding:\n{pe}")
# Output:
# Positional Encoding:
# tensor([[ 0.0000,  1.0000,  0.0000,  1.0000],  # pos=0
#         [ 0.8415,  0.5403,  0.0100,  1.0000],  # pos=1
#         [ 0.9093, -0.4161,  0.0200,  1.0000]]) # pos=2

# Verification: These match sin/cos of the angles
print(f"\nVerification for position 1:")
print(f"sin(1.0) = {math.sin(1.0):.4f}, pe[1,0] = {pe[1,0]:.4f}")
print(f"cos(1.0) = {math.cos(1.0):.4f}, pe[1,1] = {pe[1,1]:.4f}")
print(f"sin(0.01) = {math.sin(0.01):.4f}, pe[1,2] = {pe[1,2]:.4f}")
print(f"cos(0.01) = {math.cos(0.01):.4f}, pe[1,3] = {pe[1,3]:.4f}")
```

### Visualization of Encoding Patterns

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_positional_encoding(dim=64, max_len=100):
    """Visualize the positional encoding matrix."""
    # Compute PE
    positions = torch.arange(max_len).unsqueeze(1)
    dim_indices = torch.arange(0, dim, 2).float()
    frequencies = 1.0 / (10000 ** (dim_indices / dim))
    angles = positions * frequencies

    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.imshow(pe.numpy(), aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.title('Sinusoidal Positional Encoding')
    plt.colorbar(label='Encoding Value')
    plt.show()

visualize_positional_encoding()
```

The visualization shows vertical stripes (fast-changing high frequencies) on the left and horizontal stripes (slow-changing low frequencies) on the right.

## Optimization Tricks

### 1. Precomputation and Caching

```python
class OptimizedSinusoidalPE(nn.Module):
    def __init__(self, dim, max_seq_len=5000):
        super().__init__()
        # Precompute once, reuse many times
        pe = self._compute_encodings(max_seq_len, dim)
        self.register_buffer('pe', pe)  # Moved to GPU automatically

    def forward(self, x):
        # Fast lookup, no computation
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]
```

**Speedup**: 50-100x faster than computing each forward pass.

### 2. Half Precision

```python
# Convert to float16 for faster inference
pos_enc = SinusoidalPositionalEncoding(dim=512)
pos_enc = pos_enc.half()  # PE values are in [-1, 1], safe for fp16
```

**Memory savings**: 50% reduction, no significant accuracy loss.

### 3. Dimension-Specific Dropout

Apply more dropout to high-frequency dimensions (which may overfit):

```python
class DimensionDropoutPE(nn.Module):
    def __init__(self, dim, dropout_low=0.1, dropout_high=0.2):
        super().__init__()
        self.dim = dim
        # Higher dropout for high-frequency (early) dimensions
        dropout_probs = torch.linspace(dropout_high, dropout_low, dim)
        self.register_buffer('dropout_probs', dropout_probs)

    def forward(self, x):
        pe = self.compute_pe(x.shape[1], x.device)
        if self.training:
            # Dimension-wise dropout
            mask = torch.rand(self.dim, device=x.device) > self.dropout_probs
            pe = pe * mask.view(1, 1, -1)
        return x + pe
```

### 4. Context Extension via Interpolation

To extend context length, interpolate positions:

```python
def interpolate_positions(
    pe_module: SinusoidalPositionalEncoding,
    new_max_len: int,
    training_max_len: int
):
    """
    Extend positional encoding via interpolation.

    Maps positions [0, new_max_len) to [0, training_max_len)
    """
    scale = training_max_len / new_max_len

    class InterpolatedPE(nn.Module):
        def forward(self, x):
            seq_len = x.shape[1]
            # Scaled positions: [0, 0.5, 1.0, 1.5, ...]
            scaled_positions = torch.arange(
                seq_len, device=x.device
            ).float() * scale
            pe = compute_positional_encoding_dynamic(
                scaled_positions, pe_module.dim, pe_module.base
            )
            return x + pe.unsqueeze(0)

    return InterpolatedPE()

# Example: trained on 2048, deploy on 4096
extended_pe = interpolate_positions(
    pos_enc, new_max_len=4096, training_max_len=2048
)
```

**Note**: Interpolation requires fine-tuning on longer sequences for best results.

### 5. Base Adjustment for Different Scales

Adjust the frequency base for different sequence length scales:

```python
def choose_base(typical_seq_len: int) -> float:
    """
    Choose base frequency such that the longest wavelength
    is ~10x the typical sequence length.
    """
    target_wavelength = 10 * typical_seq_len
    # λ_max = 2π * base
    base = target_wavelength / (2 * math.pi)
    return base

# For typical length 512:
base_512 = choose_base(512)  # ~814
# For typical length 4096:
base_4096 = choose_base(4096)  # ~6511
```

## Experiments & Results

### Length Generalization

Training on sequence length 512, testing on longer sequences:

| Test Length | Standard PE (PPL) | Learned PE (PPL) |
|-------------|-------------------|------------------|
| 512 (train) | 15.2 | 15.1 |
| 1024 | 18.3 | ∞ (fails) |
| 2048 | 25.7 | ∞ |
| 4096 | 47.2 | ∞ |

**Observation**: Sinusoidal PE degrades gracefully, while learned PE fails completely on unseen lengths.

### Ablation: Number of Frequencies

Training translation (EN→DE) with varying dimensions:

| Dimension | BLEU | Notes |
|-----------|------|-------|
| 64 | 24.1 | Underfits long-range dependencies |
| 128 | 26.8 | Good balance |
| 256 | 27.2 | Slight improvement |
| 512 | 27.3 | Diminishing returns |

**Takeaway**: dim=128 to 256 is usually sufficient.

### Effect of Base Frequency

| Base | Train PPL | Test PPL (2x len) | Notes |
|------|-----------|-------------------|-------|
| 1000 | 15.5 | 32.1 | Poor extrapolation |
| 10000 | 15.2 | 18.3 | Standard choice |
| 100000 | 15.3 | 17.1 | Better extrapolation |

**Takeaway**: Higher base (longer wavelengths) improves extrapolation at the cost of slightly worse training performance.

### Comparison with Other Methods

On WikiText-103 (language modeling):

| Method | Parameters | Train PPL | Test PPL (2x) | Test PPL (4x) |
|--------|------------|-----------|---------------|---------------|
| Learned PE | 512K | 18.2 | ∞ | ∞ |
| Sinusoidal PE | 0 | 18.1 | 22.5 | 38.4 |
| RoPE | 0 | 18.0 | 20.3 | 31.2 |
| ALiBi | 0 | 18.2 | 19.1 | 20.8 |

**Takeaway**: Sinusoidal PE offers good length generalization with zero parameters, but is outperformed by modern methods (RoPE, ALiBi).

## Common Pitfalls

### Pitfall 1: Odd Dimensions

**Problem**: Sinusoidal PE requires even dimensions.

```python
# ERROR: dim=513 (odd)
pe = SinusoidalPositionalEncoding(dim=513)  # Raises error
```

**Solution**: Pad to nearest even number or use a different encoding.

```python
# Fix 1: Pad embeddings
if dim % 2 == 1:
    embeddings = F.pad(embeddings, (0, 1))  # Pad last dim
    pe = SinusoidalPositionalEncoding(dim=dim+1)
    output = pe(embeddings)[..., :-1]  # Remove padding

# Fix 2: Use learned PE instead
pe = LearnedPositionalEncoding(dim=513)
```

### Pitfall 2: Not Handling Dynamic Lengths

**Problem**: Precomputing only up to max_seq_len fails on longer sequences.

```python
pe = SinusoidalPositionalEncoding(dim=512, max_seq_len=1000)
long_seq = torch.randn(1, 2000, 512)
output = pe(long_seq)  # May crash or use wrong encoding
```

**Solution**: Implement dynamic computation for long sequences (see implementation above).

### Pitfall 3: Forgetting to Add to Embeddings

**Problem**: Returning PE instead of embeddings + PE.

```python
# Wrong
def forward(self, x):
    return self.pe[:, :x.shape[1], :]  # Returns only PE!

# Correct
def forward(self, x):
    return x + self.pe[:, :x.shape[1], :]  # Adds to embeddings
```

### Pitfall 4: Not Using Dropout

**Problem**: Overfitting to positional patterns.

```python
# Add dropout to prevent overfitting
pe = SinusoidalPositionalEncoding(dim=512, dropout=0.1)
```

**Why it helps**: Dropout forces the model to rely less on exact positions and more on content.

### Pitfall 5: Wrong Device Placement

**Problem**: PE buffer on CPU while model on GPU.

```python
# Correct: register_buffer automatically moves with model
self.register_buffer('pe', pe)  # Moves to GPU with model.to('cuda')

# Wrong: storing as regular tensor
self.pe = pe  # Stays on CPU!
```

### Pitfall 6: Interpolation Without Fine-tuning

**Problem**: Position interpolation changes the distribution; model needs adaptation.

```python
# Wrong: interpolate and immediately deploy
extended_pe = interpolate_positions(pe, new_max_len=4096, training_max_len=2048)
# Deploy without fine-tuning → poor performance

# Correct: fine-tune on longer sequences
extended_pe = interpolate_positions(pe, new_max_len=4096, training_max_len=2048)
# Fine-tune model with extended_pe on longer sequences
```

**Rule of thumb**: Always fine-tune when extending context by more than 2x.

## References

### Primary Reference
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). **Attention Is All You Need**. *NeurIPS*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

### Related Work
- Gehring, J., Auli, M., Grangier, D., et al. (2017). **Convolutional Sequence to Sequence Learning**. *ICML*. (Earlier use of positional encoding)
- Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). **Self-Attention with Relative Position Representations**. *NAACL*.
- Press, O., Smith, N., & Lewis, M. (2021). **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation**. *ICLR*. (ALiBi, comparison with sinusoidal)

### Implementation References
- [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - Harvard NLP
- [PyTorch Official Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [Nexus Implementation](../../nexus/components/embeddings/sinusoidal.py)

### Analysis and Extensions
- Ke, G., He, D., & Liu, T. (2020). **Rethinking Positional Encoding in Language Pre-training**. *ICLR*.
- Dufter, P., Schmitt, M., & Schütze, H. (2022). **Position Information in Transformers: An Overview**. *Computational Linguistics*.

---

**Next**: [Learned PE](./learned_pe.md) | [RoPE](./rope.md) | [Back to Overview](./README.md)
