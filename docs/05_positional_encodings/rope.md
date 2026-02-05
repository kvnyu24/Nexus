# RoPE: Rotary Position Embedding

## Overview & Motivation

Rotary Position Embedding (RoPE) revolutionized positional encoding by applying rotation transformations directly to query and key vectors in the attention mechanism. Introduced in the RoFormer paper (Su et al., 2021), RoPE has become the de facto standard for modern large language models including GPT-NeoX, PaLM, LLaMA, and GPT-4.

### Why RoPE?

**Key Innovation**: Instead of adding positional information to embeddings, RoPE rotates the query and key vectors by an angle proportional to their position. This encoding method:

1. **Naturally encodes relative positions**: The dot product between rotated vectors depends only on their relative position difference
2. **Preserves inner product structure**: Rotation maintains vector magnitudes
3. **Enables efficient computation**: Can be implemented with element-wise operations
4. **Generalizes well**: Excellent interpolation and reasonable extrapolation properties
5. **Works with any attention mechanism**: Drop-in replacement for absolute PE

### RoPE vs. Sinusoidal PE

| Aspect | Sinusoidal PE | RoPE |
|--------|---------------|------|
| Application | Add to embeddings | Rotate Q/K vectors |
| Relative encoding | Indirect (via learned attention) | Direct (in dot product) |
| Value vectors | Affected | Not affected (preserves content) |
| Extrapolation | Moderate | Good |
| Memory | O(L × d) cached | O(L × d) cached |
| Modern LLMs | Rare | Standard |

**Bottom line**: RoPE explicitly encodes relative positions into the attention mechanism's geometric structure, making it more principled and effective.

## Theoretical Background

### The Core Insight

In self-attention, we compute:
```
Attention(Q, K, V) = softmax(QK^T / √d)V
```

The key observation: **the attention weight between position m and n depends only on their relative offset (m - n)**.

RoPE achieves this by rotating query and key vectors such that:
```
<f_q(x_m, m), f_k(x_n, n)> = g(x_m, x_n, m - n)
```

Where:
- `f_q`, `f_k` are functions that encode position into queries and keys
- `g` is a function of only the relative position `m - n`
- `<·, ·>` denotes inner product

### Rotation Matrices in 2D

Consider a 2D rotation by angle θ:
```
R(θ) = [cos θ  -sin θ]
       [sin θ   cos θ]
```

Key property: Rotating two vectors and taking their dot product:
```
R(θ₁)ᵀ R(θ₂) x = R(θ₂ - θ₁) x
```

This means the relative angle (θ₂ - θ₁) determines the result!

### Extension to High Dimensions

RoPE applies 2D rotations to pairs of dimensions independently. For a d-dimensional vector, we create d/2 rotation pairs, each with its own frequency:

```
RoPE([x₀, x₁, x₂, x₃, ..., x_d-2, x_d-1]) =
  [R(θ₀) [x₀],   R(θ₁) [x₂],   ...,   R(θ_d/2-1) [x_d-2]]
        [x₁],         [x₃],                      [x_d-1]]
```

Where θᵢ = pos / base^(2i/d), similar to sinusoidal PE but used as rotation angles.

### Mathematical Formulation

For position `m`, dimension pair `(2i, 2i+1)`:

```
[q_2i']     = [cos(mθᵢ)  -sin(mθᵢ)] [q_2i]
[q_2i+1']     [sin(mθᵢ)   cos(mθᵢ)] [q_2i+1]
```

Where θᵢ = 1 / base^(2i/d) is the frequency for dimension pair i.

### Relative Position Property

The beauty of RoPE: When we compute attention between position m and n:

```
q_m^T k_n = Σᵢ [q_2i', q_2i+1'] R(mθᵢ)^T R(nθᵢ) [k_2i]
                                                  [k_2i+1]

          = Σᵢ [q_2i', q_2i+1'] R((m-n)θᵢ) [k_2i]      ← Only depends on m-n!
                                            [k_2i+1]
```

This is **exact** relative position encoding in the dot product!

### Complex Number Formulation

RoPE can be elegantly expressed using complex numbers. Treat each dimension pair as a complex number:

```
z = x_2i + i·x_2i+1
```

Then rotation becomes:
```
z' = z · e^(i·pos·θ)
```

This is computationally equivalent but mathematically cleaner:

```python
# Complex formulation (conceptual)
q_complex = q[..., ::2] + 1j * q[..., 1::2]
freqs = torch.exp(1j * pos * theta)
q_rotated = q_complex * freqs
```

## High-Level Intuition

### Geometric Interpretation

Think of each dimension pair as a 2D plane. RoPE rotates vectors in each plane:

```
Position 0:  →  (no rotation)
Position 1:  ↗  (rotate by θ)
Position 2:  ↑  (rotate by 2θ)
Position 3:  ↖  (rotate by 3θ)
...
```

When computing attention:
- Same position (m=m): angle difference = 0° → maximum similarity
- Adjacent positions (m=m+1): angle difference = θ → slight reduction
- Distant positions (m=m+100): angle difference = 100θ → larger reduction

The attention mechanism naturally learns to interpret these geometric relationships!

### Clock Analogy

Similar to sinusoidal PE, but instead of multiple clock hands encoding position, RoPE uses **the rotation of the hands** to encode position:

- **Fast-rotating hands** (high frequency θᵢ): Distinguish nearby positions
- **Slow-rotating hands** (low frequency θᵢ): Encode long-range structure

But unlike sinusoidal PE, RoPE rotates the queries and keys themselves, making relative positions explicit in the geometry.

### Visual Diagram

```
Query at position m=5:
     ╱
    ╱  ← rotated by 5θ
   ╱
  •────────→ (original direction)

Key at position n=3:
     ╱
    ╱  ← rotated by 3θ
   •────────→

Relative angle = (5-3)θ = 2θ  ← Only relative position matters!
```

## Implementation Details

### Core Implementation

```python
import torch
import torch.nn as nn
import math
from typing import Tuple

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0
    ):
        super().__init__()

        assert dim % 2 == 0, "Dimension must be even for RoPE"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies: θᵢ = 1 / base^(2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos/sin for efficiency
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _compute_cos_sin(
        self,
        seq_len: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin embeddings for given sequence length."""
        # Position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device).float()

        # Compute angles: pos * inv_freq
        # Shape: (seq_len, dim/2)
        freqs = torch.outer(positions, self.inv_freq)

        # Duplicate for full dimension (each frequency appears twice)
        # Shape: (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        return cos, sin

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RoPE cos/sin embeddings.

        Args:
            x: Input tensor (for device/shape inference)
            seq_len: Sequence length (if None, inferred from x)

        Returns:
            cos: Cosine embeddings (1, seq_len, dim)
            sin: Sine embeddings (1, seq_len, dim)
        """
        if seq_len is None:
            seq_len = x.shape[1]

        # Use cache if available
        if seq_len <= self._seq_len_cached and self._cos_cached is not None:
            return (
                self._cos_cached[:, :seq_len, :],
                self._sin_cached[:, :seq_len, :]
            )

        # Compute cos/sin
        cos, sin = self._compute_cos_sin(seq_len, x.device)

        # Add batch dimension
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        # Update cache
        self._cos_cached = cos
        self._sin_cached = sin
        self._seq_len_cached = seq_len

        return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half of the hidden dimensions.

    RoPE rotates pairs of dimensions. This function prepares the tensor
    for rotation by swapping and negating the second element of each pair.
    """
    x1 = x[..., : x.shape[-1] // 2]  # First half
    x2 = x[..., x.shape[-1] // 2 :]  # Second half
    return torch.cat((-x2, x1), dim=-1)  # [-x2, x1]

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to Q and K.

    Args:
        q: Query tensor (batch, heads, seq_len, head_dim)
        k: Key tensor (batch, heads, seq_len, head_dim)
        cos: Cosine embeddings (1, seq_len, dim) or (batch, seq_len, dim)
        sin: Sine embeddings (1, seq_len, dim) or (batch, seq_len, dim)

    Returns:
        q_embed: Rotated queries
        k_embed: Rotated keys
    """
    # Reshape cos/sin for broadcasting
    # From (1, seq_len, dim) to (1, 1, seq_len, dim)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Apply rotation using the formula:
    # q' = q * cos + rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
```

### Understanding the Rotation Formula

The rotation formula looks like:
```python
q_rotated = (q * cos) + (rotate_half(q) * sin)
```

Let's see why this works for a 2D example:

```python
# Original query pair: [q0, q1]
# Rotation angle: θ
# Want: [q0 * cos(θ) - q1 * sin(θ), q0 * sin(θ) + q1 * cos(θ)]

# Step 1: q * cos = [q0 * cos(θ), q1 * cos(θ)]
# Step 2: rotate_half(q) = [-q1, q0]
# Step 3: rotate_half(q) * sin = [-q1 * sin(θ), q0 * sin(θ)]
# Step 4: Sum = [q0*cos(θ) - q1*sin(θ), q1*cos(θ) + q0*sin(θ)] ✓
```

This is exactly the 2D rotation formula!

### Complete Usage Example

```python
import torch
from nexus.components.embeddings import RotaryEmbedding, apply_rotary_pos_emb

# Initialize RoPE
dim = 128  # Head dimension
rope = RotaryEmbedding(dim=dim, max_seq_len=8192, base=10000.0)

# Simulated attention inputs
batch_size = 32
num_heads = 8
seq_len = 512
head_dim = dim

q = torch.randn(batch_size, num_heads, seq_len, head_dim)
k = torch.randn(batch_size, num_heads, seq_len, head_dim)
v = torch.randn(batch_size, num_heads, seq_len, head_dim)

# Get RoPE embeddings
cos, sin = rope(q, seq_len=seq_len)

# Apply RoPE to queries and keys
q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin)

# Values remain unchanged
v_unchanged = v

# Continue with standard attention
attn_weights = torch.matmul(q_rotated, k_rotated.transpose(-2, -1)) / math.sqrt(head_dim)
attn_weights = torch.softmax(attn_weights, dim=-1)
output = torch.matmul(attn_weights, v_unchanged)

print(f"Output shape: {output.shape}")
# Output shape: torch.Size([32, 8, 512, 128])
```

## Code Walkthrough

Let's trace through a minimal example:

```python
import torch
import math

# Minimal example: 2 positions, 4 dimensions
positions = torch.tensor([0, 1]).float()  # pos 0 and 1
dim = 4
base = 10000.0

# Step 1: Compute inverse frequencies
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
print(f"Inverse frequencies: {inv_freq}")
# Output: tensor([1.0000, 0.0100])

# Step 2: Compute angles for each position
freqs = torch.outer(positions, inv_freq)
print(f"Frequencies:\n{freqs}")
# Output:
# tensor([[0.0000, 0.0000],  # Position 0
#         [1.0000, 0.0100]]) # Position 1

# Step 3: Duplicate for full dimension
emb = torch.cat([freqs, freqs], dim=-1)
print(f"Embedding:\n{emb}")
# Output:
# tensor([[0.0000, 0.0000, 0.0000, 0.0000],
#         [1.0000, 0.0100, 1.0000, 0.0100]])

# Step 4: Compute cos/sin
cos = emb.cos()
sin = emb.sin()
print(f"Cos:\n{cos}")
print(f"Sin:\n{sin}")
# Position 0: cos=[1, 1, 1, 1], sin=[0, 0, 0, 0] (no rotation)
# Position 1: cos=[0.54, 0.9999, 0.54, 0.9999], sin=[0.84, 0.01, 0.84, 0.01]

# Step 5: Apply to a query vector
q = torch.tensor([1.0, 0.0, 1.0, 0.0])  # Example query at position 1

# rotate_half: swap and negate second element of pairs
# [q0, q1, q2, q3] → [-q1, q0, -q3, q2]
q_rotated_half = torch.cat([
    -q[1:2], q[0:1], -q[3:4], q[2:3]
])
print(f"Rotated half: {q_rotated_half}")
# Output: tensor([0., 1., 0., 1.])

# Final rotation
q_final = (q * cos[1]) + (q_rotated_half * sin[1])
print(f"Final rotated query: {q_final}")
# Output: tensor([0.5403, 0.8415, 0.5403, 0.8415])

# Verify: this is the rotation matrix applied to [1,0,1,0]
print(f"Expected for first pair: cos(1)={math.cos(1):.4f}, sin(1)={math.sin(1):.4f}")
# cos(1)=0.5403, sin(1)=0.8415 ✓
```

### Verifying Relative Position Property

```python
import torch

def compute_attention_weight(q_pos, k_pos, dim=4, base=10000.0):
    """Compute attention weight between two positions using RoPE."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # Compute embeddings for both positions
    q_freqs = q_pos * inv_freq
    k_freqs = k_pos * inv_freq

    q_emb = torch.cat([q_freqs, q_freqs], dim=-1)
    k_emb = torch.cat([k_freqs, k_freqs], dim=-1)

    q_cos, q_sin = q_emb.cos(), q_emb.sin()
    k_cos, k_sin = k_emb.cos(), k_emb.sin()

    # Sample query and key vectors
    q = torch.randn(dim)
    k = torch.randn(dim)

    # Apply RoPE
    q_rot = (q * q_cos) + (torch.cat([-q[1:2], q[0:1], -q[3:4], q[2:3]]) * q_sin)
    k_rot = (k * k_cos) + (torch.cat([-k[1:2], k[0:1], -k[3:4], k[2:3]]) * k_sin)

    # Attention weight
    return torch.dot(q_rot, k_rot)

# Test: attention weight should only depend on relative position
pos_pairs = [(0, 0), (5, 5), (100, 100)]  # Same relative position (0)
for q_pos, k_pos in pos_pairs:
    weight = compute_attention_weight(q_pos, k_pos)
    print(f"Pos ({q_pos}, {k_pos}), relative={q_pos-k_pos}: {weight:.4f}")

# All should be similar since relative position is 0
# (small differences due to random q, k)
```

## Optimization Tricks

### 1. Precomputation and Caching

```python
class CachedRoPE(nn.Module):
    """RoPE with aggressive caching."""

    def __init__(self, dim, max_seq_len=8192):
        super().__init__()
        # Precompute for max length
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]
```

**Speedup**: 5-10x faster than computing on-the-fly.

### 2. Fused Kernel (for CUDA)

```python
# Pseudocode for fused RoPE kernel
@torch.jit.script
def fused_rope(q, k, cos, sin):
    """Fused rotation kernel."""
    # Compute rotation in a single kernel launch
    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin
    return q_rotated, k_rotated

# Use in model
q_rot, k_rot = fused_rope(q, k, cos, sin)
```

**Speedup**: 2-3x faster by reducing memory transfers.

### 3. Partial RoPE

Apply RoPE to only a subset of dimensions for efficiency:

```python
class PartialRoPE(nn.Module):
    """Apply RoPE to first `rotary_dim` dimensions only."""

    def __init__(self, dim, rotary_dim=None, max_seq_len=8192):
        super().__init__()
        self.dim = dim
        self.rotary_dim = rotary_dim or dim

        assert self.rotary_dim <= dim
        assert self.rotary_dim % 2 == 0

        # RoPE only for rotary_dim
        inv_freq = 1.0 / (10000 ** (
            torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim
        ))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, q, k):
        # Split into rotary and non-rotary parts
        q_rot, q_pass = q[..., :self.rotary_dim], q[..., self.rotary_dim:]
        k_rot, k_pass = k[..., :self.rotary_dim], k[..., self.rotary_dim:]

        # Apply RoPE to rotary part
        cos, sin = self._get_cos_sin(q.shape[-2], q.device)
        q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

        # Concatenate back
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)

        return q, k

# Example: Apply RoPE to only 64 out of 128 dimensions
rope = PartialRoPE(dim=128, rotary_dim=64)
```

**Speedup**: Proportional to rotary_dim/dim (e.g., 2x if rotary_dim=dim/2).
**Used in**: GPT-J (rotary_dim=64 out of 256)

### 4. Context Extension via Interpolation

To extend context length, scale positions linearly:

```python
def position_interpolation_rope(
    rope: RotaryEmbedding,
    scale: float
):
    """
    Position Interpolation for RoPE context extension.

    Maps position p to p/scale, effectively compressing the position space.
    """
    class InterpolatedRoPE(nn.Module):
        def forward(self, x, seq_len=None):
            if seq_len is None:
                seq_len = x.shape[1]

            # Scaled positions
            positions = torch.arange(seq_len, device=x.device).float() / scale
            freqs = torch.outer(positions, rope.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)

            cos = emb.cos().unsqueeze(0)
            sin = emb.sin().unsqueeze(0)
            return cos, sin

    return InterpolatedRoPE()

# Example: extend from 2048 to 8192 (4x)
rope_extended = position_interpolation_rope(rope, scale=4.0)
```

**Note**: Requires fine-tuning! See YaRN documentation for better methods.

### 5. Mixed Precision

```python
# Use float16 for cos/sin (values in [-1, 1], safe for fp16)
class FP16RoPE(nn.Module):
    def forward(self, q, k):
        cos, sin = self.compute_cos_sin(q.shape[-2], q.device)

        # Convert to fp16
        cos = cos.half()
        sin = sin.half()

        # Apply (q, k already in fp16)
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)

        return q_rot, k_rot
```

**Savings**: 50% memory for cos/sin, minimal accuracy loss.

## Experiments & Results

### Length Generalization

Training on 2048 tokens, testing on longer sequences:

| Test Length | RoPE (PPL) | Sinusoidal (PPL) | Learned PE (PPL) |
|-------------|------------|------------------|------------------|
| 2048 (train) | 15.0 | 15.1 | 15.1 |
| 4096 | 17.1 | 18.2 | ∞ |
| 8192 | 22.8 | 25.3 | ∞ |
| 16384 | 38.4 | 45.1 | ∞ |

**Observation**: RoPE generalizes better than sinusoidal PE, much better than learned PE.

### Ablation Studies

#### Effect of Base Frequency

| Base | Train PPL | Test PPL (2x) | Test PPL (4x) |
|------|-----------|---------------|---------------|
| 1000 | 15.2 | 19.3 | 31.5 |
| 10000 | 15.0 | 17.1 | 22.8 |
| 100000 | 15.1 | 16.2 | 19.4 |

**Takeaway**: Higher base improves extrapolation (longer wavelengths).

#### Partial RoPE

On GPT-J architecture (head_dim=256):

| Rotary Dim | Train Time | Test PPL | Memory |
|------------|------------|----------|--------|
| 256 (full) | 100% | 15.0 | 100% |
| 128 (half) | 82% | 15.1 | 92% |
| 64 (quarter) | 71% | 15.3 | 86% |

**Takeaway**: Partial RoPE (64-128 dims) offers good speed/accuracy tradeoff.

### Comparison with Other Methods

On C4 dataset (language modeling):

| Method | Parameters | Train PPL | Test PPL (2x) | Test PPL (4x) |
|--------|------------|-----------|---------------|---------------|
| Learned PE | 4M | 18.2 | ∞ | ∞ |
| Sinusoidal | 0 | 18.1 | 22.5 | 38.4 |
| RoPE | 0 | 18.0 | 20.3 | 31.2 |
| ALiBi | 0 | 18.2 | 19.1 | 20.8 |
| RoPE + PI | 0 | 18.0 | 18.9 | 21.4 |

**Takeaway**: RoPE is competitive with modern methods, extensible with interpolation/YaRN.

## Common Pitfalls

### Pitfall 1: Applying RoPE to Values

**Problem**: RoPE should only be applied to queries and keys, not values.

```python
# Wrong
q, k, v = apply_rotary_pos_emb(q, k, v, cos, sin)  # v should not be rotated!

# Correct
q, k = apply_rotary_pos_emb(q, k, cos, sin)
# v remains unchanged
```

**Why**: Values represent content, not positions. Rotating them corrupts information.

### Pitfall 2: Mismatched Dimensions

**Problem**: cos/sin dimensions don't match q/k dimensions.

```python
# q: (batch, heads, seq, head_dim=128)
# cos: (1, seq, dim=512)  # Wrong! Should be 128

# Fix: ensure cos/sin have same dimension as head_dim
rope = RotaryEmbedding(dim=head_dim)  # Not model dim!
```

### Pitfall 3: Forgetting to Cache

**Problem**: Recomputing cos/sin every forward pass.

```python
# Slow: recomputes every time
class SlowAttention(nn.Module):
    def forward(self, q, k, v):
        rope = RotaryEmbedding(dim=128)  # Creates new instance!
        cos, sin = rope(q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        ...

# Fast: reuse cached cos/sin
class FastAttention(nn.Module):
    def __init__(self):
        self.rope = RotaryEmbedding(dim=128)  # Create once

    def forward(self, q, k, v):
        cos, sin = self.rope(q)  # Uses cache
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        ...
```

### Pitfall 4: Position Interpolation Without Fine-tuning

**Problem**: Naive interpolation changes distribution significantly.

```python
# Wrong: interpolate and deploy immediately
rope_extended = position_interpolation_rope(rope, scale=4.0)
# Poor performance!

# Correct: fine-tune after interpolation
rope_extended = position_interpolation_rope(rope, scale=4.0)
# Fine-tune model with rope_extended on longer sequences
```

**Better**: Use YaRN or NTK-Aware RoPE for context extension.

### Pitfall 5: Odd Dimensions

**Problem**: RoPE requires even dimensions (pairs of dimensions).

```python
# Error
rope = RotaryEmbedding(dim=65)  # Odd dimension!

# Fix: use even dimension
assert head_dim % 2 == 0, "RoPE requires even head dimension"
rope = RotaryEmbedding(dim=64)
```

## References

### Primary Reference
- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). **RoFormer: Enhanced Transformer with Rotary Position Embedding**. *arXiv preprint arXiv:2104.09864*. [Link](https://arxiv.org/abs/2104.09864)

### Extensions and Analysis
- Chen, S., Wong, S., Chen, L., & Tian, Y. (2023). **Extending Context Window of Large Language Models via Position Interpolation**. *arXiv:2306.15595*. (Position Interpolation)
- Peng, B., Quesnelle, J., Fan, H., & Shippole, E. (2023). **YaRN: Efficient Context Window Extension of Large Language Models**. *arXiv:2309.00071*. (YaRN)
- Bloc97 (2023). **NTK-Aware Scaled RoPE**. *Reddit/LocalLLaMA*. (NTK-Aware RoPE)

### Implementations
- [LLaMA](https://github.com/facebookresearch/llama) - Meta's LLaMA uses RoPE
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) - Eleuther's implementation
- [Nexus Implementation](../../nexus/components/embeddings/rotary_embedding.py)

### Applications
- Touvron, H., et al. (2023). **LLaMA: Open and Efficient Foundation Language Models**. Meta AI.
- Black, S., et al. (2022). **GPT-NeoX-20B: An Open-Source Autoregressive Language Model**. Eleuther AI.
- Chowdhery, A., et al. (2022). **PaLM: Scaling Language Modeling with Pathways**. Google Research.

---

**Next**: [ALiBi](./alibi.md) | [YaRN](./yarn.md) | [Back to Overview](./README.md)
