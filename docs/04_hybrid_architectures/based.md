# Based: Linear Attention with Sliding Window for Extreme Throughput

## Overview

Based (pronounced "base-dee") is a hybrid architecture that combines Taylor-expanded linear attention with sliding window attention to achieve extreme inference throughput (24x faster than FlashAttention-2) while maintaining competitive quality. The key innovation is approximating softmax attention with a second-order Taylor series, enabling linear-time computation through associative recurrence.

**Key Innovation**: Replace expensive softmax attention with a Taylor approximation that decomposes into linear associative operations, achieving O(N) training and O(1) inference while using sliding windows for quality refinement.

**Paper**: [Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff](https://arxiv.org/abs/2402.18668) (ICML 2024)

## Motivation

### The Throughput Crisis

Modern LLMs face a fundamental bottleneck:
- **FlashAttention-2** is optimized but still O(N²)
- **Long contexts** (32K-128K tokens) are increasingly common
- **Real-time inference** demands <10ms per token

Based asks: Can we get transformer-like quality without quadratic attention?

### Based's Solution

**Two-pronged approach:**

1. **Taylor Linear Attention**: Replace softmax with polynomial approximation
   - Enables O(N) training via parallel cumulative sum
   - Enables O(1) inference via recurrent state updates
   - Achieves 24x throughput over FlashAttention-2

2. **Sliding Window Refinement**: Strategic softmax attention within local windows
   - Captures precise short-range dependencies
   - Keeps KV cache minimal (window size only)
   - Complements global linear attention

**Result**: 24x inference speedup with <1% perplexity degradation

## Theoretical Background

### Taylor Series Approximation of Softmax

Standard attention uses softmax for normalization:
```
Attention(Q, K, V) = softmax(QK^T) V

where softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

Based approximates softmax with a Taylor series:
```
exp(x) ≈ 1 + x + x²/2 + x³/6 + ...

For Based (2nd order):
exp(x) ≈ 1 + x + x²/2

Feature map φ(x) = [1, x, x²/2]
```

### Linear Attention via Feature Maps

With feature maps, attention becomes:
```
Standard: Attention = softmax(QK^T)V

Linear:   Attention = φ(Q) · (φ(K)^T V)
                    = φ(Q) · S

where S = Σ_t φ(K_t) ⊗ V_t  (associative operation)
```

**Key property**: S can be computed incrementally!

### Recurrent Formulation

The associative sum enables recurrent inference:

```
Training (parallel):
    for all t in parallel:
        S_t = Σ_{i=1}^t φ(K_i) ⊗ V_i
        o_t = φ(Q_t) · S_t

Inference (recurrent, O(1) per step):
    S_0 = 0
    for t = 1, 2, ...:
        S_t = S_t-1 + φ(K_t) ⊗ V_t    # O(d²) update
        o_t = φ(Q_t) · S_t             # O(d²) query
```

**Complexity:**
- Training: O(N · d² · f) where f is feature dim
- Inference: O(d² · f) per step (constant!)

## Mathematical Formulation

### Taylor Feature Map

```python
def feature_map(x):
    """Apply 2nd-order Taylor approximation.

    Args:
        x: Input (..., d)

    Returns:
        features: (..., d * 3)  # [1, x, x²/2]
    """
    ones = torch.ones_like(x)
    x_sq = x ** 2 / 2.0

    # Concatenate: [1, x, x²/2]
    features = torch.cat([ones, x, x_sq], dim=-1)

    return features
```

### Linear Attention Forward Pass

```python
def linear_attention(q, k, v, state=None):
    """Linear attention with recurrent state.

    Args:
        q: queries (B, L, d)
        k: keys (B, L, d)
        v: values (B, L, d)
        state: previous state (B, d*f, d) where f=3

    Returns:
        output: (B, L, d)
        state: updated state
    """
    # Apply feature maps
    q_feat = feature_map(q)  # (B, L, d*3)
    k_feat = feature_map(k)  # (B, L, d*3)

    # Initialize state
    if state is None:
        state = torch.zeros(B, d*3, d)

    # Recurrent accumulation
    outputs = []
    for t in range(L):
        # Update state: S = S + k ⊗ v
        state = state + torch.einsum('bd,be->bde', k_feat[:, t], v[:, t])

        # Query state: o = q · S
        o_t = torch.einsum('bd,bde->be', q_feat[:, t], state)
        outputs.append(o_t)

    output = torch.stack(outputs, dim=1)
    return output, state
```

### Sliding Window Attention

Complement linear attention with local softmax:

```python
def sliding_window_attention(x, window_size=256):
    """Standard attention within sliding window.

    Args:
        x: Input (B, L, d)
        window_size: Window size W

    Returns:
        output: (B, L, d)
    """
    # Standard QKV projection
    q, k, v = split(qkv_proj(x))

    # Create sliding window mask
    mask = create_sliding_window_mask(L, window_size)
    # mask[i, j] = 0 if |i - j| <= window_size//2, else -inf

    # Standard attention with mask
    attn = softmax(q @ k.T * scale + mask)
    output = attn @ v

    return output
```

## High-Level Intuition

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                      Based Block                         │
│                                                          │
│  Input x                                                 │
│     │                                                    │
│     ├──► Norm ──► Taylor Linear Attn ──┬──► (+) ───┐   │
│     │              (Global context)    │            │   │
│     │              [Updates state]     │            │   │
│     │                                   │            │   │
│     └───────────────────────────────────             │   │
│          │                                           │   │
│          ├──► Norm ──► Sliding Window ──┬──► (+) ───┤   │
│          │              (Local refine)  │            │   │
│          │                              │            │   │
│          └──────────────────────────────             │   │
│               │                                      │   │
│               └──► Norm ──► FFN ───────┬──► (+) ────┘   │
│                                        │                 │
│                                   Output x'              │
└──────────────────────────────────────────────────────────┘

Legend:
  Taylor Linear: O(N) global attention approximation
  Sliding Window: Precise local softmax attention
  FFN: Standard feedforward
```

### Conceptual Roles

**Linear Attention (Global, Fast):**
- Captures long-range dependencies efficiently
- Provides rough global context
- Like reading a book and keeping brief notes

**Sliding Window (Local, Precise):**
- Corrects linear attention's approximation errors
- Captures precise local patterns
- Like carefully re-reading the current page

**Combined Effect:**
- Linear attention handles bulk (80-90% of context)
- Sliding window refines critical local regions (10-20%)
- Total: Near-transformer quality at fraction of cost

## Implementation Details

### Block Configuration

```python
class BasedBlock(nn.Module):
    def __init__(
        self,
        d_model=512,
        num_heads=8,
        window_size=256,
        use_sliding_window=True  # Can disable for pure linear
    ):
        super().__init__()

        # Linear attention
        self.norm1 = nn.LayerNorm(d_model)
        self.linear_attn = TaylorLinearAttention(d_model, num_heads)

        # Sliding window (optional)
        if use_sliding_window:
            self.norm2 = nn.LayerNorm(d_model)
            self.sliding_attn = SlidingWindowAttention(
                d_model, num_heads, window_size
            )

        # FFN
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)
```

### Hyperparameter Guidelines

| Parameter | Small | Medium | Large | Notes |
|-----------|-------|--------|-------|-------|
| `d_model` | 512 | 1024 | 2048 | Model dimension |
| `num_heads` | 8 | 16 | 32 | Multi-head split |
| `window_size` | 128 | 256 | 512 | Local attention |
| `feature_dim` | 3 | 3 | 3 | Taylor order (fixed) |
| `use_sliding` | True | True | True | Almost always on |

**Rule of thumb**: Window size should cover 1-2 sentences (50-200 tokens).

## Code Walkthrough

### Taylor Linear Attention

```python
class TaylorLinearAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, feature_dim=3):
        super().__init__()
        self.head_dim = d_model // num_heads
        self.feature_dim = feature_dim  # 3 for 2nd-order Taylor

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _feature_map(self, x):
        """Taylor expansion: [1, x, x²/2]"""
        x = x * self.scale  # Scale for stability

        ones = torch.ones_like(x)
        x_sq = x ** 2 / 2.0

        # Shape: (..., head_dim * 3)
        return torch.cat([ones, x, x_sq], dim=-1)

    def forward(self, x, state=None):
        B, L, d = x.shape

        # Project to QKV
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        # Apply feature maps
        q_feat = self._feature_map(q)  # (B, L, H, head_dim*3)
        k_feat = self._feature_map(k)

        # Initialize state: (B, H, head_dim*3, head_dim)
        if state is None:
            state = torch.zeros(
                B, self.num_heads, self.head_dim * 3, self.head_dim
            )

        # Recurrent computation
        outputs = []
        for t in range(L):
            # Update: state = state + k ⊗ v
            state = state + torch.einsum(
                'bhf,bhd->bhfd',
                k_feat[:, t],  # (B, H, head_dim*3)
                v[:, t]        # (B, H, head_dim)
            )

            # Query: o = q · state
            o_t = torch.einsum(
                'bhf,bhfd->bhd',
                q_feat[:, t],
                state
            )
            outputs.append(o_t)

        # Stack and reshape
        output = torch.stack(outputs, dim=1)
        output = output.reshape(B, L, d)

        # Output projection
        output = self.out_proj(output)

        return output, state
```

### Sliding Window Implementation

```python
class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, window_size=256):
        super().__init__()
        self.window_size = window_size
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        B, L, d = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Create sliding window mask
        attn_mask = torch.full(
            (L, L), float('-inf'), device=x.device, dtype=x.dtype
        )
        for i in range(L):
            start = max(0, i - self.window_size // 2)
            end = min(L, i + self.window_size // 2 + 1)
            attn_mask[i, start:end] = 0

        # Standard attention with mask
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores + attn_mask
        attn_weights = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).reshape(B, L, d)

        return self.out_proj(output)
```

## Optimization Tricks

### 1. State Accumulation

**Efficient state updates:**

```python
# Use einsum for clarity and efficiency
state += torch.einsum('bhf,bhd->bhfd', k_feat, v)

# Equivalent to:
# state += k_feat.unsqueeze(-1) @ v.unsqueeze(-2)
# but faster and more memory-efficient
```

### 2. Feature Dimension Choice

**Why 2nd-order (feature_dim=3)?**

```python
# 1st order: [1, x]
feature_dim = 2  # Too weak, poor quality

# 2nd order: [1, x, x²/2]
feature_dim = 3  # Sweet spot: good quality, manageable size

# 3rd order: [1, x, x²/2, x³/6]
feature_dim = 4  # Marginal gains, 33% more memory

# Recommendation: Stick with 2nd order
```

### 3. Window Size Tuning

**Optimal window sizes by task:**

```python
# Short documents (news, QA)
window_size = 128  # Sufficient for paragraph context

# Medium documents (articles)
window_size = 256  # Standard, covers 1-2 paragraphs

# Long documents (books, code)
window_size = 512  # More context for complex dependencies

# Rule: Larger windows = better quality but slower
```

### 4. Mixed Precision

Based works well with mixed precision:

```python
# Safe for fp16/bf16
with torch.cuda.amp.autocast():
    output, state = based_model(input)

# Feature map computation is numerically stable
# No special scaling needed
```

### 5. Gradient Checkpointing

Reduce memory during training:

```python
model = BasedModel(...)

# Enable checkpointing
for block in model.layers:
    block = torch.utils.checkpoint.checkpoint_wrapper(block)

# Trade computation for memory (2-3x slower, 10x less memory)
```

## Experiments & Results

### Throughput Benchmarks

**Tokens/second (higher is better):**

| Model | 1K | 4K | 16K | 64K |
|-------|-----|-----|------|------|
| FlashAttn-2 | 6500 | 4200 | 1800 | 450 |
| Based (linear only) | **156K** | **156K** | **156K** | **156K** |
| Based (+ window) | **68K** | **52K** | **38K** | **28K** |

**Speedup over FlashAttn-2:**
- Based (linear): **24x** (constant throughput)
- Based (hybrid): **10-62x** (grows with context)

### Perplexity on C4

| Model | 1B params | Relative |
|-------|-----------|----------|
| Transformer | 12.3 | 100% |
| Based (linear) | 13.1 | 93.5% |
| Based (hybrid) | 12.4 | **99.2%** |

**Interpretation**: Hybrid Based recovers 99% of transformer quality while being 10-24x faster.

### Recall Tasks

**Natural Questions (exact match accuracy):**

| Model | Short | Medium | Long |
|-------|-------|--------|------|
| Transformer | 42.5 | 38.2 | 31.5 |
| Based (linear) | 35.2 | 29.8 | 25.1 |
| Based (hybrid) | **41.8** | **37.5** | **30.9** |

**Key insight**: Sliding window critical for recall quality.

## Common Pitfalls

### 1. Forgetting Feature Scaling

```python
# ❌ BAD: No scaling before feature map
q_feat = feature_map(q)  # Explodes for large q
```

```python
# ✅ GOOD: Scale by sqrt(d)
q = q * (1.0 / math.sqrt(head_dim))
q_feat = feature_map(q)
```

### 2. Wrong State Initialization

```python
# ❌ BAD: Random initialization
state = torch.randn(B, H, d*f, d)  # Nonsense state!
```

```python
# ✅ GOOD: Zero initialization
state = torch.zeros(B, H, d*f, d)  # Start with no information
```

### 3. Window Size Too Small

```python
# ❌ BAD: Tiny window
window_size = 16  # Can't capture sentence-level patterns
```

```python
# ✅ GOOD: Reasonable window
window_size = 128  # Minimum for most tasks
window_size = 256  # Safe default
```

### 4. Disabling Sliding Window

```python
# ❌ BAD: Pure linear attention
model = BasedModel(use_sliding_window=False)
# Quality drops significantly
```

```python
# ✅ GOOD: Use hybrid approach
model = BasedModel(use_sliding_window=True)
# Best quality-efficiency tradeoff
```

### 5. State Shape Confusion

```python
# ❌ BAD: Wrong state dimensions
state = torch.zeros(B, d, d)  # Missing heads and features!
```

```python
# ✅ GOOD: Correct shape
state = torch.zeros(B, num_heads, head_dim * feature_dim, head_dim)
# (batch, heads, features, value_dim)
```

## References

### Primary Paper

Arora, S., Eyuboglu, S., Zhang, M., Timalsina, A., Alberti, S., Zinsley, D., Zou, J., Rudra, A., & Ré, C. (2024). **Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff**. ICML 2024.

### Related Work

- **Linear Attention**: Katharopoulos et al. (2020)
- **Performer**: Choromanski et al. (2020)
- **Cosformer**: Qin et al. (2022)
- **FlashAttention-2**: Dao (2023)

### Code

- Reference: `nexus/models/hybrid/based.py`
- Official: [HazyResearch/based](https://github.com/HazyResearch/based)
