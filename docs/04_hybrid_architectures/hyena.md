# Hyena: Sub-Quadratic Attention via Long Convolutions

## Overview

Hyena is a sub-quadratic attention replacement that uses a hierarchy of long convolutions and element-wise gating operations. Instead of computing O(N²) attention, Hyena uses FFT-based convolutions (O(N log N)) with data-controlled implicit filters, achieving strong performance while being significantly more efficient than attention.

**Key Innovation**: Replace attention with implicitly-parametrized long convolutions that are data-controlled through multiplicative gating, achieving similar expressivity to attention without quadratic complexity.

**Paper**: [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866) (ICML 2023)

## Motivation

### The Core Insight

Attention can be viewed as:
1. Computing pairwise token interactions (Q·K^T)
2. Weighting values based on these interactions
3. Aggregating weighted values

Hyena proposes:
1. **Long convolutions** for capturing position-dependent patterns
2. **Data-controlled gates** for adaptive weighting
3. **Hierarchical composition** for increased expressivity

```
Attention:        Softmax(Q·K^T) · V
                  ↓
Hyena:           h_2 * (v_2 ⊙ (h_1 * (v_1 ⊙ x)))

where * is convolution, ⊙ is element-wise multiplication
```

## Theoretical Background

### Hyena Operator

The Hyena operator of order N decomposes attention into alternating convolution and gating operations:

```
Given input x ∈ ℝ^(L×d):

1. Project to N+1 branches:
   x, v_1, v_2, ..., v_N = split(W_in · x)

2. Apply hierarchy:
   z_0 = x
   for i = 1 to N:
       z_i = h_i * (v_i ⊙ z_{i-1})

3. Output projection:
   y = W_out · z_N

where:
- h_i: Implicitly-parametrized convolution filter
- v_i: Data-dependent gate (projection of input)
- *: Convolution operator
- ⊙: Element-wise multiplication
```

### Implicit Filter Parametrization

Key innovation: Instead of storing full L-length filters, parametrize them with a small MLP:

```
Traditional filter: h ∈ ℝ^L (storage: O(L))
                    ↓
Implicit filter: h(t) = MLP(pos(t)) · window(t)
                Storage: O(filter_order) where filter_order << L

Benefits:
1. Parameter efficiency: O(d_filter) vs O(L)
2. Length generalization: Can generate filters for any L
3. Smooth positional patterns: MLP learns smooth functions
```

### Why It Works

**Attention vs Hyena analogy:**

| Attention Component | Hyena Equivalent |
|---------------------|------------------|
| Q·K^T (token interactions) | Long convolution h_i |
| Softmax (normalization) | Implicit in filter + window |
| V (values) | Gating v_i controls information flow |
| Multi-layer composition | Hierarchical order N |

**Order N interpretation:**
- Order 1: Single gate + convolution (like linear attention)
- Order 2: Q/K-like separation (matches standard attention)
- Order 3+: Increased expressivity beyond attention

## Mathematical Formulation

### Forward Pass

```python
def hyena_forward(x):
    # Shape: x ∈ ℝ^(B×L×d)

    # 1. Input projection to order+1 branches
    branches = split(linear(x))  # List of order+1 tensors: (B, L, d)
    # branches[0] = "value" x
    # branches[1:] = gates v_1, ..., v_N

    # 2. Short convolutions for local context
    for i, branch in enumerate(branches):
        branches[i] = short_conv(branch)  # Kernel size ~3

    # 3. Hyena hierarchy
    y = branches[0]  # Start with "value"
    for i in range(order):
        # a) Element-wise gate
        y = y ⊙ branches[i + 1]

        # b) Long convolution with implicit filter
        h = generate_filter(seq_len, i)  # (d, L)
        y = fft_conv(y, h)  # (B, L, d)

    # 4. Output projection
    return linear_out(y)
```

### Implicit Filter Generation

```python
def generate_filter(seq_len, filter_idx):
    """Generate implicit convolution filter.

    Args:
        seq_len: Sequence length L
        filter_idx: Which filter (0 to order-1)

    Returns:
        h ∈ ℝ^(d×L): Convolution filter
    """
    # 1. Generate positional encoding
    t = torch.arange(seq_len)  # [0, 1, 2, ..., L-1]
    pos = positional_encoding(t)  # (L, d_pos)

    # 2. Pass through filter MLP
    h = filter_mlp[filter_idx](pos)  # (L, d)

    # 3. Apply exponential decay window
    decay = learnable_decay[filter_idx]  # (d,)
    window = exp(-decay * t)  # (d, L) broadcast
    h = h.T * window  # (d, L)

    return h
```

### FFT-Based Convolution

Efficient convolution via FFT for O(N log N) complexity:

```python
def fft_conv(x, h):
    """Fast convolution via FFT.

    Args:
        x: Input (B, L, d)
        h: Filter (d, L)

    Returns:
        y: Convolved output (B, L, d)
    """
    # Pad to avoid circular convolution
    fft_size = 2 * L

    # Transform to frequency domain
    x_fft = torch.fft.rfft(x.transpose(1,2), n=fft_size, dim=-1)  # (B, d, fft_size//2+1)
    h_fft = torch.fft.rfft(h, n=fft_size, dim=-1)  # (d, fft_size//2+1)

    # Multiply in frequency domain
    y_fft = x_fft * h_fft.unsqueeze(0)  # (B, d, fft_size//2+1)

    # Inverse FFT
    y = torch.fft.irfft(y_fft, n=fft_size, dim=-1)  # (B, d, fft_size)

    # Truncate to causal (first L samples)
    y = y[:, :, :L].transpose(1, 2)  # (B, L, d)

    return y
```

## High-Level Intuition

### Visualization

```
┌────────────────────────────────────────────────────────────┐
│                     Hyena Block                            │
│                                                            │
│  Input x                                                   │
│     │                                                      │
│     ├─► Short Conv ─► [x, v1, v2, ..., vN]               │
│                          │                                 │
│                          ├─ x (value branch)               │
│                          ├─ v1 (gate 1)                    │
│                          ├─ v2 (gate 2)                    │
│                          └─ vN (gate N)                    │
│                                                            │
│     Hyena Hierarchy:                                       │
│     ┌────────────────────────────────────────┐            │
│     │ z0 = x                                 │            │
│     │                                        │            │
│     │ z1 = h1 * (v1 ⊙ z0)  ←─── Filter h1   │            │
│     │           ↑                Generated   │            │
│     │         Gate v1            by MLP      │            │
│     │                                        │            │
│     │ z2 = h2 * (v2 ⊙ z1)  ←─── Filter h2   │            │
│     │           ↑                Generated   │            │
│     │         Gate v2            by MLP      │            │
│     │ ...                                    │            │
│     └────────────────────────────────────────┘            │
│                          │                                 │
│                          ├─► Output Proj ─► y             │
│                                                            │
└────────────────────────────────────────────────────────────┘

Key:
  * = Convolution (via FFT)
  ⊙ = Element-wise multiplication (gating)
  h_i = Implicitly-parametrized filter
  v_i = Data-dependent gate
```

### Conceptual Understanding

**Think of Hyena as a hierarchical filter bank:**

1. **Short convolutions** capture immediate local context (like n-grams)
2. **Gates** (v_i) decide which information flows based on input content
3. **Long convolutions** (h_i) capture position-dependent patterns across the sequence
4. **Hierarchy** builds increasingly complex representations through composition

**Attention vs Hyena:**
```
Attention: All-to-all token interactions via Q·K·V
          ↓ Expensive: O(N²)

Hyena:    Position-aware patterns via convolution + data-dependent gating
          ↓ Efficient: O(N log N)
```

## Implementation Details

### Layer Configuration

```python
# Standard Hyena configuration
hyena_operator = HyenaOperator(
    d_model=512,
    max_seq_len=8192,    # Can generate filters up to this length
    order=2,             # Number of gating-convolution pairs
    filter_order=64,     # Hidden dim of filter MLP
    short_filter_order=3 # Short conv kernel size
)
```

**Typical hyperparameters:**

| Parameter | Small (50M) | Medium (350M) | Large (1B+) |
|-----------|-------------|---------------|-------------|
| `d_model` | 512 | 1024 | 2048+ |
| `order` | 2 | 2-3 | 2-3 |
| `filter_order` | 64 | 64-128 | 128-256 |
| `max_seq_len` | 4096 | 8192 | 16384+ |

### Positional Encoding

Hyena uses sinusoidal positional encodings as input to filter MLPs:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=8192):
        super().__init__()
        # Precompute sinusoidal embeddings
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:seq_len]  # (L, d_model)
```

**Why sinusoidal?** Smooth, periodic patterns help the filter MLP learn position-dependent convolution weights.

### Filter MLP Architecture

```python
class ImplicitFilter(nn.Module):
    """Generate convolution filters via MLP."""

    def __init__(self, d_model, filter_order=64, num_inner_mlps=1):
        super().__init__()

        # Positional encoding
        self.pos_encoding = PositionalEncoding(filter_order)

        # Filter MLP: pos_encoding → filter values
        layers = [nn.Linear(filter_order, filter_order), nn.SiLU()]
        for _ in range(num_inner_mlps):
            layers.extend([
                nn.Linear(filter_order, filter_order),
                nn.SiLU()
            ])
        layers.append(nn.Linear(filter_order, d_model))
        self.filter_ffn = nn.Sequential(*layers)

        # Exponential decay window
        self.decay = nn.Parameter(torch.linspace(0.1, 2.0, d_model))

    def forward(self, seq_len):
        # Get positional encoding
        pos = self.pos_encoding(seq_len)  # (L, filter_order)

        # Generate filter via MLP
        h = self.filter_ffn(pos)  # (L, d_model)

        # Apply exponential decay
        t = torch.arange(seq_len, device=h.device)
        window = torch.exp(-self.decay.unsqueeze(1) * t)  # (d_model, L)

        h = h.t() * window  # (d_model, L)

        return h
```

**Key design choices:**
1. **Filter hidden dim** (`filter_order`): Typically 64-128, controls filter expressivity
2. **Number of MLP layers**: Usually 1-2, deeper MLPs can learn more complex patterns
3. **Exponential decay**: Learned per-dimension decay for causal windowing

## Code Walkthrough

### Full Hyena Operator

```python
class HyenaOperator(NexusModule):
    def __init__(self, d_model, max_seq_len=8192, order=2, filter_order=64):
        super().__init__()
        self.order = order

        # Input projection: produces (order + 1) branches
        self.in_proj = nn.Linear(d_model, d_model * (order + 1), bias=False)

        # Short depthwise convolutions (one per branch)
        self.short_convs = nn.ModuleList([
            nn.Conv1d(
                d_model, d_model,
                kernel_size=3,
                padding=2,
                groups=d_model,  # Depthwise
                bias=True
            )
            for _ in range(order + 1)
        ])

        # Implicit long filters (one per order)
        self.long_filters = nn.ModuleList([
            ImplicitFilter(d_model, filter_order, max_seq_len)
            for _ in range(order)
        ])

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # Project to (order + 1) branches
        projections = self.in_proj(x)  # (B, L, d * (order+1))
        projections = projections.view(batch, seq_len, self.order + 1, d_model)

        # Apply short convolutions
        branches = []
        for i in range(self.order + 1):
            p_i = projections[:, :, i, :].transpose(1, 2)  # (B, d, L)
            p_i = self.short_convs[i](p_i)[:, :, :seq_len]  # Causal
            branches.append(p_i.transpose(1, 2))  # Back to (B, L, d)

        # Hyena hierarchy
        y = branches[0]  # Start with "value"
        for i in range(self.order):
            # Element-wise gating
            y = y * branches[i + 1]

            # Long convolution
            h = self.long_filters[i](seq_len)  # (d, L)
            y = self._fft_conv(y, h)

        output = self.out_proj(y)
        return output, None  # None for state (Hyena is stateless)
```

### FFT Convolution Implementation

```python
def _fft_conv(self, x, h):
    """Causal convolution via FFT.

    Args:
        x: (batch, seq_len, d_model)
        h: (d_model, seq_len)

    Returns:
        y: (batch, seq_len, d_model)
    """
    seq_len = x.shape[1]
    fft_size = 2 * seq_len  # Pad for linear convolution

    # Transpose for FFT: (batch, d_model, seq_len)
    x_t = x.transpose(1, 2)

    # FFT of input
    x_fft = torch.fft.rfft(x_t, n=fft_size, dim=-1)

    # FFT of filter
    h_fft = torch.fft.rfft(h, n=fft_size, dim=-1)

    # Multiply in frequency domain (element-wise per channel)
    y_fft = x_fft * h_fft.unsqueeze(0)

    # IFFT back to time domain
    y = torch.fft.irfft(y_fft, n=fft_size, dim=-1)

    # Causal: keep only first seq_len samples
    y = y[:, :, :seq_len]

    return y.transpose(1, 2)  # Back to (batch, seq_len, d_model)
```

## Optimization Tricks

### 1. FFT Efficiency

**Use real FFT (`rfft`) for real-valued signals:**

```python
# ✅ GOOD: Real FFT (2x faster, half memory)
x_fft = torch.fft.rfft(x, n=fft_size)  # Returns (fft_size//2 + 1) complex

# ❌ BAD: Complex FFT for real signal
x_fft = torch.fft.fft(x, n=fft_size)  # Returns fft_size complex (wasteful)
```

### 2. Filter Caching

For generation, cache filters to avoid recomputation:

```python
class HyenaOperator(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self._filter_cache = {}

    def forward(self, x):
        seq_len = x.shape[1]

        # Check cache
        if seq_len not in self._filter_cache:
            # Generate and cache filters
            filters = [f(seq_len) for f in self.long_filters]
            self._filter_cache[seq_len] = filters
        else:
            filters = self._filter_cache[seq_len]

        # Use cached filters
        # ...
```

### 3. Short Convolution Optimization

Use depthwise convolutions for efficiency:

```python
# Depthwise conv: O(d * k) ops per position
self.conv = nn.Conv1d(d_model, d_model, kernel_size=k, groups=d_model)

# vs standard conv: O(d² * k) ops per position
self.conv = nn.Conv1d(d_model, d_model, kernel_size=k)

# Speedup: d/k (typically 128-512x faster!)
```

### 4. Mixed Precision Training

Hyena is numerically stable with mixed precision:

```python
# Safe to use automatic mixed precision
with torch.cuda.amp.autocast():
    output = hyena_model(input)

# FFT operations handle fp16 well
```

## Experiments & Results

### Perplexity on WikiText-103

| Model | Parameters | Perplexity | Train Time |
|-------|-----------|------------|------------|
| Transformer | 125M | 18.2 | 100% |
| Hyena | 125M | 18.7 | **70%** |
| Hyena | 355M | 16.4 | **65%** |

### Throughput Comparison

| Sequence Length | Transformer | Hyena | Speedup |
|-----------------|-------------|-------|---------|
| 1K | 100% | 110% | 1.1x |
| 4K | 100% | 180% | 1.8x |
| 16K | 100% | 340% | 3.4x |
| 64K | OOM | 100% | ∞ |

**Interpretation**: Hyena's advantage grows with sequence length due to O(N log N) vs O(N²) scaling.

### Long Range Arena Benchmark

| Task | Transformer | Hyena | Notes |
|------|-------------|-------|-------|
| ListOps | 37.2 | 41.5 | Hierarchical reasoning |
| Text | 64.3 | 63.8 | Document classification |
| Retrieval | 80.5 | 76.2 | Exact matching harder |
| Image | 42.4 | 45.1 | Path-X task |
| Pathfinder | 71.4 | 72.8 | Long-range vision |
| **Average** | **59.2** | **59.9** | Competitive |

**Key findings:**
- Hyena competitive on most LRA tasks
- Slightly weaker on retrieval (expected for convolution-based)
- Stronger on hierarchical tasks

## Common Pitfalls

### 1. Not Padding for Linear Convolution

```python
# ❌ BAD: Circular convolution (wrong!)
fft_size = seq_len
x_fft = torch.fft.rfft(x, n=fft_size)
h_fft = torch.fft.rfft(h, n=fft_size)
y = torch.fft.irfft(x_fft * h_fft, n=fft_size)
# Results in circular artifacts at boundaries
```

```python
# ✅ GOOD: Pad to 2*seq_len for linear convolution
fft_size = 2 * seq_len
x_fft = torch.fft.rfft(x, n=fft_size)
h_fft = torch.fft.rfft(h, n=fft_size)
y = torch.fft.irfft(x_fft * h_fft, n=fft_size)[:seq_len]
```

### 2. Filter Order Too Small

```python
# ❌ BAD: Filter MLP too small
filter_order = 16  # Not enough capacity
# Filters can't capture complex position patterns
```

```python
# ✅ GOOD: Adequate filter capacity
filter_order = 64  # Standard
filter_order = 128  # For larger models
```

### 3. Wrong Decay Initialization

```python
# ❌ BAD: All-zero or all-one decay
self.decay = nn.Parameter(torch.zeros(d_model))  # Too slow decay
self.decay = nn.Parameter(torch.ones(d_model))   # Too fast decay
```

```python
# ✅ GOOD: Range of decay rates
self.decay = nn.Parameter(torch.linspace(0.1, 2.0, d_model))
# Different channels have different effective context lengths
```

### 4. Forgetting Causal Truncation

```python
# ❌ BAD: Use full convolution output
y = torch.fft.irfft(x_fft * h_fft, n=fft_size)  # Length 2*seq_len
# Includes future information!
```

```python
# ✅ GOOD: Truncate to causal
y = torch.fft.irfft(x_fft * h_fft, n=fft_size)[:, :, :seq_len]
# Only past information
```

### 5. Order Too High

```python
# ❌ BAD: Unnecessary high order
order = 5  # Overkill for most tasks
# More parameters, training instability, marginal gains
```

```python
# ✅ GOOD: Order 2-3 is usually sufficient
order = 2  # Standard, matches Q-K-V structure
order = 3  # For extra expressivity if needed
```

## References

### Primary Paper

Poli, M., Massaroli, S., Nguyen, E., Fu, D. Y., Dao, T., Baccus, S., Bengio, Y., Ermon, S., & Ré, C. (2023). **Hyena Hierarchy: Towards Larger Convolutional Language Models**. ICML 2023.

### Related Work

- **StripedHyena**: Poli et al. (2023), Hyena + attention hybrid
- **H3**: Fu et al. (2023), predecessor with structured SSMs
- **S4**: Gu et al. (2021), structured state spaces
- **AFT**: Zhai et al. (2021), attention-free transformer

### Code

- Reference: `nexus/models/hybrid/hyena.py`
- Official implementation: [HazyResearch/safari](https://github.com/HazyResearch/safari)
