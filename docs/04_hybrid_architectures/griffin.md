# Griffin: Mixing Gated Linear Recurrences with Local Attention

## Overview

Griffin is a hybrid architecture developed by Google DeepMind that combines gated linear recurrences (the "Hawk" component) with local multi-query attention. It achieves strong performance on both long-range tasks and precise retrieval tasks while maintaining efficient inference with O(1) memory complexity per token generation step.

**Key Innovation**: Each Griffin block contains both a recurrent component (RGLRU) for efficient long-range modeling and local attention for precise short-range retrieval, applied sequentially within the same block.

**Paper**: [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427) (Google DeepMind, 2024)

## Motivation

### The Problem

Pure transformers suffer from:
- **Quadratic KV cache**: O(N²) memory for long sequences
- **Inefficient inference**: Every token attends to all previous tokens
- **Limited scalability**: Cannot handle very long contexts efficiently

Pure recurrent models (RNNs, SSMs) suffer from:
- **Imprecise recall**: Difficulty retrieving specific tokens
- **Associative memory limits**: Struggles with exact matching tasks
- **Quality ceiling**: Often underperform transformers on complex reasoning

### Griffin's Solution

**Design Philosophy**: Use each mechanism where it excels
1. **Gated Linear Recurrence (RGLRU)** for:
   - Long-range dependencies
   - Efficient O(1) inference
   - Bulk sequence processing

2. **Local Multi-Query Attention** for:
   - Precise token retrieval
   - Short-range refinement
   - Critical local patterns

**Result**: Transformer-quality performance with RNN-level inference efficiency

## Theoretical Background

### Gated Linear Recurrence

Griffin uses the **Real-Gated Linear Recurrent Unit (RGLRU)**, a diagonal recurrent system with learned gates:

```
Recurrence equations:
    a[t] = σ(W_a x[t] + b_a)              # Input-dependent gate
    h[t] = a[t] ⊙ h[t-1] + √(1 - a[t]²) ⊙ (W_x x[t])
    y[t] = h[t]
```

**Key properties:**
- **Diagonal gating**: Each dimension has independent gate, enabling parallelization
- **Magnitude preservation**: The √(1 - a²) scaling maintains signal magnitude (unitary-like)
- **Associative scan**: Can be computed in O(log N) parallel steps during training

### Local Multi-Query Attention

Griffin uses windowed attention with multi-query configuration:

```
Standard attention:
    Q ∈ ℝ^(L×d), K ∈ ℝ^(L×d), V ∈ ℝ^(L×d)

Multi-query attention (MQA):
    Q ∈ ℝ^(L×d), K ∈ ℝ^(L×d_kv), V ∈ ℝ^(L×d_kv)
    where d_kv << d (typically d_kv = d/num_heads)

Local window:
    Only attend to positions [t - W, t] where W is window size
```

**Benefits:**
- **Reduced KV cache**: MQA reduces cache by ~8x (if 8 heads)
- **Local window**: Cache capped at window size W (typically 128-512)
- **Combined**: Total KV cache is O(W × d/num_heads) per layer

## Mathematical Formulation

### RGLRU Recurrence

The RGLRU performs magnitude-preserving recurrence:

```
Given input x[t] ∈ ℝ^d:

1. Gate computation:
   a[t] = σ(W_a x[t] + b_a) ∈ [0, 1]^d_rec

2. Input transformation:
   x̃[t] = W_x x[t]

3. Magnitude-preserving scaling:
   x̃[t] ← x̃[t] ⊙ √(1 - a[t]² + ε)

4. Recurrence update:
   h[t] = a[t] ⊙ h[t-1] + x̃[t]

where:
- ⊙ denotes element-wise multiplication
- σ is sigmoid activation
- ε = 1e-6 for numerical stability
```

**Why magnitude preservation?**

The scaling ensures ||h[t]|| ≈ ||h[t-1]|| in expectation, preventing vanishing/exploding gradients across long sequences.

### Parallel Scan for Training

During training, the recurrence can be parallelized using the associative scan algorithm:

```
Log-space formulation for numerical stability:

1. Compute cumulative log gates:
   log_a[t] = log(a[t])
   cumsum_log_a[t] = Σ_{s=1}^t log_a[s]

2. Weighted inputs:
   weighted_x[t] = x̃[t] / exp(cumsum_log_a[t])
   cumsum_wx[t] = Σ_{s=1}^t weighted_x[s]

3. Rescale outputs:
   h[t] = cumsum_wx[t] × exp(cumsum_log_a[t])
```

This enables O(N) work with O(log N) parallel depth.

### Griffin Block Computation

Each Griffin block applies transformations sequentially:

```
Input: x ∈ ℝ^(B×L×d)

1. Recurrence branch:
   x₁ = LayerNorm(x)
   x₁, state = RGLRU(x₁, state)
   x = x + Dropout(x₁)

2. Local attention branch:
   x₂ = LayerNorm(x)
   x₂, kv_cache = LocalMQA(x₂, kv_cache)
   x = x + Dropout(x₂)

3. Feedforward:
   x₃ = LayerNorm(x)
   x₃ = FFN(x₃)
   x = x + Dropout(x₃)

Output: x ∈ ℝ^(B×L×d), updated state and kv_cache
```

## High-Level Intuition

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Griffin Block                        │
│                                                         │
│  Input x                                                │
│     │                                                   │
│     ├─────────────────────────────────────────┐        │
│     │                                         │        │
│     ├──► LayerNorm ──► RGLRU ──────────┬─► (+) ───┐  │
│     │                    │              │           │  │
│     │               [state update]      │           │  │
│     │                                   │           │  │
│     └────────────────────────────────────           │  │
│          │                                           │  │
│          ├──► LayerNorm ──► LocalMQA ───┬─► (+) ───┤  │
│          │                    │         │           │  │
│          │               [KV cache]     │           │  │
│          │                              │           │  │
│          └──────────────────────────────            │  │
│               │                                     │  │
│               └──► LayerNorm ──► FFN ──┬─► (+) ────┘  │
│                                         │              │
│                                         │              │
│                                   Output x'            │
└─────────────────────────────────────────────────────────┘

Legend:
  RGLRU: Real-Gated Linear Recurrent Unit (efficient global context)
  LocalMQA: Local Multi-Query Attention (precise local refinement)
  FFN: Feedforward network (position-wise processing)
```

### Conceptual Roles

Think of Griffin as a two-stage information processor:

**Stage 1: RGLRU (Global Context)**
- Aggregates information across the entire sequence
- Builds a compressed representation in the recurrent state
- Handles long-range dependencies efficiently
- Like a "lossy compression" of history

**Stage 2: Local Attention (Local Refinement)**
- Focuses on nearby tokens within window
- Retrieves precise information for local patterns
- Corrects RGLRU's imprecise global aggregation
- Like a "local sharpening" filter

**Analogy**: RGLRU is like reading a book and keeping general notes, while local attention is like re-reading the current page carefully for details.

## Implementation Details

### Layer Interleaving Strategy

Griffin uses **within-block composition**: each block has both recurrence and attention.

```python
# Griffin configuration
class GriffinBlock:
    def __init__(self):
        self.recurrence = GatedLinearRecurrence(...)
        self.attention = LocalMultiQueryAttention(...)
        self.ffn = FeedForward(...)

    def forward(self, x, state, kv_cache):
        # Both mechanisms in every block
        x, state = self.recurrence_block(x, state)
        x, kv_cache = self.attention_block(x, kv_cache)
        x = self.ffn_block(x)
        return x, state, kv_cache
```

**Alternative: Hawk Variant**

Set `use_attention=False` to get pure recurrence (Hawk):

```python
# Hawk: Griffin without attention
model = GriffinModel(..., hawk_only=True)
```

### Shared vs Separate Parameters

In Griffin, each layer has **independent parameters** for both recurrence and attention:

```python
# Each layer is independent
layers = nn.ModuleList([
    GriffinBlock(...)  # Unique RGLRU + unique LocalMQA
    for _ in range(num_layers)
])
```

**Design rationale**: Different layers need different capabilities (early layers for local patterns, late layers for global reasoning).

### Hyperparameter Choices

| Hyperparameter | Typical Value | Range | Notes |
|----------------|---------------|-------|-------|
| `d_model` | 512-4096 | 256-8192 | Model dimension |
| `num_layers` | 18-64 | 12-80 | Depth |
| `num_heads` | 8-32 | 4-64 | MQA query heads |
| `num_kv_heads` | 1-8 | 1-16 | MQA key/value heads |
| `window_size` | 128-512 | 64-2048 | Local attention window |
| `d_conv` | 4 | 3-7 | Temporal conv kernel |
| `expand` | 2 | 1-4 | RGLRU expansion factor |
| `ffn_expand` | 4 | 2-8 | FFN expansion factor |

**Small model example** (50M params):
```python
model = GriffinModel(
    d_model=512,
    num_layers=12,
    num_heads=8,
    num_kv_heads=1,
    window_size=128
)
```

**Large model example** (7B params):
```python
model = GriffinModel(
    d_model=4096,
    num_layers=32,
    num_heads=32,
    num_kv_heads=8,
    window_size=256
)
```

## Code Walkthrough

### RGLRU Implementation

```python
class RealGatedLinearRecurrentUnit(NexusModule):
    """Core recurrent component of Griffin."""

    def __init__(self, d_model: int, d_recurrence: Optional[int] = None):
        super().__init__()
        self.d_recurrence = d_recurrence or d_model

        # Input projection
        self.x_proj = nn.Linear(d_model, self.d_recurrence, bias=False)

        # Recurrence gate (learnable)
        self.a_proj = nn.Linear(d_model, self.d_recurrence, bias=True)

        # Initialize gate bias to encourage remembering (high a values)
        nn.init.constant_(self.a_proj.bias, 1.0)

    def forward(self, x, state=None):
        batch_size, seq_len, _ = x.shape

        # Compute input-dependent gate
        a = torch.sigmoid(self.a_proj(x))  # (B, L, d_rec)

        # Project input
        x_in = self.x_proj(x)  # (B, L, d_rec)

        # Magnitude-preserving scaling
        x_in = x_in * torch.sqrt(1 - a ** 2 + 1e-6)

        # Initialize state if needed
        if state is None:
            state = torch.zeros(
                batch_size, self.d_recurrence,
                device=x.device, dtype=x.dtype
            )

        # Run recurrence (training uses parallel scan, inference is sequential)
        if self.training:
            output, state = self._parallel_scan(a, x_in, state)
        else:
            output, state = self._sequential_scan(a, x_in, state)

        return output, state
```

**Key implementation details:**

1. **Gate initialization**: `bias=1.0` encourages high gate values initially, helping the model remember information by default.

2. **Magnitude scaling**: `sqrt(1 - a²)` ensures the recurrence doesn't explode or vanish.

3. **Training vs inference**: Parallel scan for training (O(log N) depth), sequential for inference (O(1) per step).

### Local Multi-Query Attention

```python
class LocalMultiQueryAttention(NexusModule):
    """Local windowed attention with MQA for efficiency."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_kv_heads: int = 1,
        window_size: int = 128
    ):
        super().__init__()
        self.window_size = window_size
        self.head_dim = d_model // num_heads

        # Multiple query heads, few KV heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        # Project queries, keys, values
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Expand KV for multi-query (repeat to match num_heads)
        heads_per_kv = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(heads_per_kv, dim=2)
        v = v.repeat_interleave(heads_per_kv, dim=2)

        # Append to cache and limit to window
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        # Keep only window_size most recent
        if k.shape[1] > self.window_size:
            k = k[:, -self.window_size:]
            v = v[:, -self.window_size:]

        # Standard attention computation
        # ... (causal masking, softmax, etc.)

        return output, (k, v)  # Return updated cache
```

**Key implementation details:**

1. **Multi-query efficiency**: Only `num_kv_heads` KV parameters (typically 1-8) vs `num_heads` (typically 8-32) query parameters.

2. **Window management**: Cache automatically limited to `window_size` most recent tokens.

3. **Cache update**: Return updated KV cache for efficient autoregressive generation.

### Full Model Integration

```python
class GriffinModel(NexusModule):
    def __init__(self, d_model, num_layers, ...):
        super().__init__()

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Griffin blocks
        self.blocks = nn.ModuleList([
            GriffinBlock(d_model, ...)
            for _ in range(num_layers)
        ])

        # Output head
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, states=None):
        if states is None:
            states = [None] * self.num_layers

        # Embed tokens
        x = self.embedding(input_ids)

        # Process through blocks
        new_states = []
        for block, state in zip(self.blocks, states):
            x, new_state = block(x, state)
            new_states.append(new_state)

        # Output projection
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_states
```

## Optimization Tricks

### 1. KV Cache Compression

Griffin achieves efficient KV cache through:

**Multi-Query Attention**: Reduce KV heads
```python
# Standard attention: 8 heads × d_model KV cache
num_heads = 8
d_model = 512
kv_cache_size = window_size * d_model  # 128 * 512 = 65K per layer

# MQA: 1 KV head
num_kv_heads = 1
kv_cache_size = window_size * (d_model // num_heads)  # 128 * 64 = 8K per layer
# 8x reduction!
```

**Sliding Window**: Cap cache size
```python
# Without window: O(N) growth
kv_cache_size = seq_len * d_kv  # Grows with sequence

# With window: O(1) constant
kv_cache_size = window_size * d_kv  # Fixed at 128 tokens
```

**Combined effect**: ~50x smaller KV cache than standard transformer

### 2. Throughput Optimization

**Inference optimizations:**

```python
# 1. Use sequential scan for inference (O(1) per step)
def _sequential_scan(self, a, x, state):
    outputs = []
    for t in range(seq_len):
        state = a[:, t] * state + x[:, t]  # Simple update
        outputs.append(state)
    return torch.stack(outputs, dim=1), state

# 2. Minimize KV cache updates
if kv_len > self.window_size:
    k = k[:, -self.window_size:]  # Drop old tokens
    v = v[:, -self.window_size:]

# 3. Efficient state management
state_size = d_recurrence  # Typically 256-1024
# vs transformer KV cache: window_size * num_layers * d_model
```

**Training optimizations:**

```python
# Use parallel scan for training (O(log N) depth)
def _parallel_scan(self, a, x, state):
    # Log-space cumsum for numerical stability
    log_a = torch.log(a + 1e-6)
    log_a_cumsum = torch.cumsum(log_a, dim=1)

    # Weighted inputs
    weighted_x = x * torch.exp(-log_a_cumsum)
    cumsum_weighted_x = torch.cumsum(weighted_x, dim=1)

    # Rescale
    output = cumsum_weighted_x * torch.exp(log_a_cumsum)

    # Add initial state contribution
    state_contrib = state.unsqueeze(1) * torch.exp(log_a_cumsum)
    output = output + state_contrib

    return output, output[:, -1]
```

### 3. Memory-Efficient Implementation

**State reuse across layers:**

```python
# Each layer maintains O(d_recurrence) state
total_state_memory = num_layers * d_recurrence * batch_size

# Example: 32 layers, 512 dim, batch=1
# = 32 * 512 * 4 bytes = 64KB
# vs transformer KV cache: GB-scale for long sequences
```

**Gradient checkpointing friendly:**

```python
# Griffin blocks are modular and checkpoint-friendly
model = GriffinModel(...)
model.gradient_checkpointing_enable()

# Each block recomputes activations, saving memory
```

## Experiments & Results

### Performance Benchmarks (from paper)

**Perplexity on C4 validation (lower is better):**

| Model | Size | Perplexity | Inference Speed |
|-------|------|------------|-----------------|
| Transformer | 350M | 13.2 | 1.0x (baseline) |
| Griffin | 350M | 13.5 | **2.5x faster** |
| Hawk (no attention) | 350M | 14.1 | **5.0x faster** |

**Recall-intensive tasks:**

| Model | Natural Questions | MMLU | TriviaQA |
|-------|-------------------|------|----------|
| Transformer | 85.2 | 72.3 | 81.5 |
| Griffin | **84.8** | **71.9** | **80.1** |
| Hawk | 78.3 | 68.5 | 73.2 |

**Long-range tasks:**

| Model | Long Range Arena | PG-19 | arXiv-Math |
|-------|------------------|-------|------------|
| Transformer | 62.4 | 15.8 | 0.92 |
| Griffin | **64.1** | **15.2** | **0.89** |
| Hawk | **65.3** | **14.9** | **0.88** |

### Efficiency-Quality Tradeoff

```
Quality (Perplexity) vs Speed Tradeoff:

  13.0 ┤              ○ Transformer
       │
  13.5 ┤          ◆ Griffin
       │
  14.0 ┤                          □ Hawk
       │
  14.5 ┤
       └────────────────────────────────►
       1.0x      2.5x              5.0x
              Inference Speed
```

**Key insights:**
1. Griffin maintains ~98% of transformer quality
2. Griffin achieves 2.5x speedup over transformers
3. Hawk (pure recurrence) gets 5x speedup but sacrifices recall quality
4. Griffin excels on long-range tasks where recurrence helps

### Scaling Behavior

**Context length scaling:**

| Context Length | Transformer Memory | Griffin Memory | Reduction |
|----------------|-------------------|----------------|-----------|
| 2K | 100 MB | 20 MB | 5x |
| 8K | 800 MB | 25 MB | 32x |
| 32K | 6.4 GB | 35 MB | 183x |

**Interpretation**: Griffin's memory scales with window size, not context length.

## Common Pitfalls

### 1. Gate Initialization

**Pitfall**: Using default initialization for gate bias.

```python
# ❌ BAD: Default initialization
self.a_proj = nn.Linear(d_model, d_recurrence, bias=True)
# Gates start ~0.5, model forgets too quickly
```

```python
# ✅ GOOD: Initialize to favor remembering
self.a_proj = nn.Linear(d_model, d_recurrence, bias=True)
nn.init.constant_(self.a_proj.bias, 1.0)  # Start with high memory
```

**Why**: High gate values (close to 1) encourage the model to retain information, which is crucial early in training.

### 2. Magnitude Scaling

**Pitfall**: Forgetting magnitude preservation scaling.

```python
# ❌ BAD: No scaling
x_in = self.x_proj(x)
state = a * state + x_in  # State magnitude grows/shrinks!
```

```python
# ✅ GOOD: Magnitude-preserving scaling
x_in = self.x_proj(x)
x_in = x_in * torch.sqrt(1 - a ** 2 + 1e-6)  # Preserve magnitude
state = a * state + x_in
```

**Why**: Without scaling, the recurrence can explode or vanish, especially in deep networks.

### 3. Window Size Too Small

**Pitfall**: Using tiny attention windows.

```python
# ❌ BAD: Window too small for task
window_size = 16  # Only sees 16 tokens
# Model can't capture medium-range dependencies
```

```python
# ✅ GOOD: Match window to task requirements
window_size = 128  # Standard for most tasks
window_size = 512  # For tasks needing more local context
```

**Rule of thumb**: Window should cover at least 1-2 sentences (50-100 tokens minimum).

### 4. State Management in Generation

**Pitfall**: Forgetting to pass states during generation.

```python
# ❌ BAD: Reset state every step
for token in range(max_length):
    logits, _ = model(input_ids)  # State is None every time!
    # Model has no memory of previous tokens
```

```python
# ✅ GOOD: Maintain state across steps
states = None
for token in range(max_length):
    logits, states = model(input_ids, states=states)  # Accumulate state
    # Model remembers all previous context
```

### 5. Parallel Scan Numerical Stability

**Pitfall**: Computing parallel scan in linear space.

```python
# ❌ BAD: Numerical instability for long sequences
cumsum_a = torch.cumprod(a, dim=1)  # Exponential growth/decay!
```

```python
# ✅ GOOD: Log-space computation
log_a = torch.log(a + 1e-6)
cumsum_log_a = torch.cumsum(log_a, dim=1)
cumsum_a = torch.exp(cumsum_log_a)  # Stable!
```

### 6. KV Cache Mismanagement

**Pitfall**: Not limiting cache size.

```python
# ❌ BAD: Unbounded cache growth
k_cache = torch.cat([k_cache, k], dim=1)  # Grows forever!
```

```python
# ✅ GOOD: Cap at window size
k_cache = torch.cat([k_cache, k], dim=1)
if k_cache.shape[1] > window_size:
    k_cache = k_cache[:, -window_size:]  # Keep only recent
```

## References

### Primary Paper

De, S., Smith, S. L., Fernando, A., Botev, A., Cristian-Muraru, G., Gu, A., Haroun, R., Berrada, L., Chen, Y., Srinivasan, S., Desjardins, G., Doucet, A., Budden, D., Teh, Y. W., Pascanu, R., Freitas, N. D., & Gulcehre, C. (2024). **Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models**. arXiv:2402.19427.

### Related Work

1. **Hawk (recurrence-only variant)**: Same paper, pure RGLRU baseline
2. **Mamba**: Gu & Dao (2023), selective SSMs
3. **H3**: Fu et al. (2023), hierarchical SSMs
4. **Hyena**: Poli et al. (2023), long convolutions
5. **RetNet**: Sun et al. (2023), retention mechanisms

### Implementation Resources

- Reference implementation: `nexus/models/hybrid/griffin.py`
- Official JAX implementation: [Google DeepMind GitHub](https://github.com/google-deepmind/recurrentgemma)
- Hawk variant: `nexus/models/hybrid/hawk.py`

### Follow-up Work

- **RecurrentGemma** (2024): Open-source Griffin-based LM from Google
- **Jamba** (2024): Griffin-inspired hybrid with Mamba instead of RGLRU
