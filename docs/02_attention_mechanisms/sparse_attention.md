# Sparse Attention

## Overview & Motivation

Sparse Attention addresses the quadratic complexity bottleneck of standard attention by restricting each token to attend to only a sparse subset of positions. Instead of computing an N×N attention matrix, sparse attention computes attention over carefully chosen subsets, reducing complexity from O(N²) to O(N·k) where k ≪ N.

**Key Innovation**: Not all pairs of tokens need to interact directly. By using structured sparsity patterns (local windows, strided patterns, global tokens), models can achieve strong performance while dramatically reducing computation and memory.

**Why Sparse Attention?**
- **Scalability**: Enables context lengths of 16K-256K+ tokens
- **Efficiency**: Reduces memory from O(N²) to O(N·k)
- **Performance**: Often matches or exceeds dense attention quality
- **Flexibility**: Multiple sparsity patterns for different use cases
- **Production Use**: Powers Longformer, BigBird, DeepSeek V3, and many long-context models

## Theoretical Background

### The Sparsity Hypothesis

Key insight: **Most attention weights are small and contribute little to the output**.

Analysis of trained transformers shows:
- ~90% of attention weight mass concentrates on 10-20% of positions
- Local context (nearby tokens) gets most attention
- A few global tokens (like [CLS], [SEP]) receive broad attention
- Distant token pairs rarely need direct interaction

### Sparsity Patterns

**1. Local (Sliding Window)**
```
Token i attends to: [i - w, i + w]
Complexity: O(N·w)
Used by: Mistral, Gemma, Longformer (local component)
```

**2. Strided (Fixed)**
```
Token i attends to: {i, i-s, i-2s, i-3s, ...}
Complexity: O(N·(N/s)) = O(N²/s)
Used by: Sparse Transformer (OpenAI)
```

**3. Local + Global**
```
Token i attends to:
  - Local window: [i - w, i + w]
  - Global tokens: [0, ..., g]
Complexity: O(N·(w + g))
Used by: Longformer, ETC
```

**4. BigBird Pattern**
```
Token i attends to:
  - Local window: [i - w, i + w]
  - Global tokens: [0, ..., g]
  - Random tokens: r random positions
Complexity: O(N·(w + g + r))
Used by: BigBird
```

### Theoretical Guarantees

BigBird (Zaheer et al., 2020) proved:
- Sparse attention with local + global + random is a **universal approximator**
- Can approximate any function that dense attention can
- With w=3, g=2, r=3: theoretically sufficient for any task

### Connectivity

Sparse attention creates a **connectivity graph**:
- Nodes: Tokens
- Edges: Attention connections
- Path length: Max hops to connect any two tokens

Example with 3-token local window:
```
Dense attention: All tokens 1 hop apart
Sparse attention: Distant tokens ≤ N/3 hops apart
With L layers: Effective receptive field = (2w+1)^L
```

For w=256, L=12: Receptive field covers 6144 tokens.

## Mathematical Formulation

### Standard Attention (Dense)

```
S = QK^T / √d_k ∈ ℝ^(N×N)    [All N² pairs]
A = softmax(S) ∈ ℝ^(N×N)
O = AV ∈ ℝ^(N×d)
```

### Sparse Attention (General)

```
S_sparse = QK^T / √d_k ∈ ℝ^(N×k)    [Only k pairs per token]
A_sparse = softmax(S_sparse) ∈ ℝ^(N×k)
O = A_sparse V_subset ∈ ℝ^(N×d)
```

Where:
- k: Average number of positions each token attends to
- V_subset: Only values for attended positions

### Sparsity Mask

Implemented via masking:
```
M ∈ {0, -∞}^(N×N)
M_ij = 0    if i attends to j (allowed)
M_ij = -∞   if i does not attend to j (masked)

S_masked = QK^T / √d_k + M
A = softmax(S_masked)    [Results in sparse A]
```

### Local Window Pattern (Formal)

```
M_ij = 0     if |i - j| ≤ w
M_ij = -∞    otherwise

For causal: M_ij = 0 if 0 ≤ i - j ≤ w
```

### Local + Global Pattern

```
M_ij = 0     if |i - j| ≤ w OR j < g
M_ij = -∞    otherwise

Where g = number of global tokens
```

### BigBird Pattern

```
For token i:
  Attend to:
    - Local: {i - w, ..., i + w}
    - Global: {0, ..., g-1}
    - Random: R_i (r random indices)

M_ij = 0     if j ∈ Local(i) ∪ Global ∪ R_i
M_ij = -∞    otherwise
```

### Complexity Analysis

| Pattern | Complexity | Memory | Receptive Field (1 layer) |
|---------|-----------|--------|---------------------------|
| Dense | O(N²) | O(N²) | N |
| Local (w) | O(N·w) | O(N·w) | 2w+1 |
| Strided (s) | O(N²/s) | O(N²/s) | N/s |
| Local+Global | O(N·(w+g)) | O(N·(w+g)) | 2w+1+g |
| BigBird | O(N·(w+g+r)) | O(N·(w+g+r)) | 2w+1+g+r |

## High-Level Intuition

### Mental Model

Think of sparse attention like a **social network**:

**Dense Attention**: Everyone knows everyone directly
- N people, N² friendships
- Expensive to maintain
- Often unnecessary

**Sparse Attention**: Strategic friendships
- Local friends: Your neighbors (local window)
- Global influencers: Everyone knows them (global tokens)
- Random connections: Weak ties for diversity (random)
- Still connected, but far fewer edges

### Example: Document Processing

```
Document: 10,000 words

Dense Attention:
- Word 1 attends to all 10,000 words
- 10,000² = 100M attention computations
- Most attention weights are near zero

Sparse Attention (w=256, g=2):
- Word 1 attends to:
  * Words 1-257 (local)
  * Global tokens (document start)
- ~258 attention computations per word
- 10,000 × 258 = 2.58M computations
- 40x reduction, minimal quality loss
```

### Why It Works

1. **Locality Bias**: Most relevant context is nearby
   - "The cat sat on the ___" → "mat" is close to "on"

2. **Global Anchors**: Some tokens summarize broader context
   - [CLS] token aggregates sentence information
   - Section headers provide global structure

3. **Transitivity**: Distant tokens connect via intermediate hops
   - Token 0 → Token 256 → Token 512 (2 hops)
   - After L layers, any tokens connected in O(L) hops

## Implementation Details

### Core Implementation

See `Nexus/nexus/components/attention/sparse_attention.py`

```python
class SparseAttention(NexusModule):
    """
    Sparse Attention with configurable sparsity patterns.

    Supports: local, strided, local_global, bigbird patterns.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sparsity_pattern: Literal['local', 'strided', 'local_global', 'bigbird'] = 'local',
        local_window: int = 256,
        global_tokens: int = 1,
        num_random: int = 3,
        stride: int = 64,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.sparsity_pattern = sparsity_pattern
        self.local_window = local_window
        self.global_tokens = global_tokens
        self.num_random = num_random
        self.stride = stride
        self.causal = causal
        self.scale = self.head_dim ** -0.5

        # Standard projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
```

### Sparsity Pattern Creation

```python
def _create_sparse_mask(
    self,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """Create sparse attention mask based on pattern."""

    if self.sparsity_pattern == 'local':
        return self._create_local_mask(seq_len, device, dtype)
    elif self.sparsity_pattern == 'strided':
        return self._create_strided_mask(seq_len, device, dtype)
    elif self.sparsity_pattern == 'local_global':
        return self._create_local_global_mask(seq_len, device, dtype)
    elif self.sparsity_pattern == 'bigbird':
        return self._create_bigbird_mask(seq_len, device, dtype)
    else:
        raise ValueError(f"Unknown pattern: {self.sparsity_pattern}")

def _create_local_mask(self, seq_len, device, dtype):
    """Local sliding window mask."""
    # Position indices
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)

    # Window condition
    if self.causal:
        mask = (col_idx <= row_idx) & (row_idx - col_idx < self.local_window)
    else:
        mask = torch.abs(row_idx - col_idx) < self.local_window

    # Convert to additive mask
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    mask.masked_fill_(~mask, float('-inf'))
    return mask

def _create_local_global_mask(self, seq_len, device, dtype):
    """Local window + global tokens mask."""
    # Start with local mask
    mask = self._create_local_mask(seq_len, device, dtype)

    # Add global token connections
    # Global tokens attend to all, all attend to global
    mask[:self.global_tokens, :] = 0  # Global tokens see all
    mask[:, :self.global_tokens] = 0  # All see global tokens

    return mask

def _create_bigbird_mask(self, seq_len, device, dtype):
    """BigBird: Local + Global + Random mask."""
    # Start with local + global
    mask = self._create_local_global_mask(seq_len, device, dtype)

    # Add random connections
    for i in range(seq_len):
        # Skip global tokens (already connected)
        if i < self.global_tokens:
            continue

        # Sample random positions
        candidates = list(range(seq_len))
        # Remove already connected (local window + global)
        candidates = [
            j for j in candidates
            if abs(i - j) >= self.local_window and j >= self.global_tokens
        ]

        if len(candidates) > 0:
            random_indices = torch.randperm(len(candidates))[:self.num_random]
            random_positions = [candidates[idx] for idx in random_indices]
            mask[i, random_positions] = 0

    return mask
```

### Forward Pass

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    return_attention: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    B, N, D = hidden_states.shape

    # Project Q, K, V
    q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    # Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

    # Apply sparsity mask
    sparsity_mask = self._create_sparse_mask(N, hidden_states.device, attn_scores.dtype)
    attn_scores = attn_scores + sparsity_mask

    # Apply optional attention mask (e.g., padding)
    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask

    # Softmax and apply to values
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_weights = self.attn_dropout(attn_weights)

    output = torch.matmul(attn_weights, v)
    output = output.transpose(1, 2).reshape(B, N, -1)
    output = self.o_proj(output)

    if return_attention:
        return output, attn_weights
    return output, None
```

## Code Walkthrough

### Example 1: Local Window Attention

```python
from nexus.components.attention import SparseAttention

# Mistral-style sliding window
sparse_attn = SparseAttention(
    dim=4096,
    num_heads=32,
    sparsity_pattern='local',
    local_window=4096,  # 4K window
    causal=True
)

# Process long sequence
x = torch.randn(1, 16384, 4096, device='cuda')  # 16K context
output = sparse_attn(x)

print(f"Input: {x.shape}")
print(f"Output: {output.shape}")
print(f"Effective attention: {16384 * 4096 / (16384**2):.1%} of dense")
# 25% of dense attention computation
```

### Example 2: BigBird for Documents

```python
# Long document understanding
bigbird_attn = SparseAttention(
    dim=768,
    num_heads=12,
    sparsity_pattern='bigbird',
    local_window=256,
    global_tokens=2,  # [CLS] and [SEP]
    num_random=3,
    causal=False  # Bidirectional for understanding
)

# Process document
document = torch.randn(1, 4096, 768, device='cuda')  # 4K tokens
output = bigbird_attn(document)

# Attention per token: 2*256 + 2 + 3 = 517
# vs 4096 for dense (8x reduction)
```

### Example 3: Longformer-style Local+Global

```python
# Question answering with long context
longformer_attn = SparseAttention(
    dim=512,
    num_heads=8,
    sparsity_pattern='local_global',
    local_window=512,
    global_tokens=1,  # [CLS] for classification
    causal=False
)

# Context + question
x = torch.randn(1, 8192, 512, device='cuda')
output = longformer_attn(x)
```

### Example 4: Visualize Sparsity Pattern

```python
def visualize_pattern(pattern, seq_len=100):
    attn = SparseAttention(
        dim=64, num_heads=1,
        sparsity_pattern=pattern,
        local_window=10,
        global_tokens=2,
        num_random=2
    )

    mask = attn._create_sparse_mask(seq_len, 'cpu', torch.float32)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(mask.numpy(), cmap='RdYlGn', vmin=-1, vmax=0)
    plt.title(f'{pattern} Attention Pattern')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar(label='0=Attend, -inf=Masked')
    plt.show()

visualize_pattern('bigbird')
```

## Optimization Tricks

### 1. Block-Sparse Kernels

Use optimized sparse attention kernels:

```python
# Triton block-sparse attention
from triton.ops.blocksparse import matmul as blocksparse_matmul

# Define sparsity layout (block-level)
# Much faster than element-wise sparse
```

### 2. Cache Sparsity Masks

```python
# Don't recompute mask every forward pass
class SparseAttention(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self._mask_cache = {}

    def _get_mask(self, seq_len, device):
        key = (seq_len, device)
        if key not in self._mask_cache:
            self._mask_cache[key] = self._create_sparse_mask(...)
        return self._mask_cache[key]
```

### 3. Flash-Sparse Attention

Combine with FlashAttention tiling:

```python
# FlashAttention with sparsity mask
# Only compute attention for non-masked blocks
# Best of both worlds: IO-efficiency + sparsity
```

### 4. Dynamic Sparsity

Learn which positions to attend to:

```python
# Top-k selection
k = self.window_size
scores = (q @ k.T).mean(dim=1)  # (B, N, N) → (B, N)
_, top_k_indices = scores.topk(k, dim=-1)
# Create mask from top_k_indices
```

### 5. Chunked Processing

Process long sequences in chunks:

```python
chunk_size = 2048
for i in range(0, seq_len, chunk_size):
    chunk = x[:, i:i+chunk_size]
    # Apply sparse attention to chunk
    # Overlapping windows for continuity
```

## Experiments & Results

### Long Document Understanding

**Dataset**: WikiHop (multi-hop QA), avg length 4,000 tokens

| Model | Attention | Accuracy | Memory | Speed |
|-------|-----------|----------|--------|-------|
| BERT-Base | Dense (512) | Truncated | 16 GB | 1.0x |
| Longformer | Local+Global | 68.2% | 18 GB | 0.9x |
| BigBird | BigBird | 69.5% | 19 GB | 0.85x |
| Sparse Transformer | Strided | 66.1% | 17 GB | 1.1x |

Sparse attention enables 8x longer context with minimal quality loss.

### Language Modeling

**Dataset**: PG-19 (long books), context 8192 tokens

| Model | Perplexity | Params | Context | Throughput |
|-------|------------|--------|---------|------------|
| Transformer-XL | 36.3 | 277M | 1024 | 1.0x |
| Sparse Transformer | 35.1 | 277M | 8192 | 0.7x |
| Longformer | 34.8 | 149M | 8192 | 0.8x |

Longer context improves perplexity significantly.

### Sparsity Ablations

**Model**: Longformer-base on classification

| Window Size | Accuracy | Tokens/Sec |
|-------------|----------|------------|
| 128 | 82.1% | 2400 |
| 256 | 84.3% | 1800 |
| 512 | 85.1% | 1200 |
| 1024 | 85.3% | 750 |

Diminishing returns after w=512 for most tasks.

### Global Tokens Impact

| # Global Tokens | Accuracy | Added Cost |
|-----------------|----------|------------|
| 0 (local only) | 82.5% | 0% |
| 1 | 84.1% | +0.5% |
| 2 | 84.8% | +1.0% |
| 4 | 84.9% | +2.0% |

2-4 global tokens provide best cost/benefit ratio.

## Common Pitfalls

### 1. Insufficient Window Size

```python
# Wrong: Window too small for task
sparse_attn = SparseAttention(local_window=64)  # Only sees ±64 tokens
# For long-range dependencies, may need larger window

# Correct: Tune window to task
sparse_attn = SparseAttention(local_window=512)  # Better coverage
```

### 2. Forgetting Global Tokens

```python
# Wrong: Pure local attention, no global anchors
sparse_attn = SparseAttention(
    sparsity_pattern='local',
    global_tokens=0  # No global context!
)

# Correct: Include global tokens for classification/summary
sparse_attn = SparseAttention(
    sparsity_pattern='local_global',
    global_tokens=2  # [CLS], [SEP]
)
```

### 3. Random Pattern Not Fixed

```python
# Wrong: Different random pattern every forward pass
# Breaks gradient flow and caching
def _create_bigbird_mask(self, ...):
    random_indices = torch.randperm(...)  # Changes every time!

# Correct: Fix random seed or cache pattern
def _create_bigbird_mask(self, ...):
    if self._cached_random is None:
        torch.manual_seed(42)
        self._cached_random = torch.randperm(...)
    random_indices = self._cached_random
```

### 4. Not Handling Padding

```python
# Wrong: Sparse mask doesn't account for padding
sparse_mask = self._create_sparse_mask(...)
# Padding tokens may still be attended to!

# Correct: Combine sparsity mask with padding mask
sparse_mask = self._create_sparse_mask(...)
combined_mask = sparse_mask + padding_mask
```

### 5. Inefficient Mask Implementation

```python
# Wrong: Creating full N×N mask
mask = torch.zeros(N, N)
for i in range(N):
    for j in range(N):
        if abs(i - j) < window:
            mask[i, j] = 0  # Slow!

# Correct: Vectorized mask creation
row_idx = torch.arange(N).unsqueeze(1)
col_idx = torch.arange(N).unsqueeze(0)
mask = (torch.abs(row_idx - col_idx) < window).float()
```

### 6. Wrong Complexity Assumption

```python
# Wrong: Thinking sparse attention is always faster
# For small N, sparse overhead may dominate
if N < 1000:
    use_dense_attention()  # Faster for short sequences
else:
    use_sparse_attention()  # Better for long sequences
```

## References

### Original Papers

1. **Generating Long Sequences with Sparse Transformers**
   Child, R., Gray, S., Radford, A., & Sutskever, I. (2019)
   OpenAI
   [arxiv.org/abs/1904.10509](https://arxiv.org/abs/1904.10509)

2. **Longformer: The Long-Document Transformer**
   Beltagy, I., Peters, M. E., & Cohan, A. (2020)
   Allen AI
   [arxiv.org/abs/2004.05150](https://arxiv.org/abs/2004.05150)

3. **Big Bird: Transformers for Longer Sequences**
   Zaheer, M., et al. (2020)
   Google Research
   [arxiv.org/abs/2007.14062](https://arxiv.org/abs/2007.14062)

### Recent Applications

4. **DeepSeek-V3 Technical Report**
   DeepSeek-AI (2024)
   Uses sparse attention with MLA
   [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)

5. **Mistral 7B**
   Jiang, A. Q., et al. (2023)
   Sliding window attention
   [arxiv.org/abs/2310.06825](https://arxiv.org/abs/2310.06825)

### Analysis

6. **Do Transformer Attention Heads Provide Transparency in Abstractive Summarization?**
   Dou, Z.-Y., et al. (2021)
   Analysis of attention patterns

### Related Mechanisms

- [Sliding Window Attention](./sliding_window_attention.md) - Pure local window
- [Linear Attention](./linear_attention.md) - Different approach to O(N) complexity
- [Flash Attention](./flash_attention.md) - Can be combined with sparsity
- [Ring Attention](./ring_attention.md) - Distributed long context

## See Also

- **Implementation**: `Nexus/nexus/components/attention/sparse_attention.py`
- **Block-Sparse Kernels**: Triton, xFormers
- **Longformer Repo**: [github.com/allenai/longformer](https://github.com/allenai/longformer)
- **BigBird Repo**: [github.com/google-research/bigbird](https://github.com/google-research/bigbird)
