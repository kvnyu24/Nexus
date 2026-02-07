# Self Attention

## Overview & Motivation

Self-Attention is the foundational mechanism that enables sequences to attend to themselves, allowing each position to gather contextual information from all other positions in the sequence. It's the core building block of transformer architectures and the key to their success in capturing long-range dependencies.

**Key Innovation**: Unlike RNNs that process sequences sequentially, or CNNs that have fixed receptive fields, self-attention allows every position to directly attend to every other position in a single operation. This enables parallel processing and unlimited receptive fields.

**Why Self-Attention?**
- **Unlimited Context**: Each token can attend to any other token, regardless of distance
- **Parallel Processing**: All positions computed simultaneously (vs sequential in RNNs)
- **Interpretable**: Attention weights show which tokens influence each other
- **Foundation**: Powers BERT, GPT, T5, and virtually all modern language models
- **Flexible**: Works for any sequential data (text, audio, time series, DNA sequences)

## Theoretical Background

### The Core Idea

Given a sequence of tokens, each token needs to understand its context. Self-attention computes a context-aware representation for each token by:

1. Comparing it with all other tokens (including itself)
2. Computing attention scores based on similarity
3. Taking a weighted average of all token representations

### Self-Attention vs Cross-Attention

**Self-Attention**: Q, K, V all from same sequence
```
Input: X ∈ ℝ^(n×d)
Q = XW^Q,  K = XW^K,  V = XW^V
Output = Attention(Q, K, V) ∈ ℝ^(n×d)
```

**Cross-Attention**: Q from one sequence, K and V from another
```
Query input: X ∈ ℝ^(n×d_x)
KV input: Y ∈ ℝ^(m×d_y)
Q = XW^Q,  K = YW^K,  V = YW^V
```

### Information Flow

```
Input Sequence: X ∈ ℝ^(n×d)
       ↓
   ┌───┴───┬───────┬───────┐
   │       │       │       │
   ↓       ↓       ↓       ↓
   Q       K       V    (Linear projections)
   │       │       │
   └───┬───┘       │
       │           │
    QK^T/√d_k      │    (Similarity scores)
       │           │
   Softmax         │    (Attention weights)
       │           │
       └─────┬─────┘
             │
          Attn·V        (Weighted combination)
             │
           Output       (Context-aware representations)
```

### Attention Matrix Properties

For self-attention with n tokens:
- **Shape**: n×n (square matrix)
- **Row i**: How token i attends to all tokens (including itself)
- **Column j**: How all tokens attend to token j
- **Row sum**: 1.0 (normalized distribution after softmax)
- **Diagonal**: Self-attention (token attending to itself)

### Bidirectional vs Causal

**Bidirectional** (BERT-style):
```
Each token attends to all tokens (past, present, future)
Attention matrix: Full n×n matrix
Use case: Understanding tasks (classification, NER, QA)
```

**Causal** (GPT-style):
```
Each token only attends to itself and previous tokens
Attention matrix: Lower triangular
Use case: Generation tasks (language modeling, completion)
```

## Mathematical Formulation

### Standard Self-Attention

Given input sequence X ∈ ℝ^(n×d_model):

1. **Linear Projections**:
   ```
   Q = XW^Q ∈ ℝ^(n×d_k)
   K = XW^K ∈ ℝ^(n×d_k)
   V = XW^V ∈ ℝ^(n×d_v)

   where W^Q, W^K ∈ ℝ^(d_model×d_k), W^V ∈ ℝ^(d_model×d_v)
   ```

2. **Attention Scores**:
   ```
   S = QK^T / √d_k ∈ ℝ^(n×n)

   S_ij = (q_i · k_j) / √d_k
        = similarity between position i and position j
   ```

3. **Optional Masking** (for causal or padding):
   ```
   S_masked = S + M

   where M_ij = 0    if allowed
                -∞   if masked
   ```

4. **Attention Weights**:
   ```
   A = softmax(S_masked) ∈ ℝ^(n×n)

   A_ij = exp(S_ij) / Σ_k exp(S_ik)
        = attention weight from position i to position j
   ```

5. **Output**:
   ```
   O = AV ∈ ℝ^(n×d_v)

   o_i = Σ_j A_ij · v_j
       = weighted average of all value vectors
   ```

### Multi-Head Self-Attention

```
head_i = SelfAttention(XW_i^Q, XW_i^K, XW_i^V)

MultiHeadSelfAttn(X) = Concat(head_1, ..., head_h)W^O
```

Where:
- h: number of heads
- d_k = d_v = d_model / h (typical setting)
- Each head learns different patterns
- W^O ∈ ℝ^(h·d_v×d_model): Output projection

### Complexity Analysis

**Time Complexity**: O(n²d)
- Computing QK^T: O(n²d_k) per head × h heads = O(n²d_model)
- Softmax: O(n²)
- Attention·V: O(n²d_v) per head × h heads = O(n²d_model)
- Dominated by n² term for long sequences

**Space Complexity**: O(n² + nd)
- Attention matrices: O(n²h)
- Activations: O(nd_model)
- Parameters: O(d²_model) for projections

**Bottleneck**: The n² term makes self-attention expensive for long sequences, motivating efficient variants (Flash, Linear, Sparse, etc.)

### Position Information

Self-attention is **position-invariant** by default (permutation equivariant):
```
SelfAttn(Permute(X)) = Permute(SelfAttn(X))
```

To encode position, add positional encodings:
1. **Absolute**: Add position embeddings to input
2. **Relative**: Add position bias to attention scores
3. **Rotary (RoPE)**: Rotate Q and K based on position

## High-Level Intuition

### Mental Model

Think of self-attention as a **social network** where tokens communicate:

```
Sentence: "The cat sat on the mat"

Token "sat" asks: "What's my context?"
- Looks at "The": Low relevance (0.05)
- Looks at "cat": High relevance (0.40) ← Subject of "sat"
- Looks at "sat": Medium (0.10) ← Self-attention
- Looks at "on": Medium (0.15) ← Preposition
- Looks at "the": Low (0.05)
- Looks at "mat": High (0.25) ← Object via "on"

Result: "sat" representation is enhanced with information from "cat" and "mat"
```

### Visualization

Attention matrix for "The cat sat on the mat":

```
           The   cat   sat   on    the   mat
The        0.6   0.2   0.0   0.0   0.1   0.1   (Determiner looks at itself, noun)
cat        0.1   0.5   0.3   0.0   0.0   0.1   (Noun looks at itself, verb)
sat        0.0   0.4   0.2   0.2   0.0   0.2   (Verb looks at subject, object)
on         0.0   0.0   0.3   0.3   0.1   0.3   (Prep looks at verb, object)
the        0.0   0.0   0.0   0.1   0.6   0.3   (Det looks at itself, noun)
mat        0.0   0.1   0.1   0.2   0.1   0.5   (Noun looks at itself, prep)
```

Each row sums to 1.0 (normalized distribution)

### Why Three Matrices (Q, K, V)?

- **Query (Q)**: "What am I looking for?"
  - Represents the information needs of each position
  - Example: Token "sat" queries for subject and object

- **Key (K)**: "What information do I offer?"
  - Represents what each position can provide
  - Example: Token "cat" offers "I'm a noun, potential subject"

- **Value (V)**: "What information do I actually provide?"
  - The actual information to be aggregated
  - Example: Token "cat" provides its semantic embedding

This separation allows flexible matching and retrieval patterns.

## Implementation Details

### Core Implementation

See `Nexus/nexus/components/attention/self_attention.py`

```python
class SelfAttention(BaseAttention):
    """
    Self-Attention: Sequence attends to itself

    The foundation of transformer models. Each position can attend
    to all positions in the same sequence.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
        is_causal: bool = False,
        max_seq_length: int = 2048
    ):
        super().__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.is_causal = is_causal
        self.scale = self.head_dim ** -0.5

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Causal mask (if needed)
        if is_causal:
            self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones(max_seq_length, max_seq_length), diagonal=1).bool()
            )
```

### Forward Pass Walkthrough

```python
def forward(
    self,
    hidden_states: torch.Tensor,  # (B, N, D)
    attention_mask: Optional[torch.Tensor] = None,
    position_bias: Optional[torch.Tensor] = None,
    return_attention: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Args:
        hidden_states: Input sequence (batch, seq_len, hidden_size)
        attention_mask: Optional mask (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
        position_bias: Optional relative position bias (1, num_heads, seq_len, seq_len)
        return_attention: Whether to return attention weights

    Returns:
        output: Context-aware representations (batch, seq_len, hidden_size)
        attention_weights: Optional (batch, num_heads, seq_len, seq_len)
    """
    B, N, D = hidden_states.shape

    # 1. Project to Q, K, V in one go
    qkv = self.qkv_proj(hidden_states)  # (B, N, 3*D)
    qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d)
    q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, N, d)

    # 2. Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

    # 3. Add position bias (e.g., relative position encodings)
    if position_bias is not None:
        attn_scores = attn_scores + position_bias

    # 4. Apply causal mask (for autoregressive)
    if self.is_causal:
        causal_mask = self.causal_mask[:N, :N]
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

    # 5. Apply attention mask (e.g., padding)
    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask

    # 6. Softmax to get attention weights
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_weights = self.dropout(attn_weights)

    # 7. Apply attention to values
    attn_output = torch.matmul(attn_weights, v)  # (B, H, N, d)

    # 8. Concatenate heads and project
    attn_output = attn_output.transpose(1, 2).reshape(B, N, D)  # (B, N, D)
    output = self.out_proj(attn_output)

    if return_attention:
        return output, attn_weights
    return output, None
```

## Code Walkthrough

### Example 1: Basic Self-Attention

```python
from nexus.components.attention import SelfAttention

# Initialize
self_attn = SelfAttention(
    hidden_size=512,
    num_heads=8,
    dropout=0.1,
    is_causal=False  # Bidirectional
)

# Forward pass
x = torch.randn(2, 20, 512)  # (batch, seq_len, hidden_size)
output, attn = self_attn(x, return_attention=True)

print(f"Input: {x.shape}")
print(f"Output: {output.shape}")
print(f"Attention: {attn.shape}")  # (2, 8, 20, 20)
```

### Example 2: Causal Self-Attention (GPT-style)

```python
# Autoregressive language modeling
causal_attn = SelfAttention(
    hidden_size=768,
    num_heads=12,
    is_causal=True,  # Each token only sees past
    max_seq_length=1024
)

# Generate sequence
x = torch.randn(1, 50, 768)
output = causal_attn(x)

# Verify causality: token i doesn't depend on tokens > i
print("Attention pattern is lower triangular")
```

### Example 3: With Padding Mask

```python
# Sentence: "The cat sat <pad> <pad>"
# Need to mask padding tokens

tokens = torch.tensor([[1, 2, 3, 0, 0]])  # 0 = padding
embeddings = embedding_layer(tokens)  # (1, 5, 512)

# Create padding mask
padding_mask = (tokens != 0)  # (1, 5): [True, True, True, False, False]
attention_mask = padding_mask[:, None, None, :]  # (1, 1, 1, 5)
attention_mask = (~attention_mask).float() * -1e9  # Convert to additive mask

# Apply self-attention with mask
output = self_attn(embeddings, attention_mask=attention_mask)

# Padding positions don't influence real tokens
```

### Example 4: BERT-style Encoder Block

```python
class BERTEncoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, ff_dim=3072):
        super().__init__()

        # Bidirectional self-attention
        self.self_attn = SelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            is_causal=False  # Bidirectional
        )

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_size)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        # Self-attention with residual
        x = x + self.self_attn(self.norm1(x), attention_mask=mask)[0]

        # Feed-forward with residual
        x = x + self.ffn(self.norm2(x))

        return x
```

### Example 5: GPT-style Decoder Block

```python
class GPTDecoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()

        # Causal self-attention
        self.self_attn = SelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            is_causal=True  # Causal for autoregressive
        )

        self.ffn = FeedForward(hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Causal self-attention
        x = x + self.self_attn(self.norm1(x))[0]

        # Feed-forward
        x = x + self.ffn(self.norm2(x))

        return x
```

## Optimization Tricks

### 1. Fused QKV Projection

```python
# Inefficient: Three separate matrix multiplications
q = self.q_proj(x)
k = self.k_proj(x)
v = self.v_proj(x)

# Efficient: Single matmul, then split
qkv = self.qkv_proj(x)  # (B, N, 3*D)
q, k, v = qkv.chunk(3, dim=-1)

# Speedup: ~1.3x
```

### 2. Flash Attention Integration

```python
# Use FlashAttention for long sequences
if has_flash_attn and not return_attention:
    output = flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        causal=self.is_causal
    )
else:
    # Standard attention
    output = self._standard_attention(q, k, v)
```

### 3. Pre-Computed Causal Mask

```python
# Inefficient: Create mask every forward pass
causal_mask = torch.triu(torch.ones(N, N), diagonal=1).bool()

# Efficient: Register as buffer (created once)
self.register_buffer("causal_mask", ...)
# Reuse in forward: self.causal_mask[:N, :N]
```

### 4. Attention Dropout Only

```python
# Correct: Drop attention weights, not output
attn_weights = softmax(scores)
attn_weights = dropout(attn_weights)  # Drop here
output = attn_weights @ v

# Incorrect: Dropping output
output = attn_weights @ v
output = dropout(output)  # Wrong place
```

### 5. KV Caching for Generation

```python
# During autoregressive generation
class SelfAttentionWithCache(SelfAttention):
    def forward(self, x, past_kv=None, use_cache=False):
        q, k, v = self.compute_qkv(x)

        if past_kv is not None:
            # Concatenate with cached k, v
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        output = self.compute_attention(q, k, v)

        if use_cache:
            return output, (k, v)
        return output, None
```

### 6. Mixed Precision

```python
# BF16 for better numerical stability in attention
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = self_attn(x)
```

## Experiments & Results

### Language Modeling (GPT-2)

| Model | Params | Context | Perplexity | Attention Type |
|-------|--------|---------|------------|----------------|
| LSTM | 124M | - | 35.8 | N/A |
| GPT-2 Small | 124M | 1024 | 29.4 | Causal Self-Attn |
| GPT-2 Medium | 355M | 1024 | 26.5 | Causal Self-Attn |
| GPT-2 Large | 774M | 1024 | 24.2 | Causal Self-Attn |

Self-attention enables ~20% perplexity improvement over LSTMs.

### Sentence Understanding (BERT)

| Task | LSTM | BERT-Base | Improvement |
|------|------|-----------|-------------|
| SST-2 (Sentiment) | 89.2 | 93.5 | +4.3 |
| CoLA (Grammar) | 45.4 | 60.5 | +15.1 |
| SQuAD (QA F1) | 80.4 | 88.5 | +8.1 |

Bidirectional self-attention crucial for understanding tasks.

### Attention Head Analysis

Study on BERT (Clark et al., 2019):
- **50% of heads** attend to next token
- **30% of heads** attend to previous token
- **10% of heads** attend to specific syntax patterns (e.g., [SEP] tokens)
- **5% of heads** attend to punctuation
- **5% of heads** have diffuse attention patterns

### Pruning Studies

Michel et al. (2019) found:
- Can remove 40% of attention heads with <1% performance drop
- Some heads are redundant
- But remaining heads are crucial

### Context Length Scaling

```
Model: GPT-2 Small
Task: Language Modeling

Context | Perplexity | Training Time
--------|------------|---------------
512     | 31.2       | 1.0x
1024    | 29.4       | 1.9x
2048    | 28.1       | 3.7x
4096    | 27.3       | 7.2x

Time scales as O(n²) due to attention
```

## Common Pitfalls

### 1. Forgetting Position Information

```python
# Wrong: Self-attention is position-invariant
output = self_attn(embeddings)  # Can't distinguish "cat sat" from "sat cat"!

# Correct: Add positional encodings
embeddings = embeddings + positional_encoding
output = self_attn(embeddings)
```

### 2. Incorrect Causal Masking

```python
# Wrong: Upper triangular instead of lower
mask = torch.triu(torch.ones(N, N), diagonal=0).bool()  # Masks diagonal!

# Correct: Mask upper triangle, keep lower + diagonal
mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
```

### 3. Broadcasting Errors with Masks

```python
# Wrong: Mask shape doesn't broadcast
mask = torch.ones(N, N)  # (N, N)
attn_scores = attn_scores + mask  # Error! attn_scores is (B, H, N, N)

# Correct: Add batch and head dimensions
mask = mask[None, None, :, :]  # (1, 1, N, N)
```

### 4. Not Handling Variable Lengths

```python
# Wrong: Padding affects attention
batch = ["The cat sat", "Hello", "A"]  # Different lengths
# Padded: [3, 1, 1] → [3, 1, 1, <pad>, <pad>]
output = self_attn(padded_embeddings)  # Padding contaminates representations!

# Correct: Use padding mask
padding_mask = create_padding_mask(lengths)
output = self_attn(padded_embeddings, attention_mask=padding_mask)
```

### 5. Memory Issues with Long Sequences

```python
# Wrong: OOM with long sequences
x = torch.randn(1, 32000, 768)  # 32K tokens
output = self_attn(x)  # OOM! Attention matrix is 32K×32K

# Correct: Use efficient attention
from nexus.components.attention import FlashAttention
flash_attn = FlashAttention(...)
output = flash_attn(x)  # Memory-efficient
```

### 6. Mixing Bidirectional and Causal

```python
# Wrong: Using bidirectional attention for generation
attn = SelfAttention(is_causal=False)
# During generation, model sees future tokens!

# Correct: Causal for generation
attn = SelfAttention(is_causal=True)
```

## References

### Original Papers

1. **Attention Is All You Need**
   Vaswani, A., et al. (2017)
   NeurIPS 2017
   [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

2. **BERT: Pre-training of Deep Bidirectional Transformers**
   Devlin, J., et al. (2018)
   NAACL 2019
   [arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
   Bidirectional self-attention

3. **Improving Language Understanding by Generative Pre-Training (GPT)**
   Radford, A., et al. (2018)
   Causal self-attention for generation

### Analysis Papers

4. **What Does BERT Look At?**
   Clark, K., et al. (2019)
   ACL 2019
   Attention pattern analysis

5. **Are Sixteen Heads Really Better than One?**
   Michel, P., Levy, O., & Neubig, G. (2019)
   NeurIPS 2019
   Head pruning study

6. **Analyzing Multi-Head Self-Attention**
   Voita, E., et al. (2019)
   ACL 2019
   Head specialization

### Efficient Variants

7. **FlashAttention: Fast and Memory-Efficient Exact Attention**
   Dao, T., et al. (2022)
   NeurIPS 2022

8. **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention**
   Katharopoulos, A., et al. (2020)
   ICML 2020

### Related Mechanisms

- [Multi-Head Attention](./multi_head_attention.md) - Multi-head variant
- [Cross Attention](./cross_attention.md) - Attend to different sequence
- [Flash Attention](./flash_attention.md) - Memory-efficient implementation
- [Sparse Attention](./sparse_attention.md) - Reduce O(n²) complexity

## See Also

- **Implementation**: `Nexus/nexus/components/attention/self_attention.py`
- **Base Class**: `Nexus/nexus/components/attention/base_attention.py`
- **Positional Encodings**: See `docs/05_positional_encodings/` for position information methods
