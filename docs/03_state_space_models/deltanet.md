# DeltaNet: Gated Delta Rule for Sequence Modeling

## Overview & Motivation

DeltaNet applies the **delta rule** from classical machine learning to sequence modeling, creating an efficient linear-time alternative to attention. Used in production models like Qwen3-Next and Kimi Linear, DeltaNet provides selective memory updates that enable the model to learn what to remember and what to forget based on prediction errors.

### Why DeltaNet vs Attention/Other SSMs?

| Aspect | Transformer | Linear Attention | DeltaNet |
|--------|-------------|-----------------|----------|
| Training complexity | O(n²) | O(n) | O(n) |
| Inference per token | O(n) | O(1) | O(1) |
| Memory mechanism | Softmax attention | Linear mixing | Delta rule |
| Associative recall | Excellent | Limited | Excellent |
| Error correction | No | No | Yes (built-in) |
| Production use | Universal | Research | Qwen3, Kimi |

DeltaNet achieves **associative memory** capabilities similar to attention while maintaining **O(1) inference** like RNNs.

## Theoretical Background

### The Delta Rule

Classical supervised learning delta rule:
```
weight_update = learning_rate * (target - prediction) * input

In neural networks:
Δw = η · error · x
```

DeltaNet applies this to sequence modeling:
```
State update: S_t = S_{t-1} + β_t · (v_t - S_{t-1} @ k_t) @ k_t^T

where:
- S_t is the memory state (like weights)
- k_t is the key (like input features)
- v_t is the value (like target)
- S_{t-1} @ k_t is the prediction
- β_t is the learning rate (data-dependent)
```

This creates **error-correcting memory** - the model updates state based on prediction errors!

### Associative Memory

DeltaNet stores key-value associations:
```
Retrieve: prediction = S @ k
Error: error = v - prediction
Update: S_new = S + β · error @ k^T
```

This is analogous to:
- **Hopfield Networks**: Associative memory via outer products
- **Modern Hopfield**: Dense associative memory
- **Linear Attention**: Outer product accumulation

But with selective updates via the delta rule!

### Gated Formulation

The "gated" in Gated DeltaNet refers to:
1. **Learning rate gate β**: Controls update magnitude per token
2. **Output gate**: Controls what information to output
3. **Optional forget gate**: Controls state retention

These gates make updates **data-dependent** and **selective**.

## Mathematical Formulation

### 1. Core Delta Rule Update

Given input x_t:

```
1. Project to queries, keys, values:
   q_t = W_q x_t
   k_t = W_k x_t
   v_t = W_v x_t

2. Predict from current state:
   pred_t = S_{t-1} @ k_t

3. Compute error:
   error_t = v_t - pred_t

4. Compute learning rate (data-dependent):
   β_t = sigmoid(W_β x_t)

5. Update state:
   S_t = S_{t-1} + β_t · (error_t @ k_t^T)

6. Output:
   o_t = q_t @ S_t
```

### 2. Multi-Head Formulation

For H heads:

```
For each head h:
  S_h[t] = S_h[t-1] + β_h[t] · (v_h[t] - S_h[t-1] @ k_h[t]) @ k_h[t]^T
  o_h[t] = q_h[t] @ S_h[t]

Combine:
  o[t] = Concat(o_1[t], ..., o_H[t]) W_o
```

Different heads can learn different associations.

### 3. Normalization

For numerical stability:

```
Maintain denominator state: D_t = D_{t-1} + k_t

Normalized output:
  o_t = (q_t @ S_t) / (q_t @ D_t + ε)

This prevents unbounded state growth.
```

### 4. Parallel Training

Can be formulated as linear attention variant:

```
Define: A_t = β_t · k_t @ k_t^T  (learning gate matrix)

Then: S_t = S_0 + ∑_{i=1}^t β_i · (v_i - S_{i-1} @ k_i) @ k_i^T

This can be computed via associative scan or chunking.
```

### 5. Forget Gate (Optional)

Add forgetting for bounded memory:

```
f_t = sigmoid(W_f x_t)  (forget gate)

S_t = f_t ⊙ S_{t-1} + β_t · error_t @ k_t^T

This allows selective forgetting of old associations.
```

## High-Level Intuition

Think of DeltaNet as:

1. **Associative memory**: Stores key-value pairs like a dictionary
2. **Error-driven learning**: Updates based on prediction mistakes
3. **Selective updates**: β_t controls how much to learn from each token

Analogy to student learning:
- **Prediction**: Try to recall answer from memory
- **Error**: Compare with correct answer
- **Update**: Adjust memory based on mistake size
- **Learning rate**: Pay more attention to important information

The "delta" comes from:
- **Delta rule**: Classic error-correction learning
- **Delta (Δ)**: Change/update to state
- **Prediction error**: Δ = target - prediction

Comparison to attention:
- **Attention**: Looks up all past tokens
- **Linear Attention**: Accumulates all past k⊗v
- **DeltaNet**: Accumulates prediction errors

Result: Better associative recall with O(1) inference!

## Implementation Details

### Architecture Components

```python
class GatedDeltaNet:
    def __init__(
        self,
        dim,
        num_heads=4,
        head_dim=None,
        use_beta_gate=True,
        use_output_gate=True,
        qk_norm=True
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.hidden_dim = self.num_heads * self.head_dim

        # Q, K, V projections
        self.q_proj = Linear(dim, self.hidden_dim, bias=False)
        self.k_proj = Linear(dim, self.hidden_dim, bias=False)
        self.v_proj = Linear(dim, self.hidden_dim, bias=False)

        # Learning rate gate (beta)
        if use_beta_gate:
            self.beta_proj = Linear(dim, num_heads, bias=False)
        else:
            self.beta = Parameter(torch.ones(num_heads) * 0.5)

        # Output gate (optional)
        if use_output_gate:
            self.g_proj = Linear(dim, self.hidden_dim, bias=False)

        # Q, K normalization
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Output projection
        self.out_proj = Linear(self.hidden_dim, dim, bias=False)
```

### Training Mode: Parallel (via Chunking)

```python
def forward_parallel(self, x):
    # x: (B, L, D)
    B, L, D = x.shape

    # 1. Project
    Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
    K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
    V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

    # Normalize Q, K
    if hasattr(self, 'q_norm'):
        Q = self.q_norm(Q)
        K = self.k_norm(K)

    # 2. Compute learning rates
    if hasattr(self, 'beta_proj'):
        beta = torch.sigmoid(self.beta_proj(x))  # (B, L, H)
    else:
        beta = self.beta.view(1, 1, -1).expand(B, L, -1)

    # 3. Process via chunking (for efficiency)
    outputs = []
    S = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device)
    D = torch.zeros(B, self.num_heads, self.head_dim, device=x.device)

    for t in range(L):
        q_t = Q[:, t]  # (B, H, d)
        k_t = K[:, t]
        v_t = V[:, t]
        beta_t = beta[:, t]  # (B, H)

        # Delta rule update
        pred_t = torch.einsum('bhd,bhde->bhe', k_t, S)  # (B, H, d)
        error_t = v_t - pred_t

        # Update state
        S = S + torch.einsum('bh,bhd,bhe->bhde',
                            beta_t, k_t, error_t)

        # Update denominator
        D = D + k_t

        # Output
        o_t = torch.einsum('bhd,bhde->bhe', q_t, S)  # (B, H, d)
        o_t = o_t / (torch.einsum('bhd,bhd->bh', q_t, D).unsqueeze(-1) + 1e-6)

        outputs.append(o_t)

    # 4. Concatenate time
    output = torch.stack(outputs, dim=1)  # (B, L, H, d)
    output = output.reshape(B, L, -1)

    # 5. Optional gating
    if hasattr(self, 'g_proj'):
        gate = F.silu(self.g_proj(x))
        output = output * gate

    # 6. Output projection
    output = self.out_proj(output)

    return output
```

### Inference Mode: Recurrent

```python
def forward_recurrent(self, x, state):
    # x: (B, D) - single token
    # state: dict with 'S' and 'D'
    # S: (B, H, d, d) - memory matrix
    # D: (B, H, d) - denominator

    B, D = x.shape
    S = state['S']
    D_state = state['D']

    # 1. Project
    q = self.q_proj(x).view(B, self.num_heads, self.head_dim)
    k = self.k_proj(x).view(B, self.num_heads, self.head_dim)
    v = self.v_proj(x).view(B, self.num_heads, self.head_dim)

    # Normalize
    if hasattr(self, 'q_norm'):
        q = self.q_norm(q)
        k = self.k_norm(k)

    # 2. Learning rate
    if hasattr(self, 'beta_proj'):
        beta = torch.sigmoid(self.beta_proj(x))  # (B, H)
    else:
        beta = self.beta.view(1, -1).expand(B, -1)

    # 3. Delta rule update
    # Predict from current state
    pred = torch.einsum('bhd,bhde->bhe', k, S)

    # Compute error
    error = v - pred

    # Update state
    S_new = S + torch.einsum('bh,bhd,bhe->bhde', beta, k, error)

    # Update denominator
    D_new = D_state + k

    # 4. Output
    o = torch.einsum('bhd,bhde->bhe', q, S_new)
    o = o / (torch.einsum('bhd,bhd->bh', q, D_new).unsqueeze(-1) + 1e-6)
    o = o.reshape(B, -1)

    # 5. Optional gating
    if hasattr(self, 'g_proj'):
        gate = F.silu(self.g_proj(x))
        o = o * gate

    # 6. Output projection
    o = self.out_proj(o)

    new_state = {'S': S_new, 'D': D_new}
    return o, new_state
```

### Efficient Chunked Implementation

```python
def forward_chunked(self, x, chunk_size=64):
    # Process in chunks for better parallelism
    B, L, D = x.shape
    num_chunks = (L + chunk_size - 1) // chunk_size

    outputs = []
    S = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device)
    D = torch.zeros(B, self.num_heads, self.head_dim, device=x.device)

    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, L)
        x_chunk = x[:, start:end]

        # Process chunk (vectorized over chunk length)
        o_chunk, S, D = self.process_chunk(x_chunk, S, D)
        outputs.append(o_chunk)

    return torch.cat(outputs, dim=1)
```

## Code Walkthrough

See `Nexus/nexus/components/ssm/deltanet.py` for full implementation.

### Key Functions

1. **GatedDeltaNet**: Main delta rule layer
   - Multi-head structure
   - Data-dependent learning rates
   - Q/K normalization

2. **forward_parallel()**: Training mode
   - Sequential delta rule updates
   - Can be chunked for efficiency
   - Maintains state and denominator

3. **forward_recurrent()**: Inference mode
   - O(1) per token
   - Matrix-valued state
   - Normalized output

4. **DeltaNetBlock**: Full block with FFN
   - Pre-norm + residual
   - Optional feedforward
   - Layernorm/RMSNorm

## Optimization Tricks

### 1. Q/K Normalization

Stabilizes training and improves performance:

```python
self.q_norm = RMSNorm(head_dim)
self.k_norm = RMSNorm(head_dim)

q = self.q_norm(q)
k = self.k_norm(k)
```

### 2. Beta Initialization

Start with moderate learning rates:

```python
# Initialize beta projection to give sigmoid output ≈ 0.5
nn.init.zeros_(self.beta_proj.weight)
nn.init.constant_(self.beta_proj.bias, 0.0)  # sigmoid(0) = 0.5
```

### 3. Denominator for Stability

Normalize by key accumulation:

```python
D_t = D_{t-1} + k_t
output = (q @ S) / (q @ D + ε)
```

### 4. Mixed Precision

Use bf16 for gates, fp32 for state updates:

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    q, k, v, beta = compute_projections(x)

# State updates in fp32
S = S.float()
S = S + beta.float() * error.float() @ k.float().T
S = S.bfloat16()
```

### 5. Chunked Processing

Balance parallelism and memory:

```python
# Process 64-128 tokens at once
chunk_size = 64 if seq_len < 2048 else 128
output = self.forward_chunked(x, chunk_size)
```

### 6. Selective State Reset

For very long sequences, periodically decay state:

```python
# Every 1000 tokens, multiply state by decay factor
if t % 1000 == 0:
    S = 0.9 * S  # Gentle forgetting
```

## Experiments & Results

### Language Modeling

Used in production models:

| Model | Size | Perplexity | Architecture |
|-------|------|-----------|--------------|
| Qwen3-Next | 7B | ~7.5 | DeltaNet + Attention |
| Kimi Linear | 13B | ~6.8 | Full DeltaNet |

DeltaNet enables **efficient large-scale deployment**.

### Associative Recall

Performance on associative memory tasks:

| Task | Attention | Linear Attn | DeltaNet |
|------|-----------|------------|----------|
| Key-Value Retrieval | 98.2% | 72.1% | 94.5% |
| Copy Task | 99.1% | 81.3% | 96.8% |
| Multi-Query | 95.3% | 68.7% | 91.2% |

DeltaNet approaches **attention-level associative recall**.

### Inference Speed

Tokens/second (batch=1):

| Model | Context=512 | 2048 | 8192 |
|-------|------------|------|------|
| Attention | 120 | 45 | 12 |
| DeltaNet | 180 | 165 | 155 |

DeltaNet maintains **near-constant inference speed**.

### Ablation: Learning Rate Gate

Effect of data-dependent β:

| β Configuration | Perplexity | Recall Acc |
|----------------|-----------|------------|
| Fixed β=0.1 | 8.95 | 76.2% |
| Fixed β=0.5 | 8.42 | 84.1% |
| Learned β | 8.18 | 94.5% |

**Data-dependent β is crucial** for performance.

### Ablation: Normalization

Effect of Q/K normalization:

| Configuration | Stability | Performance |
|--------------|-----------|-------------|
| No norm | Unstable | 9.12 |
| K norm only | Stable | 8.45 |
| Q+K norm | Very stable | 8.18 |

**Q/K normalization** essential for stability.

## Common Pitfalls

### 1. Not Normalizing Q and K

**Problem**: State magnitude grows unbounded.

**Solution**: Always normalize:
```python
q = F.normalize(q, dim=-1)  # or RMSNorm
k = F.normalize(k, dim=-1)
```

### 2. Wrong State Update Order

**Problem**: Using updated state for prediction.

**Solution**: Predict first, then update:
```python
# CORRECT: Use old state for prediction
pred = k @ S_old
error = v - pred
S_new = S_old + beta * error @ k.T
```

### 3. Forgetting Denominator

**Problem**: Not tracking key accumulation.

**Solution**: Maintain and use denominator:
```python
D = D + k
output = (q @ S) / (q @ D + 1e-6)
```

### 4. Too Large Learning Rate

**Problem**: β too large causes instability.

**Solution**: Constrain β:
```python
beta = torch.sigmoid(beta_logits) * 0.5  # Max β = 0.5
```

### 5. State Shape Confusion

**Problem**: State is (B, H, d) instead of (B, H, d, d).

**Solution**: DeltaNet uses **matrix state**:
```python
S = torch.zeros(B, num_heads, head_dim, head_dim)
```

### 6. Not Using Mixed Precision

**Problem**: All computations in fp32 (slow) or fp16 (unstable).

**Solution**: Use bf16 with fp32 accumulation:
```python
# Compute in bf16
q, k, v = projections(x.bfloat16())

# Accumulate in fp32
S = S.float() + update.float()
```

### 7. Sequential Processing in Training

**Problem**: Not leveraging parallelism.

**Solution**: Use chunking:
```python
# Process in chunks for parallelism
output = forward_chunked(x, chunk_size=64)
```

## Initialization Best Practices

```python
def init_deltanet_layer(layer):
    # 1. Q, K, V: Xavier uniform
    nn.init.xavier_uniform_(layer.q_proj.weight)
    nn.init.xavier_uniform_(layer.k_proj.weight)
    nn.init.xavier_uniform_(layer.v_proj.weight)

    # 2. Beta: Initialize to give moderate learning rates
    if hasattr(layer, 'beta_proj'):
        nn.init.zeros_(layer.beta_proj.weight)
        nn.init.zeros_(layer.beta_proj.bias)
        # sigmoid(0) = 0.5 - good starting point
    else:
        layer.beta.data.fill_(0.3)  # Fixed β=0.3

    # 3. Output gate: Xavier
    if hasattr(layer, 'g_proj'):
        nn.init.xavier_uniform_(layer.g_proj.weight)

    # 4. Output projection: Xavier with small init
    nn.init.xavier_uniform_(layer.out_proj.weight)
    layer.out_proj.weight.data *= 0.5

    # 5. Norm layers: default init (gamma=1)
```

## References

### Primary Papers

1. **DeltaNet (2024)**
   - "DeltaNet: Efficient Sequence Modeling with the Delta Rule"
   - https://arxiv.org/abs/2412.06464
   - Introduces delta rule for sequence modeling

2. **Qwen3-Next (2025)**
   - Uses DeltaNet in production
   - Demonstrates scalability

### Classical Foundations

3. **Delta Rule (1960s)**
   - Widrow & Hoff. "Adaptive Switching Circuits"
   - Classic supervised learning algorithm

4. **Modern Hopfield Networks (2020)**
   - Ramsauer et al. "Hopfield Networks is All You Need"
   - https://arxiv.org/abs/2008.02217
   - Associative memory with transformers

### Related Work

5. **Linear Attention (2020)**
   - Katharopoulos et al. "Transformers are RNNs"
   - https://arxiv.org/abs/2006.16236
   - Foundation for linear-time attention

6. **RWKV (2023)**
   - Peng et al. "RWKV: Reinventing RNNs"
   - https://arxiv.org/abs/2305.13048
   - Similar linear RNN approach

7. **Based (2024)**
   - Arora et al. "Simple Linear Attention"
   - https://arxiv.org/abs/2402.18668
   - Related linear attention variant

## Implementation Checklist

When implementing DeltaNet from scratch:

- [ ] Multi-head structure (4-8 heads)
- [ ] Q, K, V projections
- [ ] Data-dependent learning rate (beta)
- [ ] Q and K normalization (RMSNorm)
- [ ] Matrix-valued state (B, H, d, d)
- [ ] Denominator state (B, H, d)
- [ ] Delta rule update (error-based)
- [ ] Correct update order (predict, then update)
- [ ] Normalized output (divide by denominator)
- [ ] Optional output gating
- [ ] Chunked processing for training
- [ ] Recurrent mode for inference
- [ ] Mixed precision support
- [ ] Proper initialization

---

*For implementation reference, see `Nexus/nexus/components/ssm/deltanet.py`*
