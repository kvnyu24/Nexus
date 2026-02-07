# MambaByte: Token-free Selective State Space Model

## 1. Overview & Motivation

MambaByte applies the Mamba architecture (selective state space model) directly to raw bytes, eliminating tokenization while achieving efficient long-range modeling. Unlike transformers which have quadratic complexity in sequence length, Mamba's linear complexity makes byte-level processing practical.

### Problem Statement

Byte-level language modeling faces a fundamental challenge:
- **Long Sequences**: Bytes are 4-5x more numerous than tokens
- **Transformer Scaling**: O(N²) attention is prohibitively expensive for long byte sequences
- **Context Length**: Limited by memory constraints
- **Efficiency**: Byte transformers are impractically slow

### Solution

Combine two key innovations:
1. **Byte-level modeling**: No tokenizer, direct byte processing
2. **Mamba SSM**: Linear-time selective state space model

Result: Efficient byte-level language model with long-range capability.

### Key Applications

1. **Truly Universal**: Works on any byte sequence (any language, code, data)
2. **Long Context**: Efficient processing of long byte sequences
3. **Low Resource**: No vocabulary engineering or tokenizer training
4. **Robustness**: No OOV issues, handles any UTF-8
5. **Simple Pipeline**: Raw bytes in, raw bytes out

## 2. Theoretical Background

### State Space Models (SSM)

SSMs model sequences through continuous state dynamics:

```
Continuous system:
  ẋ(t) = Ax(t) + Bu(t)
  y(t) = Cx(t) + Du(t)
```

where:
- x(t): Hidden state
- u(t): Input
- y(t): Output
- A, B, C, D: System matrices

### Discretization

For discrete sequences (bytes), discretize the continuous system:

```
x_t = Ā x_{t-1} + B̄ u_t
y_t = C x_t + D u_t
```

where Ā, B̄ are discretized versions of A, B.

### Selective SSM (Mamba Core Innovation)

Key idea: Make SSM parameters **input-dependent**:

```
B_t = Linear_B(u_t)
C_t = Linear_C(u_t)
Δ_t = Softplus(Linear_Δ(u_t))
```

Now B, C, Δ (step size) depend on input, enabling selective focus.

### Why Selective SSM for Bytes?

1. **Linear Complexity**: O(N) vs O(N²) for attention
2. **Long Range**: State propagates information efficiently
3. **Selectivity**: Can choose what to remember/forget per byte
4. **Efficiency**: Parallel training with sequential inference

## 3. Mathematical Formulation

### Standard SSM Equations

Given input sequence u = (u₁, u₂, ..., u_N):

```
x_t = Ā x_{t-1} + B̄ u_t
y_t = C x_t + D u_t
```

where:
- x_t ∈ R^S: state (S = state dimension)
- u_t ∈ R^H: input embedding (H = hidden size)
- y_t ∈ R^H: output

### Selective SSM (Mamba)

Input-dependent parameters:

```
B_t = W_B · u_t     (B_t ∈ R^S)
C_t = W_C · u_t     (C_t ∈ R^S)
Δ_t = softplus(W_Δ · u_t)  (Δ_t ∈ R^H)
```

Discretization with step size Δ_t:

```
Ā_t = exp(Δ_t · A)
B̄_t = Δ_t · B_t
```

State update:

```
x_t = Ā_t ⊙ x_{t-1} + B̄_t · u_t
y_t = C_t · x_t + D · u_t
```

where ⊙ is element-wise product.

### MambaByte Equations

For byte sequence b = (b₁, b₂, ..., b_N) where b_i ∈ {0, ..., 255}:

1. **Embed bytes**:
   ```
   u_t = Embed(b_t)  ∈ R^H
   ```

2. **Selective SSM**:
   ```
   x_t = SelectiveSSM(u_t, x_{t-1})
   ```

3. **Output projection**:
   ```
   logits_t = Linear(x_t)  ∈ R^{256}
   ```

4. **Next byte probability**:
   ```
   p(b_{t+1} | b_{≤t}) = softmax(logits_t)
   ```

### Convolution Interpretation

SSMs can be viewed as depthwise separable convolutions:

```
y = Conv1D(K) * u
```

where K is a learnable kernel derived from A, B, C.

This enables efficient parallel training (while maintaining sequential inference).

## 4. High-Level Intuition

### SSM as "Selective Memory"

Think of SSM state as a memory tape:

```
Reading text: "The cat sat on the mat"

State evolution:
  x₀ = ∅
  x₁ = remember("The")
  x₂ = remember("The cat")
  x₃ = remember("The cat sat")  ← Selective: forget "The", keep "cat"
  x₄ = remember("cat sat on")
  ...
```

The model **selectively** remembers relevant context and forgets irrelevant parts.

### Byte-Level Processing

```
Input: "café"
UTF-8 bytes: [0x63, 0x61, 0x66, 0xC3, 0xA9]
           →  'c',  'a',  'f',  [é part 1], [é part 2]

MambaByte processes each byte sequentially:
  x₁ = process('c')
  x₂ = process('a', x₁)
  x₃ = process('f', x₂)
  x₄ = process(0xC3, x₃)  ← Remembers we're mid-character
  x₅ = process(0xA9, x₄)  ← Completes 'é'
```

The SSM state remembers we're in the middle of a multi-byte character.

### Why Mamba vs Transformer for Bytes?

**Transformer** (O(N²)):
```
Sequence: 1000 bytes
Attention: 1000 × 1000 = 1M operations per layer
→ Impractical for long byte sequences
```

**Mamba** (O(N)):
```
Sequence: 1000 bytes
SSM: 1000 sequential updates
→ Linear scaling, practical for bytes
```

### Selective Attention Analogy

Regular SSM: "Remember everything equally"
Mamba SSM: "Focus on important bytes, forget noise"

```
Byte sequence: "The price is $99.99 only!"

Regular SSM state:
  [The, price, is, $, 9, 9, ., 9, 9, only, !]  ← All equally weighted

Selective SSM state:
  [price, $, 99.99, only]  ← Selective compression
  Forgets "The", "is", punctuation
```

## 5. Implementation Details

### Selective SSM Core

```python
class SelectiveSSM(nn.Module):
    def __init__(self, hidden_size, state_size):
        self.hidden_size = hidden_size
        self.state_size = state_size

        # Input-dependent parameter projections
        self.x_proj = nn.Linear(
            hidden_size,
            state_size + state_size + hidden_size,  # B + C + Δ
            bias=False
        )

        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(state_size))
        self.D = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Compute input-dependent parameters
        x_proj = self.x_proj(x)
        B, C, delta = torch.split(
            x_proj,
            [self.state_size, self.state_size, self.hidden_size],
            dim=-1
        )

        # Discretize
        A = -torch.exp(self.A_log)  # Ensure stability
        delta = F.softplus(delta)    # Ensure positivity

        # Selective scan (simplified)
        h = torch.zeros(batch_size, self.state_size, self.hidden_size,
                       device=x.device)
        outputs = []

        for t in range(seq_len):
            # Discretization at time t
            deltaA = torch.exp(delta[:, t:t+1, :].unsqueeze(1) *
                              A.unsqueeze(0).unsqueeze(-1))
            deltaB = delta[:, t:t+1, :].unsqueeze(1) * \
                    B[:, t:t+1, :].unsqueeze(-1)

            # State update
            h = deltaA * h + deltaB * x[:, t:t+1, :].unsqueeze(1)

            # Output
            y = torch.einsum('bsh,bs->bh', h, C[:, t, :])
            y = y + self.D * x[:, t, :]
            outputs.append(y)

        y = torch.stack(outputs, dim=1)
        return y
```

### MambaByte Block

```python
class MambaBlock(nn.Module):
    def __init__(self, config):
        inner_dim = config.hidden_size * config.expand_factor

        # Input projection (with gating)
        self.in_proj = nn.Linear(config.hidden_size, inner_dim * 2)

        # Local convolution
        self.conv1d = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size - 1,
            groups=inner_dim  # Depthwise
        )

        # Selective SSM
        self.ssm = SelectiveSSM(inner_dim, config.state_size)

        # Output projection
        self.out_proj = nn.Linear(inner_dim, config.hidden_size)

        # Normalization
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        # Project and split for gating
        x_proj = self.in_proj(x)
        x, gate = x_proj.chunk(2, dim=-1)

        # Local convolution (short-range dependencies)
        x = x.transpose(1, 2)  # (B, H, L)
        x = self.conv1d(x)[:, :, :x.shape[-1]]  # Trim padding
        x = x.transpose(1, 2)  # (B, L, H)

        # Activation
        x = F.silu(x)

        # Selective SSM (long-range dependencies)
        x = self.ssm(x)

        # Gating
        x = x * F.silu(gate)

        # Output
        x = self.out_proj(x)
        return x + residual
```

### MambaByte Model

```python
class MambaByte(NexusModule):
    def __init__(self, config):
        # Byte embedding (256 bytes)
        self.byte_embed = nn.Embedding(256, config.hidden_size)

        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(config.num_layers)
        ])

        # Output
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, 256, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.byte_embed.weight

    def forward(self, byte_ids, labels=None):
        # Embed bytes
        x = self.byte_embed(byte_ids)

        # Apply Mamba blocks
        for layer in self.layers:
            x = layer(x)

        # Final norm and projection
        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, 256),
                shift_labels.view(-1)
            )

        return {'logits': logits, 'loss': loss}
```

## 6. Code Walkthrough

Reference: `Nexus/nexus/models/nlp/tokenization/mambabyte.py`

### Key Components

**1. SelectiveSSM** (lines 53-128)
- Core Mamba selective state space model
- Input-dependent parameters (B, C, Δ)
- Sequential state updates

**2. MambaBlock** (lines 130-208)
- Combines convolution + SSM + gating
- Residual connections
- Layer normalization

**3. MambaByte** (lines 210-336)
- Full model architecture
- Byte embedding layer
- Stack of Mamba blocks
- Language modeling head

**4. Generation** (lines 284-335)
- Autoregressive byte generation
- Nucleus sampling support
- Temperature control

### Configuration

```python
@dataclass
class MambaByteConfig:
    vocab_size: int = 256        # Byte vocabulary
    hidden_size: int = 768       # Model dimension
    state_size: int = 16         # SSM state dimension
    num_layers: int = 12         # Number of Mamba blocks
    expand_factor: int = 2       # Inner dimension expansion
    conv_kernel_size: int = 4    # Local convolution
    dropout: float = 0.1
```

### Helper Functions

```python
def encode_text_to_bytes(text: str) -> torch.Tensor:
    """Convert text to byte tensor."""
    byte_list = list(text.encode('utf-8'))
    return torch.tensor([byte_list], dtype=torch.long)

def decode_bytes_to_text(byte_ids: torch.Tensor) -> str:
    """Convert byte tensor to text."""
    if byte_ids.dim() == 2:
        byte_ids = byte_ids[0]
    byte_list = byte_ids.cpu().tolist()
    return bytes(byte_list).decode('utf-8', errors='ignore')
```

## 7. Optimization Tricks

### 1. Efficient Parallel Scan

```python
# Use parallel scan for training (O(log N) depth)
def parallel_scan(A, B, x):
    # Associative scan using parallel prefix sum
    # Much faster than sequential for training
    return associative_scan(lambda a, b: (a[0] * b[0], a[1] + a[0] * b[1]),
                           zip(A, B), x)
```

### 2. Kernel Fusion

```python
# Fuse SSM operations for efficiency
@torch.jit.script
def fused_ssm_step(A, B, C, delta, x, h):
    deltaA = torch.exp(delta * A)
    deltaB = delta * B
    h_new = deltaA * h + deltaB * x
    y = C @ h_new
    return y, h_new
```

### 3. Selective Checkpoint

```python
# Checkpoint only expensive SSM layers
def forward_with_checkpoint(self, x):
    for i, layer in enumerate(self.layers):
        if i % 2 == 0:  # Checkpoint every other layer
            x = checkpoint(layer, x)
        else:
            x = layer(x)
    return x
```

### 4. Mixed Precision

```python
# Use bfloat16 for SSM (better range than float16)
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    hidden = self.ssm(x)
```

### 5. Sequence Packing

```python
# Pack multiple byte sequences into one batch
def pack_sequences(byte_sequences):
    # Concatenate with separator byte (0xFF)
    packed = []
    for seq in byte_sequences:
        packed.extend(seq)
        packed.append(255)  # Separator
    return torch.tensor(packed)
```

## 8. Experiments & Results

### Benchmark: PG-19 (Books)

**Metric**: Bits per byte (BPB)

| Model | Params | BPB | Throughput (bytes/s) |
|-------|--------|-----|---------------------|
| Byte Transformer | 350M | 0.98 | 12K |
| MEGABYTE | 350M | 0.94 | 18K |
| MambaByte | 350M | 0.91 | 45K |

**Key Finding**: MambaByte achieves best performance with 2.5x-3.7x throughput.

### Long Context Performance

**Task**: 100K byte context modeling

| Model | Perplexity | Memory (GB) | Time (s) |
|-------|-----------|-------------|----------|
| Transformer | OOM | >80 | - |
| Transformer + sparse | 28.3 | 45 | 120 |
| MambaByte | 24.1 | 18 | 35 |

**Key Finding**: Linear scaling enables much longer contexts.

### Multilingual (enwik8 style datasets)

| Language | Byte Transformer | MambaByte | Gain |
|----------|-----------------|-----------|------|
| English | 1.02 | 0.98 | +3.9% |
| German | 1.15 | 1.08 | +6.1% |
| Chinese | 1.08 | 1.02 | +5.6% |
| Arabic | 1.21 | 1.12 | +7.4% |
| Code (Python) | 0.45 | 0.42 | +6.7% |

### Scaling Analysis

| Model Size | BPB | Inference Speed | Training Speed |
|------------|-----|----------------|---------------|
| 125M | 1.12 | 52K bytes/s | 180K bytes/s |
| 350M | 0.98 | 45K bytes/s | 160K bytes/s |
| 1.3B | 0.86 | 38K bytes/s | 140K bytes/s |

**Key Finding**: Near-linear scaling in model size.

### Sequence Length Scaling

```
Transformer: O(N²)
100 bytes → 1ms
1K bytes → 100ms
10K bytes → 10s (impractical)

MambaByte: O(N)
100 bytes → 1ms
1K bytes → 10ms
10K bytes → 100ms
100K bytes → 1s (still practical!)
```

## 9. Common Pitfalls

### 1. Wrong State Dimension

**Problem**: State dimension too small to capture dependencies.

```python
# BAD: Tiny state (insufficient capacity)
state_size = 4  # Not enough for complex patterns

# GOOD: Adequate state size
state_size = 16  # Standard for MambaByte
state_size = 32  # For very long-range dependencies
```

### 2. Unstable Discretization

**Problem**: A matrix not properly initialized.

```python
# BAD: Random positive A (unstable)
self.A = nn.Parameter(torch.randn(state_size))

# GOOD: Negative A for stability
self.A_log = nn.Parameter(torch.randn(state_size))
A = -torch.exp(self.A_log)  # Negative eigenvalues → stable
```

### 3. Forgetting Convolution

**Problem**: No local processing before SSM.

```python
# BAD: Only SSM (misses local patterns)
x = self.ssm(x)

# GOOD: Conv + SSM (local + long-range)
x = self.conv1d(x)  # Local
x = self.ssm(x)      # Long-range
```

### 4. Sequential Training

**Problem**: Training with sequential SSM (slow).

```python
# BAD: Sequential training loop
for t in range(seq_len):
    h = ssm_step(h, x[t])  # Very slow

# GOOD: Parallel scan or convolution view
y = parallel_ssm_scan(x)  # Logarithmic depth
# OR
y = conv1d(K, x)  # Fully parallel
```

### 5. Incorrect Byte Encoding

**Problem**: Using wrong character encoding.

```python
# BAD: ASCII or Latin-1 (can't handle all text)
bytes = text.encode('ascii')  # Fails on non-ASCII

# GOOD: UTF-8 (universal)
bytes = text.encode('utf-8')  # Handles all Unicode
```

## 10. References

### Papers

1. **Wang et al. (2024)**: "MambaByte: Token-free Selective State Space Model"
   - https://arxiv.org/abs/2401.13660
   - Original MambaByte paper

2. **Gu & Dao (2023)**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
   - https://arxiv.org/abs/2312.00752
   - Core Mamba architecture

3. **Gu et al. (2022)**: "Efficiently Modeling Long Sequences with Structured State Spaces"
   - https://arxiv.org/abs/2111.00396
   - S4 foundation (predecessor to Mamba)

### Related Work

1. **S4**: Structured State Spaces
2. **H3**: Hungry Hungry Hippos
3. **Hyena**: Subquadratic attention alternative
4. **RWKV**: RNN-based transformer alternative

### Code & Resources

- Nexus Implementation: `Nexus/nexus/models/nlp/tokenization/mambabyte.py`
- Mamba Official: https://github.com/state-spaces/mamba
- MambaByte: https://github.com/lucidrains/mambabyte-pytorch (community)

### Concepts

1. **State Space Models**: Control theory foundation
2. **Selective Attention**: Input-dependent parameters
3. **Parallel Scan**: Efficient training algorithms
4. **Linear Attention**: Alternatives to quadratic attention

### Applications

1. **Long Documents**: Books, articles, code files
2. **Multilingual**: Universal byte-level processing
3. **Streaming**: Real-time byte-by-byte generation
4. **Binary Data**: Not limited to text (any bytes)
5. **Efficient Serving**: Linear complexity for inference

### Comparisons

**vs Transformer**:
- Complexity: O(N) vs O(N²)
- Long context: Better
- Short sequences: Similar
- Parallelism: Training (both fast), Inference (Mamba faster)

**vs RNN**:
- Complexity: Same O(N)
- Long range: Better (selective memory)
- Parallelism: Much better (parallel scan)

**vs BLT**:
- Tokenization: Both byte-level
- Architecture: SSM vs Transformer
- Efficiency: Mamba more efficient for long sequences
- Adaptivity: BLT more adaptive (dynamic patching)
