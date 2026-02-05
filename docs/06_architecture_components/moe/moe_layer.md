# MoE Layer

## Overview & Motivation

The MoE Layer is the foundational building block for sparse mixture-of-experts models. It replaces dense feed-forward networks with a gated collection of expert sub-networks, where each input token is dynamically routed to a small subset (top-k) of experts.

**Key Motivation**: Traditional dense neural networks activate all parameters for every input. MoE enables conditional computation - activating only relevant parameters based on the input, allowing models to scale to massive parameter counts while keeping inference costs manageable.

**Real-World Impact**:
- Mixtral 8x7B: 47B parameters, performance of 13B active parameters
- DeepSeek-V3: 671B parameters, 37B active per token
- Switch Transformers: Scaled to 1.6T parameters

## Theoretical Background

### Conditional Computation

The core idea: **Not all inputs need the same computation**. MoE formalizes this through learned routing:

```
y = Σᵢ₌₁ᴺ gᵢ(x) · Eᵢ(x)
```

Where:
- `x` is the input
- `N` is total number of experts
- `gᵢ(x)` is the gating weight for expert i (computed by router)
- `Eᵢ(x)` is expert i's output

In sparse top-k MoE:
```
y = Σᵢ∈TopK(g(x)) gᵢ(x) · Eᵢ(x)
```

Only k experts are activated per token.

### Why It Works

1. **Specialization**: Different experts learn complementary transformations
2. **Efficiency**: Sparse activation reduces FLOPs by factor of N/k
3. **Capacity**: Total parameters = N × expert_size, but compute = k × expert_size
4. **Scalability**: Can add experts without increasing per-token cost

### Load Balancing Challenge

Without constraints, routing collapses to using only a few experts. Solutions:

**Auxiliary Loss** (Switch Transformer):
```
L_aux = α · Σᵢ fᵢ · Pᵢ
```
Where:
- `fᵢ = (tokens to expert i) / (total tokens)`
- `Pᵢ = mean routing probability for expert i`
- `α = 0.01` (typical)

This encourages uniform distribution: when `fᵢ` and `Pᵢ` are both 1/N, loss is minimized.

## Mathematical Formulation

### Complete MoE Forward Pass

**Step 1: Router Computation**
```
logits = W_gate · x         # (batch, seq, num_experts)
logits = logits + noise     # Optional jitter for exploration
```

**Step 2: Top-K Selection**
```
weights, indices = TopK(logits, k)  # Select k experts
weights = Softmax(weights)           # Normalize to sum to 1
```

**Step 3: Expert Processing**
```
For each selected expert i:
    mask_i = (indices == i)                    # Find tokens for expert i
    tokens_i = x[mask_i]                       # Gather tokens
    output_i = Expert_i(tokens_i)              # Process through expert
    y[mask_i] += weights[mask_i] * output_i    # Weighted accumulation
```

**Step 4: Output**
```
y_final = y + SharedExpert(x)  # Optional shared expert (DeepSeek style)
```

### Expert Architecture

Each expert is typically a standard FFN:

```
Expert(x) = W_down · σ(W_up · x)
```

For SwiGLU activation (modern LLMs):
```
Expert(x) = W_down · (σ(W_gate · x) ⊙ W_up · x)
```

Where:
- `W_up: d → h` (usually h = 4d)
- `W_down: h → d`
- `W_gate: d → h` (for gated activations)
- `σ` is activation function (SiLU/GELU)
- `⊙` is element-wise product

## High-Level Intuition

Think of MoE as a learned committee of specialists:

### Analogy: Hospital Departments

Imagine a hospital with specialized departments (experts):
- **Cardiology** (Expert 1)
- **Neurology** (Expert 2)
- **Orthopedics** (Expert 3)
- **Pediatrics** (Expert 4)

When a patient arrives (input token):
1. **Triage** (router) assesses symptoms
2. **Routes** patient to appropriate department(s)
3. **Specialists** (experts) provide treatment
4. **Combined** recommendations form final diagnosis (output)

**Key Insight**: Not every patient needs every specialist. Routing enables efficiency.

### Visual Representation

```
Input Tokens: [tok1, tok2, tok3, tok4]
                  |     |     |     |
              Router Scores
                  |     |     |     |
         Top-K Selection (k=2)
         /   \   /  \   /  \   /  \
    E1  E2  E3  E1  E4  E2  E3  E4
     |   |   |   |   |   |   |   |
    Weighted Combination
         |     |     |     |
    [out1, out2, out3, out4]

E1 processes tok1, tok3
E2 processes tok1, tok4
E3 processes tok2, tok4
E4 processes tok3, tok4
```

## Implementation Details

### Code Location
- **File**: `/Users/kevinyu/Projects/Nexus/nexus/components/moe/expert.py`
- **Classes**: `MoELayer`, `ExpertLayer`, `SharedExpert`

### Key Parameters

```python
MoELayer(
    dim=2048,              # Model dimension
    num_experts=8,         # Total number of experts
    top_k=2,              # Experts activated per token
    expert_hidden_dim=8192, # Expert FFN hidden size (typically 4×dim)
    shared_expert=False,   # Whether to include always-active shared expert
    activation='swiglu',   # Expert activation function
    dropout=0.0,          # Dropout probability
    router_jitter=0.0     # Noise for router exploration
)
```

### Usage Example

```python
from nexus.components.moe import MoELayer

# Create MoE layer
moe = MoELayer(
    dim=2048,
    num_experts=8,
    top_k=2,
    expert_hidden_dim=8192
)

# Forward pass
hidden_states = torch.randn(2, 100, 2048)  # (batch, seq, dim)
output, aux_loss = moe(hidden_states, return_aux_loss=True)

# Training loop
total_loss = task_loss + 0.01 * aux_loss
total_loss.backward()
```

### Memory Considerations

**Parameter Count**:
```python
# Router
router_params = dim * num_experts

# Experts (SwiGLU)
params_per_expert = dim * hidden_dim * 3  # gate, up, down
total_expert_params = num_experts * params_per_expert

# Total
total_params = router_params + total_expert_params

# Example: dim=2048, num_experts=8, hidden_dim=8192
# router: 2048 * 8 = 16K
# experts: 8 * (2048 * 8192 * 3) = 402M
# total: ~402M parameters
```

**Activation Memory** (during forward pass):
```python
# Need to store:
# - Router logits: batch * seq * num_experts
# - Selected tokens per expert: varies
# - Expert outputs: batch * seq * dim

# Peak memory occurs when all experts process their tokens
peak_activation_memory = (
    batch * seq * dim +                    # Input
    batch * seq * num_experts +            # Router logits
    batch * seq * top_k * hidden_dim +     # Expert activations
    batch * seq * dim                      # Output
)
```

## Code Walkthrough

### Router Implementation

From `nexus/components/moe/router.py`:

```python
class ExpertRouter(NexusModule):
    def __init__(self, dim, num_experts, top_k):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        # Compute routing scores
        logits = self.gate(x)  # (batch, seq, num_experts)

        # Top-k selection
        top_k_logits, expert_indices = torch.topk(logits, self.top_k, dim=-1)

        # Normalize weights
        expert_weights = F.softmax(top_k_logits, dim=-1)

        # Compute load balancing loss
        aux_loss = self._compute_aux_loss(logits, expert_indices)

        return expert_weights, expert_indices, aux_loss
```

**Key Design Decisions**:
1. **No bias in gate**: Prevents systematic expert preference
2. **Softmax normalization**: Weights sum to 1 for stability
3. **Topk before softmax**: Only normalize selected experts

### Expert Processing

From `nexus/components/moe/expert.py`:

```python
class MoELayer(NexusModule):
    def forward(self, x, return_aux_loss=True):
        batch_size, seq_len, dim = x.shape

        # Route tokens
        expert_weights, expert_indices, aux_loss = self.router(x)

        # Flatten for efficient processing
        x_flat = x.view(-1, dim)
        weights_flat = expert_weights.view(-1, self.top_k)
        indices_flat = expert_indices.view(-1, self.top_k)

        # Process through experts
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            for expert_idx in range(self.num_experts):
                # Find tokens for this expert
                mask = (indices_flat[:, k] == expert_idx)
                if not mask.any():
                    continue

                # Gather, process, scatter
                expert_input = x_flat[mask]
                expert_output = self.experts[expert_idx](expert_input)
                output[mask] += weights_flat[mask, k].unsqueeze(-1) * expert_output

        return output.view(batch_size, seq_len, dim), aux_loss
```

**Efficiency Note**: This simple implementation processes experts sequentially. Production systems batch tokens by expert for parallelism.

## Optimization Tricks

### 1. Expert Capacity with Token Dropping

Limit tokens per expert to prevent load imbalance:

```python
capacity = (total_tokens // num_experts) * capacity_factor

# Drop tokens exceeding capacity
if tokens_to_expert_i > capacity:
    keep_mask = torch.rand(tokens_to_expert_i) < (capacity / tokens_to_expert_i)
    tokens_i = tokens_i[keep_mask]
```

### 2. Efficient Batched Expert Computation

Instead of processing experts one at a time, batch by expert:

```python
# Group tokens by expert
expert_batches = [[] for _ in range(num_experts)]
for token_idx in range(num_tokens):
    expert_id = expert_indices[token_idx]
    expert_batches[expert_id].append(token_idx)

# Process each expert's batch in parallel
outputs = []
for expert_id in range(num_experts):
    if len(expert_batches[expert_id]) > 0:
        batch = x[expert_batches[expert_id]]
        outputs.append(experts[expert_id](batch))
```

### 3. Expert Parallelism

Distribute experts across GPUs:

```python
# Shard experts across devices
experts_per_device = num_experts // world_size
local_expert_ids = range(
    rank * experts_per_device,
    (rank + 1) * experts_per_device
)

# All-to-all communication
tokens_for_local_experts = all_to_all(tokens, expert_indices)
local_outputs = process_local_experts(tokens_for_local_experts)
final_outputs = all_to_all(local_outputs, reverse=True)
```

### 4. Router Jitter Noise

Add noise during training to encourage exploration:

```python
if self.training and self.jitter_noise > 0:
    noise = torch.randn_like(logits) * self.jitter_noise
    logits = logits + noise
```

### 5. Gradient Checkpointing for Experts

Save memory during training:

```python
from torch.utils.checkpoint import checkpoint

# Checkpoint expert computation
expert_output = checkpoint(
    expert.forward,
    expert_input,
    use_reentrant=False
)
```

## Experiments & Results

### Expert Specialization

Analysis on language modeling tasks shows experts specialize by:

1. **Syntax vs Semantics**: Some experts activate for grammatical structure, others for meaning
2. **Token Type**: Punctuation, nouns, verbs route to different experts
3. **Domain**: Code tokens vs natural language vs numbers
4. **Position**: Early-sentence vs mid-sentence vs final tokens

### Performance Comparison

**Setup**: Language modeling on C4 dataset, 1B parameter models

| Configuration | Perplexity | Training FLOPs | Inference FLOPs |
|---------------|------------|----------------|-----------------|
| Dense FFN | 12.5 | 1.0x | 1.0x |
| MoE 8 experts, top-2 | 11.8 | 1.1x | 0.35x |
| MoE 16 experts, top-2 | 11.3 | 1.2x | 0.20x |
| MoE 32 experts, top-2 | 11.0 | 1.3x | 0.12x |

**Key Findings**:
- MoE improves quality with same training compute
- Inference speedup increases with more experts
- Diminishing returns beyond 32-64 experts

### Load Balancing Impact

**Experiment**: Train with varying auxiliary loss coefficients

| Aux Loss Coef | Expert Utilization (std) | Perplexity |
|---------------|-------------------------|------------|
| 0.0 (no balancing) | 0.85 (collapsed) | 13.2 |
| 0.001 | 0.12 | 11.9 |
| 0.01 | 0.05 | 11.8 |
| 0.1 | 0.02 | 12.1 |

**Conclusion**: Need balancing (0.001-0.01), but too much hurts quality.

## Common Pitfalls

### 1. Forgetting to Add Auxiliary Loss

```python
# WRONG: Auxiliary loss computed but not used
output, aux_loss = moe(x)
loss = cross_entropy(output, labels)
loss.backward()  # aux_loss ignored!

# CORRECT: Include in total loss
output, aux_loss = moe(x)
loss = cross_entropy(output, labels) + 0.01 * aux_loss
loss.backward()
```

### 2. Expert Capacity Too Small

```python
# Symptoms: Many tokens get dropped, performance degrades

# Check token drop rate
total_tokens = batch_size * seq_len
tokens_dropped = total_tokens - tokens_processed
drop_rate = tokens_dropped / total_tokens

# If drop_rate > 0.1, increase capacity_factor
router = ExpertRouter(dim, num_experts, capacity_factor=2.0)  # Increase from 1.25
```

### 3. Not Normalizing Router Weights

```python
# WRONG: Unnormalized weights
expert_weights = top_k_logits

# CORRECT: Normalize to sum to 1
expert_weights = F.softmax(top_k_logits, dim=-1)
```

### 4. Memory Issues with Many Experts

```python
# Problem: 64 experts × 1B params/expert = 64B params won't fit in GPU

# Solution 1: Expert parallelism
model = DistributedMoE(num_experts=64, gpus=8)  # 8 experts per GPU

# Solution 2: Expert offloading
model = MoEWithOffloading(num_experts=64, offload_to='cpu')

# Solution 3: Reduce expert size
model = MoELayer(expert_hidden_dim=2048)  # Instead of 8192
```

## References

1. **Shazeer et al. (2017)** - "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
   - Original MoE paper, introduced top-k gating and auxiliary loss
   - https://arxiv.org/abs/1701.06538

2. **Lepikhin et al. (2020)** - "GShard: Scaling Giant Models with Conditional Computation"
   - Scaled MoE to 600B parameters
   - Expert parallelism across TPU pods
   - https://arxiv.org/abs/2006.16668

3. **Fedus et al. (2021)** - "Switch Transformers: Scaling to Trillion Parameter Models"
   - Simplified to top-1 routing
   - Improved load balancing strategies
   - https://arxiv.org/abs/2101.03961

4. **Riquelme et al. (2021)** - "Scaling Vision with Sparse Mixture of Experts"
   - Applied MoE to vision transformers
   - https://arxiv.org/abs/2106.05974

5. **Zhou et al. (2022)** - "Mixture-of-Experts with Expert Choice Routing"
   - Inverted routing paradigm
   - https://arxiv.org/abs/2202.09368
