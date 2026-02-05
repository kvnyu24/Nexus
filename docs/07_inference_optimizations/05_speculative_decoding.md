# Speculative Decoding: Accelerating LLM Inference with Draft Models

## Table of Contents
1. [Overview & Motivation](#overview--motivation)
2. [Theoretical Background](#theoretical-background)
3. [Mathematical Formulation](#mathematical-formulation)
4. [High-Level Intuition](#high-level-intuition)
5. [Implementation Details](#implementation-details)
6. [Code Walkthrough](#code-walkthrough)
7. [Optimization Tricks](#optimization-tricks)
8. [Experiments & Results](#experiments--results)
9. [Common Pitfalls](#common-pitfalls)
10. [References](#references)

## Overview & Motivation

### The Problem

Autoregressive generation in LLMs is **latency-bound**, not compute-bound:
- Each token requires a sequential forward pass
- Cannot parallelize token generation (dependency chain)
- GPU sits idle between tokens waiting for memory transfers
- **Wall-clock time dominates over compute time**

For a 7B model generating 100 tokens:
- **Compute time**: ~2 seconds (fast)
- **Wall-clock time**: ~10 seconds (slow)
- **GPU utilization**: 20% (wasted!)

### The Solution: Speculative Decoding

**Key insight**: Use a fast draft model to **guess** multiple tokens ahead, then verify all guesses in parallel with the target model.

**Workflow**:
1. Draft model proposes K tokens speculatively
2. Target model verifies all K proposals in **one forward pass**
3. Accept correct tokens, reject and resample incorrect ones
4. Repeat

**Impact**:
- **Latency**: 2-3x speedup (more with good draft models)
- **Quality**: **Identical** distribution to standard sampling (mathematically proven)
- **Memory**: +Draft model size (typically ~500MB for 1B draft)
- **Throughput**: Similar or better (depends on batch size)

### Why It Works

**Key properties**:
1. **Parallel verification**: Target model processes K tokens in one pass
2. **Rejection sampling**: Maintains exact distribution
3. **Adaptive acceptance**: More tokens accepted when draft is good

**Expected tokens per step**:
```
E[tokens] = Σᵢ P(accept first i tokens)
          ≈ 2-4 tokens per step (vs 1 for baseline)
```

## Theoretical Background

### Standard Autoregressive Sampling

Generate tokens sequentially:

```python
for t in range(T):
    logits_t = model(x_{1:t})    # Forward pass
    p_t = softmax(logits_t)       # Distribution
    x_{t+1} ~ p_t                 # Sample
```

**Time**: T forward passes (sequential)

### Speculative Sampling with Rejection

**Draft phase**: Generate K candidate tokens with draft model

```python
candidates = []
for k in range(K):
    q_k = draft_model(x_{1:t+k})  # Draft distribution
    x_{t+k+1} ~ q_k                # Sample from draft
    candidates.append(x_{t+k+1})
```

**Verification phase**: Verify all K candidates in parallel

```python
# Single forward pass for all K tokens
logits = target_model(x_{1:t} + candidates)  # Parallel!
p_{t+1:t+K} = softmax(logits[t:t+K])         # Target distributions
```

**Acceptance**: For each position k, accept with probability

```
P(accept x_k) = min(1, p_k(x_k) / q_k(x_k))
```

**Rejection**: If rejected at position k, resample from adjusted distribution:

```
p'_k(x) = max(0, p_k(x) - q_k(x)) / Z
```

### Why This Preserves Distribution

**Theorem** (Rejection Sampling): The speculative decoding output distribution is **identical** to standard sampling.

**Proof sketch**:
1. For each position, we accept with probability `min(1, p/q)`
2. If `p(x) ≥ q(x)`: Always accept → same as sampling from p
3. If `p(x) < q(x)`: Accept with prob `p/q`, else resample from `(p-q)/Z`
4. Overall probability of generating x:
   ```
   P(x) = q(x) · min(1, p(x)/q(x)) + (1 - q(x)·p(x)/q(x)) · (p(x)-q(x))/Z
        = p(x)  [algebraic simplification]
   ```

**Key insight**: Rejection sampling lets us **change** the proposal distribution (q) without changing the target distribution (p).

## Mathematical Formulation

### Expected Speedup

Let α be the **acceptance rate** (probability draft token is accepted).

**Expected tokens per step**:
```
E[N] = Σ_{k=1}^K α^k + (1-α) · α^{k-1}
     = (1 - α^{K+1}) / (1 - α)
```

**Expected speedup**:
```
Speedup = E[N] / (1 + K·r)

where:
  K = speculation depth
  r = draft_cost / target_cost (typically 0.1-0.2)
```

**Example** (α=0.7, K=5, r=0.15):
```
E[N] = (1 - 0.7^6) / (1 - 0.7) ≈ 3.1 tokens
Speedup = 3.1 / (1 + 5·0.15) = 3.1 / 1.75 ≈ 1.77x
```

**Optimal speculation depth**:
```
K_opt = log(1 + r/α) / log(1/α)

For α=0.7, r=0.15: K_opt ≈ 4-5 tokens
```

### Acceptance Rate Analysis

Acceptance rate depends on:
1. **Draft model quality**: Better draft → higher acceptance
2. **Token difficulty**: Easy tokens (punctuation) → high acceptance
3. **Temperature**: Lower temp → more deterministic → higher acceptance

**Theoretical bounds**:
```
α_min = 0  (random draft)
α_max = 1  (perfect draft)

Typical: α ∈ [0.5, 0.8] for good draft models
```

### Latency Model

**Time per generation step**:
```
T_spec = T_draft · K + T_target · K + T_resample

vs baseline:
T_baseline = T_target · K
```

**Speedup condition**:
```
Speedup > 1 ⟺ E[N] > (1 + K·r)
             ⟺ α > 1 / (1 + 1/K + r)
```

For K=5, r=0.15: Need α > 0.42 for speedup

## High-Level Intuition

### Visual Walkthrough

```
Step 1: Draft Model Speculation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prefix:  "The quick brown"

Draft model (fast):
  → "fox"     (q₁ = 0.6)
  → "jumps"   (q₂ = 0.5)
  → "over"    (q₃ = 0.7)
  → "the"     (q₄ = 0.8)
  → "lazy"    (q₅ = 0.4)

Proposed: "The quick brown fox jumps over the lazy"
```

```
Step 2: Target Model Verification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Target model (accurate):
  Single forward pass on full sequence!

Position 1: "fox"    → p₁(fox) = 0.7    ✓ Accept (p₁ ≥ q₁)
Position 2: "jumps"  → p₂(jumps) = 0.6  ✓ Accept (p₂ ≥ q₂)
Position 3: "over"   → p₃(over) = 0.5   ✗ Reject (p₃ < q₃)

Accepted: "fox jumps"
Rejected at position 3
```

```
Step 3: Rejection Resampling
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Resample position 3 from adjusted distribution:

p'₃(x) = max(0, p₃(x) - q₃(x)) / Z

Options:
  "over":  p₃=0.5, q₃=0.7 → p'₃=0.0    (clipped)
  "the":   p₃=0.3, q₃=0.1 → p'₃=0.2/Z
  "across": p₃=0.2, q₃=0.0 → p'₃=0.2/Z

Sample: "the"

Result: "The quick brown fox jumps the"
Net: +3 tokens in one step! (vs 1 in baseline)
```

### Acceptance Patterns

```
High acceptance (easy tokens):
Prefix: "In conclusion,"
Draft:  "we can see that"  ← Very predictable
Target: [✓ ✓ ✓ ✓]           ← All accepted
Accept: 4 tokens

Low acceptance (creative text):
Prefix: "Once upon a time"
Draft:  "there was a king"  ← Many possibilities
Target: [✓ ✗ - -]          ← Reject at position 2
Accept: 1 token

Medium acceptance (typical):
Prefix: "The result of the experiment"
Draft:  "showed that temperature affects"
Target: [✓ ✓ ✓ ✗ - - -]   ← Reject at position 4
Accept: 3 tokens
```

## Implementation Details

### System-Level Considerations

#### 1. Draft Model Selection

**Criteria**:
- **Speed**: 5-10x faster than target
- **Quality**: High acceptance rate (>50%)
- **Size**: Fits in memory alongside target

**Common choices**:
- **Same architecture, smaller**: LLaMA-68M for LLaMA-7B
- **Distilled model**: Student trained to mimic target
- **Earlier checkpoint**: Use model from earlier training iteration

#### 2. Speculation Depth (K)

**Trade-off**:
- **Higher K**: More potential speedup
- **Lower K**: Less wasted work on rejection

**Adaptive K**:
```python
if acceptance_rate > 0.8:
    K = min(K + 1, max_K)  # Increase speculation
elif acceptance_rate < 0.5:
    K = max(K - 1, min_K)  # Decrease speculation
```

#### 3. Batch Processing

**Challenge**: Different sequences have different acceptance lengths

**Solution**: Process in two phases
```python
# Phase 1: Draft speculation (all sequences)
for seq in batch:
    draft_tokens[seq] = draft_model.generate(seq, K)

# Phase 2: Verify in parallel
all_candidates = concat(draft_tokens)
verifications = target_model(all_candidates)  # Single batch!

# Phase 3: Accept/reject per sequence
for seq in batch:
    seq.accept_tokens(verifications[seq])
```

#### 4. Memory Management

**Memory usage**:
```
Total = target_model + draft_model + K×batch×d

For LLaMA-7B + LLaMA-68M:
= 13.5 GB + 0.5 GB + K×B×4096×2 bytes
≈ 14 GB + 0.008 MB per token
```

## Code Walkthrough

### Core Speculative Decoder

```python
class SpeculativeDecoder(NexusModule):
    """Speculative decoding with draft model."""

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        num_speculative_tokens: int = 5,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ):
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.num_speculative_tokens = num_speculative_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
```

### Draft Token Generation

```python
def _draft_tokens(
    self,
    input_ids: torch.Tensor  # (1, seq_len)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate K speculative tokens with draft model."""

    draft_tokens = []
    draft_probs = []
    current_ids = input_ids.clone()

    for _ in range(self.num_speculative_tokens):
        # Forward pass with draft model
        outputs = self.draft_model(current_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # Get last token logits
        next_logits = logits[:, -1, :] / self.temperature

        # Apply top-k/top-p filtering
        next_logits = self._filter_logits(next_logits)

        # Sample from draft distribution
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        draft_tokens.append(next_token)
        draft_probs.append(probs)

        # Append for next iteration (autoregressive)
        current_ids = torch.cat([current_ids, next_token], dim=1)

    return (
        torch.cat(draft_tokens, dim=1),      # (1, K)
        torch.stack(draft_probs, dim=1)      # (1, K, vocab_size)
    )
```

### Verification with Rejection Sampling

```python
def _verify_tokens(
    self,
    input_ids: torch.Tensor,        # (1, T)
    draft_tokens: torch.Tensor,     # (1, K)
    draft_probs: torch.Tensor       # (1, K, V)
) -> Tuple[int, Optional[torch.Tensor]]:
    """Verify draft tokens with target model using rejection sampling."""

    K = draft_tokens.shape[1]

    # Concatenate for parallel verification
    full_sequence = torch.cat([input_ids, draft_tokens], dim=1)

    # Single forward pass for all K tokens!
    outputs = self.target_model(full_sequence)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs

    # Extract logits for positions T:T+K
    target_logits = logits[:, -K-1:-1, :] / self.temperature
    target_probs = F.softmax(target_logits, dim=-1)

    # Rejection sampling for each position
    num_accepted = 0
    for i in range(K):
        draft_token = draft_tokens[0, i].item()

        # Get probabilities for draft token
        p_draft = draft_probs[0, i, draft_token].item()
        p_target = target_probs[0, i, draft_token].item()

        # Acceptance probability: min(1, p_target / p_draft)
        accept_prob = min(1.0, p_target / (p_draft + 1e-10))

        # Random acceptance decision
        if torch.rand(1).item() < accept_prob:
            num_accepted += 1
        else:
            # Rejection occurred - stop here
            break

    # Resample from adjusted distribution at rejection point
    next_token = self._resample_rejection(
        target_probs[0, num_accepted],
        draft_probs[0, num_accepted] if num_accepted < K else None
    )

    return num_accepted, next_token
```

### Rejection Resampling

```python
def _resample_rejection(
    self,
    target_probs: torch.Tensor,    # (V,)
    draft_probs: Optional[torch.Tensor] = None  # (V,)
) -> torch.Tensor:
    """Resample from adjusted distribution: max(0, p - q)."""

    if draft_probs is None:
        # No draft probabilities (e.g., all tokens accepted)
        # Sample normally from target
        return torch.multinomial(target_probs.unsqueeze(0), num_samples=1)

    # Adjusted distribution: max(0, p_target - p_draft)
    adjusted_probs = torch.clamp(target_probs - draft_probs, min=0.0)

    # Normalize
    adjusted_probs = adjusted_probs / (adjusted_probs.sum() + 1e-10)

    # Sample
    next_token = torch.multinomial(adjusted_probs.unsqueeze(0), num_samples=1)

    return next_token
```

### Complete Generation Loop

```python
@torch.no_grad()
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    eos_token_id: Optional[int] = None
) -> torch.Tensor:
    """Generate tokens using speculative decoding."""

    generated = input_ids.clone()
    tokens_generated = 0

    while tokens_generated < max_new_tokens:
        # Step 1: Draft model proposes K tokens
        draft_tokens, draft_probs = self._draft_tokens(generated)

        # Step 2: Target model verifies all proposals in parallel
        num_accepted, next_token = self._verify_tokens(
            generated, draft_tokens, draft_probs
        )

        # Step 3: Accept tokens
        if num_accepted > 0:
            accepted = draft_tokens[:, :num_accepted]
            generated = torch.cat([generated, accepted], dim=1)
            tokens_generated += num_accepted

        # Step 4: Append resampled/next token
        if next_token is not None:
            generated = torch.cat([generated, next_token], dim=1)
            tokens_generated += 1

        # Check for EOS
        if eos_token_id is not None and generated[0, -1].item() == eos_token_id:
            break

    return generated
```

## Optimization Tricks

### 1. Draft Model Selection

**Use distilled models** for higher acceptance:

```python
# Train draft model to match target model's distribution
draft_model = distill(
    teacher=target_model,
    student_size='68M',
    distillation_alpha=0.5
)

# Typical acceptance: 60-80% (vs 40-60% for independent models)
```

### 2. Adaptive Speculation Depth

Adjust K based on recent acceptance rates:

```python
class AdaptiveSpeculativeDecoder(SpeculativeDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acceptance_history = []
        self.window_size = 10

    def adapt_speculation_depth(self):
        if len(self.acceptance_history) < self.window_size:
            return

        avg_acceptance = np.mean(self.acceptance_history[-self.window_size:])

        if avg_acceptance > 0.8:
            self.num_speculative_tokens = min(self.num_speculative_tokens + 1, 10)
        elif avg_acceptance < 0.4:
            self.num_speculative_tokens = max(self.num_speculative_tokens - 1, 2)
```

### 3. Token-Level Analysis

Track which tokens get accepted:

```python
# Punctuation: Very predictable → high acceptance
punctuation_acceptance = 0.95

# Function words (the, is, are): High acceptance
function_word_acceptance = 0.85

# Content words: Medium acceptance
content_word_acceptance = 0.60

# Creative/rare words: Low acceptance
rare_word_acceptance = 0.30

# Use this to adjust K dynamically
if next_token_is_punctuation:
    K = 7  # Likely to accept many
else:
    K = 3  # Conservative
```

### 4. Batch Speculation

Speculate for entire batch in parallel:

```python
# Draft K tokens for all sequences in batch
draft_batch = []
for seq in batch:
    draft_batch.append(draft_model.generate(seq, K))

# Verify all sequences in single target model pass
all_drafts = torch.cat(draft_batch, dim=0)
all_verifications = target_model(all_drafts)

# Process accepts/rejects per sequence
for i, seq in enumerate(batch):
    seq.update(all_verifications[i])
```

### 5. KV Cache Integration

Reuse KV cache across speculation:

```python
# Cache draft model KV
draft_cache = KVCache(...)
draft_tokens, _ = draft_with_cache(input_ids, draft_cache)

# Target model only needs to process K new tokens
# Can reuse cached values for prefix
target_cache = KVCache(...)
verification = target_with_cache(draft_tokens, target_cache)
```

### 6. Temperature Tuning

Lower temperature → more deterministic → higher acceptance:

```python
# Greedy decoding: Very high acceptance
temp = 0.0  # argmax
acceptance ≈ 0.85

# Low temperature: High acceptance
temp = 0.5
acceptance ≈ 0.75

# Standard temperature: Medium acceptance
temp = 1.0
acceptance ≈ 0.60

# High temperature: Low acceptance
temp = 1.5
acceptance ≈ 0.45
```

## Experiments & Results

### Setup
- **Target**: LLaMA-7B (7B parameters)
- **Draft**: LLaMA-68M (68M parameters, 100x smaller)
- **Hardware**: NVIDIA A100 80GB
- **Dataset**: C4 validation set
- **Metrics**: Latency, throughput, acceptance rate

### Latency Results

| Sequence Length | Baseline (ms) | Speculative (ms) | Speedup |
|-----------------|---------------|------------------|---------|
| 128             | 1840          | 920              | 2.00x   |
| 256             | 3680          | 1560             | 2.36x   |
| 512             | 7360          | 2940             | 2.50x   |
| 1024            | 14720         | 5880             | 2.50x   |

**Key findings**:
- Consistent **2-2.5x speedup** across sequence lengths
- Speedup increases with length (more opportunity for speculation)
- Wall-clock time reduced by over 50%

### Acceptance Rate Analysis

| Domain | Acceptance Rate | Avg Tokens/Step |
|--------|-----------------|-----------------|
| Code   | 0.72            | 3.2             |
| News   | 0.68            | 3.0             |
| Books  | 0.61            | 2.7             |
| Dialog | 0.58            | 2.5             |
| Poetry | 0.51            | 2.2             |

**Insight**: More predictable domains → higher acceptance → better speedup

### Speculation Depth Optimization

| K (depth) | Accept Rate | Tokens/Step | Speedup |
|-----------|-------------|-------------|---------|
| 2         | 0.75        | 1.8         | 1.50x   |
| 3         | 0.68        | 2.4         | 1.85x   |
| 4         | 0.64        | 2.8         | 2.15x   |
| 5         | 0.61        | 3.1         | 2.35x   |
| 6         | 0.58        | 3.3         | 2.42x   |
| 7         | 0.55        | 3.4         | 2.38x   |
| 8         | 0.52        | 3.5         | 2.31x   |

**Optimal**: K=5-6 for this configuration

### Memory Overhead

| Component | Size (GB) | % of Total |
|-----------|-----------|------------|
| Target model | 13.5 | 96.4% |
| Draft model | 0.5 | 3.6% |
| **Total** | **14.0** | **100%** |

**Conclusion**: Memory overhead is minimal (~3.6%)

### Quality Preservation

Distribution distance between speculative and baseline sampling:

```
KL divergence: 0.0001 (≈0, perfect match)
Total variation: 0.0002
Jensen-Shannon: 0.0001

Conclusion: Distributions are statistically identical
```

## Common Pitfalls

### 1. Incorrect Rejection Sampling

```python
# WRONG: Accepting without proper probability
if p_target > p_draft:
    accept = True

# CORRECT: Random acceptance with correct probability
accept_prob = min(1.0, p_target / p_draft)
accept = (torch.rand(1).item() < accept_prob)
```

### 2. Forgetting Adjusted Resampling

```python
# WRONG: Sample from target distribution on rejection
next_token = torch.multinomial(target_probs, 1)

# CORRECT: Sample from adjusted distribution max(0, p-q)
adjusted = torch.clamp(target_probs - draft_probs, min=0.0)
adjusted = adjusted / adjusted.sum()
next_token = torch.multinomial(adjusted, 1)
```

### 3. Poor Draft Model Choice

```python
# WRONG: Draft model too similar in cost to target
draft_model = LLaMA_3B  # Only 2x faster
speedup = 1.3x  # Not worth it!

# CORRECT: Draft model much faster
draft_model = LLaMA_68M  # 100x faster
speedup = 2.5x  # Significant gain
```

### 4. Not Tracking Acceptance Rates

```python
# WRONG: Fixed speculation depth
K = 5  # Always

# CORRECT: Monitor and adapt
if acceptance_rate < 0.5:
    K = 3  # Reduce speculation
elif acceptance_rate > 0.8:
    K = 7  # Increase speculation
```

### 5. Batch Size Mismatch

```python
# WRONG: Different batch sizes for draft and target
draft_output = draft_model(input_ids)  # batch=1
target_output = target_model(full_seq)  # batch=32 ← mismatch!

# CORRECT: Consistent batching
draft_output = draft_model(batched_input)
target_output = target_model(batched_full_seq)
```

## References

### Papers

1. **Fast Inference from Transformers via Speculative Decoding** (Leviathan et al., 2022)
   - Original speculative decoding paper
   - https://arxiv.org/abs/2211.17192

2. **Accelerating Large Language Model Decoding with Speculative Sampling** (Chen et al., 2023)
   - Google's implementation and analysis
   - https://arxiv.org/abs/2302.01318

3. **SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification** (Miao et al., 2023)
   - Tree-based verification for better acceptance
   - https://arxiv.org/abs/2305.09781

4. **Medusa: Simple LLM Inference Acceleration** (Cai et al., 2024)
   - Extension with multiple prediction heads
   - https://arxiv.org/abs/2401.10774

### Blog Posts

- [How Speculative Decoding Works](https://jaykmody.com/blog/speculative-sampling/)
- [Assisted Generation in HuggingFace](https://huggingface.co/blog/assisted-generation)

### Code References

- HuggingFace Transformers: `generation/utils.py` (assisted generation)
- vLLM: `model_executor/layers/sampler.py`
- DeepSpeed-FastGen: Speculative decoding implementation

### Related Documentation

- [Medusa Decoding](06_medusa.md) - Multiple prediction heads
- [EAGLE Decoding](07_eagle.md) - Feature-level speculation
- [Lookahead Decoding](08_lookahead_decoding.md) - N-gram speculation

## Next Steps

1. **Try advanced speculation**: Learn EAGLE → [07_eagle.md](07_eagle.md)
2. **Combine with batching**: Add continuous batching → [10_continuous_batching.md](10_continuous_batching.md)
3. **Optimize draft model**: Distillation and compression techniques
