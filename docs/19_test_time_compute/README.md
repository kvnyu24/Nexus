# Test-Time Compute Methods

This directory contains comprehensive documentation for test-time compute techniques that improve model performance by spending additional computation during inference. These methods enable models to adapt, search, and reason more effectively at test time.

## Table of Contents

1. [Test-Time Training (TTT) Layers](#ttt-layers)
2. [Compute-Optimal Scaling](#compute-optimal-scaling)
3. [Best-of-N with Process Reward Models (PRM)](#best-of-n-prm)

## Overview

Test-time compute is a paradigm shift in how we think about model deployment:

**Traditional Paradigm**:
- Fixed computation at inference
- Model outputs single prediction
- No adaptation to test distribution

**Test-Time Compute Paradigm**:
- Adaptive computation budget at inference
- Multiple predictions evaluated and selected
- Model adapts to test-time inputs
- Performance improves with more compute

### Core Motivation

**Training Compute** is expensive but:
- Amortized over millions of queries
- Can use large GPU clusters
- One-time cost

**Test-Time Compute** allows:
- User control over speed vs. quality
- Adaptation to individual inputs
- Scaling performance post-deployment
- Handling distribution shift

## Algorithm Overview

### 1. Test-Time Training (TTT) Layers
- **File**: [ttt_layers.md](ttt_layers.md)
- **Difficulty**: Advanced
- **Key Concept**: Self-supervised adaptation during forward pass
- **Use Case**: Distribution shift, long sequences, continual adaptation

### 2. Compute-Optimal Scaling
- **File**: [compute_optimal_scaling.md](compute_optimal_scaling.md)
- **Difficulty**: Intermediate-Advanced
- **Key Concept**: Scaling laws for test-time compute
- **Use Case**: Resource allocation, performance prediction

### 3. Best-of-N with PRM
- **File**: [best_of_n_prm.md](best_of_n_prm.md)
- **Difficulty**: Intermediate
- **Key Concept**: Generate multiple outputs, select best
- **Use Case**: Reasoning, code generation, creative tasks

## Comparison Table

| Method | Adaptation | Multi-Sample | Complexity | Compute Cost | Use Case |
|--------|-----------|--------------|------------|--------------|----------|
| TTT Layers | ✅ Online | ❌ Single | High | O(T·K) steps | Distribution shift |
| Compute Scaling | ❌ Fixed | ✅ Multiple | Medium | O(N·inference) | Performance boost |
| Best-of-N PRM | ❌ Fixed | ✅ Multiple | Low | O(N·inference) | Quality improvement |

**Notation**:
- T: Sequence length
- K: TTT optimization steps
- N: Number of samples

## Core Concepts

### Test-Time Adaptation

**TTT Layers** adapt the model during inference:

```python
# Traditional forward pass
output = model(input)

# TTT forward pass
hidden_state = initialize()
for token in sequence:
    # Self-supervised update at each step
    hidden_state = hidden_state.update_via_gradient(token)
    output = hidden_state.predict(next_token)
```

**Key Benefit**: Model continuously adapts to test sequence.

### Test-Time Search

**Best-of-N** generates multiple outputs and selects best:

```python
# Generate N candidates
candidates = [model.generate() for _ in range(N)]

# Score with process reward model
scores = [prm.score(candidate) for candidate in candidates]

# Select best
best_output = candidates[argmax(scores)]
```

**Key Benefit**: Quality improves with more samples (up to limit).

### Compute-Performance Scaling

**Scaling Laws** predict performance vs. compute:

```python
# Empirical scaling law
performance = a * compute^b

# Optimal allocation
compute_per_sample = optimal_allocation(
    total_budget=budget,
    num_samples=N,
    scaling_exponent=b
)
```

**Key Benefit**: Principled resource allocation.

## When to Use Each Method

### Use TTT Layers when:
- **Distribution Shift**: Test data differs from training
- **Long Sequences**: Benefits accumulate over time
- **Continual Learning**: Need ongoing adaptation
- **Few-Shot Learning**: Limited labeled test data
- **Online Learning**: Data arrives sequentially

**Examples**:
- Time series prediction with drift
- Video understanding over long clips
- Language models for code completion
- Personalization to user writing style

### Use Compute-Optimal Scaling when:
- **Resource Budgets**: Fixed compute budget to allocate
- **Performance Targets**: Need to hit specific accuracy
- **Cost Optimization**: Minimize cost for target quality
- **Scaling Decisions**: Choosing between model size and samples
- **Production Planning**: Capacity planning for deployment

**Examples**:
- API service with latency SLAs
- Batch processing with deadline
- Multi-model ensemble optimization
- A/B testing different inference strategies

### Use Best-of-N with PRM when:
- **Quality Matters**: Correctness more important than speed
- **Verifiable Tasks**: Can evaluate output quality
- **Asymmetric Cost**: Wrong answers are expensive
- **Reasoning Tasks**: Multi-step problem solving
- **Creative Tasks**: Multiple valid solutions

**Examples**:
- Code generation (run tests on N samples)
- Math problem solving (verify solutions)
- Question answering (consistency checking)
- Text generation (quality scoring)
- Planning and scheduling

## Combining Methods

These techniques can be combined for maximum benefit:

**TTT + Best-of-N**:
```python
# Adapt model with TTT
adapted_model = ttt_adapt(model, test_prefix)

# Generate N samples from adapted model
candidates = [adapted_model.generate() for _ in range(N)]

# Select best
best = select_best(candidates, prm)
```

**Compute Scaling + Best-of-N**:
```python
# Determine optimal N given budget
optimal_N = compute_optimal_samples(budget, cost_per_sample)

# Generate and select
candidates = [model.generate() for _ in range(optimal_N)]
best = select_best(candidates, prm)
```

**All Three**:
```python
# Compute-optimal allocation
budget_per_sample, N = allocate_compute(total_budget)

# TTT adaptation per sample
adapted_models = [
    ttt_adapt(model, prompt, steps=budget_per_sample)
    for _ in range(N)
]

# Generate and select
candidates = [m.generate() for m in adapted_models]
best = select_best(candidates, prm)
```

## Performance vs. Compute Trade-offs

### TTT Layers

**Compute Cost**: O(sequence_length × optimization_steps)

**Performance Curve**:
```
Performance
    ^
    |     ___---  Diminishing returns
    |   _/
    |  /
    | /
    |/______________> Optimization Steps
```

**Sweet Spot**: 1-5 gradient steps per token

### Best-of-N

**Compute Cost**: O(N × generation_cost)

**Performance Curve**:
```
Performance
    ^
    |         ___---  Saturation
    |     __/
    |   _/
    |  /
    |/______________> Number of Samples (N)
```

**Sweet Spot**: N = 10-50 for most tasks

### Compute-Optimal Scaling

**Goal**: Maximize performance per dollar

**Trade-off Curves**:
```
Performance
    ^
    | A: Large model, few samples
    |  \
    |   \__ B: Medium model, medium samples
    |     \
    |      \__ C: Small model, many samples
    |        \
    |_________\______> Total Compute Cost
```

**Optimal Point**: Depends on scaling exponents

## Best Practices

### TTT Layers

1. **Learning Rate Selection**:
   - Too high: Instability, catastrophic forgetting
   - Too low: No adaptation
   - Typical: 0.001 - 0.1

2. **Number of Steps**:
   - More steps = more adaptation but slower
   - Start with 1-3 steps
   - Increase for harder distribution shift

3. **Loss Function**:
   - Self-supervised (reconstruction, prediction)
   - Task-agnostic (no labels at test time)
   - Stable gradients (avoid NaN)

4. **State Management**:
   - Reinitialize or carry over state?
   - Trade-off: Adaptation vs. stability

### Best-of-N with PRM

1. **Sample Diversity**:
   - Use temperature > 0 for sampling
   - Avoid duplicate outputs
   - Consider nucleus/top-k sampling

2. **PRM Quality**:
   - Train on diverse correct/incorrect examples
   - Validate on held-out set
   - Monitor calibration

3. **Number of Samples**:
   - Start with N=10
   - Increase until diminishing returns
   - Consider compute budget

4. **Selection Strategy**:
   - Use PRM scores (best-of-N)
   - Majority voting (consistency)
   - Weighted ensemble

### Compute-Optimal Scaling

1. **Measure Scaling Exponents**:
   - Run experiments at multiple scales
   - Fit power law: perf = a × compute^b
   - Validate on held-out budgets

2. **Account for Fixed Costs**:
   - Model loading time
   - Prompt processing
   - Output decoding

3. **Dynamic Allocation**:
   - Easy inputs: less compute
   - Hard inputs: more compute
   - Use uncertainty estimates

4. **Multi-Objective Optimization**:
   - Pareto front of latency vs. accuracy
   - User preference modeling
   - Cost-benefit analysis

## Implementation Patterns

### TTT Layer Integration

```python
class TransformerWithTTT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            TTTLayer(config) if i % 2 == 0 else AttentionLayer(config)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # TTT layers adapt during forward pass
        return x
```

### Best-of-N Pipeline

```python
def best_of_n_generate(model, prompt, prm, N=10):
    # Generate N candidates
    candidates = []
    for _ in range(N):
        output = model.generate(prompt, temperature=0.8)
        candidates.append(output)

    # Score all candidates
    scores = prm.score_batch(candidates)

    # Select best
    best_idx = scores.argmax()
    return candidates[best_idx]
```

### Compute-Optimal Allocation

```python
def allocate_compute(total_budget, cost_per_sample, scaling_exponent):
    """Determine optimal number of samples given budget."""

    # Scaling law: perf = a * (budget/N)^b where N is num samples
    # Optimal N maximizes: N * (budget/N)^b = budget^b / N^(b-1)
    # This gives: N_optimal ∝ budget^(b/(1+b))

    optimal_N = (total_budget / cost_per_sample) ** (scaling_exponent / (1 + scaling_exponent))

    return int(optimal_N)
```

## Monitoring and Debugging

### TTT Layers

**Key Metrics**:
- Adaptation loss per step
- Parameter drift magnitude
- Output distribution shift
- Inference latency

**Warning Signs**:
- Loss increases → learning rate too high
- No adaptation → learning rate too low
- NaN values → numerical instability
- Slow inference → too many steps

### Best-of-N

**Key Metrics**:
- Score distribution (PRM outputs)
- Best vs. average score gap
- Inter-sample diversity
- Selection accuracy (if ground truth available)

**Warning Signs**:
- All scores similar → PRM not discriminative
- Low diversity → sampling temperature too low
- Best score not correlated with quality → PRM miscalibrated

### Compute Scaling

**Key Metrics**:
- Performance vs. compute curve
- Scaling exponent estimates
- Utilization efficiency
- Cost per correct output

**Warning Signs**:
- Flat scaling curve → saturated performance
- Negative returns → redundant computation
- High variance → need more measurements

## Research Directions

### Open Problems

1. **Adaptive TTT Steps**: How many steps per token?
2. **Multi-Task TTT**: Sharing adaptation across tasks
3. **TTT Stability**: Preventing catastrophic forgetting
4. **PRM Generalization**: Transfer across domains
5. **Scaling Law Prediction**: Extrapolate to unseen budgets
6. **Compute Allocation**: Dynamic per-input budgeting

### Future Work

- **Learned Compute Allocation**: Meta-learning optimal budgets
- **Hierarchical TTT**: Adaptation at multiple scales
- **Process Supervision**: Finer-grained reward modeling
- **Hybrid Methods**: Combining multiple test-time techniques
- **Efficient Implementation**: Reducing overhead of test-time compute

## Key Papers

### TTT Layers

1. **Test-Time Training** (NeurIPS 2020)
   - Sun et al.
   - https://arxiv.org/abs/1909.13231

2. **TTT with Self-Supervision** (2024)
   - Sun et al.
   - https://arxiv.org/abs/2407.04620

### Best-of-N

3. **Process Reward Models** (2023)
   - Lightman et al.
   - OpenAI

4. **Scaling Test-Time Compute** (2024)
   - Snell et al.
   - https://arxiv.org/abs/2408.03314

### Compute Scaling

5. **Scaling Laws** (2020)
   - Kaplan et al.
   - https://arxiv.org/abs/2001.08361

6. **Chinchilla Scaling** (2022)
   - Hoffmann et al.
   - https://arxiv.org/abs/2203.15556

## File Structure

```
19_test_time_compute/
├── README.md                      # This file
├── ttt_layers.md                 # Test-Time Training Layers
├── compute_optimal_scaling.md    # Compute-Optimal Scaling Laws
└── best_of_n_prm.md             # Best-of-N with PRM
```

## Getting Started

1. **New to test-time compute?** Start with [Best-of-N with PRM](best_of_n_prm.md)
2. **Interested in adaptation?** Read [TTT Layers](ttt_layers.md)
3. **Optimizing resources?** Study [Compute-Optimal Scaling](compute_optimal_scaling.md)

Each documentation includes theory, implementation, code walkthroughs, and practical tips.
