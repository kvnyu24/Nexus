# DoRA: Weight-Decomposed Low-Rank Adaptation

## Overview & Motivation

DoRA (Weight-Decomposed Low-Rank Adaptation) improves upon LoRA by decomposing weight matrices into magnitude and directional components, then applying low-rank adaptation only to the directional component. This design more closely mirrors how full fine-tuning updates weights, achieving better performance with the same number of trainable parameters as LoRA.

### Efficiency Gains

- **Parameters**: Same as LoRA (+ small magnitude vector)
- **Performance**: Consistently outperforms LoRA by 1-3% across tasks
- **Compatibility**: Drop-in replacement for LoRA, similar training speed
- **Theory**: Better aligned with full fine-tuning behavior

### Key Insight

Analysis of full fine-tuning reveals two patterns:
1. **Magnitude changes**: The norm of weight vectors changes
2. **Directional changes**: The direction of weight vectors rotates

LoRA applies updates that conflate magnitude and direction. DoRA separates these, applying LoRA only to direction while making magnitude explicitly trainable.

## Theoretical Background

### Weight Decomposition

Any weight matrix W ∈ ℝ^(d×k) can be decomposed as:

```
W = m ⊙ (V / ||V||)
```

where:
- m ∈ ℝ^d: Magnitude vector (one scalar per output neuron)
- V ∈ ℝ^(d×k): Directional matrix
- ||V||: Column-wise (or row-wise) norm
- ⊙: Element-wise multiplication (broadcast)

This is analogous to polar coordinates: m is the radius, V/||V|| is the unit direction.

### Fine-Tuning Dynamics

Liu et al. (2024) analyzed how weights change during full fine-tuning:

**Observation 1**: Both magnitude and direction change significantly
```
Δm / m ≈ 10-30% (magnitude change)
cos(θ_before, θ_after) ≈ 0.8-0.95 (directional change)
```

**Observation 2**: LoRA primarily affects direction but implicitly changes magnitude
- LoRA update: W' = W + BA
- This changes both ||W'|| and direction(W') in a coupled way
- Suboptimal: the low-rank constraint applies to the coupled update

**Observation 3**: Full fine-tuning adjusts magnitude and direction somewhat independently

DoRA explicitly separates these transformations, allowing more efficient adaptation.

### Mathematical Formulation

#### DoRA Weight Update

For pre-trained weight W₀, DoRA computes:

```
W_DoRA = m ⊙ ((V₀ + ΔV) / ||V₀ + ΔV||)
```

where:
- V₀: Pre-trained directional matrix
- ΔV = B @ A: Low-rank directional update (same as LoRA)
- m: Trainable magnitude vector (initialized from ||V₀||)

#### Forward Pass

```
h = W_DoRA @ x
  = m ⊙ ((V₀ + BA) / ||V₀ + BA||) @ x
```

Steps:
1. Compute directional update: V' = V₀ + BA
2. Normalize: V_hat = V' / ||V'||_columns
3. Scale by magnitude: W_final = m ⊙ V_hat
4. Apply: h = W_final @ x

### Comparison with LoRA

| Aspect | LoRA | DoRA |
|--------|------|------|
| Update form | W + BA | m ⊙ (V + BA) / \|\|V + BA\|\| |
| Magnitude | Implicit | Explicit (trainable m) |
| Direction | Low-rank | Low-rank + normalized |
| Trainable params | rank×(d+k) | rank×(d+k) + d |
| Alignment with FT | Partial | Better |

DoRA adds only d extra parameters (one magnitude scalar per output dimension), negligible compared to the LoRA matrices.

## High-Level Intuition

### Why Separate Magnitude and Direction?

Consider adapting a weight vector from the pre-trained value w to a task-specific value w':

**Full fine-tuning** might:
1. Increase its magnitude from ||w|| = 1.0 to ||w'|| = 1.2
2. Rotate its direction by 15°

**LoRA** applies a low-rank update Δw, changing both magnitude and direction together. But the low-rank constraint limits the space of possible updates. If the optimal update requires a specific magnitude change AND a specific rotation, LoRA might not be able to represent it well with low rank.

**DoRA** separates these:
1. Magnitude: m learns to scale to 1.2 (just one parameter)
2. Direction: BA learns the 15° rotation (low-rank)

This separation makes the low-rank bottleneck less restrictive.

### Analogy: Navigation

Imagine navigating to a destination:
- **Full fine-tuning**: "Walk 120 meters northeast" (free to choose any magnitude and direction)
- **LoRA**: "Take a low-rank path" (constrained path, both distance and direction limited)
- **DoRA**: "Walk northeast (low-rank direction), but choose your own distance" (more flexibility)

### Normalization Effect

The normalization ||V + BA||⁻¹ ensures that the directional component remains on the unit sphere. This:
1. Prevents magnitude from interfering with direction learning
2. Stabilizes training (gradients for direction don't explode with norm)
3. Allows m to independently control overall scaling

## Implementation Details

### Creating a DoRA Model

```python
from nexus.models.compression.peft import apply_dora, DoRAConfig

config = DoRAConfig(
    rank=8,
    alpha=16.0,
    dropout=0.1,
    target_modules=['q_proj', 'v_proj'],
    magnitude_trainable=True  # Default: True
)

model = apply_dora(model, config=config)
```

### DoRA Forward Pass

Reference: `/Users/kevinyu/Projects/Nexus/nexus/models/compression/peft/dora.py`

```python
class DoRALinear(NexusModule):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.linear.weight  # V₀ (frozen)

        # Compute LoRA directional update
        delta_v = self.lora_B @ self.lora_A  # (out_features, in_features)

        # Updated directional component
        v_updated = weight + delta_v * self.scaling

        # Column-normalize (unit direction)
        v_norm = torch.norm(v_updated, dim=1, keepdim=True).clamp(min=1e-8)
        v_hat = v_updated / v_norm

        # Apply magnitude scaling
        weight_final = self.magnitude.unsqueeze(1) * v_hat

        # Compute output
        return F.linear(x, weight_final, self.bias)
```

### Initialization

Critical: Initialize magnitude from pre-trained weight norms:

```python
def _initialize_magnitude_from_weight(self) -> None:
    with torch.no_grad():
        weight = self.linear.weight.data
        # Compute row-wise (per-output) norms
        col_norms = torch.norm(weight, dim=1)
        self.magnitude.data.copy_(col_norms)
```

This ensures that at initialization (when BA=0), DoRA exactly reproduces the pre-trained model:

```
W_DoRA = m ⊙ (V₀ / ||V₀||) = ||V₀|| ⊙ (V₀ / ||V₀||) = V₀ ✓
```

### Gradient Flow

During backpropagation, gradients flow through:

1. **Magnitude**: `∂L/∂m` (d parameters, one per output neuron)
2. **LoRA A**: `∂L/∂A` through the normalization and magnitude scaling
3. **LoRA B**: `∂L/∂B` through the normalization and magnitude scaling

The normalization introduces non-linearity in the gradient flow, which helps decorrelate magnitude and direction learning.

## Code Walkthrough

### Core Components

```python
class DoRALinear(NexusModule):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        # Frozen pre-trained weight
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False

        # Trainable magnitude vector (one per output)
        self.magnitude = nn.Parameter(
            torch.ones(out_features),
            requires_grad=True
        )

        # LoRA matrices for directional update
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.scaling = alpha / rank
```

### Forward Computation

Step-by-step:

```python
def forward(self, x):
    # 1. Get frozen weight V₀
    V0 = self.linear.weight  # (out_features, in_features)

    # 2. Compute low-rank update
    delta_V = self.lora_B @ self.lora_A  # (out_features, in_features)

    # 3. Updated directional matrix
    V_updated = V0 + self.scaling * delta_V

    # 4. Normalize to unit direction
    # Norm along dim=1 (per output neuron)
    norms = torch.norm(V_updated, dim=1, keepdim=True)
    V_hat = V_updated / norms.clamp(min=1e-8)

    # 5. Scale by learned magnitude
    # magnitude: (out_features,) → (out_features, 1) for broadcasting
    W_final = self.magnitude.unsqueeze(1) * V_hat

    # 6. Apply linear transformation
    return F.linear(x, W_final, self.linear.bias)
```

### Magnitude Initialization

```python
@classmethod
def from_linear(cls, linear: nn.Linear, rank=8, alpha=16.0):
    dora = cls(
        in_features=linear.in_features,
        out_features=linear.out_features,
        rank=rank,
        alpha=alpha,
    )

    # Copy frozen weight
    dora.linear.weight.data.copy_(linear.weight.data)

    # Initialize magnitude from weight norms
    weight_norms = torch.norm(linear.weight.data, dim=1)
    dora.magnitude.data.copy_(weight_norms)

    return dora
```

## Optimization Tricks

### 1. Rank Selection

DoRA benefits from slightly lower ranks than LoRA due to the decomposition:

| Task Complexity | LoRA Rank | DoRA Rank | Reasoning |
|----------------|-----------|-----------|-----------|
| Simple | r=8 | r=4 | Direction alone is simpler |
| Medium | r=16 | r=8 | Magnitude handles part of adaptation |
| Complex | r=32 | r=16 | Still benefits from decomposition |

**Guideline**: Try half the LoRA rank first, then increase if needed.

### 2. Magnitude Learning Rate

The magnitude vector often benefits from a different learning rate than LoRA matrices:

```python
lora_params = [p for n, p in model.named_parameters() if 'lora' in n]
mag_params = [p for n, p in model.named_parameters() if 'magnitude' in n]

optimizer = torch.optim.AdamW([
    {'params': lora_params, 'lr': 1e-3},
    {'params': mag_params, 'lr': 3e-3}  # 3× higher for magnitude
])
```

### 3. Normalization Epsilon

The clamping epsilon in normalization affects stability:

```python
v_hat = v_updated / torch.norm(v_updated, dim=1, keepdim=True).clamp(min=1e-8)
```

- Too small (1e-10): Risk of numerical instability
- Too large (1e-6): Prevents learning very small directional updates
- **Recommended**: 1e-8 (default)

### 4. Freezing Magnitude

For some tasks, you may want to freeze magnitude and only adapt direction:

```python
config = DoRAConfig(
    rank=8,
    alpha=16.0,
    magnitude_trainable=False  # Freeze magnitude
)
```

This reduces DoRA to normalized LoRA, useful when you want to preserve pre-trained scaling.

### 5. Combining with Quantization

DoRA works with QLoRA-style quantization:

```python
# 1. Quantize base model
model = quantize_model(model, bits=4)

# 2. Apply DoRA
model = apply_dora(model, config=config)
```

The magnitude vector remains in full precision while the base directional matrix is quantized.

### 6. Gradient Clipping

The normalization can sometimes lead to large gradients. Use gradient clipping:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Experiments & Results

### Original DoRA Paper Results

**RoBERTa-base on GLUE**:

| Method | MNLI | QQP | QNLI | SST-2 | Average |
|--------|------|-----|------|-------|---------|
| Full FT | 87.6 | 91.9 | 92.8 | 94.8 | 91.8 |
| LoRA (r=8) | 86.4 | 91.3 | 92.0 | 93.7 | 90.9 |
| **DoRA (r=8)** | **87.2** | **91.7** | **92.5** | **94.3** | **91.4** |

DoRA recovers 62% of the gap between LoRA and full fine-tuning.

**LLaMA-7B on Commonsense Reasoning**:

| Method | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Avg |
|--------|-------|------|------|-----------|------------|-------|-------|------|-----|
| Base | 75.7 | 79.1 | 48.9 | 76.0 | 69.2 | 74.6 | 46.1 | 57.8 | 65.9 |
| LoRA | 77.1 | 79.5 | 49.2 | 76.8 | 70.0 | 75.8 | 47.3 | 59.0 | 66.8 |
| **DoRA** | **78.4** | **80.2** | **49.8** | **77.6** | **71.1** | **76.9** | **48.7** | **60.4** | **67.9** |

DoRA provides consistent improvements of +1.1% average over LoRA.

### Efficiency Comparison

**Training Speed** (LLaMA-7B, batch size 8):

| Method | Params (M) | Memory (GB) | Time/Iter (s) | Speedup vs Full FT |
|--------|-----------|-------------|---------------|---------------------|
| Full FT | 7,000 | 28.0 | 2.1 | 1× |
| LoRA | 4.2 | 9.5 | 0.8 | 2.6× |
| **DoRA** | **4.2** | **9.6** | **0.9** | **2.3×** |

DoRA has negligible overhead over LoRA (only the magnitude vector + normalization).

### Ablation Studies

**Effect of Magnitude Training**:

| Variant | MNLI Acc | Notes |
|---------|----------|-------|
| DoRA (full) | 87.2 | Magnitude trainable |
| DoRA (frozen mag) | 86.8 | Magnitude fixed at init |
| LoRA | 86.4 | No decomposition |

Training magnitude is crucial for performance.

**Effect of Normalization**:

| Variant | MNLI Acc | Notes |
|---------|----------|-------|
| DoRA (normalized) | 87.2 | V / \|\|V\|\| |
| DoRA (no norm) | 86.5 | Just m ⊙ (V + BA) |
| LoRA | 86.4 | W + BA |

Normalization is essential to decouple magnitude and direction.

## Common Pitfalls

### 1. Forgetting Magnitude Initialization

**Symptom**: Training starts with high loss or diverges immediately.

**Diagnosis**:
```python
# Check if magnitude was initialized
print(model.magnitude)  # Should not be all ones or random
```

**Solution**: Always call `_initialize_magnitude_from_weight()` after loading pre-trained weights:
```python
dora_layer = DoRALinear.from_linear(pretrained_linear, ...)
# from_linear automatically initializes magnitude
```

### 2. Magnitude Explosion

**Symptom**: Loss becomes NaN or inf after a few steps.

**Cause**: Magnitude learning rate too high, causing unbounded growth.

**Solution**:
- Reduce magnitude learning rate
- Apply gradient clipping
- Use weight decay on magnitude: `weight_decay=0.01`

### 3. Incorrect Normalization Dimension

**Symptom**: Shapes don't match or poor performance.

**Solution**: Normalize along the input dimension (dim=1 for weight matrices of shape (out, in)):
```python
# Correct:
v_hat = v / torch.norm(v, dim=1, keepdim=True)

# Incorrect:
v_hat = v / torch.norm(v, dim=0, keepdim=True)
```

### 4. Combining with Certain Optimizers

**Symptom**: Training unstable with some optimizers.

**Cause**: The normalization creates a non-linear constraint on the LoRA parameters. Some optimizers (e.g., SGD without momentum) may struggle.

**Solution**: Use adaptive optimizers like AdamW:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

### 5. Not Accounting for Extra Compute

**Symptom**: Training slower than expected compared to LoRA.

**Cause**: DoRA adds normalization overhead (~10-15%).

**Solution**: This is expected. DoRA trades slight compute increase for better accuracy. If speed is critical, stick with LoRA.

### 6. Ignoring Magnitude in Evaluation

**Symptom**: Merged model performs differently than expected.

**Cause**: Magnitude vector wasn't properly incorporated during merging.

**Solution**: DoRA doesn't support simple weight merging like LoRA (due to normalization). For inference, keep the DoRALinear module or implement custom merging:
```python
# DoRA "merged" weight is:
W_merged = m ⊙ ((V₀ + BA) / ||V₀ + BA||)

# This is not a simple addition, so must be computed at inference time
```

### 7. Quantization Order

**Symptom**: Errors when combining with quantization.

**Solution**: Quantize first, then apply DoRA:
```python
# Correct order:
model = quantize_model(model, bits=4)
model = apply_dora(model, config=config)

# Incorrect:
model = apply_dora(model, config=config)
model = quantize_model(model, bits=4)  # May quantize DoRA params!
```

## References

1. **Original Paper**: Liu, S., et al. "DoRA: Weight-Decomposed Low-Rank Adaptation." ICML 2024. https://arxiv.org/abs/2402.09353

2. **LoRA**: Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.

3. **Weight Magnitude Analysis**: Frankle, J., & Carbin, M. "The Lottery Ticket Hypothesis." ICLR 2019.

4. **Normalization in Neural Networks**: Ba, J. L., et al. "Layer Normalization." arXiv 2016.

5. **Nexus Implementation**: `/Users/kevinyu/Projects/Nexus/nexus/models/compression/peft/dora.py`

## See Also

- [LoRA](lora.md): Foundation of DoRA
- [QLoRA](qlora.md): Quantized LoRA (compatible with DoRA)
- [AdaLoRA](adalora.md): Adaptive rank allocation
- [LoRA+](lora_plus.md): Improved LoRA optimization
