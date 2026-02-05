# LoRA: Low-Rank Adaptation of Large Language Models

## Overview & Motivation

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning (PEFT) technique that drastically reduces the number of trainable parameters when adapting large pre-trained language models to downstream tasks. Instead of fine-tuning all model parameters, LoRA freezes the pre-trained weights and injects trainable low-rank decomposition matrices into each targeted layer.

### Efficiency Gains

- **Memory**: Reduces trainable parameters by 10,000x (e.g., from 175B to 4.7M for GPT-3)
- **Compute**: Faster training with fewer parameters to update
- **Storage**: Model checkpoints are <1MB instead of hundreds of GB
- **Multi-task**: Can store multiple task-specific adapters with minimal overhead

### Key Motivation

Full fine-tuning of billion-parameter models is prohibitively expensive. Prior work showed that the weight updates during fine-tuning have low intrinsic dimensionality - they can be represented in a much lower-rank subspace. LoRA exploits this observation by constraining the update to a low-rank form from the start.

## Theoretical Background

### Intrinsic Dimensionality

Aghajanyan et al. (2020) demonstrated that pre-trained language models have a low "intrinsic dimension" - the parameter updates during fine-tuning reside in a low-dimensional subspace despite the high-dimensional parameter space. This suggests that effective fine-tuning doesn't require updating all parameters in their full dimensionality.

### Low-Rank Matrix Factorization

Any matrix update ΔW ∈ ℝ^(d×k) can be decomposed as:

```
ΔW = B @ A
```

where:
- A ∈ ℝ^(r×k) is the down-projection matrix (k → r)
- B ∈ ℝ^(d×r) is the up-projection matrix (r → d)
- r << min(d, k) is the rank

This decomposition reduces parameters from d×k to (d+k)×r. For example, with d=4096, k=4096, and r=8:
- Full update: 16,777,216 parameters
- LoRA update: 65,536 parameters (256× reduction)

## Mathematical Formulation

### Forward Pass

For a pre-trained weight matrix W₀ ∈ ℝ^(d×k), the adapted forward pass is:

```
h = W₀x + ΔWx = W₀x + BAx
```

More precisely, with scaling factor α:

```
h = W₀x + (α/r) · B · A · x
```

where:
- W₀: Frozen pre-trained weight
- B ∈ ℝ^(d×r): Trainable up-projection (initialized to zeros)
- A ∈ ℝ^(r×k): Trainable down-projection (initialized with Kaiming)
- α: Scaling hyperparameter (tuned independent of r)
- r: Adapter rank

### Initialization Strategy

**Matrix A**: Initialized with Kaiming uniform initialization
```
A ~ U(-√(5/k), √(5/k))
```

**Matrix B**: Initialized to zeros
```
B = 0
```

This ensures ΔW = BA = 0 at initialization, so the model starts with the exact pre-trained behavior.

### Scaling Factor

The scaling α/r is crucial:
- **α** acts as a learning rate multiplier for the adapter
- Larger α → larger effective updates
- **r** in denominator normalizes updates by rank
- Decouples learning rate from rank choice
- Typical values: α = 16 or α = 32 with r = 8

## High-Level Intuition

### Why Low-Rank Works

Think of neural network layers as performing transformations in high-dimensional space. During pre-training, the model learns a general-purpose transformation. Fine-tuning adjusts this transformation for a specific task, but the adjustment is typically "simple" - it doesn't need to explore the full high-dimensional space.

LoRA hypothesizes that this task-specific adjustment lies in a low-dimensional subspace. By constraining updates to rank-r matrices, we're forcing the model to find efficient, low-dimensional adjustments.

### Analogy

Consider a 1000-dimensional function. To adapt it to a new task, you don't need to adjust all 1000 dimensions independently. Perhaps adjusting along just 8 carefully chosen directions (the low-rank subspace) is sufficient. LoRA finds these 8 directions via gradient descent.

### Target Module Selection

LoRA can be applied to any weight matrix, but common choices are:
- **Attention projections**: q_proj, k_proj, v_proj, o_proj
- **FFN layers**: up_proj, down_proj, gate_proj

Empirically, adapting query and value projections (q_proj, v_proj) provides a good accuracy/efficiency trade-off.

## Implementation Details

### Module Replacement

LoRA implementation wraps existing `nn.Linear` layers:

```python
from nexus.models.compression.peft import LoRALinear

# Original layer
original_layer = nn.Linear(768, 768)

# Replace with LoRA layer
lora_layer = LoRALinear.from_linear(
    original_layer,
    rank=8,
    alpha=16.0,
    dropout=0.1
)
```

### Applying to Entire Model

Use `apply_lora()` to inject adapters into all matching layers:

```python
from nexus.models.compression.peft import apply_lora, LoRAConfig

config = LoRAConfig(
    rank=8,
    alpha=16.0,
    dropout=0.1,
    target_modules=['q_proj', 'v_proj'],
    bias='none'
)

model = apply_lora(model, config=config)
```

### Parameter Freezing

After applying LoRA, verify that only adapter parameters are trainable:

```python
# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
```

### Merging for Inference

For deployment, merge adapters into the base weights to eliminate overhead:

```python
from nexus.models.compression.peft import merge_lora

# Merge adapters into weights
model = merge_lora(model)

# Now forward pass has no LoRA overhead
output = model(input)
```

After merging, the adapted model is indistinguishable from a fully fine-tuned model in terms of architecture.

## Code Walkthrough

### LoRALinear Implementation

Reference: `/Users/kevinyu/Projects/Nexus/nexus/models/compression/peft/lora.py`

```python
class LoRALinear(NexusModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        # Frozen base layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False

        # Trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Scaling
        self.scaling = alpha / rank

        # Dropout on adapter path
        self.lora_dropout = nn.Dropout(p=dropout)
```

### Forward Pass

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Base output (frozen)
    base_output = self.linear(x)

    # LoRA adapter path
    lora_input = self.lora_dropout(x)
    lora_output = F.linear(
        F.linear(lora_input, self.lora_A),
        self.lora_B
    )

    # Combine with scaling
    return base_output + lora_output * self.scaling
```

The forward pass computes:
1. Base transformation: `W₀x`
2. Adapter transformation: `BAx` (two sequential linear ops)
3. Scaled sum: `W₀x + (α/r)BAx`

### Weight Merging

```python
def merge_weights(self) -> None:
    with torch.no_grad():
        # Compute ΔW = (α/r) * B @ A
        delta_w = (self.lora_B @ self.lora_A) * self.scaling

        # Add to frozen weight
        self.linear.weight.data += delta_w

    self.merged = True
```

After merging, `self.linear.weight` contains `W₀ + ΔW`, and the forward pass can bypass the adapter computation.

## Optimization Tricks

### 1. Rank Selection

**Trade-off**: Lower rank → fewer parameters but may limit expressiveness

| Rank | Parameters (per 4096×4096 layer) | Typical Use Case |
|------|----------------------------------|------------------|
| r=1  | 8,192                            | Extremely limited tasks |
| r=4  | 32,768                           | Simple classification |
| r=8  | 65,536                           | General tasks (recommended) |
| r=16 | 131,072                          | Complex reasoning |
| r=64 | 524,288                          | Nearly full expressiveness |

**Guideline**: Start with r=8. Increase if validation performance plateaus too early.

### 2. Alpha Tuning

The scaling factor α/r controls the magnitude of adapter updates:

- **α = r**: No scaling (equivalent to learning rate × 1)
- **α = 2r**: Double the effective learning rate for adapters
- **α >> r**: Strong adapter influence (useful for large distribution shifts)

**Best Practice**: Set α = 2×r for most tasks. For fine-tuning on very different distributions, try α = 4×r.

### 3. Target Module Selection

Not all layers benefit equally from LoRA:

| Module Set | Efficiency | Accuracy | Notes |
|------------|-----------|----------|-------|
| q_proj, v_proj | High | Good | Recommended default |
| q_proj, k_proj, v_proj, o_proj | Medium | Better | Balanced choice |
| All linear layers | Low | Best | Diminishing returns |

**Recommendation**: Start with {q_proj, v_proj}. Add k_proj and o_proj if accuracy is insufficient.

### 4. Dropout for Regularization

LoRA layers can overfit on small datasets. Use dropout on the adapter path:

```python
config = LoRAConfig(
    rank=8,
    alpha=16.0,
    dropout=0.1  # 10% dropout on adapter inputs
)
```

Typical values: 0.05 to 0.2 for small datasets (<10k examples).

### 5. Learning Rate Adjustment

LoRA adapters often benefit from higher learning rates than full fine-tuning:

```python
# Separate learning rates for base model and LoRA
from nexus.models.compression.peft import get_lora_parameters

lora_params = get_lora_parameters(model)
base_params = [p for p in model.parameters() if p not in lora_params]

optimizer = torch.optim.AdamW([
    {'params': base_params, 'lr': 1e-5},  # If training base at all
    {'params': lora_params, 'lr': 1e-3}   # 100× higher for adapters
])
```

### 6. Gradient Checkpointing

For very large models, combine LoRA with gradient checkpointing:

```python
model.gradient_checkpointing_enable()
model = apply_lora(model, config=config)
```

This trades compute for memory, enabling larger batch sizes.

## Experiments & Results

### Reference Results (LoRA Paper)

**GPT-3 175B on Various Tasks**:

| Method | # Trainable Params | MNLI (Acc) | SQuAD (F1) | SAMSum (R-L) |
|--------|-------------------|------------|------------|--------------|
| Full FT | 175B (100%) | 89.5 | 91.9 | 53.8 |
| Adapter | 40M (0.023%) | 89.4 | 90.1 | 53.0 |
| Prefix | 40M (0.023%) | 88.5 | 88.7 | 52.7 |
| **LoRA** | **4.7M (0.003%)** | **91.7** | **93.2** | **54.9** |

LoRA achieves better accuracy than full fine-tuning with 37,000× fewer trainable parameters.

**Rank Ablation (RoBERTa-base on MNLI)**:

| Rank (r) | Accuracy | # Trainable Params |
|----------|----------|-------------------|
| r=1      | 86.8     | 0.1M              |
| r=2      | 88.1     | 0.2M              |
| r=4      | 89.2     | 0.4M              |
| r=8      | 90.7     | 0.8M              |
| r=64     | 90.9     | 6.4M              |

Diminishing returns beyond r=8 for this task.

### Compression Ratios

For a typical 7B parameter model (e.g., LLaMA-7B):

| Configuration | Trainable Params | Reduction Factor | Checkpoint Size |
|---------------|-----------------|------------------|-----------------|
| Full Fine-tune | 7B | 1× | 28 GB |
| LoRA r=8, {q,v} | 4.2M | 1,667× | 17 MB |
| LoRA r=16, {q,k,v,o} | 33.5M | 209× | 134 MB |

### Performance Retention

Across diverse tasks, LoRA typically retains 95-102% of full fine-tuning performance:

```
Performance Ratio = (LoRA Accuracy) / (Full Fine-tune Accuracy)
```

| Task Category | Avg. Retention |
|---------------|---------------|
| Text Classification | 98-100% |
| Question Answering | 97-101% |
| Summarization | 96-99% |
| Translation | 99-102% |

## Common Pitfalls

### 1. Rank Too Low

**Symptom**: Model fails to converge or plateaus at poor accuracy.

**Solution**: Increase rank from 8 → 16 → 32. Monitor validation loss to find the sweet spot.

### 2. Forgetting to Freeze Base Model

**Symptom**: Memory usage unexpectedly high, training slower than expected.

**Diagnosis**:
```python
# Check if base model is frozen
for name, param in model.named_parameters():
    if 'lora' not in name and param.requires_grad:
        print(f"WARNING: {name} is trainable but shouldn't be!")
```

**Solution**: Ensure `apply_lora()` properly froze base weights.

### 3. Incorrect Scaling

**Symptom**: Training is unstable, loss spikes, or model doesn't learn.

**Solution**: Verify α/r scaling is applied. Default α=16 with r=8 gives scaling of 2.0, which is reasonable.

### 4. Not Merging for Deployment

**Symptom**: Inference is slower than expected, extra memory overhead.

**Solution**: Always call `merge_lora(model)` before deployment to eliminate adapter overhead.

### 5. Target Module Mismatch

**Symptom**: No parameters are trainable, or training has no effect.

**Diagnosis**:
```python
# Check what was matched
for name, module in model.named_modules():
    if isinstance(module, LoRALinear):
        print(f"LoRA applied to: {name}")
```

**Solution**: Verify `target_modules` patterns match your model's layer names. Use regex for flexible matching:
```python
target_modules=['.*q_proj', '.*v_proj']  # Matches any q_proj, v_proj
```

### 6. Incompatible with Model Parallelism

**Symptom**: Errors when using multi-GPU training or model sharding.

**Solution**: Apply LoRA after model parallelism setup, or use FSDP-compatible LoRA implementations.

### 7. Overfitting on Small Datasets

**Symptom**: Training accuracy high, validation accuracy low.

**Solution**:
- Increase dropout: `dropout=0.2`
- Reduce rank: try r=4 instead of r=8
- Use weight decay: `weight_decay=0.01` in optimizer
- Early stopping based on validation loss

## References

1. **Original Paper**: Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. https://arxiv.org/abs/2106.09685

2. **Intrinsic Dimensionality**: Aghajanyan, A., et al. "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." ACL 2021. https://arxiv.org/abs/2012.13255

3. **Implementation**: Hugging Face PEFT library: https://github.com/huggingface/peft

4. **Nexus Implementation**: `/Users/kevinyu/Projects/Nexus/nexus/models/compression/peft/lora.py`

5. **Related Work**:
   - Adapter Layers (Houlsby et al., 2019): https://arxiv.org/abs/1902.00751
   - Prefix Tuning (Li & Liang, 2021): https://arxiv.org/abs/2101.00190
   - Prompt Tuning (Lester et al., 2021): https://arxiv.org/abs/2104.08691

## See Also

- [QLoRA](qlora.md): Quantized LoRA for 4-bit training
- [DoRA](dora.md): Weight-decomposed LoRA variant
- [LoRA+](lora_plus.md): Improved learning rate scheduling for LoRA
- [AdaLoRA](adalora.md): Adaptive rank allocation
- [GaLore](galore.md): Gradient low-rank projection (alternative approach)
