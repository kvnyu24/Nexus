# Parameter-Efficient Fine-Tuning (PEFT)

Parameter-Efficient Fine-Tuning techniques enable adapting large pre-trained models to downstream tasks while training only a tiny fraction of the model's parameters. These methods dramatically reduce memory requirements, training time, and storage costs while maintaining competitive or superior performance compared to full fine-tuning.

## Overview

| Method | Trainable Params | Key Innovation | Best For |
|--------|------------------|----------------|----------|
| [LoRA](lora.md) | 0.1-1% | Low-rank weight updates | General-purpose adaptation |
| [QLoRA](qlora.md) | 0.1-1% | 4-bit base + LoRA adapters | Memory-constrained training |
| [DoRA](dora.md) | 0.1-1% | Magnitude-direction decomposition | Improved accuracy over LoRA |
| [LoRA+](lora_plus.md) | 0.1-1% | Optimized learning rates for A/B | Faster convergence |
| [GaLore](galore.md) | 0% (optimizer) | Gradient low-rank projection | Full-rank updates, low memory |
| [LISA](lisa.md) | 0.5-2% | Layerwise importance sampling | Long-context adaptation |
| [AdaLoRA](adalora.md) | 0.1-1% | Adaptive rank allocation | Heterogeneous layer importance |
| [rsLoRA](rslora.md) | 0.1-1% | Rank-stabilized scaling | Training stability |

## When to Use PEFT

### Use Cases

**Ideal for PEFT:**
- Fine-tuning models >1B parameters
- Limited GPU memory (single consumer GPU)
- Multi-task learning (store many adapters)
- Rapid experimentation (fast training)
- Domain adaptation with limited data

**Consider Full Fine-Tuning:**
- Models <1B parameters (PEFT overhead not worth it)
- Unlimited compute budget
- Extreme distribution shift requiring architectural changes
- Tasks where every 0.1% accuracy matters

### Efficiency Comparison

For a 7B parameter model on a classification task:

| Method | Memory (GB) | Training Time | Accuracy | Checkpoint Size |
|--------|-------------|---------------|----------|-----------------|
| Full FT | 28 | 10 hours | 92.5% | 28 GB |
| LoRA (r=8) | 9.5 | 3 hours | 92.1% | 17 MB |
| QLoRA (r=16) | 7.5 | 4 hours | 92.3% | 34 MB |
| DoRA (r=8) | 9.6 | 3.5 hours | 92.7% | 17 MB |
| GaLore | 12 | 8 hours | 92.4% | 28 GB (full rank) |

## Quick Start

### Basic LoRA

```python
from nexus.models.compression.peft import apply_lora, LoRAConfig

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Apply LoRA
config = LoRAConfig(
    rank=8,
    alpha=16.0,
    target_modules=['q_proj', 'v_proj'],
    dropout=0.1
)
model = apply_lora(model, config=config)

# Train normally
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Merge for deployment
from nexus.models.compression.peft import merge_lora
model = merge_lora(model)
```

### Memory-Efficient QLoRA

```python
from nexus.models.compression.peft import apply_qlora, QLoRAConfig

# Load in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Apply QLoRA
config = QLoRAConfig(
    rank=16,
    alpha=32.0,
    bits=4,
    double_quant=True,
    compute_dtype=torch.bfloat16
)
model = apply_qlora(model, config=config)

# Train with paged optimizer
from bitsandbytes.optim import AdamW8bit
optimizer = AdamW8bit(model.parameters(), lr=2e-4)
```

### Improved Accuracy with DoRA

```python
from nexus.models.compression.peft import apply_dora, DoRAConfig

config = DoRAConfig(
    rank=8,  # Can use lower rank than LoRA
    alpha=16.0,
    magnitude_trainable=True
)
model = apply_dora(model, config=config)
```

## Method Selection Guide

### Decision Tree

```
Start
  ├─ Memory unlimited?
  │   ├─ Yes → Use Full Fine-Tuning
  │   └─ No → Continue
  │
  ├─ Can fit model in GPU memory (16-bit)?
  │   ├─ Yes
  │   │   ├─ Need best accuracy? → DoRA
  │   │   └─ Need speed? → LoRA
  │   └─ No → QLoRA (4-bit + adapters)
  │
  ├─ Need full-rank updates?
  │   └─ Yes → GaLore
  │
  ├─ Long context (>4k tokens)?
  │   └─ Yes → LISA
  │
  └─ Heterogeneous layer importance?
      └─ Yes → AdaLoRA
```

### Performance vs Efficiency

```
High Performance
     │
     │  DoRA ●
     │
     │  LoRA+ ●  AdaLoRA ●
     │
     │  LoRA ●    QLoRA ●
     │
     │           GaLore ●
     │
     │  LISA ●
Low  └────────────────────────────────→ High
    Performance         Efficiency
```

## Hyperparameter Guidelines

### Rank Selection

| Model Size | Simple Tasks | Medium Tasks | Complex Tasks |
|------------|--------------|--------------|---------------|
| <1B | r=4 | r=8 | r=16 |
| 1-10B | r=8 | r=16 | r=32 |
| 10-70B | r=16 | r=32 | r=64 |
| >70B | r=32 | r=64 | r=128 |

**Rule of thumb**: Start with r=8 for most tasks. Double if validation loss plateaus.

### Alpha Scaling

```
Effective_LR = alpha / rank * base_LR
```

| Configuration | Alpha | Use Case |
|---------------|-------|----------|
| α = r | 8 | Conservative updates |
| α = 2r | 16 | Default (recommended) |
| α = 4r | 32 | Large distribution shift |

### Target Modules

**Attention Only (Recommended)**:
```python
target_modules = ['q_proj', 'v_proj']  # Minimum
target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']  # Better
```

**Attention + FFN (Maximum)**:
```python
target_modules = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'up_proj', 'down_proj'
]
```

**Trade-off**:
- More modules → Better accuracy, slower training
- Attention-only → Good accuracy, fast training

### Learning Rates

| Method | Base Model LR | Adapter LR | Ratio |
|--------|---------------|------------|-------|
| LoRA | 1e-5 (frozen) | 1e-3 | 100× |
| QLoRA | frozen | 2e-4 | N/A |
| DoRA | frozen | 1e-3 | 100× |
| GaLore | 1e-4 | N/A | N/A |

**Tip**: PEFT adapters can tolerate 10-100× higher learning rates than full fine-tuning.

## Advanced Topics

### Combining Methods

**QLoRA + DoRA**: Best of both worlds
```python
# 1. Quantize base model
model = quantize_model(model, bits=4)

# 2. Apply DoRA
model = apply_dora(model, config=DoRAConfig(rank=16, alpha=32.0))
```

**LoRA + Pruning**: Sparse adapters
```python
# 1. Apply LoRA
model = apply_lora(model, config)

# 2. Prune LoRA matrices
prune_lora_adapters(model, sparsity=0.5)
```

### Multi-Task Adapters

Store multiple task-specific adapters:

```python
# Train task 1
model = apply_lora(model, config, task_name="task1")
train(model)
save_lora_state_dict(model, "task1.pth")

# Train task 2 (reload base model)
model = reload_base_model()
model = apply_lora(model, config, task_name="task2")
train(model)
save_lora_state_dict(model, "task2.pth")

# Inference: load specific adapter
model = reload_base_model()
model = apply_lora(model, config)
load_lora_state_dict(model, "task1.pth")  # Switch tasks
```

### Adapter Merging

Combine multiple adapters:

```python
# Arithmetic mean
W_merged = W_base + (ΔW_task1 + ΔW_task2) / 2

# Task vectors (Ilharco et al., 2022)
W_merged = W_base + λ₁*ΔW_task1 + λ₂*ΔW_task2

# DARE (Yadav et al., 2023)
W_merged = W_base + drop_and_rescale([ΔW_task1, ΔW_task2])
```

## Common Issues & Solutions

### Issue 1: Poor Convergence

**Symptoms**: Loss doesn't decrease, accuracy stuck at random guess.

**Solutions**:
1. Increase learning rate (try 1e-3 for adapters)
2. Increase rank (r=8 → r=16)
3. Add more target modules (q,v → q,k,v,o)
4. Check if base model is properly frozen

### Issue 2: Overfitting

**Symptoms**: Training accuracy high, validation accuracy low.

**Solutions**:
1. Add dropout: `dropout=0.1` or `0.2`
2. Reduce rank: r=16 → r=8
3. Use weight decay: `weight_decay=0.01`
4. Early stopping based on validation loss

### Issue 3: Out of Memory

**Solutions**:
1. Switch to QLoRA (4-bit quantization)
2. Enable gradient checkpointing: `model.gradient_checkpointing_enable()`
3. Reduce batch size, increase gradient accumulation
4. Use paged optimizers (bitsandbytes)

### Issue 4: Training Slower Than Expected

**Solutions**:
1. Merge adapters before deployment (eliminate forward pass overhead)
2. Use compiled LoRA kernels (e.g., from Hugging Face PEFT)
3. Reduce number of target modules
4. Use smaller rank

## Benchmarks

### Accuracy (LLaMA-7B on MMLU)

| Method | Rank | Accuracy | vs Full FT |
|--------|------|----------|------------|
| Full FT | - | 42.1% | 100% |
| LoRA | r=8 | 41.3% | 98.1% |
| LoRA | r=64 | 41.8% | 99.3% |
| DoRA | r=8 | 41.7% | 99.0% |
| QLoRA | r=16 | 41.5% | 98.6% |
| AdaLoRA | r_avg=8 | 41.6% | 98.8% |

### Memory Usage (7B Model, Batch Size 8)

| Method | GPU Memory | Reduction vs Full FT |
|--------|------------|---------------------|
| Full FT | 28 GB | 1.0× |
| LoRA (r=8) | 9.5 GB | 2.9× |
| DoRA (r=8) | 9.6 GB | 2.9× |
| QLoRA (r=16) | 7.5 GB | 3.7× |
| GaLore | 12 GB | 2.3× |

### Training Speed (7B Model, 1k Steps)

| Method | Time | Speedup vs Full FT |
|--------|------|-------------------|
| Full FT | 10.0 hrs | 1.0× |
| LoRA | 3.2 hrs | 3.1× |
| DoRA | 3.5 hrs | 2.9× |
| QLoRA | 4.1 hrs | 2.4× |
| GaLore | 8.2 hrs | 1.2× |

## References

### Papers

1. **LoRA**: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
2. **QLoRA**: Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.
3. **DoRA**: Liu et al. "DoRA: Weight-Decomposed Low-Rank Adaptation." ICML 2024.
4. **LoRA+**: Hayou et al. "LoRA+: Efficient Low Rank Adaptation of Large Models." arXiv 2024.
5. **GaLore**: Zhao et al. "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection." ICML 2024.
6. **LISA**: Pan et al. "LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning." arXiv 2024.
7. **AdaLoRA**: Zhang et al. "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning." ICLR 2023.
8. **rsLoRA**: Kalajdzievski "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA." arXiv 2023.

### Implementations

- **Hugging Face PEFT**: https://github.com/huggingface/peft
- **bitsandbytes**: https://github.com/TimDettmers/bitsandbytes
- **Nexus**: `Nexus/nexus/models/compression/peft/`

### Tutorials

- [LoRA from Scratch](lora.md)
- [QLoRA Guide](qlora.md)
- [DoRA Deep Dive](dora.md)

## See Also

- [Quantization Methods](../quantization/README.md): Compress base model weights
- [Pruning Methods](../pruning/README.md): Remove unnecessary parameters
- [Distillation](../distillation/README.md): Transfer knowledge to smaller models
