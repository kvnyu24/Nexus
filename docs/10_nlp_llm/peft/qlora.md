# QLoRA: Efficient Finetuning of Quantized LLMs

## Overview & Motivation

QLoRA (Quantized Low-Rank Adaptation) combines 4-bit quantization with LoRA to enable efficient fine-tuning of extremely large language models on consumer hardware. While LoRA reduces trainable parameters, QLoRA additionally compresses the frozen base model weights to 4 bits, dramatically reducing memory requirements.

### Efficiency Gains

- **Memory**: 4× reduction over 16-bit LoRA (8× over full fine-tuning)
- **Hardware Access**: Fine-tune 65B models on a single 48GB GPU
- **Storage**: Base model requires ~4 bits per parameter
- **Performance**: Maintains ~99% of 16-bit LoRA quality

### Key Innovation

QLoRA introduces three key techniques:
1. **4-bit NormalFloat (NF4)**: Information-theoretically optimal quantization for normally distributed weights
2. **Double Quantization**: Quantize the quantization constants themselves
3. **Paged Optimizers**: Use NVIDIA unified memory to handle memory spikes

## Theoretical Background

### Information-Theoretic Quantization

Standard uniform quantization divides the weight range into equal bins. However, neural network weights typically follow a normal distribution. NF4 uses quantization bins sized according to the quantiles of a standard normal distribution, minimizing expected quantization error.

For a normally distributed weight w ~ N(0, σ²), the optimal k-bit quantization minimizes:

```
E[(w - Q(w))²]
```

where Q(w) is the quantized value. NF4 pre-computes the 2^k quantiles that minimize this expectation.

### Double Quantization

In block-wise quantization, each block of weights shares a floating-point scaling constant. For 64-element blocks in a 7B model, these constants consume:

```
Memory = (7B / 64) * 32 bits = 3.5B bits ≈ 437 MB
```

Double quantization quantizes these constants to 8-bit, reducing overhead to ~55 MB (8× reduction).

### NormalFloat4 (NF4) Data Type

NF4 represents 16 discrete values corresponding to the quantiles of N(0,1):

```python
NF4_QUANT_TABLE = [
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844,
    -0.1848, -0.0911, 0.0, 0.0796, 0.1609,
    0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
]
```

Each weight is:
1. Normalized to [-1, 1] using the block's absmax
2. Mapped to the nearest NF4 quantile
3. Stored as a 4-bit index

## Mathematical Formulation

### Quantization

For a weight tensor W, block-wise NF4 quantization proceeds as:

1. **Block partitioning**: Divide W into blocks of size B (typically 64)
   ```
   W = [W₁, W₂, ..., Wₙ] where each Wᵢ ∈ ℝᴮ
   ```

2. **Per-block normalization**:
   ```
   absmax_i = max(|Wᵢ|)
   W_norm_i = Wᵢ / absmax_i  ∈ [-1, 1]
   ```

3. **Quantization**:
   ```
   Q_i = argmin_j |W_norm_i - NF4[j]|
   ```
   where Q_i ∈ {0, 1, ..., 15} is the 4-bit index

4. **Storage**: Pack two 4-bit indices per uint8:
   ```
   packed = (Q[2i] << 4) | Q[2i+1]
   ```

### Dequantization

To reconstruct the approximate weight:

```
W_reconstructed_i = NF4[Q_i] * absmax_i
```

### Double Quantization

Quantize the absmax scaling constants:

1. Group scaling constants into super-blocks of size S (typically 256)
2. For each super-block:
   ```
   s_min = min(absmax_i)
   s_max = max(absmax_i)
   scale = (s_max - s_min) / 255
   absmax_quantized_i = round((absmax_i - s_min) / scale)
   ```

3. Store:
   - `absmax_quantized`: uint8 array
   - `scale`: float32 per super-block
   - `s_min`: float32 per super-block

### QLoRA Forward Pass

For a weight matrix W and input x:

```
W_dequant = Dequantize_NF4(W_packed, absmax)
h = W_dequant @ x + (α/r) · B @ A @ x
```

where:
- W_dequant: On-the-fly dequantized weights (in fp16/bf16)
- B, A: LoRA adapter matrices (in fp16/bf16)
- α/r: Scaling factor

### Memory Calculation

For a layer with d×k weights:

| Component | Precision | Memory |
|-----------|-----------|--------|
| Base weights (NF4) | 4-bit | d×k×4 bits |
| Absmax constants | fp32 or uint8 | (d×k/64)×32 bits or 8 bits |
| LoRA A | fp16 | r×k×16 bits |
| LoRA B | fp16 | d×r×16 bits |

**Example** (4096×4096 layer, r=16):
- Base: 16M weights × 4 bits = 8 MB
- Absmax: 262K constants × 4 bytes = 1 MB (with double quant)
- LoRA: (16×4096 + 4096×16) × 2 bytes = 256 KB
- **Total**: ~9.3 MB (vs. 32 MB for fp16)

## High-Level Intuition

### Why NF4 Works

Neural network weights after pre-training tend to follow a bell-curve distribution centered around zero. Most weights are small, with few large outliers. Standard uniform quantization wastes bins on rarely-used regions and under-samples the dense central region.

NF4 allocates more bins where weights are dense (near zero) and fewer bins in the tails. Think of it like dynamic range compression in audio: quiet sounds get more resolution, loud sounds less.

### Double Quantization Intuition

Consider a 7B model with 64-element blocks. That's ~110M blocks, each needing a 32-bit scaling constant = 440 MB just for scaling factors. But these scaling factors are themselves somewhat regular (they're all positive, bounded values). We can compress them with another layer of quantization, reducing overhead by 8×.

### Paged Optimizers

During fine-tuning, optimizer states (momentum, variance) can cause memory spikes. Paged optimizers automatically offload to CPU memory when GPU memory is tight, similar to how operating systems page memory to disk. This prevents out-of-memory errors during training.

## Implementation Details

### Creating a QLoRA Model

```python
from nexus.models.compression.peft import apply_qlora, QLoRAConfig

config = QLoRAConfig(
    rank=16,
    alpha=32.0,
    bits=4,
    double_quant=True,
    quant_type='nf4',
    compute_dtype=torch.bfloat16,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
)

model = apply_qlora(model, config=config)
```

### NF4 Quantization Process

Reference: `Nexus/nexus/models/compression/peft/qlora.py`

```python
class NF4Quantize(NexusModule):
    NF4_QUANT_TABLE = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844,
        -0.1848, -0.0911, 0.0, 0.0796, 0.1609,
        0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
    ])

    def quantize(self, weight: torch.Tensor):
        # Reshape into blocks
        blocks = weight.reshape(-1, self.blocksize)

        # Per-block absmax
        absmax = blocks.abs().max(dim=1, keepdim=True).values

        # Normalize to [-1, 1]
        normalized = blocks / absmax.clamp(min=1e-12)

        # Find nearest NF4 quantile
        distances = normalized.unsqueeze(-1) - self.NF4_QUANT_TABLE
        indices = distances.abs().argmin(dim=-1)

        # Pack 2×4-bit into uint8
        packed = (indices[::2] << 4) | indices[1::2]

        return packed, absmax
```

### Double Quantization

```python
class DoubleQuantization(NexusModule):
    def quantize_constants(self, absmax: torch.Tensor):
        # Group into super-blocks
        blocks = absmax.reshape(-1, self.blocksize)

        # Compute per-superblock min/max
        block_min = blocks.min(dim=1, keepdim=True).values
        block_max = blocks.max(dim=1, keepdim=True).values

        # Scale to [0, 255]
        scale = (block_max - block_min) / 255.0
        quantized = ((blocks - block_min) / scale).round().to(torch.uint8)

        return quantized, scale, block_min
```

### QLoRA Forward Pass

```python
class QLoRALinear(NexusModule):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize base weights on-the-fly
        weight = self._dequantize_weight()  # NF4 → fp16/bf16

        # Base linear computation
        base_output = F.linear(x, weight, self.bias)

        # LoRA adapter path
        lora_output = F.linear(
            F.linear(x, self.lora_A),
            self.lora_B
        )

        return base_output + lora_output * self.scaling
```

### Training Setup

```python
import torch
from transformers import LlamaForCausalLM
from nexus.models.compression.peft import apply_qlora, QLoRAConfig

# Load model in fp16
model = LlamaForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    torch_dtype=torch.float16,
    device_map='auto'
)

# Apply QLoRA
config = QLoRAConfig(
    rank=64,
    alpha=16,
    bits=4,
    double_quant=True,
    compute_dtype=torch.bfloat16,
)
model = apply_qlora(model, config=config)

# Standard training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
# ... training code ...
```

## Code Walkthrough

### NF4 Quantization Core

The quantization table is pre-computed for N(0,1):

```python
# 16 quantiles for 4-bit (2^4 = 16 values)
NF4_QUANT_TABLE = [
    -1.0,      # 0th percentile (min)
    -0.6962,   # 6.68th percentile
    -0.5251,   # 15.87th percentile
    -0.3949,   # 25th percentile
    -0.2844,   # 34.13th percentile
    -0.1848,   # 43.32th percentile
    -0.0911,   # 50th percentile
    0.0,       # Exact zero (56.68th percentile)
    0.0796,    # 65.87th percentile
    0.1609,    # 75th percentile
    0.2461,    # 84.13th percentile
    0.3379,    # 93.32th percentile
    0.4407,    # 97.72th percentile
    0.5626,    # 99.38th percentile
    0.7230,    # 99.87th percentile
    1.0        # 100th percentile (max)
]
```

These values are chosen such that if w ~ N(0,1), quantizing w to the nearest NF4 value minimizes E[(w - Q(w))²].

### Packing and Unpacking

Two 4-bit values fit in one uint8:

```python
def pack(indices: torch.Tensor) -> torch.Tensor:
    # indices: [i0, i1, i2, i3, ...] each in [0, 15]
    # packed: [(i0<<4)|i1, (i2<<4)|i3, ...]
    high = indices[0::2]  # i0, i2, i4, ...
    low = indices[1::2]   # i1, i3, i5, ...
    return ((high << 4) | low).to(torch.uint8)

def unpack(packed: torch.Tensor) -> torch.Tensor:
    # Extract high and low 4-bit values
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    # Interleave: [high[0], low[0], high[1], low[1], ...]
    return torch.stack([high, low], dim=-1).flatten()
```

## Optimization Tricks

### 1. Compute Dtype Selection

**BFloat16 vs Float16**:

| Dtype | Range | Precision | Best For |
|-------|-------|-----------|----------|
| fp16 | ±65,504 | 3.3 decimal | Most tasks |
| bf16 | ±3.4×10³⁸ | 2.3 decimal | Large gradients, mixed precision |

**Recommendation**: Use `compute_dtype=torch.bfloat16` for stability, especially with large models.

### 2. Block Size Tuning

| Block Size | Quantization Accuracy | Memory Overhead | Speed |
|------------|----------------------|-----------------|-------|
| 32 | Best | High | Slower |
| 64 | Very Good | Medium | Balanced |
| 128 | Good | Low | Faster |
| 256 | Acceptable | Very Low | Fastest |

**Trade-off**: Smaller blocks → better accuracy, larger overhead. Default: 64.

### 3. Rank Selection for QLoRA

QLoRA typically needs higher ranks than regular LoRA due to base weight quantization:

| Model Size | LoRA Rank | QLoRA Rank | Reason |
|------------|-----------|------------|--------|
| <1B | r=8 | r=16 | Compensate quantization loss |
| 1-10B | r=8 | r=32 | More capacity needed |
| 10-70B | r=16 | r=64 | Large model, more parameters |

**Guideline**: Double the LoRA rank when using 4-bit quantization.

### 4. Gradient Checkpointing

Essential for fitting large models:

```python
model.gradient_checkpointing_enable()
model = apply_qlora(model, config=config)
```

Reduces activation memory by ~2-3×, with ~20% slowdown.

### 5. Precision in Optimizer States

Use paged optimizers for large models:

```python
from bitsandbytes.optim import AdamW8bit

optimizer = AdamW8bit(
    model.parameters(),
    lr=2e-4,
    optim_bits=32,  # Keep optimizer states in fp32
    percentile_clipping=100,
    min_8bit_size=4096
)
```

### 6. Mixed Batch Sizes

During training, use smaller batch sizes for forward pass, larger for gradient accumulation:

```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 16  # Effective batch size = 16
```

This reduces memory spikes while maintaining large effective batch size.

## Experiments & Results

### Original QLoRA Paper Results

**Guanaco 65B (LLaMA-65B + QLoRA on OASST1)**:

| Method | Memory (GB) | MMLU (5-shot) | Human Eval (Pass@1) |
|--------|-------------|---------------|---------------------|
| Full FT (fp16) | >780 | - | - |
| LoRA (fp16) | 780 | 63.4 | - |
| **QLoRA (4-bit)** | **48** | **63.2** | **20.1** |

QLoRA enables training on a single consumer GPU while maintaining 99.7% of LoRA performance.

### Compression Ratios

| Model | Full FT | LoRA (fp16) | QLoRA (4-bit) | Reduction |
|-------|---------|-------------|---------------|-----------|
| LLaMA-7B | 28 GB | 28 GB | 7.5 GB | 3.7× |
| LLaMA-13B | 52 GB | 52 GB | 13.8 GB | 3.8× |
| LLaMA-33B | 132 GB | 132 GB | 34.5 GB | 3.8× |
| LLaMA-65B | 260 GB | 260 GB | 68 GB | 3.8× |

### Quality Retention

**MMLU Benchmark (65B model)**:

| Quantization | Accuracy | Retention |
|--------------|----------|-----------|
| Full Precision | 63.4% | 100% |
| 8-bit | 63.1% | 99.5% |
| **NF4** | **63.2%** | **99.7%** |
| NF4 (no double quant) | 62.9% | 99.2% |
| Uniform 4-bit | 61.8% | 97.5% |

NF4 significantly outperforms naive uniform quantization.

### Training Speed

| Configuration | Time per Iteration | Throughput (tokens/sec) |
|---------------|-------------------|-------------------------|
| Full FT (impossible on 48GB) | - | - |
| LoRA 16-bit (4×A100) | 2.1s | 1,920 |
| **QLoRA 4-bit (1×A100)** | **2.3s** | **1,750** |

QLoRA is only ~10% slower than 16-bit LoRA despite 4× memory reduction.

## Common Pitfalls

### 1. Compute Dtype Mismatch

**Symptom**: NaN losses, training divergence.

**Diagnosis**:
```python
# Check if input dtype matches compute_dtype
print(f"Input dtype: {x.dtype}")
print(f"Compute dtype: {model.lora_A.dtype}")
```

**Solution**: Ensure inputs are cast to compute_dtype:
```python
x = x.to(model.compute_dtype)
```

### 2. Block Size Too Large

**Symptom**: Poor model quality despite successful training.

**Solution**: Reduce block size from 128 → 64 → 32. Smaller blocks improve quantization fidelity.

### 3. Insufficient Rank

**Symptom**: Model underfits, validation loss plateaus early.

**Solution**: Increase QLoRA rank. Try r=32 or r=64 for 4-bit quantization.

### 4. Missing Double Quantization

**Symptom**: Higher memory usage than expected.

**Diagnosis**:
```python
config = QLoRAConfig(double_quant=True)  # Must be explicitly enabled
```

**Solution**: Always enable double quantization for maximum memory savings.

### 5. Quantization After Model Parallelism

**Symptom**: Errors about mismatched devices or shapes.

**Solution**: Apply QLoRA before distributing the model across devices:
```python
# Correct order:
model = apply_qlora(model, config)
model = model.to('cuda')
```

### 6. Non-Normal Weight Distributions

**Symptom**: Poor performance on architectures with non-Gaussian weights (e.g., some vision models).

**Solution**: NF4 assumes normal distributions. For non-normal weights, consider:
- Using 8-bit quantization instead
- Applying only to specific layers (transformers) rather than all

### 7. Optimizer State Overflow

**Symptom**: Out-of-memory during optimizer.step().

**Solution**: Use paged optimizers:
```python
from bitsandbytes.optim import AdamW8bit
optimizer = AdamW8bit(model.parameters(), lr=2e-4)
```

## References

1. **Original Paper**: Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023. https://arxiv.org/abs/2305.14314

2. **NormalFloat Theory**: Optimal quantization for Gaussian sources. Cover & Thomas, "Elements of Information Theory."

3. **Paged Optimizers**: Dettmers, T., et al. "8-bit Optimizers via Block-wise Quantization." ICLR 2022.

4. **bitsandbytes Library**: https://github.com/TimDettmers/bitsandbytes

5. **Nexus Implementation**: `Nexus/nexus/models/compression/peft/qlora.py`

6. **Guanaco Models**: https://huggingface.co/timdettmers/guanaco-65b

## See Also

- [LoRA](lora.md): Foundation of QLoRA
- [GPTQ](../quantization/gptq.md): Alternative post-training quantization
- [AWQ](../quantization/awq.md): Activation-aware quantization
- [DoRA](dora.md): Weight-decomposed adaptation (compatible with quantization)
