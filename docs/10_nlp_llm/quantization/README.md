# Quantization Methods for Large Language Models

Quantization reduces the precision of model weights and activations from floating-point (32-bit or 16-bit) to lower bit-widths (8-bit, 4-bit, or even 2-bit), dramatically reducing memory footprint and improving inference speed with minimal accuracy loss.

## Overview

| Method | Type | Bits | Key Innovation | Best For |
|--------|------|------|----------------|----------|
| [GPTQ](gptq.md) | PTQ | 2-4 | OBS-based layer-wise quantization | General-purpose 4-bit |
| [AWQ](awq.md) | PTQ | 3-4 | Activation-aware weight quantization | Protecting salient weights |
| [QuIP#](quip_sharp.md) | PTQ | 2-4 | Lattice codebook quantization | Extreme compression (2-bit) |
| [SqueezeLLM](squeezellm.md) | PTQ | 3-4 | Dense-and-sparse quantization | Outlier handling |
| [AQLM](aqlm.md) | PTQ | 2-4 | Additive quantization | Better 2-bit quality |

**PTQ** = Post-Training Quantization (no retraining required)

## When to Use Quantization

### Use Cases

**Ideal for Quantization:**
- Deploying large models on edge devices
- Reducing inference costs in production
- Fitting models in limited GPU memory
- Serving many models on shared infrastructure
- Real-time inference requirements

**Consider Full Precision:**
- Research/development phase (full precision for debugging)
- Tasks extremely sensitive to numerical precision
- Models <1B parameters (quantization overhead dominates)
- When compute is unlimited and latency doesn't matter

### Memory & Speed Comparison

For a 7B parameter model:

| Precision | Memory | Inference Speed | Accuracy Loss |
|-----------|--------|-----------------|---------------|
| FP32 | 28 GB | 1.0× | 0% (baseline) |
| FP16 | 14 GB | 1.8× | <0.1% |
| INT8 | 7 GB | 2.5× | <0.5% |
| **INT4 (GPTQ)** | **3.5 GB** | **3.5×** | **<1%** |
| INT4 (AWQ) | 3.5 GB | 3.8× | <0.5% |
| INT3 (GPTQ) | 2.6 GB | 4.2× | 1-3% |
| INT2 (QuIP#) | 1.8 GB | 5.0× | 3-5% |

## Quantization Fundamentals

### What is Quantization?

Converting continuous floating-point values to discrete integer values:

```
Quantization:   w_float → w_int
Dequantization: w_float ≈ (w_int - zero) * scale
```

**Key parameters:**
- **Bits**: Number of bits per value (4-bit = 16 possible values)
- **Scale**: Multiplication factor for dequantization
- **Zero-point**: Offset for asymmetric quantization
- **Granularity**: Per-tensor, per-channel, or per-group

### Symmetric vs Asymmetric

**Symmetric** (zero-point = 0):
```
scale = max(|w|) / (2^(bits-1) - 1)
q = clamp(round(w / scale), -2^(bits-1), 2^(bits-1)-1)
```

**Asymmetric** (with zero-point):
```
scale = (max(w) - min(w)) / (2^bits - 1)
zero = round(-min(w) / scale)
q = clamp(round(w / scale) + zero, 0, 2^bits - 1)
```

**Trade-off**:
- Symmetric: Faster inference (no zero-point subtraction)
- Asymmetric: Better accuracy for non-centered distributions

### Quantization Granularity

| Granularity | Scale/Zero-Point | Accuracy | Overhead |
|-------------|------------------|----------|----------|
| Per-tensor | 1 per tensor | Worst | Lowest |
| Per-channel | 1 per output channel | Good | Low |
| Per-group | 1 per N elements | Better | Medium |
| Per-element | 1 per element | Best (=FP) | Highest |

**Sweet spot**: Per-group with group_size = 64-128.

## Method Comparison

### GPTQ vs AWQ vs QuIP#

**GPTQ** (Optimal Brain Surgeon):
- **Pros**: Robust, widely supported, good 4-bit quality
- **Cons**: Slower quantization process, requires calibration data
- **Use when**: You need reliable 4-bit quantization

**AWQ** (Activation-Aware):
- **Pros**: Better accuracy than GPTQ at same bit-width, faster quantization
- **Cons**: Slightly more complex setup
- **Use when**: You have activation statistics and need best 4-bit quality

**QuIP#** (Lattice Codebook):
- **Pros**: Best 2-bit and 3-bit quality
- **Cons**: More complex implementation, limited hardware support
- **Use when**: You need extreme compression (2-bit)

### Performance Comparison (LLaMA-7B on WikiText2 Perplexity)

| Method | Bits | Perplexity | Memory (GB) | Speed (tokens/s) |
|--------|------|------------|-------------|------------------|
| FP16 | 16 | 5.68 | 13.0 | 32 |
| GPTQ | 4 | 6.02 | 3.5 | 78 |
| GPTQ | 3 | 6.85 | 2.6 | 94 |
| AWQ | 4 | 5.88 | 3.5 | 85 |
| AWQ | 3 | 6.38 | 2.6 | 98 |
| QuIP# | 2 | 7.23 | 1.8 | 112 |
| SqueezeLLM | 4 | 5.94 | 3.5 | 81 |

## Quick Start

### GPTQ Quantization

```python
from nexus.models.compression.quantization import GPTQQuantizer, GPTQConfig

# Configure quantization
config = GPTQConfig(
    bits=4,
    group_size=128,
    act_order=True,
    damp_percent=0.01
)

quantizer = GPTQQuantizer(gptq_config=config)

# Collect calibration data (128 samples recommended)
for batch in calibration_loader:
    outputs = model(**batch, output_hidden_states=True)
    quantizer.add_calibration_data(outputs.hidden_states[-1])

# Quantize model
model_quantized, metrics = quantizer.quantize_model(
    model,
    calibration_data,
    nsamples=128,
    replace_with_quantized=True  # Use INT4 kernels
)

# Save quantized model
torch.save(model_quantized.state_dict(), "model_gptq_4bit.pth")
```

### AWQ Quantization

```python
from nexus.models.compression.quantization import AWQQuantizer, AWQConfig

config = AWQConfig(
    bits=4,
    group_size=128,
    zero_point=True
)

quantizer = AWQQuantizer(awq_config=config)

# AWQ automatically collects activation statistics
model_quantized = quantizer.quantize_model(
    model,
    calibration_dataloader,
    n_samples=512  # More samples for better activation stats
)
```

### Loading Quantized Models

```python
from transformers import AutoModelForCausalLM

# Load GPTQ-quantized model from Hugging Face
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto",
    quantization_config={"bits": 4}
)

# Inference as usual
outputs = model.generate(**inputs)
```

## Hyperparameter Guidelines

### Bit-Width Selection

| Application | Recommended Bits | Method | Notes |
|-------------|-----------------|--------|-------|
| Production serving | 4-bit | GPTQ or AWQ | Best quality/size trade-off |
| Edge devices | 3-bit | AWQ | Acceptable degradation |
| Extreme compression | 2-bit | QuIP# | Research/experimental |
| High-accuracy tasks | 8-bit | Standard PTQ | Minimal degradation |

### Group Size

| Group Size | Accuracy | Memory Overhead | Recommendation |
|------------|----------|-----------------|----------------|
| -1 (per-channel) | Low | <0.1% | Not recommended |
| 256 | Medium | 0.4% | Fast quantization |
| 128 | Good | 0.8% | **Default choice** |
| 64 | Better | 1.6% | High-accuracy tasks |
| 32 | Best | 3.2% | Research |

**Rule of thumb**: group_size = 128 for most cases.

### Calibration Data

**Number of samples:**
- Minimum: 32 samples (quick experiments)
- Recommended: 128 samples (production)
- Maximum benefit: 512 samples (diminishing returns beyond this)

**Data quality:**
- Use samples from target domain when possible
- Random Wikipedia/C4 works for general models
- Diverse samples > many similar samples

### Symmetric vs Asymmetric

| Distribution | Symmetric | Asymmetric | Recommendation |
|--------------|-----------|------------|----------------|
| Centered around 0 | ✓ Better | ✗ | Use symmetric |
| Skewed/shifted | ✗ | ✓ Better | Use asymmetric |
| Don't know | ✓ Faster | ✓ More accurate | Benchmark both |

## Advanced Topics

### Mixed-Precision Quantization

Quantize different layers to different bit-widths:

```python
# Keep sensitive layers in higher precision
quantization_config = {
    "embed_tokens": 16,  # Embeddings in FP16
    "layers.0": 8,       # First layer in INT8
    "layers.1-30": 4,    # Middle layers in INT4
    "layers.31": 8,      # Last layer in INT8
    "lm_head": 16        # Output head in FP16
}

model = quantize_with_config(model, quantization_config)
```

### Dynamic Quantization

Quantize activations at runtime based on their range:

```python
# Per-token dynamic quantization
def dynamic_quantize(activation):
    scale = activation.abs().max() / 127
    quant = (activation / scale).round().clamp(-128, 127)
    return quant, scale
```

### Combining with PEFT

Quantize base model, add LoRA adapters:

```python
from nexus.models.compression.quantization import apply_gptq
from nexus.models.compression.peft import apply_qlora

# 1. Quantize base model
model = apply_gptq(model, bits=4)

# 2. Add LoRA adapters (QLoRA)
model = apply_qlora(model, rank=16, alpha=32)

# 3. Fine-tune adapters
train(model)
```

### Quantization-Aware Training (QAT)

For maximum quality, fine-tune after quantization:

```python
# 1. Quantize
model = quantize_model(model, bits=4)

# 2. Fine-tune with straight-through estimator
for batch in dataloader:
    # Forward: use quantized weights
    outputs = model(**batch)

    # Backward: gradients flow through as if continuous
    loss = outputs.loss
    loss.backward()  # STE: ∂q/∂w ≈ 1

    optimizer.step()
```

## Common Issues & Solutions

### Issue 1: Severe Accuracy Drop

**Symptoms**: >5% accuracy loss after quantization.

**Solutions**:
1. Use more calibration data (32 → 128 → 512)
2. Reduce bit-width (4-bit → 3-bit) may paradoxically help if rounding is problematic
3. Decrease group size (128 → 64)
4. Try different method (GPTQ → AWQ)
5. Use activation ordering (`act_order=True`)

### Issue 2: Quantization Takes Forever

**Symptoms**: GPTQ quantization >1 hour for 7B model.

**Solutions**:
1. Reduce calibration samples (512 → 128)
2. Increase block size (`block_size=256`)
3. Disable activation ordering (`act_order=False`)
4. Use static groups (`static_groups=True`)
5. Try AWQ (faster than GPTQ)

### Issue 3: Out of Memory During Quantization

**Solutions**:
1. Reduce calibration batch size
2. Quantize layer-by-layer and save intermediate results
3. Use CPU for quantization (slower but more memory)
4. Reduce number of calibration samples

### Issue 4: Inference Slower Than Expected

**Symptoms**: Quantized model not faster than FP16.

**Causes & Solutions**:
- **No kernel support**: Use optimized libraries (AutoGPTQ, exllama)
- **CPU inference**: INT ops not optimized on CPU, use GPU
- **Dequantization overhead**: Enable kernel fusion
- **Wrong dtype**: Ensure weights stored as INT, not FP

### Issue 5: Different Results Each Run

**Symptoms**: Quantized model output varies between runs.

**Causes**:
- Calibration data order affects Hessian
- Random initialization in some methods

**Solutions**:
```python
# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Use deterministic algorithms
torch.use_deterministic_algorithms(True)
```

## Benchmarks

### Model Size Comparison (LLaMA-2)

| Model | FP16 (GB) | INT8 (GB) | INT4 (GB) | INT3 (GB) | INT2 (GB) |
|-------|-----------|-----------|-----------|-----------|-----------|
| 7B | 13.0 | 6.5 | 3.5 | 2.6 | 1.8 |
| 13B | 24.0 | 12.0 | 6.5 | 4.9 | 3.4 |
| 70B | 130.0 | 65.0 | 35.0 | 26.3 | 18.2 |

### Accuracy Retention (Average across MMLU, HellaSwag, Arc)

| Model | Method | Bits | Accuracy | vs FP16 |
|-------|--------|------|----------|---------|
| LLaMA-7B | FP16 | 16 | 58.3% | 100.0% |
| LLaMA-7B | GPTQ | 4 | 57.8% | 99.1% |
| LLaMA-7B | AWQ | 4 | 58.0% | 99.5% |
| LLaMA-7B | GPTQ | 3 | 56.7% | 97.3% |
| LLaMA-7B | QuIP# | 2 | 54.9% | 94.2% |

### Inference Speed (A100 GPU, Batch Size 1)

| Model | Precision | Tokens/sec | Speedup |
|-------|-----------|------------|---------|
| LLaMA-7B | FP16 | 32 | 1.0× |
| LLaMA-7B | INT8 | 54 | 1.7× |
| LLaMA-7B | INT4 (GPTQ) | 78 | 2.4× |
| LLaMA-7B | INT4 (AWQ) | 85 | 2.7× |
| LLaMA-13B | FP16 | 18 | 1.0× |
| LLaMA-13B | INT4 (GPTQ) | 42 | 2.3× |

## Tools & Libraries

### Production-Ready

- **AutoGPTQ**: https://github.com/PanQiWei/AutoGPTQ
  - Optimized GPTQ implementation
  - Supports CUDA kernels for fast inference
  - Integration with Hugging Face

- **exllama/exllamav2**: https://github.com/turboderp/exllamav2
  - Fastest INT4 inference
  - Optimized for GPTQ models
  - Excellent for real-time generation

- **llama.cpp**: https://github.com/ggerganov/llama.cpp
  - Cross-platform (CPU and GPU)
  - GGUF format quantization
  - Good for edge deployment

### Research

- **Nexus**: `Nexus/nexus/models/compression/quantization/`
- **QuIP**: https://github.com/Cornell-RelaxML/quip-sharp
- **AWQ**: https://github.com/mit-han-lab/llm-awq

## References

### Papers

1. **GPTQ**: Frantar et al. "GPTQ: Accurate Post-Training Quantization for GPT." ICLR 2023.
2. **AWQ**: Lin et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys 2024.
3. **QuIP#**: Tseng et al. "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks." arXiv 2024.
4. **SqueezeLLM**: Kim et al. "SqueezeLLM: Dense-and-Sparse Quantization." ICML 2024.
5. **AQLM**: Egiazarian et al. "Extreme Compression of Large Language Models via Additive Quantization." arXiv 2024.

### Surveys

- Dettmers et al. "A Survey on Model Compression for Large Language Models." arXiv 2023.
- Gholami et al. "A Survey of Quantization Methods for Efficient Neural Network Inference." 2021.

## See Also

- [PEFT Methods](../peft/README.md): Parameter-efficient fine-tuning
- [Pruning Methods](../pruning/README.md): Structured and unstructured sparsity
- [Distillation](../distillation/README.md): Knowledge transfer to smaller models
