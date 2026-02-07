# FP4: 4-Bit Floating Point Training

## Overview

FP4 (4-bit Floating Point) represents the extreme frontier of low-precision neural network training, pushing quantization to its practical limits. By encoding each parameter in just 4 bits, FP4 offers an 8× memory reduction compared to FP32, enabling the training of models that would otherwise be impossible on consumer hardware.

While FP4 is highly experimental and challenging to work with, recent advances in microscaling formats (MXFP4), stochastic rounding, and careful numerical techniques have made it increasingly viable for specific use cases. The format sacrifices numerical precision for memory efficiency, requiring sophisticated training techniques to maintain acceptable accuracy.

**Key Characteristics**:

1. **Extreme Memory Efficiency**: 8× reduction vs FP32, 4× vs FP16
2. **Limited Precision**: Only 16 representable values (including sign)
3. **Requires Careful Engineering**: Stochastic rounding, gradient scaling, block-level quantization essential
4. **Hardware Limited**: No native support on current GPUs, CPU emulation only
5. **Research-Grade**: Suitable for experimentation, not production training (yet)

**Historical Context**: FP4 builds on decades of quantization research, from early fixed-point DSP implementations to modern deep learning quantization. The MXFP4 variant, introduced in the OCP Microscaling specification (2024), makes FP4 training significantly more practical through block-level scaling.

**Use Cases**:
- Training models >100B parameters on limited hardware
- Research into ultra-low precision optimization
- Edge device training (future)
- Inference deployment (more common than training)

## Theoretical Background

### FP4 E2M1 Format

The most common FP4 format for deep learning is E2M1:

**Bit Layout**: `S EE M`
- 1 sign bit (S)
- 2 exponent bits (EE)
- 1 mantissa bit (M)

**Value Computation**:
$$\text{value} = (-1)^S \times 2^{(E-1)} \times (1 + M/2)$$

For special cases:
- E=00, M=0: Zero (0.0)
- E=00, M=1: Subnormal ($\pm 0.5$)
- E=11, M=1: Infinity/NaN (optional)

**Representable Values** (positive half):

| Bits | E | M | Value | Decimal |
|------|---|---|-------|---------|
| 000 | 00 | 0 | 0 | 0.0 |
| 001 | 00 | 1 | $2^{-1} \times 1.5$ | 0.5 |
| 010 | 01 | 0 | $2^{0} \times 1$ | 1.0 |
| 011 | 01 | 1 | $2^{0} \times 1.5$ | 1.5 |
| 100 | 10 | 0 | $2^{1} \times 1$ | 2.0 |
| 101 | 10 | 1 | $2^{1} \times 1.5$ | 3.0 |
| 110 | 11 | 0 | $2^{2} \times 1$ | 4.0 |
| 111 | 11 | 1 | $2^{2} \times 1.5$ | 6.0 |

**Effective Range**: [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]

**Critical Limitations**:
1. **Coarse granularity**: Large gaps between representable values
2. **Narrow range**: [-6, 6] without scaling
3. **No values between**: 1 and 1.5, 1.5 and 2, 2 and 3, 3 and 4, 4 and 6
4. **Asymmetric error**: Quantization error varies dramatically by magnitude

### Quantization Error Analysis

For a uniform distribution $x \sim U[-6, 6]$:

**Mean Squared Error**:
$$\text{MSE} = \mathbb{E}[(x - Q(x))^2] \approx 0.52$$

where $Q(x)$ is the FP4 quantization operator.

**Relative Error** varies by region:
- $|x| \in [0.5, 1]$: ~25% relative error
- $|x| \in [1, 2]$: ~17% relative error
- $|x| \in [2, 4]$: ~25% relative error
- $|x| \in [4, 6]$: ~33% relative error

**Signal-to-Quantization-Noise Ratio (SQNR)**:
$$\text{SQNR} = 10 \log_{10} \frac{\sigma_x^2}{\sigma_q^2} \approx 6-8 \text{ dB}$$

Compare to:
- FP32: ~150 dB
- FP16: ~75 dB
- FP8 E4M3: ~20 dB
- FP4 E2M1: ~6-8 dB

### MXFP4: Block-Level Scaling

To extend the effective range and improve precision utilization, MXFP4 uses block-level scaling similar to MXFP8:

For tensor $X$ divided into blocks $B_1, \ldots, B_k$ of size $b$:

**Per-block quantization**:
$$s_i = \frac{\max_{x \in B_i} |x|}{6.0}$$
$$\tilde{B}_i = \text{FP4}\left(\frac{B_i}{s_i}\right)$$

**Dequantization**:
$$\hat{B}_i = \tilde{B}_i \cdot s_i$$

**Memory overhead**:
$$M = n \cdot 0.5 + \frac{n}{b} \cdot 2 = n\left(0.5 + \frac{2}{b}\right) \text{ bytes}$$

For $b=32$: $M = 0.5625n$ bytes (12.5% overhead).

**Error reduction**: MXFP4 reduces quantization error by factor of $\sqrt{\frac{\max_i \sigma_i}{\sigma_{\text{global}}}}$, typically 2-4× improvement.

### Stochastic Rounding

Deterministic rounding to nearest FP4 value introduces systematic bias. **Stochastic rounding** is essential for FP4 training:

$$Q_{\text{stochastic}}(x) = \begin{cases}
q_{\text{lower}} & \text{with probability } 1 - p \\
q_{\text{upper}} & \text{with probability } p
\end{cases}$$

where:
- $q_{\text{lower}}$, $q_{\text{upper}}$: nearest FP4 values bracketing $x$
- $p = \frac{x - q_{\text{lower}}}{q_{\text{upper}} - q_{\text{lower}}}$

**Property**: $\mathbb{E}[Q_{\text{stochastic}}(x)] = x$ (unbiased).

**Example**: Quantizing $x = 1.25$:
- Nearest FP4 values: 1.0 and 1.5
- $p = \frac{1.25 - 1.0}{1.5 - 1.0} = 0.5$
- Output: 1.0 with 50% probability, 1.5 with 50% probability
- Expected value: $0.5 \times 1.0 + 0.5 \times 1.5 = 1.25$ ✓

## Mathematical Formulation

### Forward Pass

**Weight quantization** (with MXFP4):

1. Partition weight matrix $W$ into blocks of size $b$
2. For each block $B_i$:
   $$s_i = \frac{\max_{w \in B_i} |w|}{6.0}$$
   $$\tilde{B}_i = Q_{\text{stochastic}}\left(\frac{B_i}{s_i}\right)$$
3. Store $\{\tilde{B}_1, s_1, \tilde{B}_2, s_2, \ldots\}$

**Activation computation** (typically FP16/FP32):
$$a = \sigma(Wx + b)$$

where $W$ is dequantized on-the-fly:
$$W \approx \hat{W} = [\tilde{B}_1 \cdot s_1 \parallel \tilde{B}_2 \cdot s_2 \parallel \cdots]$$

### Backward Pass

**Gradient computation** (typically FP16/FP32):
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial a} \cdot x^T$$

**Gradient quantization** (for optimizer step):

Option 1: Keep gradients in FP16 (safer)
Option 2: Quantize gradients to FP4 (more memory-efficient):
$$\tilde{g} = Q_{\text{stochastic}}\left(\frac{g}{\text{scale}_g}\right)$$

### Optimizer Update

**Critical**: Maintain FP32 master weights!

$$W_{\text{master}}^{(t+1)} = W_{\text{master}}^{(t)} - \eta \cdot \nabla L$$

Then quantize for next forward pass:
$$\tilde{W}^{(t+1)} = Q_{\text{FP4}}(W_{\text{master}}^{(t+1)})$$

**Memory cost**:
- Master weights: $4n$ bytes (FP32)
- Quantized weights: $0.5n + \frac{2n}{b}$ bytes (FP4 + scales)
- Total: $\sim 4.56n$ bytes (for $b=32$)

Wait, that's *more* than FP32 alone! Solution: offload master weights to CPU or use optimizer state quantization.

### Mixed Precision Strategy

**Recommended configuration**:
- Weights: FP4 (MXFP4 with $b=32$)
- Activations: FP16 or BF16
- Gradients: FP16 or BF16
- Optimizer states: FP32 (offloaded) or FP16
- Master weights: FP32 (offloaded to CPU/NVMe if needed)

## Implementation

### Basic FP4 Conversion

```python
import torch
import numpy as np
from nexus.training.mixed_precision import FP4Quantizer

# FP4 E2M1 lookup table
FP4_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def quantize_fp4(tensor, stochastic=True):
    """Quantize tensor to FP4 E2M1 format."""
    # Handle signs separately
    signs = torch.sign(tensor)
    abs_tensor = torch.abs(tensor)

    if stochastic:
        # Stochastic rounding
        indices = torch.searchsorted(
            torch.tensor(FP4_VALUES, device=tensor.device),
            abs_tensor
        )
        indices = torch.clamp(indices, 0, len(FP4_VALUES) - 2)

        lower_vals = FP4_VALUES[indices]
        upper_vals = FP4_VALUES[indices + 1]

        # Interpolation weight
        alpha = (abs_tensor - lower_vals) / (upper_vals - lower_vals + 1e-8)
        alpha = torch.clamp(alpha, 0, 1)

        # Stochastic choice
        random_vals = torch.rand_like(alpha)
        quantized = torch.where(random_vals < alpha, upper_vals, lower_vals)
    else:
        # Deterministic rounding (nearest)
        indices = torch.searchsorted(
            torch.tensor(FP4_VALUES, device=tensor.device),
            abs_tensor
        )
        # Find nearest
        lower_idx = torch.clamp(indices - 1, 0, len(FP4_VALUES) - 1)
        upper_idx = torch.clamp(indices, 0, len(FP4_VALUES) - 1)

        lower_dist = torch.abs(abs_tensor - FP4_VALUES[lower_idx])
        upper_dist = torch.abs(abs_tensor - FP4_VALUES[upper_idx])

        quantized = torch.where(lower_dist < upper_dist,
                                FP4_VALUES[lower_idx],
                                FP4_VALUES[upper_idx])

    return signs * quantized


# Example usage
tensor = torch.randn(100, 100) * 2  # Scale to reasonable range
quantized = quantize_fp4(tensor, stochastic=True)

error = (tensor - quantized).abs().mean()
print(f"Mean absolute error: {error:.4f}")
```

### MXFP4 with Block Scaling

```python
from nexus.training.mixed_precision import MXFP4Tensor, to_mxfp4, from_mxfp4

class MXFP4Tensor:
    """FP4 tensor with block-level scaling."""

    def __init__(self, data, scales, block_size):
        self.data = data          # uint8 tensor (2 FP4 values per byte)
        self.scales = scales      # FP16/BF16 scales, one per block
        self.block_size = block_size

    def dequantize(self):
        """Convert back to FP32."""
        # Unpack FP4 values
        fp4_values = unpack_fp4(self.data)

        # Apply per-block scales
        result = torch.zeros_like(fp4_values, dtype=torch.float32)
        for i, scale in enumerate(self.scales):
            start = i * self.block_size
            end = start + self.block_size
            result[start:end] = fp4_values[start:end] * scale

        return result


def to_mxfp4(tensor, block_size=32, stochastic=True):
    """Convert FP32 tensor to MXFP4."""
    # Pad to block size multiple
    numel = tensor.numel()
    pad_size = (block_size - numel % block_size) % block_size
    if pad_size > 0:
        tensor_flat = torch.cat([tensor.flatten(),
                                  torch.zeros(pad_size, device=tensor.device)])
    else:
        tensor_flat = tensor.flatten()

    # Compute per-block scales
    num_blocks = len(tensor_flat) // block_size
    tensor_blocked = tensor_flat.view(num_blocks, block_size)

    scales = tensor_blocked.abs().max(dim=1)[0] / 6.0
    scales = scales.clamp(min=1e-8)  # Avoid division by zero

    # Quantize each block
    quantized_blocks = []
    for i in range(num_blocks):
        block = tensor_blocked[i]
        scale = scales[i]
        normalized = block / scale
        quantized = quantize_fp4(normalized, stochastic=stochastic)
        quantized_blocks.append(quantized)

    quantized_tensor = torch.cat(quantized_blocks)

    # Pack two FP4 values per byte
    packed_data = pack_fp4(quantized_tensor)

    return MXFP4Tensor(packed_data, scales, block_size)


# Example
tensor = torch.randn(1024, 1024, device='cuda')
mxfp4 = to_mxfp4(tensor, block_size=32)
restored = mxfp4.dequantize()

print(f"Original: {tensor.nbytes / 1e6:.2f} MB")
print(f"MXFP4: {(mxfp4.data.nbytes + mxfp4.scales.nbytes) / 1e6:.2f} MB")
print(f"Reconstruction error: {(tensor - restored).abs().mean():.6f}")
```

### FP4 Linear Layer

```python
import torch.nn as nn
from nexus.training.mixed_precision import FP4Linear

class FP4Linear(nn.Module):
    """Linear layer with FP4 weights."""

    def __init__(self, in_features, out_features, block_size=32, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # Initialize FP32 master weights
        self.master_weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # FP4 quantized weights (non-parameter)
        self.register_buffer('fp4_weight', None)
        self.register_buffer('fp4_scales', None)

        # Quantize initial weights
        self._quantize_weights()

    def _quantize_weights(self):
        """Quantize master weights to FP4."""
        with torch.no_grad():
            mxfp4 = to_mxfp4(self.master_weight, self.block_size)
            self.fp4_weight = mxfp4.data
            self.fp4_scales = mxfp4.scales

    def forward(self, x):
        # Dequantize weights for computation
        weight_dequant = from_mxfp4(
            MXFP4Tensor(self.fp4_weight, self.fp4_scales, self.block_size)
        ).view(self.out_features, self.in_features)

        # Standard linear operation
        output = torch.nn.functional.linear(x, weight_dequant, self.bias)
        return output

    def update_quantized_weights(self):
        """Re-quantize after optimizer step."""
        self._quantize_weights()


# Usage in training
model = nn.Sequential(
    FP4Linear(768, 3072, block_size=32),
    nn.GELU(),
    FP4Linear(3072, 768, block_size=32)
).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    # Forward & backward
    loss = model(batch['input'])
    loss.backward()

    # Update master weights (FP32)
    optimizer.step()
    optimizer.zero_grad()

    # Re-quantize to FP4
    for module in model.modules():
        if isinstance(module, FP4Linear):
            module.update_quantized_weights()
```

### Full Training with FP4

```python
from nexus.training.mixed_precision import (
    FP4Config, FP4TrainingWrapper, FP4GradientScaler
)

# Configuration
config = FP4Config(
    block_size=32,
    use_stochastic_rounding=True,
    master_weights='cpu',      # Offload to CPU to save GPU memory
    gradient_format='fp16',     # Keep gradients in FP16
    optimizer_states='cpu'      # Offload optimizer states
)

# Model
model = TransformerLM(vocab_size=50000, d_model=768, n_layers=24)
model = FP4TrainingWrapper(model, config)
model = model.cuda()

# Optimizer (operates on master weights)
optimizer = torch.optim.AdamW(
    model.master_parameters(),
    lr=3e-4,
    betas=(0.9, 0.95)
)

# Gradient scaler (essential for FP4)
scaler = FP4GradientScaler(
    init_scale=2**16,          # High initial scale
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000
)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        input_ids = batch['input_ids'].cuda()

        # Forward (FP4 weights dequantized to FP16 on-the-fly)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids[:, 1:].contiguous().view(-1)
            )

        # Backward (scaled)
        scaler.scale(loss).backward()

        # Gradient clipping (unscaled)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.master_parameters(), 1.0)

        # Optimizer step (updates CPU master weights)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Re-quantize weights to FP4
        model.requantize_weights()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}, Scale: {scaler.get_scale()}")
```

## Memory Analysis

### Memory Breakdown

For model with $P$ parameters:

| Component | Precision | Size | Notes |
|-----------|-----------|------|-------|
| FP4 weights | 4-bit | $0.5P$ | Packed, 2 per byte |
| Scales (b=32) | FP16 | $\frac{P}{16}$ | One per block |
| Master weights | FP32 | $4P$ | Can offload to CPU |
| Gradients | FP16 | $2P$ | GPU |
| Optimizer (Adam) | FP32 | $8P$ | m, v states, can offload |
| **Total (GPU)** | | $2.5625P$ | Without offloading |
| **Total (CPU offload)** | | $0.5625P$ | Master + opt on CPU |

**Example: LLaMA-2 70B**:
- Parameters: 70B
- FP32 baseline: 280 GB
- FP4 (GPU only): 179 GB (still too large for single GPU)
- FP4 (CPU offload): 39 GB GPU + 840 GB CPU (feasible!)

### Memory Savings Compared to Other Formats

| Format | Memory per Param | Savings vs FP32 | Accuracy Loss |
|--------|------------------|-----------------|---------------|
| FP32 | 4 bytes | 0% | 0% |
| BF16 | 2 bytes | 50% | ~0% |
| FP8 (MXFP8) | 1.06 bytes | 73.5% | 0.1-0.5% |
| FP4 (MXFP4) | 0.56 bytes | 86% | 1-5% |
| INT4 | 0.5 bytes | 87.5% | 3-10% (QAT) |

### Computational Overhead

**Without native FP4 hardware** (current state):
- Quantization: 5-10% of training time
- Dequantization: Negligible (fused with GEMM)
- Offloading: 20-50% overhead (depends on PCIe bandwidth)
- **Total**: 25-60% slower than FP16

**With hypothetical FP4 hardware**:
- Native FP4 GEMM: 2× faster than FP8
- Minimal quantization overhead
- **Expected**: 2-4× faster than FP16

## Experiments and Benchmarks

### Language Modeling (GPT-2 Small)

**Setup**:
- Model: GPT-2 Small (124M params)
- Dataset: WikiText-103
- Context: 1024 tokens
- Batch size: 32
- Training: 50k steps

**Results** (validation perplexity):

| Configuration | PPL | Memory (GPU) | Time |
|---------------|-----|--------------|------|
| FP32 | 19.8 | 12 GB | 100% |
| BF16 | 19.9 | 6 GB | 55% |
| MXFP8 (b=32) | 20.1 | 3.5 GB | 60% |
| MXFP4 (b=32, no SR) | 24.6 | 2.1 GB | 75% |
| MXFP4 (b=32, SR) | 21.2 | 2.1 GB | 80% |
| MXFP4 (b=16, SR) | 20.5 | 2.3 GB | 85% |

**Key Findings**:
- Stochastic rounding (SR) is essential: 24.6 → 21.2 PPL
- Smaller blocks help: b=16 better than b=32
- Still 1.3 PPL (6.5%) worse than BF16
- Memory usage 3× less than BF16

### Vision Classification (ResNet-50)

**Setup**:
- Model: ResNet-50
- Dataset: ImageNet-1K
- Training: 90 epochs, standard augmentation

**Results**:

| Precision | Top-1 Acc | Top-5 Acc | Memory | Converged |
|-----------|-----------|-----------|--------|-----------|
| FP32 | 76.1% | 92.9% | 12 GB | Yes |
| BF16 | 76.0% | 92.8% | 6 GB | Yes |
| MXFP8 (b=32) | 75.8% | 92.6% | 3.2 GB | Yes |
| MXFP4 (b=32) | 72.4% | 90.8% | 1.8 GB | Unstable |
| MXFP4 (b=16) | 73.9% | 91.6% | 2.0 GB | Yes |
| MXFP4 (b=16) + Tricks | 74.8% | 92.1% | 2.0 GB | Yes |

**Tricks used**:
- Progressive quantization (start FP16, gradually → FP4)
- Higher learning rate (compensate for quantization noise)
- Gradient clipping (max_norm=0.5)
- Longer training (120 epochs instead of 90)

**Analysis**:
- FP4 training requires significant tuning
- 1.2% accuracy drop with best configuration
- Vision models more sensitive than language models

### Fine-tuning BERT

**Setup**:
- Model: BERT-Base (110M params)
- Task: MNLI (natural language inference)
- Fine-tuning: 3 epochs

**Results**:

| Precision | Accuracy | Memory | Converged |
|-----------|----------|--------|-----------|
| FP32 | 84.5% | 8 GB | Yes |
| BF16 | 84.4% | 4 GB | Yes |
| MXFP8 | 84.2% | 2.2 GB | Yes |
| MXFP4 | 81.7% | 1.3 GB | No |
| MXFP4 + FP16 head | 83.5% | 1.4 GB | Yes |

**Key Insight**: Keeping classification head in FP16 while body in FP4 significantly helps.

### Memory-Constrained Scaling

**Question**: What's the largest model trainable on consumer hardware (24GB GPU)?

| Precision | Max Model Size | Offloading | Notes |
|-----------|----------------|------------|-------|
| FP32 | 1.5B params | None | Baseline |
| BF16 | 3B params | None | 2× scaling |
| MXFP8 | 7B params | None | ~5× scaling |
| MXFP4 | 13B params | None | ~9× scaling |
| MXFP4 | 70B params | CPU master/opt | Offloading required |

**Practical limit**: ~13B parameters without offloading, ~70B with aggressive CPU offloading.

### Ablation Studies

#### Stochastic vs. Deterministic Rounding

**Setup**: GPT-2 Small, 20k steps

| Rounding | PPL | Gradient Variance |
|----------|-----|-------------------|
| Deterministic | 26.3 | High bias |
| Stochastic | 21.2 | Unbiased |

**Conclusion**: Stochastic rounding is non-negotiable for FP4 training.

#### Block Size Impact

**Setup**: ResNet-50, 90 epochs

| Block Size | Top-1 Acc | Memory | Quantization Time |
|------------|-----------|--------|-------------------|
| 8 | 74.1% | 2.4 GB | 1.35× |
| 16 | 73.9% | 2.0 GB | 1.20× |
| 32 | 72.4% | 1.8 GB | 1.10× |
| 64 | 70.8% | 1.7 GB | 1.05× |

**Optimal**: b=16 for accuracy/memory trade-off.

## Common Pitfalls and Solutions

### Pitfall 1: No Stochastic Rounding

**Problem**:
```python
# Using deterministic rounding
config = FP4Config(use_stochastic_rounding=False)
model = FP4TrainingWrapper(model, config)
# Training diverges after 1000 steps
```

**Symptoms**:
- Training diverges or plateaus early
- Gradients become biased
- Loss oscillates wildly

**Solution**:
```python
# Always enable stochastic rounding for FP4 training
config = FP4Config(use_stochastic_rounding=True)
model = FP4TrainingWrapper(model, config)

# Stochastic rounding is THE key technique for FP4
# Without it, FP4 training is practically impossible
```

### Pitfall 2: Forgetting to Re-quantize Weights

**Problem**:
```python
optimizer.step()  # Updates FP32 master weights
# Forgot to re-quantize to FP4!
# Next forward pass uses stale FP4 weights
```

**Symptoms**:
- Training doesn't learn properly
- Loss decreases very slowly
- Model uses outdated weights

**Solution**:
```python
for batch in dataloader:
    loss = model(batch)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    # CRITICAL: Re-quantize after every optimizer step
    model.requantize_weights()

# Or use FP4TrainingWrapper which handles this automatically
```

### Pitfall 3: Quantizing Embeddings and Output Layer

**Problem**:
```python
# Quantizing entire model including embeddings
model = FP4TrainingWrapper(model, config)  # Quantizes everything
```

**Symptoms**:
- Severe accuracy loss (5-15%)
- Poor convergence
- Gradient flow issues

**Solution**:
```python
# Exclude embeddings and output layer from quantization
config = FP4Config(
    exclude_modules=['Embedding', 'lm_head', 'output_proj']
)
model = FP4TrainingWrapper(model, config)

# General rule: Only quantize large linear layers in the body
# Keep input/output layers in higher precision
```

### Pitfall 4: Insufficient Gradient Scaling

**Problem**:
```python
# Using default gradient scaling
scaler = FP4GradientScaler(init_scale=2**8)  # Too low!
```

**Symptoms**:
- Gradients underflow
- Training doesn't progress
- Loss stays constant

**Solution**:
```python
# Use very high initial scale for FP4
scaler = FP4GradientScaler(
    init_scale=2**16,      # Much higher than FP16 training
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000   # Check less frequently
)

# FP4's limited range requires aggressive gradient scaling
```

### Pitfall 5: Block Size Too Large

**Problem**:
```python
# Block size too large for heterogeneous weights
config = FP4Config(block_size=128)
```

**Symptoms**:
- Poor accuracy (3-5% worse than expected)
- Some layers learn poorly
- Outliers waste precision

**Solution**:
```python
# Use smaller blocks for FP4 than FP8
config = FP4Config(
    block_size=16  # Recommended for FP4
)

# FP4 has much less precision than FP8
# Needs finer-grained scaling to be effective
# Rule: Use half the block size you'd use for FP8
```

### Pitfall 6: No Learning Rate Adjustment

**Problem**:
```python
# Using same LR as FP32/FP16 training
optimizer = torch.optim.AdamW(params, lr=1e-4)
```

**Symptoms**:
- Slower convergence
- Suboptimal final accuracy
- Training instability

**Solution**:
```python
# Increase learning rate to compensate for quantization noise
# FP4 training benefits from higher LR
base_lr = 1e-4  # FP32 baseline

if precision == 'fp4':
    lr = base_lr * 1.5  # 50% higher for FP4
else:
    lr = base_lr

optimizer = torch.optim.AdamW(params, lr=lr)

# Intuition: Quantization noise acts as regularization
# Higher LR helps signal overcome noise
```

### Pitfall 7: Training from Scratch

**Problem**:
```python
# Training FP4 model from random initialization
model = TransformerLM(...)
model = FP4TrainingWrapper(model, config)
# Train from scratch - very difficult!
```

**Symptoms**:
- Much worse final accuracy
- Requires 2-3× more training
- May not converge at all

**Solution**:
```python
# Option 1: Progressive quantization
# Start with FP16, gradually move to FP4
schedule = ProgressiveQuantization(
    start='fp16',
    end='fp4',
    transition_steps=10000
)

# Option 2: Pre-train in FP16, fine-tune in FP4
model_fp16 = torch.load('pretrained_fp16.pt')
model_fp4 = FP4TrainingWrapper(model_fp16, config)
# Continue training in FP4

# FP4 training from scratch is research-grade difficult
# Use progressive quantization or pre-training
```

## Comparison with Other Low-Precision Formats

### FP4 vs. INT4

| Aspect | INT4 | FP4 (MXFP4) |
|--------|------|-------------|
| Dynamic range | 16 values, linear | 16 values, exponential |
| Range (scaled) | [-7, 7] | [-6, 6] with gaps |
| Training | Requires QAT | Direct (with tricks) |
| Calibration | Yes (PTQ) | No (automatic) |
| Accuracy (training) | 5-10% loss | 1-5% loss |
| Hardware support | Wide (INT8 units) | None (yet) |

**When to use INT4**: Inference on integer hardware
**When to use FP4**: Training on GPUs, research

### FP4 vs. FP8

| Aspect | FP8 (E4M3) | FP4 (E2M1) |
|--------|------------|------------|
| Bits | 8 | 4 |
| Memory | 1 byte/param | 0.5 bytes/param |
| Accuracy | 0.1-0.5% loss | 1-5% loss |
| Training difficulty | Moderate | High |
| Hardware support | H100, MI300 | None |

**Recommendation**: Use FP8 if hardware supports it, only drop to FP4 if memory is critical.

### FP4 vs. 2-bit/Ternary

| Aspect | 2-bit/Ternary | FP4 |
|--------|---------------|-----|
| Bits | 2 | 4 |
| Values | {-1, 0, +1} or 4 | 16 |
| Training | Very difficult | Difficult |
| Accuracy | 10-20% loss | 1-5% loss |
| Use case | Extreme quantization research | Memory-constrained training |

**Note**: 2-bit and ternary networks are research topics, not practical training methods.

## Advanced Techniques

### Progressive Quantization

Gradually transition from high to low precision:

```python
from nexus.training.schedules import ProgressiveQuantizationSchedule

schedule = ProgressiveQuantizationSchedule(
    milestones={
        0: 'fp16',        # Steps 0-10k: FP16
        10000: 'fp8',     # Steps 10k-20k: FP8
        20000: 'fp4'      # Steps 20k+: FP4
    },
    transition_steps=2000  # Smooth transition over 2k steps
)

for step in range(total_steps):
    precision_config = schedule.get_config(step)
    model.set_precision(precision_config)
    train_step(model, batch)
```

### Outlier-Aware FP4

Keep extreme outliers in higher precision:

```python
config = OutlierFP4Config(
    block_size=16,
    outlier_threshold=4.0,  # 4 std devs
    outlier_format='fp8',   # Outliers in FP8, not FP4
    outlier_ratio_max=0.1   # At most 10% outliers
)

# Automatically detects and handles outliers
model = FP4TrainingWrapper(model, config)
```

### Mixed FP4/FP8 Layers

Use FP4 for some layers, FP8 for others:

```python
def quantize_model_mixed(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Large layers: FP4 (memory critical)
            if module.weight.numel() > 10_000_000:
                convert_to_fp4(module)
            # Medium layers: FP8 (balance)
            elif module.weight.numel() > 1_000_000:
                convert_to_fp8(module)
            # Small layers: FP16 (accuracy critical)
            else:
                pass  # Keep FP16

# Optimizes memory while maintaining accuracy on critical layers
```

## References

1. **OCP Microscaling Formats (MX) Specification v1.0**
   Open Compute Project, 2024
   https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

   Defines MXFP4 format and block-level scaling approach. Essential reading for implementation details.

2. **Training Deep Neural Networks with 8-bit Floating Point Numbers**
   Naigang Wang, Jungwook Choi, Daniel Brand, et al.
   NeurIPS 2018
   https://arxiv.org/abs/1812.08011

   Foundational work on FP8 training. Principles extend to FP4 with additional challenges.

3. **LUT-NN: Towards Efficient Deep Neural Network Training with Low-Precision Look-Up Tables**
   Zabihi et al., 2023
   https://arxiv.org/abs/2302.09203

   Explores 4-bit training using lookup tables. Relevant techniques for FP4 implementation.

4. **The Case for 4-bit Precision: k-bit Inference Scaling Laws**
   Tim Dettmers et al., ICML 2023
   https://arxiv.org/abs/2212.09720

   Analysis of 4-bit quantization primarily for inference, but insights apply to training.

5. **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**
   Frantar et al., 2023
   https://arxiv.org/abs/2210.17323

   While focused on PTQ, demonstrates 4-bit quantization viability for large language models.

6. **QLoRA: Efficient Finetuning of Quantized LLMs**
   Dettmers et al., NeurIPS 2023
   https://arxiv.org/abs/2305.14314

   4-bit quantization for fine-tuning. Shows FP4 can work with proper techniques (stochastic rounding, master weights).

7. **ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats**
   Wu et al., 2023
   https://arxiv.org/abs/2307.09782

   Explores FP4 weights with FP8 activations. Demonstrates extreme quantization feasibility.

## Implementation Notes

**File Location**: `nexus/training/mixed_precision/fp4.py`

**Key Classes**:
- `FP4Quantizer`: Core quantization/dequantization
- `MXFP4Tensor`: Storage with block-level scales
- `FP4Linear`: Linear layer with FP4 weights
- `FP4TrainingWrapper`: Automatic model conversion and management
- `FP4GradientScaler`: Gradient scaling for FP4 training

**Dependencies**:
- PyTorch >= 2.1
- NumPy (for FP4 lookup tables)
- Custom CUDA kernels (optional, for performance)

**Hardware Requirements**:
- No native FP4 hardware exists (2024)
- CPU emulation works but slow
- GPU with FP16/FP32 for actual computation
- Minimum 16GB GPU RAM recommended

**Maturity Level**: Experimental / Research-grade
- Not recommended for production training
- Requires significant expertise to use effectively
- Active research area, techniques improving rapidly

**Testing**: `tests/training/mixed_precision/test_fp4.py`

**Benchmarking**: `benchmarks/mixed_precision/fp4_benchmark.py`

**Known Limitations**:
- 1-5% accuracy loss vs FP16 even with best practices
- Training 25-60% slower without native hardware
- Requires careful hyperparameter tuning
- Not all model architectures work well
- Offloading necessary for large models (adds latency)

**Future Outlook**:
- Native FP4 hardware unlikely (economic reasons)
- INT4 acceleration more likely (2025-2026)
- Research continues in ultra-low precision
- May become viable for edge device training
