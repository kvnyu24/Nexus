# GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

## Overview & Motivation

GPTQ is a post-training quantization method that compresses large language models to 3-4 bits per weight with minimal accuracy loss. Unlike quantization-aware training (QAT), GPTQ requires no retraining—just a small calibration dataset and a one-time quantization procedure.

### Efficiency Gains

- **Memory**: 4× reduction (4-bit) to 5.3× (3-bit) vs fp16
- **Speed**: 2-3× faster inference on GPU (INT4 kernels)
- **Storage**: 175B models fit in 44GB instead of 350GB
- **No Training**: Post-training only, uses ~128 calibration samples

### Key Innovation

GPTQ adapts the Optimal Brain Surgeon (OBS) framework for trillion-parameter models by:
1. **Layer-wise quantization**: Process one layer at a time (scalable)
2. **Lazy batching**: Update weights in blocks, not individually (efficient)
3. **Cholesky-based**: Use Cholesky decomposition for inverse Hessian (numerically stable)

## Theoretical Background

### Optimal Brain Surgeon (OBS)

Classical pruning/quantization asks: "How do I round a weight to reduce error?"

OBS provides the optimal answer by considering second-order effects. When quantizing weight w_j:

```
Error = (w_j - q_j)² / (2 * [H⁻¹]_jj)
```

where:
- q_j: Quantized value
- H: Hessian matrix (second derivatives of loss)
- [H⁻¹]_jj: Diagonal element of inverse Hessian

**Key insight**: Quantizing w_j creates error that propagates to other weights. Compensate by updating remaining weights:

```
Δw = -(w_j - q_j) / [H⁻¹]_jj * H⁻¹[:, j]
```

This update minimizes the second-order approximation of the loss increase.

### Hessian Approximation

For layer with weight W ∈ ℝ^(d×k) and input activations X ∈ ℝ^(n×k):

```
H = 2 * X^T X / n
```

This is the Gauss-Newton approximation, valid when outputs are close to predictions (true after pre-training).

**Why it works**: The layer-wise loss is approximately:

```
L ≈ ||Y - WX||² where Y = W_pretrained X
```

The Hessian of this quadratic is exactly 2 X^T X.

### GPTQ Algorithm

**Challenge**: Naive OBS is O(k³) per weight due to H⁻¹ computation. For k=4096, this is intractable.

**GPTQ solution**:
1. Compute H⁻¹ once using Cholesky decomposition: O(k³)
2. Process weights in blocks (e.g., 128 columns at a time)
3. Accumulate errors within each block
4. Propagate block error to remaining blocks: O(k²)

Total complexity: O(k³) + O(k² * blocks) ≈ O(k³) once per layer.

## Mathematical Formulation

### Layer-wise Quantization

For weight matrix W ∈ ℝ^(d×k):

1. **Hessian computation**:
   ```
   H = (2/n) * Σ_{i=1}^n x_i x_i^T
   ```
   where x_i are input activations from calibration data.

2. **Dampening** (numerical stability):
   ```
   H ← H + λ * diag(H)
   ```
   Typical λ = 0.01 * mean(diag(H)).

3. **Inverse Cholesky**:
   ```
   L = cholesky(H)
   H⁻¹ = (L^T L)⁻¹
   U = cholesky(H⁻¹)  # Upper triangular
   ```

4. **Block-wise quantization**:
   For each block of B columns (j = b*B to (b+1)*B):
   ```
   For each column c in block:
       q_c = quantize(w_c, scale, zero)
       err_c = (w_c - q_c) / U[c, c]

       # Compensate remaining columns in block
       w_{c+1:} -= err_c * U[c, c+1:]

   # Compensate subsequent blocks
   W_{next_blocks} -= Err_block @ U[block, next_blocks]
   ```

### Quantization Function

**Symmetric (no zero-point)**:
```
scale = max(|w|) / (2^(bits-1) - 1)
q = clamp(round(w / scale), -2^(bits-1), 2^(bits-1)-1)
dequant = q * scale
```

**Asymmetric (with zero-point)**:
```
scale = (max(w) - min(w)) / (2^bits - 1)
zero = round(-min(w) / scale)
q = clamp(round(w / scale) + zero, 0, 2^bits - 1)
dequant = (q - zero) * scale
```

### Activation Ordering (act_order)

Insight: Quantize high-magnitude columns first to minimize accumulated error.

```
perm = argsort(diag(H), descending=True)
W_reordered = W[:, perm]
# Apply GPTQ to W_reordered
# Store permutation for inference
```

### Group Quantization

Instead of per-channel (one scale per output), use groups:

```
For g = 0 to num_groups:
    g_start = g * group_size
    g_end = (g+1) * group_size

    scale[g] = max(|W[:, g_start:g_end]|) / max_int
    Q[:, g_start:g_end] = quantize(W[:, g_start:g_end], scale[g])
```

Smaller groups → better accuracy, larger overhead.

| Group Size | Overhead | Quality |
|------------|----------|---------|
| -1 (per-channel) | Lowest | Poor |
| 128 | 0.8% | Good |
| 64 | 1.6% | Better |
| 32 | 3.1% | Best |

## High-Level Intuition

### Why OBS-style Compensation Works

Imagine a balance scale with 1000 weights. You round weight #1, causing imbalance. Naive quantization ignores this. OBS rebalances by slightly adjusting the other 999 weights.

The Hessian tells you HOW MUCH each weight should adjust. Weights highly correlated with weight #1 (high H⁻¹[1, :]) adjust more.

### Why Process Columns in Blocks?

Processing one weight at a time:
- Pro: Optimal error compensation
- Con: O(k³) operations per weight → too slow

Blocking (process 128 weights together):
- Pro: Amortize inverse Hessian lookups, ~100× faster
- Con: Slightly suboptimal (errors within block don't compensate each other)

In practice, block size 128 loses <0.1% accuracy vs. column-by-column.

### Why Activation Ordering Helps

Early quantization errors accumulate. By quantizing high-activation columns first:
1. Most "important" weights quantized with least accumulated error
2. Low-activation columns (less critical) absorb more error
3. Net effect: better accuracy

Analogy: If you're building a tower, place the strongest blocks at the bottom (they bear the most weight).

## Implementation Details

### Calibration Data Collection

```python
from nexus.models.compression.quantization import GPTQQuantizer, GPTQConfig

quantizer = GPTQQuantizer(gptq_config=GPTQConfig(
    bits=4,
    group_size=128,
    act_order=True,
    damp_percent=0.01
))

# Collect ~128 samples
for batch in calibration_loader:
    activations = model(batch, output_hidden_states=True)
    quantizer.add_calibration_data(activations.hidden_states[-1])
```

### Quantizing a Model

```python
# Option 1: Quantize weights in-place
model, metrics = quantizer.quantize_model(
    model,
    calibration_data,
    nsamples=128,
    replace_with_quantized=False  # Keep as float
)

# Option 2: Replace with QuantizedLinear (for deployment)
model, metrics = quantizer.quantize_model(
    model,
    calibration_data,
    nsamples=128,
    replace_with_quantized=True  # Use INT4 kernels
)
```

### QuantizedLinear Layer

Reference: `Nexus/nexus/models/compression/quantization/gptq.py`

```python
class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bits=4, group_size=128):
        # Packed integer weights
        self.register_buffer("qweight", torch.zeros(..., dtype=torch.int32))

        # Per-group scales and zeros
        self.register_buffer("scales", torch.zeros(n_groups, out_features))
        self.register_buffer("zeros", torch.zeros(n_groups, out_features))

    def forward(self, x):
        # Dequantize on-the-fly
        weight_dequant = self._dequantize()
        return F.linear(x, weight_dequant)
```

### Weight Packing

4-bit weights are packed 8 per int32:

```python
def pack_weights(int_weight: torch.Tensor, bits: int) -> torch.Tensor:
    # int_weight: (out, in) with values in [0, 15]
    vals_per_int32 = 32 // bits  # 8 for 4-bit

    packed = torch.zeros(out, in // vals_per_int32, dtype=torch.int32)
    for i in range(vals_per_int32):
        packed |= int_weight[:, i::vals_per_int32] << (bits * i)

    return packed
```

## Code Walkthrough

### Core GPTQ Loop

```python
def _quantize_weight(self, W, H):
    # 1. Prepare inverse Hessian
    H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)

    # 2. Optionally reorder by activation
    if self.act_order:
        perm = torch.argsort(H.diag(), descending=True)
        W = W[:, perm]
        H_inv_chol = H_inv_chol[perm][:, perm]

    # 3. Process in blocks
    for block_start in range(0, in_features, self.block_size):
        block_end = min(block_start + self.block_size, in_features)

        W_block = W[:, block_start:block_end]
        Err_block = torch.zeros_like(W_block)

        # 4. Quantize each column in block
        for j in range(block_end - block_start):
            col_idx = block_start + j
            w_col = W_block[:, j]
            h_inv_diag = H_inv_chol[col_idx, col_idx]

            # 5. Compute group quantization parameters
            group_idx = col_idx // self.group_size
            scale, zero = self._compute_scale_zero(
                W[:, group_idx*self.group_size:(group_idx+1)*self.group_size]
            )

            # 6. Quantize
            q_col = torch.clamp(torch.round(w_col / scale) + zero, 0, 2**bits-1)
            dq_col = (q_col - zero) * scale
            W_block[:, j] = dq_col

            # 7. Compute error
            err = (w_col - dq_col) / h_inv_diag.clamp(min=1e-10)
            Err_block[:, j] = err

            # 8. Compensate remaining columns in block
            if j + 1 < block_len:
                W_block[:, j+1:] -= err.unsqueeze(1) * H_inv_chol[col_idx, col_idx+1:block_end]

        # 9. Compensate subsequent blocks
        if block_end < in_features:
            W[:, block_end:] -= Err_block @ H_inv_chol[block_start:block_end, block_end:]

    return W, scales, zeros
```

## Optimization Tricks

### 1. Dampening Tuning

The dampening factor λ affects numerical stability and accuracy:

| damp_percent | Stability | Accuracy | Use Case |
|--------------|-----------|----------|----------|
| 0.001 | Low | Best | Well-conditioned models |
| 0.01 | Good | Good | Default |
| 0.1 | Excellent | Acceptable | Ill-conditioned models |

**Guideline**: Start with 0.01. If Cholesky fails, increase to 0.1.

### 2. Block Size Selection

| Block Size | Speed | Accuracy | Memory |
|------------|-------|----------|--------|
| 32 | Slow | Best | Low |
| 128 | Medium | Good | Medium |
| 256 | Fast | Acceptable | High |

**Recommendation**: 128 for most cases. Use 256 for speed, 32 for maximum quality.

### 3. Static vs Dynamic Groups

**Static groups**: Compute scale/zero before OBS, don't update them.
**Dynamic groups**: Recompute after each weight update.

```python
GPTQConfig(static_groups=True)  # Faster, slightly less accurate
GPTQConfig(static_groups=False)  # Slower, slightly more accurate
```

Trade-off: Static is 20-30% faster with ~0.1% accuracy loss.

### 4. Calibration Data Quality

Number of samples:

| Samples | Time | Accuracy |
|---------|------|----------|
| 32 | Fast | Acceptable |
| 128 | Medium | Good (recommended) |
| 512 | Slow | Best |

Diminishing returns beyond 128 samples.

**Data diversity**: Use samples from the target domain. Random Wikipedia is okay for general models, but domain-specific data helps.

### 5. Mixed Precision

Not all layers benefit equally from quantization. Keep critical layers in higher precision:

```python
# Keep first and last layers in fp16
target_layers = [
    name for name in model_layer_names
    if 'embed' not in name and 'lm_head' not in name
]

model, metrics = quantizer.quantize_model(
    model,
    calibration_data,
    target_layers=target_layers
)
```

### 6. Activation Ordering Trade-offs

| act_order | Accuracy | Inference Speed | Complexity |
|-----------|----------|-----------------|------------|
| False | Good | Fast (no permutation) | Simple |
| True | Best | Slower (must un-permute) | Complex |

**Recommendation**: Always use `act_order=True` during quantization. For inference, either:
- Store permutation and apply it (small overhead)
- Pre-permute activations (rearrange model structure)

## Experiments & Results

### Original GPTQ Paper Results

**OPT-175B Perplexity on WikiText2**:

| Method | Bits | Perplexity | Memory (GB) |
|--------|------|------------|-------------|
| FP16 | 16 | 10.13 | 350 |
| RTN (round-to-nearest) | 4 | 35.77 | 87.5 |
| GPTQ | 4 | 10.38 | 87.5 |
| GPTQ | 3 | 11.54 | 65.6 |

GPTQ at 4-bit has only +2.5% perplexity vs FP16, while RTN degrades by +253%.

**BLOOM-176B Zero-shot Accuracy**:

| Task | FP16 | GPTQ 4-bit | GPTQ 3-bit |
|------|------|------------|------------|
| LAMBADA | 67.6 | 67.2 (-0.4) | 66.1 (-1.5) |
| HellaSwag | 73.0 | 72.7 (-0.3) | 71.2 (-1.8) |
| WinoGrande | 70.1 | 69.8 (-0.3) | 68.5 (-1.6) |
| Arc-Easy | 77.3 | 77.0 (-0.3) | 75.8 (-1.5) |
| Arc-Challenge | 45.1 | 44.8 (-0.3) | 43.2 (-1.9) |

**Average degradation**: -0.3% (4-bit), -1.7% (3-bit).

### Group Size Ablation (LLaMA-7B)

| Group Size | Perplexity | Memory Overhead |
|------------|------------|-----------------|
| -1 (per-channel) | 6.35 | 0.01% |
| 128 | 6.02 | 0.78% |
| 64 | 5.95 | 1.56% |
| 32 | 5.92 | 3.13% |
| FP16 baseline | 5.68 | - |

Sweet spot: group_size=128 balances accuracy and overhead.

### Speed Benchmarks (A100 GPU)

**LLaMA-13B Generation Speed**:

| Precision | Tokens/sec | Speedup |
|-----------|----------|---------|
| FP16 | 32 | 1.0× |
| GPTQ 4-bit | 78 | 2.4× |
| GPTQ 3-bit | 94 | 2.9× |

Int4/Int3 kernels provide significant speedup.

### Compression Ratios

| Model | FP16 Size | GPTQ 4-bit | GPTQ 3-bit | Reduction |
|-------|-----------|------------|------------|-----------|
| LLaMA-7B | 13 GB | 3.5 GB | 2.7 GB | 3.7× / 4.8× |
| LLaMA-13B | 24 GB | 6.5 GB | 5.0 GB | 3.7× / 4.8× |
| LLaMA-65B | 120 GB | 33 GB | 25 GB | 3.6× / 4.8× |
| OPT-175B | 350 GB | 87.5 GB | 65.6 GB | 4.0× / 5.3× |

## Common Pitfalls

### 1. Insufficient Calibration Data

**Symptom**: Poor accuracy, high perplexity.

**Diagnosis**:
```python
# Check Hessian condition number
H = quantizer._compute_hessian(calibration_inputs)
cond = torch.linalg.cond(H)
print(f"Condition number: {cond}")  # Should be < 1e6
```

**Solution**: Increase calibration samples from 32 → 128 → 512.

### 2. Ill-conditioned Hessian

**Symptom**: Cholesky decomposition fails with "not positive definite" error.

**Solution**:
- Increase dampening: `damp_percent=0.1`
- Use more calibration data
- Check for zero/constant activations

### 3. Quantizing Wrong Layers

**Symptom**: Severe accuracy drop.

**Solution**: Skip embedding and output layers:
```python
target_layers = [n for n in layer_names if 'embed' not in n and 'lm_head' not in n]
```

These layers are more sensitive to quantization.

### 4. Group Size Too Large

**Symptom**: Accuracy degradation despite 4-bit quantization.

**Solution**: Reduce group size from 128 → 64 → 32. Larger models can tolerate larger groups.

### 5. Not Using Activation Ordering

**Symptom**: Lower accuracy than expected.

**Solution**: Always enable `act_order=True`. The overhead is minor compared to the accuracy gain.

### 6. Bit-packing Errors

**Symptom**: Inference outputs nonsensical results.

**Diagnosis**:
```python
# Verify pack/unpack is invertible
original = torch.randint(0, 16, (100, 100))
packed = pack_weights(original, bits=4)
unpacked = unpack_weights(packed, bits=4, out_features=100, in_features=100)
assert torch.equal(original, unpacked)
```

**Solution**: Ensure pack/unpack functions are correctly implemented and tested.

### 7. Mixing Quantized and Non-Quantized Layers

**Symptom**: Runtime errors about dtype mismatches.

**Solution**: Cast inputs appropriately:
```python
# Before quantized layer
x = x.to(torch.float16)  # Match dequantized weight dtype
```

## References

1. **Original Paper**: Frantar, E., et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers." ICLR 2023. https://arxiv.org/abs/2210.17323

2. **Optimal Brain Surgeon**: Hassibi, B., & Stork, D. G. "Second Order Derivatives for Network Pruning: Optimal Brain Surgeon." NeurIPS 1992.

3. **AutoGPTQ**: https://github.com/PanQiWei/AutoGPTQ (Efficient implementation)

4. **exllama/exllamav2**: Fast INT4 inference kernels for GPTQ models.

5. **Nexus Implementation**: `Nexus/nexus/models/compression/quantization/gptq.py`

## See Also

- [AWQ](awq.md): Activation-aware quantization (alternative to GPTQ)
- [QLoRA](../peft/qlora.md): Quantized LoRA (combines quantization + PEFT)
- [SparseGPT](../pruning/sparse_gpt.md): Similar OBS-based method for pruning
- [QuIP#](quip_sharp.md): Lattice-based quantization (better than GPTQ at 2-bit)
