# FP4: 4-Bit Floating Point Training

## Overview

FP4 enables training with 4 bits per parameter through an extremely low-precision floating-point format. MXFP4 extends this with block-level scaling for better accuracy. Provides 8x memory reduction vs FP32.

## FP4 E2M1 Format

**Bit Layout**:
- 1 sign bit
- 2 exponent bits
- 1 mantissa bit

**Representable Values** (positive half):
```
0b000 → 0.0
0b001 → 0.5
0b010 → 1.0
0b011 → 1.5
0b100 → 2.0
0b101 → 3.0
0b110 → 4.0
0b111 → 6.0
```

**Range**: [-6, 6] with very limited precision

## MXFP4: Block-Level Scaling

Similar to MXFP8 but with 4-bit quantization:

$$\\text{scale}_i = \\frac{\\max|B_i|}{6.0}$$

Each block of 16-32 elements shares one FP16/BF16 scale factor.

## Mathematical Details

### Quantization

**Stochastic rounding** (recommended for training):
```python
nearest = argmin |x - fp4_val|
next_nearest = nearest + 1
prob = (x - fp4_val[nearest]) / (fp4_val[next_nearest] - fp4_val[nearest])
quantized = next_nearest if random() < prob else nearest
```

**Deterministic rounding**:
```python
quantized = argmin |x - fp4_val|
```

### Dequantization

```python
value = fp4_lookup_table[quantized_bits]
dequantized = value * scale
```

## Implementation

### Basic Usage

```python
from nexus.training.mixed_precision import to_fp4, FP4Linear, FP4Config

# Configuration
config = FP4Config(
    use_microscaling=True,
    block_size=32,
    stochastic_rounding=True,
    clip_value=6.0
)

# Convert tensor to FP4
tensor = torch.randn(1024, 1024)
fp4_tensor = to_fp4(tensor, config)

# Use FP4 linear layer
layer = FP4Linear(768, 3072, config=config)
output = layer(input)
```

### Training with FP4

```python
from nexus.training.mixed_precision import FP4Config, FP4Linear, FP4GradientScaler

# Replace linear layers with FP4 versions
model = MyModel()  # With FP4Linear layers
optimizer = torch.optim.AdamW(model.parameters())
scaler = FP4GradientScaler(init_scale=2**10)

for batch in dataloader:
    # Forward (FP4 weights)
    loss = model(batch)
    
    # Backward
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    
    # Unscale and step
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

## Memory Savings

**Calculation**:
- FP32: 4 bytes per parameter
- FP4: 0.5 bytes per parameter + scales
- Scales: ~2 bytes per 32 parameters

**Example** (1B parameters):
- FP32: 4GB
- MXFP4: 0.5GB + 62.5MB = 0.5625GB
- **Savings**: 86%

## Accuracy Considerations

**Expected Accuracy Loss**:
- Standard FP4: 5-15% degradation
- MXFP4: 1-5% degradation
- With careful tuning: <1%

**Key techniques**:
1. **Stochastic rounding**: Essential for training
2. **Gradient scaling**: Prevent underflow
3. **Mixed precision**: FP4 weights, FP16/BF16 activations
4. **Block scaling**: MXFP4 much better than global scaling

## When to Use

**Best for**:
- Extreme memory constraints (>100B params on limited hardware)
- Research into ultra-low precision
- Edge deployment (inference)

**Not recommended for**:
- Most production training (FP16/BF16/FP8 sufficient)
- Small models (overhead not worth it)
- High-accuracy requirements

**Reality check**: FP4 training is experimental. FP8/BF16 are more practical for most use cases.

## Configuration Guidelines

**Block Size**:
- Smaller (16): Better accuracy, more overhead
- Larger (64): Less overhead, lower accuracy
- **Recommended**: 32

**Stochastic Rounding**:
- **Always enable** for training
- Adds randomness that prevents quantization bias

**Gradient Scaling**:
- Start with `init_scale=2**10`
- Adjust based on overflow/underflow

## Performance

**Memory**: 8x reduction vs FP32  
**Accuracy**: 85-99% of FP32 (highly variable)  
**Compute**: No speed benefit without specialized hardware  
**Maturity**: Experimental (use FP8 in production)

## References

**FP8 Formats for Deep Learning (adapted to FP4)**  
Micikevicius et al., 2022

**OCP Microscaling Formats (MX) Specification**  
Open Compute Project, 2024

**Implementation**: `nexus/training/mixed_precision/fp4.py`
