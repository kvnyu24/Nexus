# Activation Offloading: CPU Memory for GPU Compute

## Overview

Activation offloading (Chen et al., 2021) reduces GPU memory usage by offloading activations to CPU memory during forward pass and asynchronously prefetching them back before backward pass.

## Key Concept

**Problem**: Activations consume significant GPU memory (stored for backward pass).

**Solution**: 
1. **Forward**: Store activations on CPU (slower but abundant memory)
2. **Backward**: Prefetch activations back to GPU just-in-time
3. **Async**: Overlap prefetch with computation

**Trade-off**: CPU ↔ GPU transfer time vs GPU memory savings

## When It Works

### Good Scenarios
- **Large activations**: High memory cost
- **Fast CPU-GPU interconnect**: PCIe 4.0, NVLink
- **CPU memory available**: Plenty of RAM
- **Compute-bound model**: Transfer latency hidden by compute

### Bad Scenarios
- **Small activations**: Overhead > savings
- **Slow interconnect**: Transfer becomes bottleneck
- **Memory-bound model**: Can't hide latency
- **Limited CPU memory**: Defeats the purpose

## Implementation

### Basic Usage

```python
from nexus.training.gradient_methods import ActivationOffloader

offloader = ActivationOffloader(
    enabled=True,
    offload_threshold_mb=10.0,  # Only offload activations >10MB
    prefetch_ahead=2,  # Prefetch 2 layers ahead
    pin_memory=True  # Use pinned memory for faster transfer
)

# Use as context manager
with offloader.offload_context():
    # Forward pass: Activations offloaded to CPU
    output = model(input)
    
    # Backward pass: Activations prefetched from CPU
    loss = criterion(output, target)
    loss.backward()

optimizer.step()
optimizer.zero_grad()
```

### Manual Offloading

```python
offloader = ActivationOffloader()

# Offload specific tensor
storage_id = offloader._offload_to_cpu(activation_tensor)

# Later, prefetch it back
device = torch.device('cuda:0')
activation_restored = offloader._prefetch_to_gpu(storage_id, device)
```

### Integration with Training Loop

```python
model = MyLargeModel().cuda()
optimizer = torch.optim.AdamW(model.parameters())
offloader = ActivationOffloader(enabled=True)

for epoch in range(num_epochs):
    for batch in dataloader:
        with offloader.offload_context():
            # Forward with offloading
            output = model(batch['input'])
            loss = criterion(output, batch['target'])
            
            # Backward with prefetching
            loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
```

## Configuration

### Offload Threshold

**Purpose**: Only offload large activations (avoid overhead for small tensors)

```python
offloader = ActivationOffloader(
    offload_threshold_mb=10.0  # Offload if activation >10MB
)
```

**Tuning**:
- **Lower (e.g., 1MB)**: Offload more, save more memory, higher overhead
- **Higher (e.g., 50MB)**: Offload less, less overhead, less savings
- **Recommended**: 10-20MB

### Prefetch Ahead

**Purpose**: Prefetch activations before they're needed (hide latency)

```python
offloader = ActivationOffloader(
    prefetch_ahead=2  # Prefetch 2 layers ahead during backward
)
```

**Tuning**:
- **Higher**: Better latency hiding, more memory used temporarily
- **Lower**: Less latency hiding, lower memory usage
- **Recommended**: 1-3

### Pinned Memory

**Purpose**: Use pinned (page-locked) CPU memory for faster transfers

```python
offloader = ActivationOffloader(
    pin_memory=True  # Enable pinned memory
)
```

**Effect**:
- **True**: Faster CPU ↔ GPU transfer, uses page-locked memory
- **False**: Slower transfer, regular CPU memory
- **Recommended**: True (unless CPU memory limited)

## Memory Savings

### Estimation

```python
from nexus.training.gradient_methods import estimate_offloading_memory_savings

savings = estimate_offloading_memory_savings(
    activation_memory_gpu_mb=8000,  # 8GB activations on GPU
    offload_ratio=0.5  # Offload 50% to CPU
)

print(f"GPU memory: {savings['gpu_memory_mb']:.0f} MB")
print(f"GPU savings: {savings['gpu_savings_mb']:.0f} MB")
print(f"CPU usage: {savings['cpu_usage_mb']:.0f} MB")
print(f"Savings: {savings['savings_percent']:.1f}%")
```

### Typical Results

**Example**: GPT-3 175B, batch size 32
- Activation memory: 60GB (GPU)
- Offload 50%: 30GB moved to CPU
- **GPU savings**: 30GB
- **Overhead**: ~5-10% slower training

## Performance Characteristics

### Transfer Bandwidth

**PCIe 4.0 x16**: ~32 GB/s  
**PCIe 3.0 x16**: ~16 GB/s  
**NVLink**: ~300 GB/s

**Latency hiding**: Prefetch ahead to overlap transfer with compute

### Memory Breakdown

**GPU**:
- Model parameters: No change
- Optimizer states: No change
- Activations: Reduced by offload_ratio
- Gradients: No change

**CPU**:
- Offloaded activations: Added

### Time Overhead

**Transfer time per activation**:
$$t_{\\text{transfer}} = \\frac{\\text{size}_{\\text{MB}} \\times 1024}{\\text{bandwidth}_{\\text{GB/s}}} \\text{ ms}$$

**Example**: 100MB activation, PCIe 4.0
$$t = \\frac{100 \\times 1024}{32 \\times 1024} = 3.125 \\text{ ms}$$

**Hidden if**: Compute time per layer > transfer time

## Advanced Usage

### Selective Offloading

Only offload specific layers:

```python
class ModelWithOffloading(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1024, 4096)  # Large, offload
        self.layer2 = nn.Linear(4096, 1024)  # Large, offload
        self.layer3 = nn.Linear(1024, 10)  # Small, don't offload
        
        self.offloader = ActivationOffloader()
    
    def forward(self, x):
        # Offload layer1 activations
        if self.offloader._should_offload(x):
            storage_id = self.offloader._offload_to_cpu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # No offload
        
        return x
```

### Combine with Checkpointing

```python
from nexus.training.gradient_methods import ActivationOffloader, SelectiveCheckpoint

# Checkpoint + offload for maximum memory savings
checkpoint_fn = SelectiveCheckpoint(policy="heavy_ops")
offloader = ActivationOffloader(enabled=True)

with offloader.offload_context():
    # Checkpointed operations recompute
    # Non-checkpointed activations offloaded
    output = checkpoint_fn(model.heavy_layer, input)
```

**Result**: ~75% memory reduction (50% from checkpoint, 50% of remainder from offload)

## Comparison to Other Techniques

| Technique | Memory Savings | Compute Overhead | Transfer Overhead |
|-----------|----------------|------------------|-------------------|
| **Activation Checkpointing** | 50-90% | 50-100% | 0% |
| **Activation Offloading** | 30-60% | 0% | 5-15% |
| **Mixed Precision (FP16)** | 50% | 0% | 0% |
| **Gradient Checkpointing + Offloading** | 70-80% | 50-100% | 5-15% |

**Best Combination**: Checkpointing + Mixed Precision + Offloading = 80-90% reduction

## When to Use

**Use activation offloading when**:
- GPU memory is bottleneck
- CPU memory is abundant
- Have fast CPU-GPU interconnect
- Training very large models

**Don't use when**:
- Small models (overhead not worth it)
- Slow interconnect (bottleneck)
- Limited CPU memory
- Inference (no backward pass to benefit from)

## Limitations

1. **Transfer Overhead**: Always adds some latency
2. **CPU Memory Required**: Needs spare CPU RAM
3. **Complexity**: More moving parts, harder to debug
4. **Async Issues**: Potential race conditions if not careful

## Troubleshooting

### Slower Than Expected

1. **Check bandwidth**: Verify PCIe speed with `nvidia-smi topo -m`
2. **Increase prefetch_ahead**: Hide more latency
3. **Enable pinned memory**: Faster transfers
4. **Profile transfers**: Use torch.profiler to see transfer costs

### Out of CPU Memory

1. **Reduce offload_ratio**: Don't offload everything
2. **Increase offload_threshold**: Only offload largest activations
3. **Use swap**: Allow OS to use disk (slow!)

### Incorrect Results

1. **Verify offload/restore**: Ensure activations restored correctly
2. **Check RNG state**: Dropout may behave differently
3. **Test without offloading**: Compare to baseline

## References

**ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compression**  
Chen et al., 2021

**ZeRO-Offload: Democratizing Billion-Scale Model Training**  
Ren et al., Microsoft, 2021

**Implementation**: `nexus/training/gradient_methods.py` (ActivationOffloader class)
