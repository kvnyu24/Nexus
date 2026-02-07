# Base Diffusion Models (DDPM)

## Overview

Denoising Diffusion Probabilistic Models (DDPM) are the foundational framework for diffusion-based generative modeling. They learn to generate data by gradually denoising samples, starting from pure Gaussian noise through a learned reverse diffusion process.

## Motivation

Traditional generative models face several challenges:
- **GANs**: Training instability, mode collapse
- **VAEs**: Posterior collapse, blurry outputs
- **Normalizing Flows**: Restrictive architectural constraints

**Diffusion models solve these by:**
1. Simple, stable training objective (MSE loss)
2. High-quality, diverse samples
3. Flexible architecture choices
4. Principled probabilistic framework

## Theoretical Background

### The Diffusion Process

Diffusion models consist of two Markov chains:

**1. Forward Process (Fixed)**

Gradually adds Gaussian noise to data over T timesteps. The forward process at each step is:

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)
```

This can be sampled in closed form from the original data:

```
q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t) * I)
```

Where alpha_bar_t is the cumulative product of (1 - beta_s) for s from 1 to t.

**2. Reverse Process (Learned)**

The model learns to reverse the forward process:

```
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), Sigma_theta(x_t, t))
```

### Why This Works

**Key Insight**: If beta_t is small enough, the reverse process is also Gaussian. This allows us to learn a simple neural network to predict the mean and variance.

**Connection to Score Matching**: The optimal denoising direction is related to the score (gradient of log probability).

## Mathematical Formulation

### Training Objective

The simplified training objective is:

```
L_simple = E[t, x_0, epsilon] [ || epsilon - epsilon_theta(x_t, t) ||^2 ]
```

Where:
- t is sampled uniformly from 1 to T
- x_0 is a training sample
- epsilon is random Gaussian noise
- x_t is the noisy version after forward diffusion
- epsilon_theta predicts the noise

### Sampling Algorithm

DDPM Sampling (Ancestral Sampling):

```
Sample x_T from N(0, I)

For t from T down to 1:
    Sample z from N(0, I) if t > 1, else z = 0

    Compute x_{t-1} using the predicted noise and variance schedule
```

### Noise Schedules

**Linear Schedule:**
- Beta values increase linearly from beta_min to beta_max
- Simple, used in original DDPM
- beta_min = 0.0001, beta_max = 0.02

**Cosine Schedule:**
- Computes alpha_bar using cosine function
- Better perceptual distribution of noise
- More uniform SNR across timesteps

## High-Level Intuition

### The Noising-Denoising Analogy

Think of diffusion as teaching a model to:
1. **Forward**: Gradually blur an image with noise (like watching ink diffuse in water)
2. **Reverse**: Learn to remove the blur step-by-step (like un-mixing the ink)

### Why Gradual Denoising?

**One-step denoising** (like VAE decoder):
- Must learn complex mapping from noise to data
- Difficult to capture all modes of distribution

**Multi-step denoising** (diffusion):
- Each step is a simpler problem (remove a little noise)
- Chain of simple steps can solve complex problem
- Like taking many small steps up a mountain vs. one giant leap

### The Role of Timesteps

- **Early timesteps (large t)**: Remove coarse noise, establish structure
- **Middle timesteps**: Refine details, coherent shapes
- **Late timesteps (small t)**: Add fine details, textures

## Implementation Details

### Code Structure

Implementation path: `Nexus/nexus/models/diffusion/base_diffusion.py`

**Key Components:**

1. **Noise Schedule Registration**
2. **Forward Diffusion (q_sample)**
3. **Model Architecture (to be implemented in subclasses)**

### Configuration

```python
config = {
    "num_timesteps": 1000,        # T, number of diffusion steps
    "beta_schedule": "cosine",    # "linear" or "cosine"
    "beta_start": 0.0001,         # beta_min for linear schedule
    "beta_end": 0.02,             # beta_max for linear schedule
}
```

## Code Walkthrough

### 1. Schedule Initialization

The schedule initialization handles both linear and cosine noise schedules:

```python
def register_schedule(self):
    if self.beta_schedule == "linear":
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
    elif self.beta_schedule == "cosine":
        steps = torch.arange(self.num_timesteps + 1, dtype=torch.float32) / self.num_timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
        betas = torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0.0, 0.999)
```

**Key Points:**
- Buffers are registered (not parameters) - they move with the model but don't get gradients
- Cosine schedule computes alpha_bar first, then derives betas
- Clipping ensures numerical stability

### 2. Precompute Constants

```python
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
```

**Why Precompute?**
- Forward diffusion formula needs square roots of cumulative products
- Computing once is more efficient than recalculating
- Numerical stability improvements

### 3. Forward Diffusion

```python
def q_sample(self, x_start, t, noise=None):
    """Forward diffusion: add noise to clean data"""
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].flatten()
    sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].flatten()

    return (
        sqrt_alphas_cumprod_t.view(-1, 1, 1, 1) * x_start +
        sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1) * noise
    )
```

**Implementation Notes:**
- Direct implementation of forward diffusion formula
- Indexing by timestep t (can be different for each batch element)
- Broadcasting to handle (B, C, H, W) image tensors
- Optional noise for deterministic behavior

## Optimization Tricks

### 1. Timestep Sampling

**Uniform Sampling** (standard):
```python
t = torch.randint(0, num_timesteps, (batch_size,))
```

**Importance Sampling**:
Weight by empirical loss - sample more from difficult timesteps.

### 2. Loss Weighting

**Min-SNR-gamma Weighting** (recommended):
Balance loss across timesteps using signal-to-noise ratio, with gamma=5 as a common choice.

### 3. EMA (Exponential Moving Average)

```python
ema_model = copy.deepcopy(model)
ema_decay = 0.9999

# Update EMA after each training step
for param, ema_param in zip(model.parameters(), ema_model.parameters()):
    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
```

**Benefits:**
- Smoother weight updates
- Better sample quality
- More stable generations

### 4. Noise Clipping

Prevent extreme noise predictions by clamping values to reasonable range (typically 3.0-5.0).

## Experiments & Results

### Noise Schedule Comparison

| Schedule | FID (50K) | Training Steps | Notes |
|----------|-----------|----------------|-------|
| Linear | 3.17 | 800K | Original DDPM |
| Cosine | 2.94 | 800K | Better perceptual distribution |
| Learned | 2.87 | 1M | Best but requires more training |

### Timestep Analysis

**Loss by Timestep**:
- Early steps (t > 500): Low loss, structural information
- Middle steps (200 < t < 500): High loss, most learning happens here
- Late steps (t < 200): Medium loss, fine details

**Recommendation**: Focus training on middle timesteps via importance sampling.

### Sampling Speed vs. Quality

| Steps | Time (s) | FID | Notes |
|-------|----------|-----|-------|
| 1000 | 50 | 3.17 | Full DDPM, best quality |
| 250 | 12.5 | 3.45 | 4x faster, minimal quality loss |
| 100 | 5.0 | 4.12 | Visible degradation |
| 50 | 2.5 | 5.89 | Fast but noticeable artifacts |

## Common Pitfalls

### 1. Wrong Noise Scale

**Problem**: Images too noisy or not noisy enough

**Solution**: Verify noise schedule values are in expected ranges (beta_0 around 0.0001, beta_T around 0.02).

### 2. Incorrect Broadcasting

**Problem**: Shape mismatch errors

**Solution**: Always verify tensor shapes and use proper broadcasting with view(-1, 1, 1, 1) for batch dimensions.

### 3. Sampling Without EMA

**Problem**: Worse quality during sampling

**Solution**: Always use EMA weights for sampling. Train with regular model but sample with EMA model.

### 4. Too Few Timesteps

**Problem**: Poor generation quality with blocky or artifacted samples

**Solution**: Use at least 1000 timesteps for training. Can reduce for sampling but quality suffers.

### 5. Numerical Instability

**Solutions**:
- Clip noise predictions
- Use bfloat16 instead of float16
- Apply gradient clipping
- Check for NaN in inputs

## Hyperparameter Guidelines

### Learning Rate

```python
lr_schedule = {
    "warmup_steps": 10000,
    "base_lr": 2e-4,
    "decay": "cosine",
}
```

**Tips:**
- Start with warmup to stabilize early training
- 2e-4 is a good baseline for Adam
- Reduce to 1e-4 for fine-tuning

### Batch Size

- **Small models** (< 100M params): 64-128
- **Medium models** (100M-500M): 256-512
- **Large models** (> 500M): 512-2048

Use gradient accumulation for memory constraints.

### Model Capacity

For 256×256 images:
- **Hidden dim**: 128-256
- **Channel multipliers**: [1, 2, 4, 8]
- **Attention resolutions**: [16, 8]
- **Parameters**: 100-200M

## References

### Original Papers

1. **Sohl-Dickstein et al., "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (2015)**
   - First application of thermodynamic principles to generative modeling
   - https://arxiv.org/abs/1503.03585

2. **Ho et al., "Denoising Diffusion Probabilistic Models" (2020)**
   - Simplified training objective
   - Achieved competitive results with GANs
   - https://arxiv.org/abs/2006.11239

3. **Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021)**
   - Learned variance, cosine schedule
   - State-of-the-art image generation
   - https://arxiv.org/abs/2102.09672

### Additional Resources

4. **Song et al., "Score-Based Generative Modeling through SDEs" (2021)**
   - Continuous-time formulation
   - https://arxiv.org/abs/2011.13456

5. **Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (2021)**
   - Architectural improvements
   - https://arxiv.org/abs/2105.05233

## Next Steps

1. Study **Conditional Diffusion** for class and text conditioning
2. Explore **UNet Architecture** for the denoising model
3. Learn **Stable Diffusion** for latent-space diffusion
4. Try **Fast Sampling Methods** for efficient generation

---

*Implementation: `Nexus/nexus/models/diffusion/base_diffusion.py`*
