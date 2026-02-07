# Diffusion Models

Comprehensive documentation for diffusion-based generative models, from foundational DDPMs to state-of-the-art flow-based and transformer architectures.

## Overview

Diffusion models are a class of generative models that learn to gradually denoise data, starting from pure Gaussian noise. They have emerged as the dominant paradigm for high-quality image, audio, and video generation, achieving state-of-the-art results across multiple domains.

## Contents

### Foundational Models

1. **[Base Diffusion](./base_diffusion.md)**
   - Denoising Diffusion Probabilistic Models (DDPM)
   - Forward and reverse diffusion processes
   - Noise schedules (linear, cosine)
   - Core training and sampling algorithms

2. **[Conditional Diffusion](./conditional_diffusion.md)**
   - Conditioning mechanisms
   - Classifier-free guidance
   - Time and condition embeddings
   - Multi-modal conditioning

3. **[Stable Diffusion](./stable_diffusion.md)**
   - Latent diffusion models
   - Text-to-image generation
   - VAE encoder/decoder
   - CLIP text encoder integration

4. **[UNet Architecture](./unet.md)**
   - U-Net backbone for diffusion
   - Skip connections and residual blocks
   - Attention mechanisms
   - Time and condition injection

### Transformer-Based Architectures

5. **[DiT - Diffusion Transformer](./dit.md)**
   - Replacing U-Net with transformers
   - Patch-based tokenization
   - AdaLN-Zero conditioning
   - Scaling properties and efficiency

6. **[MMDiT - Multimodal Diffusion Transformer](./mmdit.md)**
   - Dual-stream architecture (Stable Diffusion 3, FLUX)
   - Joint attention over image and text
   - Modality-specific parameters
   - Cross-modal information flow

7. **[PixArt-alpha](./pixart_alpha.md)**
   - Efficient high-resolution generation
   - T5 text encoder
   - Cross-attention with decomposition
   - Training efficiency improvements

### Fast Sampling Methods

8. **[Consistency Models](./consistency_models.md)** ✅
   - Single-step generation
   - Self-consistency property
   - Consistency training and distillation
   - Improved consistency training (iCT)

9. **[Latent Consistency Models (LCM)](./lcm.md)**
   - Distilling latent diffusion models
   - 2-4 step generation
   - Guidance distillation
   - Real-time generation capabilities

10. **[Flow Matching](./flow_matching.md)**
    - Continuous normalizing flows
    - Straight-line trajectories
    - Optimal transport connections
    - Simulation-free training

11. **[Rectified Flow](./rectified_flow.md)**
    - Straightening probability flows
    - Reflow procedure
    - Fast ODE sampling
    - Reduced curvature trajectories

## Key Concepts

### The Diffusion Process

**Forward Process (Noising):**

The forward process gradually adds Gaussian noise to data over T timesteps:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)

q(x_t | x_0) = N(x_t; √(ᾱ_t)x_0, (1-ᾱ_t)I)
```

Where:
- β_t is the noise schedule
- ᾱ_t = ∏_{i=1}^t (1-β_i) is the cumulative product of alphas

**Reverse Process (Denoising):**

The model learns to reverse this process:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

### Training Objective

**Noise Prediction (ε-prediction):**
```
L_simple = E_t,x_0,ε [||ε - ε_θ(√(ᾱ_t)x_0 + √(1-ᾱ_t)ε, t)||²]
```

**Velocity Prediction (v-prediction):**
```
v_t = √(ᾱ_t)ε - √(1-ᾱ_t)x_0
L_v = E_t,x_0,ε [||v_t - v_θ(x_t, t)||²]
```

**Score Matching (x_0-prediction):**
```
L_score = E_t,x_0,ε [||x_0 - x_θ(x_t, t)||²]
```

### Noise Schedules

**Linear Schedule:**
```python
β_t = β_min + (β_max - β_min) * t/T
```
- Simple but not perceptually optimal
- Used in original DDPM

**Cosine Schedule:**
```python
ᾱ_t = cos²((t/T + s)/(1 + s) * π/2)
β_t = 1 - ᾱ_t/ᾱ_{t-1}
```
- Better perceptual distribution
- More noise in middle timesteps
- Preferred in modern models

**Learned Schedules:**
- Can be learned end-to-end
- Data-dependent adaptation

### Sampling Algorithms

**DDPM (Stochastic):**
```python
x_{t-1} = 1/√(α_t) * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t z
```
- Adds noise at each step (stochastic)
- Requires many steps (1000)
- High quality but slow

**DDIM (Deterministic):**
```python
x_{t-1} = √(ᾱ_{t-1}) * x̂_0 + √(1-ᾱ_{t-1}-σ²) * ε_θ(x_t, t)
```
- Deterministic sampling (σ=0)
- Can skip timesteps
- 50-100 steps sufficient

**DPM-Solver:**
- ODE solver for diffusion
- 10-20 steps for high quality
- Fast and stable

### Classifier-Free Guidance

Core technique for controllable generation:

```python
ε̃_θ(x_t, c, t) = ε_θ(x_t, ∅, t) + w * (ε_θ(x_t, c, t) - ε_θ(x_t, ∅, t))
              = (1+w) * ε_θ(x_t, c, t) - w * ε_θ(x_t, ∅, t)
```

Where:
- c is the conditioning (text, class, etc.)
- ∅ is null conditioning
- w is the guidance scale

**Effects:**
- w = 1: No guidance (unconditional + conditional)
- w > 1: Stronger conditioning, less diversity
- w < 1: Weaker conditioning, more diversity

**Typical Values:**
- Stable Diffusion: 7.5
- DALL-E 2: 10.0
- FLUX: 3.5-4.5

## Architecture Comparison

| Model | Backbone | Params | Resolution | Speed | Key Innovation |
|-------|----------|--------|------------|-------|----------------|
| **DDPM** | U-Net | 100M | 256×256 | Slow (1000 steps) | Foundation |
| **Stable Diffusion** | U-Net | 860M | 512×512 | Medium (50 steps) | Latent space |
| **DiT** | Transformer | 675M | 256×256 | Slow | Scalability |
| **MMDiT** | Dual Transformer | 2B | 1024×1024 | Medium | Dual-stream |
| **PixArt-α** | Transformer | 600M | 1024×1024 | Fast | Efficiency |
| **Consistency** | U-Net/Transformer | 100M-2B | Various | Very Fast (1-2 steps) | Single-step |
| **LCM** | U-Net | 860M | 512×512 | Very Fast (4 steps) | Distillation |

## Implementation Patterns

### Basic Diffusion Training

```python
def train_step(model, x_0, conditioning=None):
    # Sample timestep
    t = torch.randint(0, num_timesteps, (batch_size,))

    # Sample noise
    noise = torch.randn_like(x_0)

    # Forward diffusion
    x_t = sqrt_alphas_cumprod[t] * x_0 + sqrt_one_minus_alphas_cumprod[t] * noise

    # Predict noise
    noise_pred = model(x_t, t, conditioning)

    # Compute loss
    loss = F.mse_loss(noise_pred, noise)

    return loss
```

### DDPM Sampling

```python
@torch.no_grad()
def ddpm_sample(model, shape, conditioning=None):
    # Start from noise
    x = torch.randn(shape)

    for t in reversed(range(num_timesteps)):
        # Predict noise
        noise_pred = model(x, t, conditioning)

        # Compute coefficients
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        beta_t = betas[t]

        # Denoising step
        x = (1 / sqrt(alpha_t)) * (
            x - (beta_t / sqrt(1 - alpha_bar_t)) * noise_pred
        )

        # Add noise (except last step)
        if t > 0:
            x += sqrt(beta_t) * torch.randn_like(x)

    return x
```

### DDIM Sampling

```python
@torch.no_grad()
def ddim_sample(model, shape, conditioning=None, timesteps=50):
    # Start from noise
    x = torch.randn(shape)

    # Create timestep schedule
    step_size = num_timesteps // timesteps
    schedule = list(range(0, num_timesteps, step_size))

    for i in reversed(range(len(schedule))):
        t = schedule[i]
        t_prev = schedule[i-1] if i > 0 else 0

        # Predict noise
        noise_pred = model(x, t, conditioning)

        # Predict x_0
        alpha_bar_t = alphas_cumprod[t]
        x_0_pred = (x - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)

        # Compute x_{t-1}
        if t_prev > 0:
            alpha_bar_prev = alphas_cumprod[t_prev]
            x = sqrt(alpha_bar_prev) * x_0_pred + sqrt(1 - alpha_bar_prev) * noise_pred
        else:
            x = x_0_pred

    return x
```

### Classifier-Free Guidance

```python
@torch.no_grad()
def cfg_sample(model, shape, conditioning, guidance_scale=7.5):
    x = torch.randn(shape)

    for t in reversed(range(num_timesteps)):
        # Unconditional prediction
        noise_uncond = model(x, t, None)

        # Conditional prediction
        noise_cond = model(x, t, conditioning)

        # Apply guidance
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # Denoise step
        x = denoise_step(x, noise_pred, t)

    return x
```

## Training Strategies

### Progressive Training

1. **Low resolution first** (64×64 or 128×128)
2. **Increase resolution gradually** (256×256, 512×512)
3. **Fine-tune on high resolution** (1024×1024+)

Benefits:
- Faster initial training
- Better feature learning
- More stable training

### Multi-Aspect Training

```python
aspect_ratios = {
    (1024, 1024): 1.0,  # Square
    (1152, 896): 0.8,   # Landscape
    (896, 1152): 0.8,   # Portrait
}

# Bucket training
def get_bucket(height, width):
    return min(aspect_ratios.keys(),
               key=lambda k: abs(k[0]/k[1] - height/width))
```

### Timestep Sampling Strategies

**Uniform Sampling:**
```python
t = torch.randint(0, num_timesteps, (batch_size,))
```

**Importance Sampling:**
```python
# Sample more from difficult timesteps
weights = compute_loss_weights()
t = torch.multinomial(weights, batch_size)
```

**Loss-Aware Sampling:**
```python
# Track per-timestep losses
timestep_losses = running_average_of_losses
p = timestep_losses / timestep_losses.sum()
t = torch.multinomial(p, batch_size)
```

## Optimization Tricks

### 1. EMA (Exponential Moving Average)

```python
ema_model = copy.deepcopy(model)
ema_decay = 0.9999

def update_ema():
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
```

Benefits:
- More stable generations
- Better sample quality
- Essential for good results

### 2. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Benefits:
- 2x faster training
- 2x less memory
- Minimal quality loss with bfloat16

### 3. Gradient Accumulation

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4. Min-SNR Weighting

```python
def min_snr_gamma_weight(t, gamma=5):
    snr = alphas_cumprod[t] / (1 - alphas_cumprod[t])
    weight = torch.minimum(snr, torch.ones_like(snr) * gamma) / snr
    return weight

loss = weight * F.mse_loss(noise_pred, noise, reduction='none')
```

Benefits:
- Balances loss across timesteps
- Better convergence
- Improved sample quality

## Common Pitfalls

### 1. Poor Noise Schedule

**Problem:** Linear schedule not optimal for all data types

**Solution:**
- Use cosine schedule for images
- Experiment with custom schedules
- Consider learned schedules

### 2. Insufficient Guidance

**Problem:** Generated samples don't follow conditioning

**Solution:**
- Increase guidance scale
- Ensure proper null conditioning training
- Check conditioning embedding quality

### 3. Memory Issues

**Problem:** OOM during training or sampling

**Solutions:**
- Use gradient checkpointing
- Reduce batch size
- Use mixed precision (fp16/bf16)
- Operate in latent space

### 4. Slow Sampling

**Problem:** 1000 steps too slow for inference

**Solutions:**
- Use DDIM (50-100 steps)
- Use DPM-Solver (10-20 steps)
- Distill to consistency models (1-4 steps)
- Use turbo/LCM variants

### 5. Training Instability

**Problem:** Loss spikes or NaN values

**Solutions:**
- Use EMA for stable weights
- Clip gradients (max_norm=1.0)
- Reduce learning rate
- Check for NaN in inputs
- Use bfloat16 instead of float16

## Evaluation

### Quantitative Metrics

**FID (Fréchet Inception Distance):**
```python
# Lower is better
# Measures distribution similarity
fid = compute_fid(real_images, generated_images)
```

**CLIP Score:**
```python
# Higher is better
# Measures text-image alignment
clip_score = compute_clip_score(images, prompts)
```

**Inception Score (IS):**
```python
# Higher is better
# Measures quality and diversity
is_score = compute_inception_score(generated_images)
```

### Qualitative Evaluation

1. **Visual inspection** - Overall quality
2. **Prompt adherence** - Following instructions
3. **Diversity** - Variety in generations
4. **Artifacts** - Unusual patterns or errors
5. **Consistency** - Coherence across samples

## Advanced Topics

### Latent Diffusion

Operating in compressed latent space:

```python
# Encode to latent
latent = vae.encode(image)

# Diffusion in latent space
latent_denoised = diffusion_process(latent)

# Decode to pixel space
image = vae.decode(latent_denoised)
```

Benefits:
- 4-16x memory reduction
- Faster training and sampling
- Better scaling

### ControlNet

Spatial conditioning for diffusion:

```python
# Add spatial conditioning
control = controlnet(conditioning_image, timestep)

# Inject into U-Net
output = unet(noisy_image, timestep, text_embedding, control)
```

Applications:
- Pose-guided generation
- Depth-conditioned synthesis
- Edge-guided image creation

### LoRA Fine-tuning

Efficient adaptation:

```python
# Add low-rank adapters
lora = LoRALinear(in_features, out_features, rank=4)

# Fine-tune only LoRA params
optimizer = Adam(lora.parameters(), lr=1e-4)
```

Benefits:
- 100x fewer parameters
- Fast fine-tuning
- Easy model switching

## References

### Core Papers

1. **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
2. **DDIM**: Song et al., "Denoising Diffusion Implicit Models" (2021)
3. **Score-based**: Song et al., "Score-Based Generative Modeling through SDEs" (2021)
4. **Classifier-Free Guidance**: Ho & Salimans, "Classifier-Free Diffusion Guidance" (2022)
5. **Latent Diffusion**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)

### Architecture Papers

6. **DiT**: Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023)
7. **MMDiT/SD3**: Esser et al., "Scaling Rectified Flow Transformers" (2024)
8. **PixArt-α**: Chen et al., "PixArt-α: Fast Training of Diffusion Transformer" (2023)

### Fast Sampling Papers

9. **Consistency Models**: Song et al., "Consistency Models" (2023)
10. **LCM**: Luo et al., "Latent Consistency Models" (2023)
11. **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling" (2023)
12. **Rectified Flow**: Liu et al., "Flow Straight and Fast" (2023)

---

*Each model has detailed documentation covering theory, implementation, and practical considerations.*
