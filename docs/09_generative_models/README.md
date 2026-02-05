# Generative Models

Comprehensive documentation for generative modeling approaches, from classical methods to state-of-the-art diffusion and flow-based models.

## Overview

Generative models learn to model the data distribution p(x) and can generate new samples from that distribution. This collection covers the major paradigms in generative modeling, each with different trade-offs in terms of quality, diversity, training stability, and computational efficiency.

## Categories

### 1. [Diffusion Models](./diffusion/)

Modern diffusion and flow-based models that achieve state-of-the-art generation quality.

**Foundations:**
- [Base Diffusion](./diffusion/base_diffusion.md) - Core DDPM formulation and noise scheduling
- [Conditional Diffusion](./diffusion/conditional_diffusion.md) - Conditioning mechanisms and classifier-free guidance
- [Stable Diffusion](./diffusion/stable_diffusion.md) - Latent diffusion with text-to-image generation
- [UNet Architecture](./diffusion/unet.md) - U-Net backbone for diffusion models

**Transformer-Based Architectures:**
- [DiT (Diffusion Transformer)](./diffusion/dit.md) - Scalable diffusion with transformers
- [MMDiT (Multimodal Diffusion Transformer)](./diffusion/mmdit.md) - Dual-stream architecture for SD3/FLUX
- [PixArt-alpha](./diffusion/pixart_alpha.md) - Efficient high-resolution text-to-image

**Fast Sampling:**
- [Consistency Models](./diffusion/consistency_models.md) - Single-step generation via consistency training
- [Latent Consistency Models (LCM)](./diffusion/lcm.md) - Distilled consistency models for latent diffusion
- [Flow Matching](./diffusion/flow_matching.md) - Continuous normalizing flows for generation
- [Rectified Flow](./diffusion/rectified_flow.md) - Straightened probability flows

### 2. [Audio & Video Generation](./audio_video/)

Temporal generative models for audio and video synthesis.

**Video Generation:**
- [CogVideoX](./audio_video/cogvideox.md) - Expert transformer for text-to-video
- [VideoPoet](./audio_video/videopoet.md) - Large language model for video generation

**Audio & Speech:**
- [VALL-E](./audio_video/valle.md) - Neural codec language model for TTS
- [Voicebox](./audio_video/voicebox.md) - Non-autoregressive speech generation
- [SoundStorm](./audio_video/soundstorm.md) - Parallel audio generation with confidence-based decoding
- [MusicGen](./audio_video/musicgen.md) - Text-to-music generation
- [NaturalSpeech 3](./audio_video/naturalspeech3.md) - Factorized diffusion for speech synthesis

### 3. [GANs (Generative Adversarial Networks)](./gans.md)

Classical adversarial training approaches.

- **Base GAN** - Original adversarial training framework
- **Conditional GAN** - Class-conditional image generation
- **CycleGAN** - Unpaired image-to-image translation
- **Wasserstein GAN (WGAN)** - Improved training stability with Wasserstein distance

### 4. [VAE (Variational Autoencoders)](./vae.md)

Latent variable models with explicit probabilistic formulation.

- **Standard VAE** - ELBO optimization and reparameterization trick
- **Beta-VAE** - Disentangled representations
- **Architectural Variants** - MLP and convolutional architectures

## Key Concepts

### Generative Modeling Paradigms

| Paradigm | Training | Sampling | Quality | Speed | Controllability |
|----------|----------|----------|---------|-------|-----------------|
| **GANs** | Adversarial (unstable) | Fast (1 step) | High | Fast | Medium |
| **VAEs** | ELBO maximization | Fast (1 step) | Medium | Fast | High |
| **Diffusion** | MSE on noise | Slow (50-1000 steps) | Very High | Slow | Very High |
| **Flow** | Likelihood-based | Fast-Medium | High | Medium | High |
| **Consistency** | Consistency distillation | Very Fast (1-2 steps) | High | Very Fast | High |

### Diffusion Models: Core Principles

**Forward Process (Noise Addition):**
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
```

**Reverse Process (Denoising):**
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

**Training Objective:**
```
L = E_t,x_0,ε [||ε - ε_θ(x_t, t)||²]
```

Where the model predicts the noise ε added at timestep t.

### Conditioning Mechanisms

**Classifier-Free Guidance (CFG):**
```
ε̃_θ(x_t, c, t) = ε_θ(x_t, ∅, t) + w · (ε_θ(x_t, c, t) - ε_θ(x_t, ∅, t))
```
- Trades diversity for fidelity via guidance scale w
- w = 1: no guidance, w > 1: stronger conditioning
- Typical values: 7.5 for Stable Diffusion, 4.5 for FLUX

**Cross-Attention Conditioning:**
```
Attention(Q, K, V) where:
Q = queries from image tokens
K, V = keys/values from text embeddings
```

### Latent Diffusion

Operating in compressed latent space rather than pixel space:

**Advantages:**
- 4-16x memory reduction
- Faster training and sampling
- Better scaling to high resolutions

**Pipeline:**
```
Text → Text Encoder → Conditioning
Image → VAE Encoder → Latent → Diffusion → VAE Decoder → Image
```

## Implementation Patterns

### Basic Training Loop

```python
# Sample noise and timestep
noise = torch.randn_like(x_0)
t = torch.randint(0, num_timesteps, (batch_size,))

# Forward diffusion (add noise)
x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

# Predict noise
noise_pred = model(x_t, t, conditioning)

# Compute loss
loss = F.mse_loss(noise_pred, noise)
```

### Sampling (DDPM)

```python
x = torch.randn(shape)  # Start from pure noise

for t in reversed(range(num_timesteps)):
    # Predict noise
    noise_pred = model(x, t, conditioning)

    # Compute denoising step
    alpha_t = alphas[t]
    alpha_bar_t = alphas_cumprod[t]

    # Update x
    x = (1 / sqrt(alpha_t)) * (x - ((1 - alpha_t) / sqrt(1 - alpha_bar_t)) * noise_pred)

    # Add noise (except last step)
    if t > 0:
        x += sqrt(betas[t]) * torch.randn_like(x)
```

### Classifier-Free Guidance Sampling

```python
for t in reversed(range(num_timesteps)):
    # Unconditional prediction
    noise_uncond = model(x, t, null_conditioning)

    # Conditional prediction
    noise_cond = model(x, t, conditioning)

    # Apply guidance
    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    # Denoise step
    x = denoise_step(x, noise_pred, t)
```

## Training Considerations

### Noise Schedules

**Linear Schedule:**
```python
betas = torch.linspace(beta_start, beta_end, num_timesteps)
```
- Simple but may not be perceptually optimal
- beta_start ≈ 0.0001, beta_end ≈ 0.02

**Cosine Schedule:**
```python
s = 0.008
t = torch.arange(num_timesteps + 1) / num_timesteps
alpha_bar = torch.cos((t + s) / (1 + s) * π/2) ** 2
betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
```
- Better perceptual distribution of noise
- Preferred for most modern models

### Data Augmentation

- **Random horizontal flips** for symmetric data
- **Center crops** to fixed resolution
- **Normalize to [-1, 1]** for better training stability
- **Bucketized resolutions** for variable-aspect-ratio training

### Loss Weighting

**Standard MSE:**
```python
loss = ||ε - ε_θ(x_t, t)||²
```

**SNR Weighting (Min-SNR-γ):**
```python
weight = min(SNR(t), γ) / SNR(t)
loss = weight * ||ε - ε_θ(x_t, t)||²
```
- Balances loss across timesteps
- γ = 5 is a common choice

## Evaluation Metrics

### Image Quality

- **FID (Fréchet Inception Distance)** - Measures distribution similarity
- **IS (Inception Score)** - Measures quality and diversity
- **CLIP Score** - Measures text-image alignment
- **Aesthetic Score** - Learned aesthetic quality predictor

### Sample Diversity

- **Intra-class diversity** - Variation within same conditioning
- **Multi-scale structural similarity (MS-SSIM)** - Measures diversity at multiple scales

### Human Evaluation

- **Preference studies** - A/B testing between models
- **Prompt following** - How well the model follows instructions
- **Aesthetic quality** - Overall visual appeal

## Common Pitfalls

### Training Issues

1. **Mode collapse** (GANs)
   - Use spectral normalization
   - Try Wasserstein loss
   - Increase discriminator capacity

2. **Posterior collapse** (VAEs)
   - Use beta-VAE with β < 1 initially
   - Warm up KL weight gradually
   - Use free bits constraint

3. **Slow convergence** (Diffusion)
   - Use proper noise schedule (cosine often better than linear)
   - Ensure proper normalization
   - Use adequate model capacity

### Sampling Issues

1. **Poor sample quality**
   - Try different guidance scales
   - Use more sampling steps
   - Check conditioning strength

2. **Lack of diversity**
   - Reduce guidance scale
   - Sample from earlier timesteps
   - Use stochastic samplers (DDPM vs DDIM)

3. **Memory issues**
   - Use gradient checkpointing
   - Reduce batch size
   - Use mixed precision training (fp16/bf16)

## Advanced Techniques

### Fast Sampling Methods

**DDIM (Denoising Diffusion Implicit Models):**
- Deterministic sampling
- Can skip timesteps (50 steps → 10 steps)
- Trade-off: slightly lower quality

**DPM-Solver:**
- ODE solver for diffusion ODEs
- 10-20 steps for good quality
- Faster convergence than DDPM/DDIM

**LCM/Turbo:**
- Distilled models for 1-4 step generation
- Maintains high quality
- Requires distillation training

### Conditioning Techniques

**Multi-modal Conditioning:**
- Text + image (inpainting, editing)
- Text + depth/pose (ControlNet)
- Text + style (IP-Adapter, LoRA)

**Fine-grained Control:**
- **LoRA** - Low-rank adaptation for efficient fine-tuning
- **ControlNet** - Spatial conditioning signals
- **IP-Adapter** - Image prompt conditioning

## Code Structure

```
nexus/models/
├── diffusion/          # Diffusion model implementations
│   ├── base_diffusion.py
│   ├── conditional_diffusion.py
│   ├── stable_diffusion.py
│   ├── unet.py
│   ├── dit.py
│   ├── mmdit.py
│   ├── consistency_model.py
│   ├── flow_matching.py
│   ├── rectified_flow.py
│   └── pixart_alpha.py
├── video/              # Video generation models
│   ├── cogvideox.py
│   └── videopoet.py
├── audio/              # Audio generation models
│   ├── valle.py
│   ├── voicebox.py
│   ├── soundstorm.py
│   ├── musicgen.py
│   └── naturalspeech3.py
├── gan/                # GAN implementations
│   ├── base_gan.py
│   ├── conditional_gan.py
│   ├── cycle_gan.py
│   └── wgan.py
└── cv/vae/             # VAE implementations
    └── vae.py
```

## References

### Foundational Papers

**Diffusion Models:**
- Sohl-Dickstein et al., "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (2015)
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" (2021)

**Latent Diffusion:**
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)

**Fast Sampling:**
- Song et al., "Consistency Models" (2023)
- Liu et al., "Flow Matching for Generative Modeling" (2023)
- Lipman et al., "Flow Matching for Generative Modeling" (2023)

**Architecture:**
- Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023)
- Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (2024)

**GANs:**
- Goodfellow et al., "Generative Adversarial Networks" (2014)
- Arjovsky et al., "Wasserstein GAN" (2017)

**VAEs:**
- Kingma & Welling, "Auto-Encoding Variational Bayes" (2014)
- Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (2017)

## Getting Started

1. **Start with base models** - Understand DDPM before moving to advanced variants
2. **Experiment with schedules** - Noise schedules significantly impact quality
3. **Master conditioning** - Classifier-free guidance is essential for controllability
4. **Optimize sampling** - Use fast samplers (DDIM, DPM-Solver) for inference
5. **Fine-tune wisely** - LoRA and similar methods for efficient adaptation

Each model documentation includes:
- Theoretical foundations
- Implementation walkthrough
- Training and optimization tips
- Common pitfalls and solutions
- Experimental results and ablations

---

*For detailed implementation examples, see the respective model documentation files.*
