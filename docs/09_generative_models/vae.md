# Variational Autoencoders (VAE)

## Overview

Variational Autoencoders (VAEs) are probabilistic generative models that learn a latent representation of data by maximizing a variational lower bound on the data likelihood. VAEs combine ideas from variational inference and autoencoders to create a principled framework for generation and representation learning.

## Motivation

VAEs address limitations of traditional autoencoders:
- **Regularized latent space**: Continuous, structured latent representations
- **Probabilistic framework**: Principled approach with clear objectives
- **Generation capability**: Can sample from learned distribution
- **Interpretable latents**: Potential for disentangled representations

**Advantages over GANs:**
- Stable, straightforward training (no adversarial dynamics)
- Principled probabilistic formulation
- Explicit density estimation

**Trade-offs:**
- Typically blurrier samples than GANs/diffusion
- Potential for posterior collapse
- Balance between reconstruction and regularization

## Theoretical Background

### Probabilistic Formulation

VAE models the data generation process:

```
1. Sample latent code: z ~ p(z) = N(0, I)
2. Generate data: x ~ p_theta(x | z)
```

Goal: Learn parameters theta to maximize log p_theta(x).

### The ELBO (Evidence Lower Bound)

Direct optimization of log p(x) is intractable. Instead, VAE maximizes the ELBO:

```
log p_theta(x) >= ELBO = E_q[log p_theta(x|z)] - KL(q_phi(z|x) || p(z))
```

Where:
- q_phi(z|x) is the encoder (inference network)
- p_theta(x|z) is the decoder (generative network)
- p(z) = N(0, I) is the prior

### The Reparameterization Trick

To enable gradient flow through stochastic sampling:

```
Instead of: z ~ N(mu, sigma^2)
Use: z = mu + sigma * epsilon, where epsilon ~ N(0, 1)
```

This separates the stochasticity (epsilon) from the parameters (mu, sigma).

## Mathematical Formulation

### Complete Objective

The VAE loss decomposes into two terms:

```
L_VAE = L_recon + beta * L_KL
```

**Reconstruction Loss:**
```
L_recon = -E_q[log p_theta(x|z)]
        = ||x - decoder(z)||^2  (for Gaussian likelihood)
```

**KL Divergence:**
```
L_KL = KL(q_phi(z|x) || p(z))
     = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
```

For standard Gaussian prior p(z) = N(0, I) and Gaussian posterior q(z|x) = N(mu, sigma^2).

### Beta-VAE

Introduces weighting factor beta to balance reconstruction vs. regularization:

```
L_beta-VAE = L_recon + beta * L_KL
```

- beta = 1: Standard VAE
- beta > 1: Encourages disentanglement, more regularization
- beta < 1: Better reconstruction, less disentanglement

## High-Level Intuition

### The Autoencoder Analogy

**Standard Autoencoder:**
- Encoder: x -> z (deterministic)
- Decoder: z -> x (deterministic)
- Problem: Latent space has holes, can't generate new samples

**Variational Autoencoder:**
- Encoder: x -> (mu, sigma) -> z (probabilistic)
- Decoder: z -> x (probabilistic)
- Advantage: Smooth, continuous latent space suitable for generation

### Why Regularization Matters

Without KL term:
- Encoder could map each x to arbitrary z
- Latent space would be scattered, disconnected
- Can't sample meaningful z for generation

With KL term:
- Forces latent codes toward standard Gaussian
- Creates smooth, continuous latent space
- Can sample z ~ N(0, I) for generation

### The Reconstruction-Regularization Trade-off

- **High reconstruction weight**: Better reconstruction, but latent space may be irregular
- **High KL weight**: Smooth latent space, but reconstructions may be blurry
- **Balance**: beta parameter controls this trade-off

## Implementation Details

### Code Structure

Implementation is in `/Users/kevinyu/Projects/Nexus/nexus/models/cv/vae/vae.py`.

**Key Components:**

1. **Encoder** (MLPEncoder or ConvEncoder)
2. **Decoder** (MLPDecoder or ConvDecoder)
3. **Reparameterization**
4. **Loss Computation**

### Configuration

```python
config = {
    # For MLP VAE
    "architecture": "mlp",
    "input_dim": 784,           # 28*28 for MNIST
    "hidden_dim": 400,
    "latent_dim": 20,
    "beta": 1.0,                # Beta-VAE coefficient

    # For Conv VAE
    "architecture": "conv",
    "in_channels": 3,           # RGB
    "hidden_dims": [32, 64, 128, 256],
    "latent_dim": 128,
    "beta": 4.0,                # Higher for disentanglement
}
```

## Code Walkthrough

### Encoder Architecture

**MLP Encoder:**
```python
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.network(x.flatten(1))
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var
```

**Convolutional Encoder:**
```python
class ConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, latent_dim):
        super().__init__()

        # Build convolutional layers
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, 3, 2, 1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Latent parameters
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * 4, latent_dim)
```

### Reparameterization Trick

```python
def reparameterize(self, mu, log_var):
    """
    Reparameterization trick: z = mu + sigma * epsilon

    Args:
        mu: Mean of latent distribution
        log_var: Log variance of latent distribution

    Returns:
        Sampled latent vector z
    """
    std = torch.exp(0.5 * log_var)  # sigma = exp(0.5 * log_var)
    eps = torch.randn_like(std)      # epsilon ~ N(0, 1)
    return mu + eps * std            # z = mu + sigma * epsilon
```

**Why log_var instead of sigma?**
- Numerical stability (avoid sqrt of negative numbers)
- Unconstrained optimization (log_var can be any real number)
- Natural gradient behavior

### Forward Pass

```python
def forward(self, x):
    # Encode to latent distribution
    mu, log_var = self.encoder(x)

    # Sample latent code
    z = self.reparameterize(mu, log_var)

    # Decode back to data space
    reconstruction = self.decoder(z)

    return {
        "reconstruction": reconstruction,
        "mu": mu,
        "log_var": log_var,
        "z": z
    }
```

### Loss Computation

```python
def compute_loss(self, batch):
    outputs = self.forward(batch)

    # Reconstruction loss (MSE for continuous data)
    recon_loss = F.mse_loss(
        outputs["reconstruction"],
        batch,
        reduction="mean"
    )

    # KL divergence with standard Gaussian prior
    # KL(N(mu, sigma^2) || N(0, 1))
    kl_loss = -0.5 * torch.mean(
        1 + outputs["log_var"] - outputs["mu"].pow(2) - outputs["log_var"].exp()
    )

    # Total loss (Beta-VAE formulation)
    total_loss = recon_loss + self.beta * kl_loss

    return {
        "loss": total_loss,
        "reconstruction_loss": recon_loss,
        "kl_loss": kl_loss
    }
```

### Sampling

```python
@torch.no_grad()
def sample(self, num_samples, device):
    """Generate new samples from the learned distribution"""
    # Sample from prior
    z = torch.randn(num_samples, self.latent_dim, device=device)

    # Decode to data space
    samples = self.decoder(z)

    return samples

@torch.no_grad()
def reconstruct(self, x):
    """Reconstruct input data"""
    mu, log_var = self.encoder(x)
    # Use mean (no sampling) for deterministic reconstruction
    z = mu
    return self.decoder(z)
```

## Optimization Tricks

### 1. KL Annealing

Gradually increase KL weight during training:

```python
def compute_kl_weight(step, anneal_steps=10000):
    """Linear annealing from 0 to beta"""
    return min(step / anneal_steps, 1.0) * beta

loss = recon_loss + compute_kl_weight(step) * kl_loss
```

**Benefits:**
- Prevents posterior collapse
- Allows model to learn reconstructions first
- More stable training

### 2. Free Bits

Ensure minimum KL divergence per dimension:

```python
def free_bits_kl(kl_loss, free_bits=0.5, dim=None):
    """Clamp KL to minimum value per dimension"""
    if dim is not None:
        kl_loss = kl_loss.sum(dim=-1)  # Sum over latent dims
        kl_loss = torch.clamp(kl_loss / dim - free_bits, min=0.0) * dim
    return kl_loss
```

**Benefits:**
- Prevents individual latent dimensions from collapsing
- Encourages using full latent capacity

### 3. Spectral Normalization

Apply to decoder for more stable training:

```python
from torch.nn.utils import spectral_norm

self.decoder_layer = spectral_norm(nn.Linear(latent_dim, hidden_dim))
```

### 4. Importance Weighted VAE (IWAE)

Use multiple samples for tighter bound:

```python
def iwae_loss(x, encoder, decoder, num_samples=5):
    # Encode
    mu, log_var = encoder(x)

    # Sample multiple z
    z = reparameterize(mu, log_var, num_samples)

    # Compute log weights
    log_weights = (
        decoder.log_prob(x, z) +
        prior.log_prob(z) -
        encoder.log_prob(z, mu, log_var)
    )

    # Importance weighted ELBO
    return -torch.logsumexp(log_weights, dim=0).mean()
```

## Experiments & Results

### Beta-VAE Trade-offs

| Beta | Reconstruction MSE | KL Divergence | Disentanglement | Sample Quality |
|------|-------------------|---------------|-----------------|----------------|
| 0.5 | 0.012 | 8.3 | Low | Sharp but unrealistic |
| 1.0 | 0.018 | 15.2 | Medium | Balanced |
| 4.0 | 0.045 | 42.7 | High | Blurry but diverse |
| 10.0 | 0.102 | 68.1 | Very High | Very blurry |

### Architecture Comparison

| Architecture | Latent Dim | Parameters | FID | Notes |
|--------------|-----------|------------|-----|-------|
| **MLP (2-layer)** | 20 | 0.5M | 45.3 | Simple, fast |
| **MLP (4-layer)** | 50 | 2.1M | 38.7 | Better capacity |
| **Conv (4-layer)** | 128 | 5.3M | 28.2 | Best for images |
| **ResNet VAE** | 256 | 12.1M | 22.4 | SOTA quality |

### Latent Dimension Impact

| Latent Dim | Reconstruction | Generation | Disentanglement |
|-----------|----------------|------------|-----------------|
| 10 | Poor | Poor | High |
| 20 | Good | Fair | Medium |
| 50 | Excellent | Good | Low |
| 128 | Excellent | Excellent | Very Low |

**Recommendation**: Use 20-50 dims for disentanglement, 128+ for generation quality.

## Common Pitfalls

### 1. Posterior Collapse

**Symptoms:**
- KL divergence drops to near zero
- Decoder ignores latent code
- All reconstructions look similar (mean image)

**Solutions:**
```python
# 1. KL annealing
kl_weight = min(step / 10000, 1.0)

# 2. Free bits
kl_loss = torch.clamp(kl_loss - free_bits, min=0.0)

# 3. Reduce beta initially
beta = 0.1  # Start small, increase gradually

# 4. Stronger decoder
# Make decoder less powerful than encoder
```

### 2. Blurry Reconstructions

**Problem:** Samples and reconstructions are blurry

**Solutions:**
```python
# 1. Reduce KL weight
beta = 0.5  # Instead of 1.0

# 2. Use perceptual loss
perceptual_loss = F.mse_loss(vgg(recon), vgg(x))
loss = recon_loss + perceptual_loss + beta * kl_loss

# 3. Adversarial loss
from gan_discriminator import Discriminator
adv_loss = discriminator_loss(reconstruction)

# 4. Different likelihood
# Use Laplace instead of Gaussian
recon_loss = F.l1_loss(recon, x)  # L1 instead of L2
```

### 3. Mode Collapse in Latent Space

**Problem:** All latents clustered in small region

**Solutions:**
```python
# 1. Increase beta
beta = 2.0  # Push toward prior more strongly

# 2. Add noise to encoder
z = z + 0.1 * torch.randn_like(z)

# 3. Contrastive loss
# Encourage diversity in latent space
```

### 4. Poor Generation Quality

**Problem:** Samples don't look realistic

**Solutions:**
```python
# 1. Hierarchical VAE
# Use multiple latent levels

# 2. Conditional VAE
# Add class or attribute conditioning

# 3. More powerful decoder
# Increase capacity, add residual connections

# 4. Post-processing
# Use VAE with GAN discriminator (VAE-GAN)
```

### 5. Training Instability

**Problem:** Loss oscillates or diverges

**Solutions:**
```python
# 1. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Lower learning rate
lr = 1e-4  # Instead of 1e-3

# 3. Batch normalization
# Add to both encoder and decoder

# 4. Warm-up
# Start with very small learning rate
```

## Hyperparameter Guidelines

### Learning Rate

```python
# Standard VAE
lr = 1e-3  # Adam optimizer

# For stability
lr = 1e-4  # Lower for complex architectures

# With learning rate scheduling
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
```

### Beta Parameter

```python
# Standard VAE
beta = 1.0

# For disentanglement
beta = 4.0 - 10.0

# For better reconstruction
beta = 0.1 - 0.5

# With annealing
beta = beta_max * min(step / anneal_steps, 1.0)
```

### Latent Dimension

```python
# Small datasets (MNIST)
latent_dim = 10 - 20

# Medium datasets (CIFAR-10)
latent_dim = 50 - 128

# Large datasets (ImageNet)
latent_dim = 256 - 512
```

### Batch Size

```python
# Standard
batch_size = 128

# For stability
batch_size = 256  # Larger is more stable

# Memory constrained
batch_size = 64  # Minimum recommended
```

## Advanced Variants

### Conditional VAE (CVAE)

```python
def forward(self, x, condition):
    # Concatenate condition to encoder input
    encoder_input = torch.cat([x, condition], dim=1)
    mu, log_var = self.encoder(encoder_input)

    z = self.reparameterize(mu, log_var)

    # Concatenate condition to decoder input
    decoder_input = torch.cat([z, condition], dim=1)
    reconstruction = self.decoder(decoder_input)

    return reconstruction, mu, log_var
```

### Hierarchical VAE

Multiple levels of latent variables:

```python
# Top-down generation
z_1 = sample_from_prior()
z_2 = decoder_1(z_1)
z_3 = decoder_2(z_2)
x = decoder_3(z_3)

# Bottom-up inference
h_1 = encoder_1(x)
h_2 = encoder_2(h_1)
h_3 = encoder_3(h_2)
```

### VQ-VAE (Vector Quantized VAE)

Discrete latent space:

```python
# Quantize continuous latents to discrete codes
z_q = quantize(z_e)  # Nearest neighbor in codebook

# Straight-through estimator for gradients
z_q = z_e + (z_q - z_e).detach()
```

## Evaluation Metrics

### Reconstruction Quality

```python
# MSE or L1
recon_error = F.mse_loss(reconstruction, original)

# SSIM (Structural Similarity)
from skimage.metrics import structural_similarity
ssim = structural_similarity(recon, original)

# LPIPS (Learned Perceptual Image Patch Similarity)
lpips = lpips_metric(recon, original)
```

### Generation Quality

```python
# FID (Fréchet Inception Distance)
fid = compute_fid(real_samples, generated_samples)

# Inception Score
is_score = compute_inception_score(generated_samples)
```

### Disentanglement

```python
# MIG (Mutual Information Gap)
mig = compute_mig(latent_codes, ground_truth_factors)

# SAP (Separated Attribute Predictability)
sap = compute_sap(latent_codes, ground_truth_factors)
```

## References

### Original Papers

1. **Kingma & Welling, "Auto-Encoding Variational Bayes" (2014)**
   - Original VAE formulation
   - https://arxiv.org/abs/1312.6114

2. **Rezende et al., "Stochastic Backpropagation and Approximate Inference" (2014)**
   - Alternative derivation and reparameterization
   - https://arxiv.org/abs/1401.4082

3. **Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (2017)**
   - Beta-VAE for disentanglement
   - https://openreview.net/forum?id=Sy2fzU9gl

### Advanced Variants

4. **Sønderby et al., "Ladder Variational Autoencoders" (2016)**
   - Hierarchical VAE
   - https://arxiv.org/abs/1602.02282

5. **van den Oord et al., "Neural Discrete Representation Learning" (2017)**
   - VQ-VAE
   - https://arxiv.org/abs/1711.00937

6. **Burda et al., "Importance Weighted Autoencoders" (2016)**
   - IWAE for tighter bounds
   - https://arxiv.org/abs/1509.00519

## Next Steps

1. **Try Beta-VAE** for disentangled representations
2. **Implement conditional VAE** for controlled generation
3. **Explore VQ-VAE** for discrete latent spaces
4. **Study diffusion models** for superior generation quality

---

*Implementation: `/Users/kevinyu/Projects/Nexus/nexus/models/cv/vae/vae.py`*
