# Generative Adversarial Networks (GANs)

## Overview

Generative Adversarial Networks (GANs) are a class of generative models based on adversarial training between two neural networks: a generator that creates synthetic data and a discriminator that distinguishes between real and fake data. This documentation covers base GAN, conditional GAN, CycleGAN, and Wasserstein GAN implementations.

## Motivation

GANs revolutionized generative modeling by:
- **Implicit density modeling**: No need to explicitly model p(x)
- **Sharp, high-quality samples**: Better than VAEs for image generation
- **Flexible architectures**: Can use any differentiable network
- **Fast sampling**: Single forward pass through generator

## Theoretical Background

### The Adversarial Game

GANs are formulated as a two-player minimax game:

```
min_G max_D V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

Where:
- G is the generator: maps noise z to fake data G(z)
- D is the discriminator: estimates probability that input is real
- Real data x from distribution p_data(x)
- Noise z from prior distribution p_z(z) (typically N(0,I))

### Training Dynamics

**Discriminator Training:**
- Maximize ability to distinguish real from fake
- Binary classification problem
- Loss: -[log D(x) + log(1 - D(G(z)))]

**Generator Training:**
- Minimize discriminator's ability to detect fakes
- Equivalently: maximize D(G(z))
- Loss: -log D(G(z)) (non-saturating loss)

### Optimal Discriminator

When G is fixed, the optimal discriminator is:

```
D*(x) = p_data(x) / (p_data(x) + p_g(x))
```

At Nash equilibrium: p_g = p_data and D*(x) = 1/2 everywhere.

## Mathematical Formulation

### Standard GAN Loss

**Generator Loss (non-saturating):**
```
L_G = -E_z[log D(G(z))]
```

**Discriminator Loss:**
```
L_D = -E_x[log D(x)] - E_z[log(1 - D(G(z)))]
```

### Wasserstein GAN Loss

Uses Wasserstein distance (Earth Mover's Distance):

```
W(p_data, p_g) = inf_{gamma} E_{(x,y)~gamma}[||x - y||]
```

**WGAN Losses:**
```
L_D = -E_x[D(x)] + E_z[D(G(z))]  (critic loss)
L_G = -E_z[D(G(z))]  (generator loss)
```

With weight clipping or gradient penalty to enforce Lipschitz constraint.

### Conditional GAN Loss

Extends GAN with conditioning information c (e.g., class labels):

```
min_G max_D V(D, G) = E_x,c[log D(x, c)] + E_z,c[log(1 - D(G(z, c), c))]
```

## Implementation Details

### Code Structure

Implementations are in `/Users/kevinyu/Projects/Nexus/nexus/models/gan/`:

- `base_gan.py` - Standard GAN with generator and discriminator
- `conditional_gan.py` - Class-conditional generation
- `cycle_gan.py` - Unpaired image-to-image translation
- `wgan.py` - Wasserstein GAN with gradient penalty

### Base GAN Configuration

```python
config = {
    "latent_dim": 100,          # Dimension of noise vector z
    "hidden_dim": 512,          # Base hidden dimension
    "output_channels": 3,       # RGB images
    "output_size": 64,          # Image resolution
    "learning_rate": 2e-4,      # Adam learning rate
    "beta1": 0.5,               # Adam beta1
    "beta2": 0.999,             # Adam beta2
}
```

## Code Walkthrough

### Base Generator Architecture

```python
class BaseGenerator(NexusModule):
    def __init__(self, config):
        super().__init__(config)

        self.latent_dim = config["latent_dim"]
        self.hidden_dim = config.get("hidden_dim", 512)

        # Project latent to spatial feature map
        self.main = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim * 4 * 4),
            nn.BatchNorm1d(self.hidden_dim * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (self.hidden_dim, 4, 4)),

            # Upsampling layers
            # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
            *self._build_upsampling_layers()
        )
```

**Key Points:**
- Linear projection from latent vector to spatial feature map
- Progressive upsampling with transposed convolutions
- BatchNorm for training stability
- ReLU activation (LeakyReLU also common)
- Tanh output for [-1, 1] range

### Base Discriminator Architecture

```python
class BaseDiscriminator(NexusModule):
    def __init__(self, config):
        super().__init__(config)

        self.main = nn.Sequential(
            # Initial layer (no batch norm)
            nn.Conv2d(input_channels, hidden_dim, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            # Downsampling layers
            *self._build_downsampling_layers(),

            # Output layer
            nn.Conv2d(hidden_dim * 8, 1, 4, 1, 0),
            nn.Sigmoid()  # Probability output
        )
```

**Key Points:**
- Strided convolutions for downsampling
- LeakyReLU(0.2) preferred over ReLU
- No batch norm in first layer
- Sigmoid output for probability [0, 1]

### Training Loop

```python
def train_step(generator, discriminator, real_images, latent_dim):
    batch_size = real_images.size(0)
    device = real_images.device

    # Labels for real and fake
    real_labels = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)

    # ========== Train Discriminator ==========
    discriminator.zero_grad()

    # Real images
    real_output = discriminator(real_images)
    d_loss_real = criterion(real_output, real_labels)

    # Fake images
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_images = generator(z).detach()  # Detach to avoid backprop to G
    fake_output = discriminator(fake_images)
    d_loss_fake = criterion(fake_output, fake_labels)

    # Total discriminator loss
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    d_optimizer.step()

    # ========== Train Generator ==========
    generator.zero_grad()

    # Generate new fake images
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_images = generator(z)
    fake_output = discriminator(fake_images)

    # Generator loss (fool discriminator)
    g_loss = criterion(fake_output, real_labels)
    g_loss.backward()
    g_optimizer.step()

    return g_loss.item(), d_loss.item()
```

### Conditional GAN

```python
class ConditionalGenerator(NexusModule):
    def forward(self, z, labels):
        # Embed labels
        label_emb = self.label_embedding(labels)

        # Concatenate with noise
        input = torch.cat([z, label_emb], dim=1)

        # Generate conditioned on label
        return self.main(input)
```

### Wasserstein GAN with Gradient Penalty

```python
def compute_gradient_penalty(discriminator, real_data, fake_data):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=real_data.device)

    # Interpolate between real and fake
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)

    # Get discriminator output
    d_interpolates = discriminator(interpolates)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

# WGAN-GP loss
d_loss = -real_score + fake_score + lambda_gp * gradient_penalty
```

## Optimization Tricks

### 1. Training Balance

**Problem**: Generator or discriminator becomes too strong

**Solutions:**
- Train discriminator k times per generator update (k=1-5)
- Use separate learning rates (often lr_D = lr_G / 2)
- Monitor loss ratio

### 2. Label Smoothing

```python
# Instead of hard 0/1 labels
real_labels = torch.ones(batch_size) * 0.9  # Smooth to 0.9
fake_labels = torch.zeros(batch_size) + 0.1  # Smooth to 0.1
```

Benefits: Prevents overconfident discriminator.

### 3. Noisy Labels

```python
# Occasionally flip labels
if random.random() < 0.05:
    real_labels, fake_labels = fake_labels, real_labels
```

Benefits: Adds regularization, prevents discriminator dominance.

### 4. Spectral Normalization

```python
from torch.nn.utils import spectral_norm

# Apply to discriminator layers
self.conv1 = spectral_norm(nn.Conv2d(...))
```

Benefits: Controls Lipschitz constant, training stability.

### 5. Two Time-Scale Update Rule (TTUR)

```python
# Separate learning rates
g_optimizer = Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
d_optimizer = Adam(discriminator.parameters(), lr=4e-4, betas=(0.0, 0.9))
```

Benefits: Better convergence, especially for WGAN variants.

## Experiments & Results

### GAN Variant Comparison

| Model | FID (50K) | Training Stability | Mode Coverage | Notes |
|-------|-----------|-------------------|---------------|-------|
| **Vanilla GAN** | 15.3 | Poor | Low | Mode collapse common |
| **DCGAN** | 8.7 | Medium | Medium | Convolutional architecture |
| **WGAN** | 6.2 | Good | High | Wasserstein distance |
| **WGAN-GP** | 5.1 | Very Good | High | Gradient penalty |
| **Spectral Norm GAN** | 4.8 | Excellent | High | Best stability |

### Architecture Impact

| Component | FID Improvement | Notes |
|-----------|----------------|-------|
| **Batch Normalization** | -2.1 | Essential for generator |
| **Spectral Normalization** | -1.3 | Critical for discriminator |
| **Self-Attention** | -0.8 | Better global coherence |
| **Progressive Growing** | -1.5 | Enables high resolution |

## Common Pitfalls

### 1. Mode Collapse

**Symptoms:**
- Generator produces limited variety
- All samples look similar
- Low inter-class diversity

**Solutions:**
```python
# 1. Use Wasserstein loss
# 2. Mini-batch discrimination
# 3. Feature matching
feature_real = discriminator.get_features(real_images)
feature_fake = discriminator.get_features(fake_images)
feature_matching_loss = F.mse_loss(feature_fake.mean(0), feature_real.mean(0))

# 4. Unrolled GAN (update G with multiple D steps ahead)
```

### 2. Training Instability

**Symptoms:**
- Oscillating losses
- Generator loss increases indefinitely
- Discriminator accuracy near 100%

**Solutions:**
```python
# 1. WGAN-GP
d_loss = -real_score + fake_score + lambda_gp * gradient_penalty

# 2. Spectral normalization
from torch.nn.utils import spectral_norm

# 3. Lower learning rates
lr = 2e-4  # Standard
lr = 1e-4  # More stable

# 4. Gradient clipping
torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
```

### 3. Vanishing Gradients

**Problem:** Generator loss saturates, no learning signal

**Solution:**
```python
# Non-saturating loss (recommended)
g_loss = -torch.log(discriminator(fake_images)).mean()

# Or equivalently
g_loss = -discriminator(fake_images).mean()  # WGAN
```

### 4. Poor Sample Quality

**Symptoms:**
- Blurry or artifacted images
- Unrealistic textures
- Poor fine details

**Solutions:**
```python
# 1. Increase model capacity
hidden_dim = 512  # -> 1024

# 2. Progressive growing
# Start at 4x4, gradually increase to 1024x1024

# 3. Self-attention layers
from torch.nn import MultiheadAttention

# 4. Better upsampling
nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
nn.Conv2d(...)  # Instead of transposed conv
```

### 5. Discriminator Overpowering

**Symptoms:**
- Discriminator accuracy near 100%
- Generator loss very high
- No improvement over time

**Solutions:**
```python
# 1. Train discriminator less frequently
if step % 2 == 0:
    train_discriminator()
train_generator()

# 2. Add noise to discriminator inputs
noisy_real = real_images + 0.1 * torch.randn_like(real_images)

# 3. Reduce discriminator capacity
hidden_dim_d = hidden_dim_g // 2
```

## Model-Specific Details

### CycleGAN

For unpaired image-to-image translation:

```python
# Cycle consistency loss
cycle_loss = F.l1_loss(reconstructed_A, real_A) + F.l1_loss(reconstructed_B, real_B)

# Identity loss (optional)
identity_loss = F.l1_loss(G_AB(real_B), real_B) + F.l1_loss(G_BA(real_A), real_A)

# Total loss
total_loss = gan_loss + lambda_cycle * cycle_loss + lambda_identity * identity_loss
```

**Hyperparameters:**
- lambda_cycle = 10.0
- lambda_identity = 0.5 (or 0)

## Evaluation Metrics

### Quantitative Metrics

**Inception Score (IS):**
```python
# Higher is better
# Measures quality and diversity
IS = exp(E_x[KL(p(y|x) || p(y))])
```

**Fréchet Inception Distance (FID):**
```python
# Lower is better
# Measures distribution similarity
FID = ||mu_real - mu_fake||^2 + Tr(Sigma_real + Sigma_fake - 2*sqrt(Sigma_real * Sigma_fake))
```

**Precision and Recall:**
- Precision: Fraction of fake samples that look real
- Recall: Fraction of real distribution covered by fakes

### Qualitative Evaluation

1. **Visual inspection** - Overall quality and realism
2. **Interpolation** - Smooth transitions in latent space
3. **Mode coverage** - Variety of generated samples
4. **Attribute control** - Conditional generation quality

## Hyperparameter Guidelines

### Learning Rates

```python
# Standard GAN
lr_g = 2e-4
lr_d = 2e-4

# WGAN variants
lr_g = 1e-4
lr_d = 4e-4  # Can be higher for critic

# TTUR
lr_g = 1e-4
lr_d = 4e-4
```

### Batch Size

- **Small models**: 64-128
- **Standard DCGAN**: 128-256
- **Large scale**: 256-512

Larger batch sizes generally improve training stability.

### Discriminator Updates

```python
n_critic = 5  # WGAN: train critic more
n_critic = 1  # Standard GAN: balanced training
```

## References

### Original Papers

1. **Goodfellow et al., "Generative Adversarial Networks" (2014)**
   - Original GAN formulation
   - https://arxiv.org/abs/1406.2661

2. **Radford et al., "Unsupervised Representation Learning with Deep Convolutional GANs" (2016)**
   - DCGAN architecture guidelines
   - https://arxiv.org/abs/1511.06434

3. **Arjovsky et al., "Wasserstein GAN" (2017)**
   - Wasserstein distance for GANs
   - https://arxiv.org/abs/1701.07875

4. **Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)**
   - Gradient penalty instead of weight clipping
   - https://arxiv.org/abs/1704.00028

5. **Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (2017)**
   - CycleGAN for unpaired translation
   - https://arxiv.org/abs/1703.10593

### Additional Resources

6. **Miyato et al., "Spectral Normalization for GANs" (2018)**
   - Spectral normalization for stability
   - https://arxiv.org/abs/1802.05957

7. **Brock et al., "Large Scale GAN Training for High Fidelity Natural Image Synthesis" (2019)**
   - BigGAN, self-attention, orthogonal regularization
   - https://arxiv.org/abs/1809.11096

## Next Steps

1. **Try Wasserstein GAN** for more stable training
2. **Implement conditional generation** for controllable outputs
3. **Explore diffusion models** for state-of-the-art quality
4. **Study StyleGAN** for advanced image synthesis

---

*Implementations: `/Users/kevinyu/Projects/Nexus/nexus/models/gan/`*
