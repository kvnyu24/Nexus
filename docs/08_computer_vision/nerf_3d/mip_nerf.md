# Mip-NeRF: Multiscale Representation for Anti-Aliasing

## Overview & Motivation

Mip-NeRF, published at ICCV 2021 by Barron et al., addresses a fundamental flaw in the original NeRF: aliasing artifacts when rendering at different resolutions or distances. The key innovation is replacing rays with cones and point samples with volumetric frustums, enabling anti-aliased rendering through integrated positional encoding.

### The Aliasing Problem in NeRF

**Issue**: NeRF samples discrete points along rays, but a pixel actually captures light from a cone-shaped region:

```
NeRF:    Camera → Ray (infinitesimal width) → Point samples
Reality: Camera → Cone (pixel footprint) → Volumetric samples
```

**Consequences**:
1. **Scale ambiguity**: Loses detail at distance, shows artifacts up close
2. **Aliasing**: Jagged edges, moiré patterns at different resolutions
3. **Training/test mismatch**: Trained at one resolution, fails at others

### Mip-NeRF's Solution

**Key Insight**: Treat each pixel as a cone, and each sample as a 3D Gaussian frustum whose size depends on distance from camera.

```
Integrated Positional Encoding (IPE):
Instead of: γ(x) = [sin(2^k πx), cos(2^k πx)]
Use:        E[γ(X)] where X ~ N(μ, Σ)  # Expected encoding of Gaussian
```

This allows the network to reason about volumetric regions rather than infinitesimal points.

## Theoretical Background

### From Rays to Cones

Each pixel covers a cone emanating from the camera center:

```
Cone radius at distance t:
r(t) = t · tan(θ/2)

where θ is the pixel's angular extent
```

### Conical Frustums as Gaussians

A sample along the cone between t₁ and t₂ is approximated as a 3D Gaussian:

```
Mean (μ):
μ = o + t_μ d

where t_μ = (t₁ + t₂) / 2

Covariance (Σ):
Σ = (t_σ)² (dd^T + r² I_⊥)

where:
t_σ: Spread along ray direction
r: Radius of cone at distance t_μ
I_⊥: Identity perpendicular to ray
```

### Integrated Positional Encoding

For a Gaussian X ~ N(μ, Σ), the expected positional encoding is:

```
E[sin(2^k πX)] = exp(-½ · 2^(2k) π² Σ) sin(2^k πμ)
E[cos(2^k πX)] = exp(-½ · 2^(2k) π² Σ) cos(2^k πμ)
```

**Interpretation**:
- High frequencies (large k) are attenuated by exp(-Σ)
- Larger covariance → more blur → less high-frequency content
- Natural multi-scale representation

## Mathematical Formulation

### Conical Frustum Parameterization

Given ray r(t) = o + td and interval [t₀, t₁]:

**Mean**:
```
μ(t₀, t₁) = o + ((t₀ + t₁)/2) d
```

**Variance along ray**:
```
σ_parallel² = (t₁ - t₀)² / 12
```

**Variance perpendicular to ray**:
```
σ_perp² = (t₀² + t₀t₁ + t₁²) / 12 · (r_pixel)²

where r_pixel = tan(pixel_angle / 2)
```

**Full covariance matrix**:
```
Σ = σ_parallel² (dd^T) + σ_perp² (I - dd^T)
```

### Integrated Positional Encoding (IPE)

For input Gaussian (μ, Σ) and frequency 2^k:

```
γ_k(μ, Σ) = [γ_k^sin(μ, Σ), γ_k^cos(μ, Σ)]

where:
γ_k^sin(μ, Σ) = exp(-2^(2k-1) π² diag(Σ)) ⊙ sin(2^k π μ)
γ_k^cos(μ, Σ) = exp(-2^(2k-1) π² diag(Σ)) ⊙ cos(2^k π μ)
```

**Full encoding** (concatenate over all frequencies):
```
γ(μ, Σ) = [γ₀(μ, Σ), γ₁(μ, Σ), ..., γ_L(μ, Σ)]
```

### Network Architecture

Similar to NeRF, but with IPE instead of PE:

```
Position → IPE(μ_pos, Σ_pos) → MLP → [σ, h]
[h, IPE(μ_dir, Σ_dir)] → MLP → rgb
```

**Key difference**: Position encoding now takes (μ, Σ) instead of just x.

## High-Level Intuition

### Why Integrated Encoding Works

Think of it as **adaptive blur**:

1. **Near camera**: Small Σ → all frequencies preserved → sharp details
2. **Far from camera**: Large Σ → high frequencies dampened → smooth appearance
3. **Natural LoD**: Level-of-detail built into representation

### The Cone vs Ray Analogy

```
NeRF (Ray):
  - "What color at this point?"
  - Single sample = infinitesimal
  - Scale ambiguous

Mip-NeRF (Cone):
  - "What color in this region?"
  - Sample = Gaussian frustum
  - Scale-aware
```

### Anti-Aliasing Mechanism

The exponential attenuation acts as a **low-pass filter**:

```
Large variance → exp(-½kΣ) small → high freq suppressed → smooth
Small variance → exp(-½kΣ ≈ 1 → high freq preserved → detailed
```

This is exactly what we want for proper anti-aliasing.

## Implementation Details

### Frustum Computation

```python
def compute_conical_frustum(
    ray_origin: torch.Tensor,    # [3]
    ray_direction: torch.Tensor, # [3]
    t_near: float,
    t_far: float,
    pixel_radius: float          # Cone radius per unit distance
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and covariance of conical frustum.

    Returns:
        mu: [3] - Mean position
        cov: [3, 3] - Covariance matrix
    """
    # Mean position
    t_mu = (t_near + t_far) / 2
    mu = ray_origin + t_mu * ray_direction

    # Variance along ray
    t_delta = t_far - t_near
    var_parallel = (t_delta ** 2) / 12

    # Variance perpendicular to ray
    # Average radius squared
    r_near = t_near * pixel_radius
    r_far = t_far * pixel_radius
    var_perp = ((r_near ** 2 + r_near * r_far + r_far ** 2) / 12)

    # Build covariance matrix
    # Σ = var_parallel (dd^T) + var_perp (I - dd^T)
    d = ray_direction
    d_outer = torch.outer(d, d)  # dd^T
    I = torch.eye(3, device=d.device)

    cov = var_parallel * d_outer + var_perp * (I - d_outer)

    return mu, cov
```

### Integrated Positional Encoding

```python
class IntegratedPositionalEncoding(nn.Module):
    def __init__(self, min_deg=0, max_deg=16):
        super().__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg

        # Precompute frequency scales
        self.register_buffer(
            'scales',
            2.0 ** torch.arange(min_deg, max_deg)
        )

    def forward(
        self,
        means: torch.Tensor,     # [N, 3]
        covs: torch.Tensor       # [N, 3, 3]
    ) -> torch.Tensor:
        """
        Compute integrated positional encoding.

        Returns:
            encoded: [N, 3 * 2 * num_frequencies]
        """
        # Extract diagonal covariance (variance)
        variances = torch.diagonal(covs, dim1=-2, dim2=-1)  # [N, 3]

        # Compute scaled inputs
        # scaled_means: [N, 3, F]
        # scaled_vars: [N, 3, F]
        scaled_means = means[..., None] * self.scales[None, None, :]
        scaled_vars = variances[..., None] * (self.scales[None, None, :] ** 2)

        # Compute expected sine and cosine
        # exp(-½σ²ω²) * sin(μω) and exp(-½σ²ω²) * cos(μω)
        damping = torch.exp(-0.5 * scaled_vars * (np.pi ** 2))

        sin_features = damping * torch.sin(np.pi * scaled_means)
        cos_features = damping * torch.cos(np.pi * scaled_means)

        # Concatenate [sin, cos] for each frequency
        # Shape: [N, 3, 2F] → [N, 6F]
        encoded = torch.cat([sin_features, cos_features], dim=-1)
        encoded = encoded.reshape(means.shape[0], -1)

        return encoded
```

### Mip-NeRF Network

```python
class MipNeRFNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        min_deg=0,
        max_deg=16,
        num_layers=8
    ):
        super().__init__()

        self.ipe_position = IntegratedPositionalEncoding(min_deg, max_deg)
        self.ipe_direction = IntegratedPositionalEncoding(0, 4)

        pos_dim = 3 * 2 * (max_deg - min_deg)
        dir_dim = 3 * 2 * 4

        # Position MLP with skip connection
        layers = []
        layers.append(nn.Linear(pos_dim, hidden_dim))
        layers.append(nn.ReLU())

        for i in range(1, num_layers):
            if i == 4:  # Skip connection
                layers.append(nn.Linear(hidden_dim + pos_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.position_mlp = nn.Sequential(*layers)

        # Density head
        self.density_head = nn.Linear(hidden_dim, 1)

        # Color MLP
        self.color_mlp = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

    def forward(
        self,
        pos_means: torch.Tensor,      # [N, 3]
        pos_covs: torch.Tensor,       # [N, 3, 3]
        dir_means: torch.Tensor,      # [N, 3]
        dir_covs: torch.Tensor = None # [N, 3, 3] optional
    ) -> Dict[str, torch.Tensor]:
        # Encode positions with IPE
        pos_encoded = self.ipe_position(pos_means, pos_covs)

        # Process through MLP
        h = pos_encoded
        for i, layer in enumerate(self.position_mlp):
            h = layer(h)
            # Skip connection at layer 4
            if i == 8:  # After 4th block
                h = torch.cat([h, pos_encoded], dim=-1)

        # Predict density
        density = F.softplus(self.density_head(h))

        # Encode directions
        if dir_covs is None:
            dir_covs = torch.zeros(
                dir_means.shape[0], 3, 3,
                device=dir_means.device
            )
        dir_encoded = self.ipe_direction(dir_means, dir_covs)

        # Predict color
        color_input = torch.cat([h, dir_encoded], dim=-1)
        color = self.color_mlp(color_input)

        return {
            "density": density,
            "color": color
        }
```

### Rendering with Conical Frustums

```python
def render_mip_nerf(
    model: MipNeRFNetwork,
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    near: float,
    far: float,
    num_samples: int = 128,
    pixel_radius: float = None
) -> Dict[str, torch.Tensor]:
    """Render with anti-aliasing."""

    # Compute pixel radius if not provided
    if pixel_radius is None:
        # Assume 90 degree FoV, image width 800
        pixel_radius = np.tan(np.pi / 4) / 400

    # Sample intervals along ray
    t_vals = torch.linspace(near, far, num_samples + 1, device=ray_origin.device)
    t_mids = (t_vals[:-1] + t_vals[1:]) / 2

    # Compute frustums for each interval
    means = []
    covs = []

    for i in range(num_samples):
        mu, cov = compute_conical_frustum(
            ray_origin,
            ray_direction,
            t_vals[i],
            t_vals[i + 1],
            pixel_radius
        )
        means.append(mu)
        covs.append(cov)

    means = torch.stack(means)  # [N, 3]
    covs = torch.stack(covs)    # [N, 3, 3]

    # Prepare directions (constant along ray)
    directions = ray_direction[None].expand(num_samples, -1)

    # Evaluate network
    outputs = model(means, covs, directions)
    density = outputs["density"]  # [N, 1]
    color = outputs["color"]      # [N, 3]

    # Volume rendering (same as NeRF)
    delta_t = t_vals[1:] - t_vals[:-1]
    alpha = 1 - torch.exp(-density.squeeze(-1) * delta_t)

    transmittance = torch.cumprod(
        torch.cat([
            torch.ones(1, device=alpha.device),
            1 - alpha + 1e-10
        ]),
        dim=0
    )[:-1]

    weights = alpha * transmittance

    # Final color
    rgb = (weights[:, None] * color).sum(dim=0)

    return {"rgb": rgb, "weights": weights}
```

## Code Walkthrough

Our implementation in `nexus/models/cv/nerf/mipnerf.py` provides the core functionality:

### Key Class: `IntegratedPositionalEncoding`

```python
# From nexus/models/cv/nerf/mipnerf.py (lines 8-27)

class IntegratedPositionalEncoding(PositionalEncoding):
    def __init__(self, num_frequencies: int = 10, min_deg: int = 0, max_deg: int = 16):
        super().__init__(num_frequencies)
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = 2.0 ** torch.linspace(min_deg, max_deg-1, max_deg-min_deg)

    def forward(self, means: torch.Tensor, covs: torch.Tensor) -> torch.Tensor:
        # Key innovation: encoding with covariance
        scales = self.scales[None, :].to(means.device)
        scaled_means = means[..., None] * scales
        scaled_covs = covs[..., None] * (scales ** 2)

        # Expected sine/cosine under Gaussian distribution
        exp_sin = torch.exp(-0.5 * scaled_covs) * torch.sin(scaled_means)
        exp_cos = torch.exp(-0.5 * scaled_covs) * torch.cos(scaled_means)

        encoded = torch.cat([exp_sin, exp_cos], dim=-1)
        return encoded.reshape(means.shape[0], -1)
```

**Critical detail**: The `exp(-0.5 * scaled_covs)` term implements the low-pass filtering that eliminates aliasing.

## Optimization Tricks

### 1. Proposal Network for Efficient Sampling

Use a coarse network to guide fine sampling:

```python
def sample_hierarchical_mipnerf(coarse_weights, num_fine=128):
    """Sample fine points based on coarse weights."""
    # Convert weights to PDF
    pdf = coarse_weights / (coarse_weights.sum() + 1e-5)

    # Sample from CDF
    cdf = torch.cumsum(pdf, dim=-1)
    u = torch.rand(num_fine, device=cdf.device)
    indices = torch.searchsorted(cdf, u)

    return indices
```

### 2. Multi-Resolution Training

Train on images at different resolutions:

```python
def get_random_scale():
    # Randomly scale image between 0.5x and 1.0x
    scale = np.random.uniform(0.5, 1.0)
    return scale

# During training
scale = get_random_scale()
image_scaled = F.interpolate(image, scale_factor=scale)
pixel_radius *= scale  # Adjust cone radius
```

### 3. Coarse-to-Fine Frequency Annealing

Gradually increase frequency range during training:

```python
def get_max_freq(iteration, total_iters, max_freq=16):
    # Start with low frequencies, gradually add higher
    progress = iteration / total_iters
    return int(max_freq * progress)
```

## Experiments & Results

### Quantitative Results (Multiscale Blender)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|--------|--------|---------|
| **Mip-NeRF** | **33.09** | **0.961** | **0.043** |
| NeRF | 31.01 | 0.947 | 0.081 |
| JaxNeRF | 31.78 | 0.954 | 0.067 |

**+2.08 dB improvement** over original NeRF with same architecture.

### Multi-Scale Consistency

Rendering at different resolutions:

| Resolution | NeRF PSNR | Mip-NeRF PSNR |
|------------|-----------|---------------|
| 200×200 | 27.3 | **31.2** |
| 400×400 | 31.0 | **33.1** |
| 800×800 | 31.0 | **33.0** |

Mip-NeRF maintains consistent quality across scales.

### Ablation Studies

**Effect of IPE**:
- Without IPE (standard PE): 31.0 PSNR
- With IPE: **33.1 PSNR** (+2.1 dB)

**Effect of Conical Frustums**:
- Point samples: Aliasing artifacts
- Frustums: **Smooth, anti-aliased** rendering

## Common Pitfalls

### 1. Incorrect Covariance Computation

```python
# Wrong: Isotropic covariance
cov = variance * torch.eye(3)

# Correct: Anisotropic based on cone geometry
cov = var_parallel * (d @ d.T) + var_perp * (I - d @ d.T)
```

### 2. Forgetting to Square Frequencies in Damping

```python
# Wrong: Linear frequency in exponent
damping = torch.exp(-0.5 * scaled_vars * freq)

# Correct: Squared frequency
damping = torch.exp(-0.5 * scaled_vars * (freq ** 2))
```

### 3. Not Adjusting Pixel Radius for Resolution

```python
# When rendering at different resolutions
pixel_radius_new = pixel_radius_original * (resolution_original / resolution_new)
```

## References

### Primary Paper

```bibtex
@inproceedings{barron2021mipnerf,
  title={Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields},
  author={Barron, Jonathan T and Mildenhall, Ben and Tancik, Matthew and
          Hedman, Peter and Martin-Brualla, Ricardo and Srinivasan, Pratul P},
  booktitle={ICCV},
  year={2021}
}
```

### Follow-up Work

- **Mip-NeRF 360** (2022): Unbounded scenes with IPE
- **Zip-NeRF** (2023): Combined with hash encoding
- **Mip-Splatting** (2024): IPE for Gaussian Splatting

---

**Next**: See [NeRF++](./nerf_plus_plus.md) for unbounded scenes or [Zip-NeRF](./zip_nerf.md) for SOTA quality.
