# Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields

## Overview & Motivation

Zip-NeRF (ICCV 2023) combines the best of both worlds: **Mip-NeRF's anti-aliasing** with **Instant-NGP's speed**. It achieves state-of-the-art quality while maintaining fast training times through multiresolution hash encoding combined with integrated positional encoding.

### The Gap Between Quality and Speed

Before Zip-NeRF:
- **Mip-NeRF 360**: Best quality, but slow (days of training)
- **Instant-NGP**: Fast (minutes), but aliasing artifacts
- **Trade-off**: Quality OR speed, not both

### Zip-NeRF's Innovation

Combines three key techniques:
1. **Multiresolution hash grids** (from Instant-NGP) for speed
2. **Integrated positional encoding** (from Mip-NeRF) for anti-aliasing
3. **Online distillation** for proposal networks

Result: **Near Mip-NeRF quality at Instant-NGP speeds**

## Theoretical Background

### Multisampling Anti-Aliasing (MSAA)

Traditional approach to anti-aliasing in hash grids:
```
For each sample:
  Sample multiple nearby grid points
  Average their features
```

**Problem**: Expensive, requires many grid lookups

### Integrated Encoding in Feature Space

Zip-NeRF's key insight: Apply IPE to hash grid features, not positions.

```
Standard: γ(hash(x))
Zip-NeRF: E[γ(hash(X))] where X ~ N(μ, Σ)
```

### Proposal Network Distillation

Use coarse proposal network to guide sampling:
```
Proposal network: Fast, guides where to sample
Main network: High quality, evaluates at important points
```

Online distillation: Proposal network learns from main network during training.

## Mathematical Formulation

### Hash Encoding with IPE

For a Gaussian sample (μ, Σ) at resolution level l:

```
Resolution: r_l = r_min × b^l
Encoding: E[h_l(X)] where X ~ N(μ, Σ)

Approximate as:
h_l(μ) weighted by exp(-λ_l · tr(Σ))

where λ_l controls anti-aliasing strength at each level
```

### Hierarchical Sampling with Proposals

Two-stage sampling:
```
Stage 1 (Proposal):
  t ~ p_proposal(t | r)
  Quick evaluation for sample distribution

Stage 2 (Main):
  t ~ p_main(t | r, p_proposal)
  Detailed evaluation at important locations
```

### Distillation Loss

Align proposal and main network predictions:
```
L_distill = KL(p_main || p_proposal)
          + ||w_main - w_proposal||²

Guides proposal to predict main network's importance
```

### Combined Loss

```
L_total = L_rgb + λ_distill · L_distill + λ_interlevel · L_interlevel

L_rgb: Photometric reconstruction
L_distill: Proposal network distillation
L_interlevel: Consistency across hash levels
```

## Implementation Details

### Hash Grid with Anti-Aliasing

```python
class ZipNeRFHashGrid(nn.Module):
    def __init__(
        self,
        n_levels=16,
        features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        max_resolution=2048
    ):
        super().__init__()
        self.n_levels = n_levels
        self.features_per_level = features_per_level

        # Multi-resolution hash tables
        self.hash_tables = nn.ModuleList([
            nn.Embedding(2 ** log2_hashmap_size, features_per_level)
            for _ in range(n_levels)
        ])

        # Resolution per level
        self.resolutions = torch.tensor([
            base_resolution * (max_resolution / base_resolution) ** (i / (n_levels - 1))
            for i in range(n_levels)
        ])

    def forward(self, means, covs):
        """
        Anti-aliased hash encoding.

        Args:
            means: [N, 3] Sample means
            covs: [N, 3, 3] Sample covariances

        Returns:
            features: [N, n_levels * features_per_level]
        """
        features_list = []

        for level in range(self.n_levels):
            resolution = self.resolutions[level]

            # Compute anti-aliasing weight
            # Higher variance → lower weight for this level
            trace_cov = torch.diagonal(covs, dim1=-2, dim2=-1).sum(dim=-1)
            aa_weight = torch.exp(-0.5 * trace_cov * (resolution ** 2))

            # Hash and lookup features
            scaled_means = means * resolution
            features_level = self.hash_lookup(scaled_means, level)

            # Apply anti-aliasing
            features_level = features_level * aa_weight[:, None]
            features_list.append(features_level)

        return torch.cat(features_list, dim=-1)

    def hash_lookup(self, positions, level):
        """Trilinear interpolation from hash table."""
        floor_pos = torch.floor(positions).long()
        frac_pos = positions - floor_pos

        # Sample 8 corners
        features = torch.zeros(
            len(positions), self.features_per_level,
            device=positions.device
        )

        for dx, dy, dz in itertools.product([0, 1], repeat=3):
            corner = floor_pos + torch.tensor([dx, dy, dz], device=positions.device)
            hash_idx = self.spatial_hash(corner)

            corner_features = self.hash_tables[level](hash_idx)

            # Trilinear weight
            weight = (
                (frac_pos[:, 0] if dx else 1 - frac_pos[:, 0]) *
                (frac_pos[:, 1] if dy else 1 - frac_pos[:, 1]) *
                (frac_pos[:, 2] if dz else 1 - frac_pos[:, 2])
            )

            features += weight[:, None] * corner_features

        return features

    def spatial_hash(self, coords):
        """Spatial hash function using primes."""
        primes = torch.tensor([1, 2654435761, 805459861], device=coords.device)
        return torch.sum(coords * primes, dim=-1) % (2 ** 19)
```

### Proposal Network

```python
class ProposalNetwork(nn.Module):
    """Fast coarse network for importance sampling."""

    def __init__(self, n_levels=8, features_per_level=2):
        super().__init__()
        # Smaller hash grid for speed
        self.hash_grid = ZipNeRFHashGrid(
            n_levels=n_levels,
            features_per_level=features_per_level
        )

        # Tiny MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_levels * features_per_level, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Just density
        )

    def forward(self, means, covs):
        features = self.hash_grid(means, covs)
        density = F.softplus(self.mlp(features))
        return density
```

### Online Distillation Training

```python
def train_step(
    proposal_net,
    main_net,
    ray_origins,
    ray_directions,
    target_rgb
):
    # Stage 1: Proposal sampling
    t_coarse = sample_uniform(near, far, N_coarse)
    means_coarse, covs_coarse = compute_gaussians(
        ray_origins, ray_directions, t_coarse
    )

    density_proposal = proposal_net(means_coarse, covs_coarse)
    weights_proposal = volume_rendering_weights(density_proposal, t_coarse)

    # Stage 2: Importance sampling from proposal
    t_fine = sample_pdf(t_coarse, weights_proposal, N_fine)
    means_fine, covs_fine = compute_gaussians(
        ray_origins, ray_directions, t_fine
    )

    # Main network evaluation
    outputs_main = main_net(means_fine, covs_fine)
    rgb_pred = volume_rendering(outputs_main, t_fine)

    # Compute losses
    loss_rgb = F.mse_loss(rgb_pred, target_rgb)

    # Distillation: Align proposal with main network
    with torch.no_grad():
        density_main = main_net.density(means_coarse, covs_coarse)
        weights_main = volume_rendering_weights(density_main, t_coarse)

    loss_distill = F.mse_loss(weights_proposal, weights_main)

    # Combined loss
    loss = loss_rgb + 0.01 * loss_distill

    return loss
```

## High-Level Intuition

### Why Zip-NeRF Works

**Analogy**: Think of rendering as looking through progressively finer screens:

1. **Coarse screens** (low-res hash levels): See general structure
2. **Fine screens** (high-res hash levels): See details

**Anti-aliasing**: When viewing from far away, automatically blur out fine screens (they'd just add noise). Close up, keep all detail.

### The "Zip" in Zip-NeRF

"Zip" refers to:
1. **Fast** (zip through training)
2. **Compression** (hash encoding compresses space)
3. **Bringing together** (zips IPE and hash grids)

## Optimization Tricks

### 1. Adaptive Hash Grid Resolution

```python
def compute_optimal_resolutions(scene_scale, num_levels):
    """Adapt hash grid resolutions to scene."""
    base_res = max(16, scene_scale / 100)
    max_res = min(4096, scene_scale * 10)

    return [
        base_res * (max_res / base_res) ** (i / (num_levels - 1))
        for i in range(num_levels)
    ]
```

### 2. Interlevel Loss

Ensure consistency across hash levels:

```python
def interlevel_loss(features_per_level):
    """Regularize adjacent levels to be similar."""
    loss = 0
    for i in range(len(features_per_level) - 1):
        loss += F.mse_loss(
            features_per_level[i],
            features_per_level[i+1].detach()
        )
    return loss / (len(features_per_level) - 1)
```

### 3. Progressive Training

```python
# Start with coarse levels, add fine levels gradually
def get_active_levels(iteration, total_iterations, n_levels):
    progress = iteration / total_iterations
    return max(4, int(n_levels * progress))
```

### 4. Learning Rate Scheduling

```python
# Different learning rates for hash grids and MLP
optimizer = torch.optim.Adam([
    {'params': hash_grid.parameters(), 'lr': 1e-2},
    {'params': mlp.parameters(), 'lr': 1e-3}
])

# Exponential decay
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.1 ** (1 / 50000)
)
```

## Experiments & Results

### Quantitative Results (Mip-NeRF 360 Dataset)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Training Time |
|--------|--------|--------|---------|---------------|
| **Zip-NeRF** | **28.54** | **0.828** | **0.189** | **2 hours** |
| Mip-NeRF 360 | **28.87** | **0.840** | **0.183** | 48 hours |
| Instant-NGP | 25.60 | 0.694 | 0.344 | 5 min |
| 3D Gaussian Splatting | 27.21 | 0.815 | 0.214 | 30 min |

**Key achievement**: 95% of Mip-NeRF 360 quality at 24x speedup

### Quality-Speed Tradeoff

| Samples/Ray | PSNR | Rendering Speed |
|-------------|------|-----------------|
| 48 | 27.2 | 10 FPS |
| 96 | 28.1 | 5 FPS |
| 192 | 28.5 | 2 FPS |
| 384 | 28.6 | 0.5 FPS |

### Ablation Studies

| Configuration | PSNR | Training Time |
|---------------|------|---------------|
| Full Zip-NeRF | 28.5 | 2 hours |
| - No anti-aliasing | 26.8 (-1.7) | 2 hours |
| - No distillation | 27.9 (-0.6) | 2.5 hours |
| - No interlevel loss | 28.2 (-0.3) | 2 hours |

Each component contributes to final quality.

## Common Pitfalls

### 1. Hash Collision Management

```python
# Monitor collision rate
def check_hash_collisions(hash_grid, sample_positions):
    hash_indices = hash_grid.spatial_hash(sample_positions)
    unique_ratio = len(torch.unique(hash_indices)) / len(hash_indices)

    if unique_ratio < 0.5:
        warnings.warn(
            f"High hash collision rate: {1-unique_ratio:.2%}\n"
            "Consider increasing hash table size"
        )
```

### 2. Balancing Proposal and Main Network

```python
# Don't let proposal network get too far behind
if loss_distill > 10 * loss_rgb:
    # Train proposal more aggressively
    for _ in range(5):
        proposal_optimizer.step()
```

### 3. Anti-Aliasing Weight Tuning

```python
# Anti-aliasing strength depends on scene scale
aa_strength = 0.5 / scene_scale  # Adjust empirically

aa_weight = torch.exp(-aa_strength * trace_cov * (resolution ** 2))
```

### 4. Numerical Stability

```python
# Avoid NaN in exponentials with large covariances
aa_weight = torch.exp(-torch.clamp(
    0.5 * trace_cov * (resolution ** 2),
    max=50.0  # Prevent overflow
))
```

## References

### Primary Paper

```bibtex
@inproceedings{barron2023zipnerf,
  title={Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields},
  author={Barron, Jonathan T and Mildenhall, Ben and Verbin, Dor and
          Srinivasan, Pratul P and Hedman, Peter},
  booktitle={ICCV},
  year={2023}
}
```

### Building Blocks

- **Mip-NeRF 360** (Barron et al., 2022): Integrated positional encoding
- **Instant-NGP** (Müller et al., 2022): Multiresolution hash encoding
- **Proposal Networks** (Müller et al., 2019): Importance sampling

### Follow-up Work

- **Zip-NeRF++** (2024): Extended to dynamic scenes
- **Grid-NeRF** (2024): Further optimizations for mobile deployment

---

**Summary**: Zip-NeRF represents the state-of-the-art for NeRF-based methods, achieving near-optimal quality with practical training times. For even faster rendering, see [Gaussian Splatting](./gaussian_splatting.md).
