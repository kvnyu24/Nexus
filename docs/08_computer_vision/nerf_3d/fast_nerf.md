# Fast NeRF: Acceleration Techniques

## Overview & Motivation

The original NeRF requires rendering each pixel independently by querying an MLP hundreds of times, leading to extremely slow rendering (30+ seconds per 800×800 image). Fast NeRF encompasses various techniques to accelerate NeRF training and rendering while maintaining quality.

### The Performance Bottleneck

**NeRF rendering pipeline**:
```
For each pixel (640k for 800×800 image):
  For each sample along ray (192 samples):
    Query MLP (8 layers, 256 hidden)
    → Total: 122M network queries per frame
    → 30+ seconds on GPU
```

### Acceleration Strategies

1. **Caching**: Pre-compute/store expensive operations
2. **Factorization**: Decompose radiance field into efficient components
3. **Spatial structures**: Use octrees/grids for early termination
4. **Knowledge distillation**: Bake into fast representation
5. **Hybrid approaches**: Combine neural and explicit representations

## Theoretical Background

### Neural Sparse Voxel Octree (NSVF)

Replace dense sampling with sparse voxel structure:

```
Scene = Octree of voxels
Each voxel: Neural feature vector
Empty voxels: Skipped during rendering
```

**Rendering equation**:
```
Only sample along ray within occupied voxels
Skip empty space → Massive speedup
```

### Factorization Approaches

Decompose radiance field into lower-rank components:

**Vector-Matrix (VM) Decomposition**:
```
σ(x,y,z) ≈ Σᵢ vᵢ(x,y) · mᵢ(z)

Instead of: 3D volume
Use: 2D planes × 1D lines
```

**Tensor Decomposition (TensoRF)**:
```
F(x,y,z) = Σᵣ vᵣˣ(x) ⊗ vᵣʸ(y) ⊗ vᵣᶻ(z)

Low-rank tensor factorization
```

### Caching Strategies

**Factored MLP**:
```
Standard: x → MLP → σ, c
Factored: x → MLP₁ → features
          features (cached) → MLP₂ → σ, c
```

Cache intermediate features for nearby points.

## Mathematical Formulation

### Occupancy Grid

Binary grid indicating occupied space:

```
G[i,j,k] = {
  1 if voxel contains geometry
  0 otherwise
}

Ray marching:
t_next = t_current + max(ε, G.distance_to_next_occupied(t_current))
```

### Feature Grid Interpolation

Store features at grid vertices, interpolate for queries:

```
f(x) = Σᵥ∈neighbors(x) wᵥ · fᵥ

where:
wᵥ: Trilinear interpolation weights
fᵥ: Learned feature at vertex v
```

### Multi-Resolution Hash Encoding (Instant-NGP)

Use multiple resolution hash tables:

```
For level l:
  resolution_l = resolution_min × growth^l
  h_l(x) = hash(⌊x · resolution_l⌋)
  features_l = table_l[h_l(x)]

Final encoding: concat(features_1, ..., features_L)
```

**Key property**: O(1) lookup, compact storage

## Implementation Details

### Occupancy Grid Acceleration

```python
class OccupancyGrid:
    def __init__(self, resolution=128, aabb_min=-1, aabb_max=1):
        self.resolution = resolution
        self.aabb_min = torch.tensor(aabb_min)
        self.aabb_max = torch.tensor(aabb_max)

        # Binary occupancy grid
        self.grid = torch.zeros(
            resolution, resolution, resolution,
            dtype=torch.bool
        )

    def update(self, model, threshold=0.01):
        """Update occupancy grid from current model."""
        # Sample grid points
        xs = torch.linspace(self.aabb_min[0], self.aabb_max[0], self.resolution)
        ys = torch.linspace(self.aabb_min[1], self.aabb_max[1], self.resolution)
        zs = torch.linspace(self.aabb_min[2], self.aabb_max[2], self.resolution)

        grid_points = torch.stack(
            torch.meshgrid(xs, ys, zs, indexing='ij'),
            dim=-1
        ).reshape(-1, 3)

        # Query density
        with torch.no_grad():
            density = model.query_density(grid_points)

        # Mark occupied voxels
        self.grid = (density > threshold).reshape(
            self.resolution, self.resolution, self.resolution
        )

    def ray_marching(self, ray_o, ray_d, near, far, max_steps=1024):
        """March ray through occupied voxels only."""
        t = near
        t_samples = []

        step = 0
        while t < far and step < max_steps:
            pos = ray_o + t * ray_d

            # Check if in occupied voxel
            voxel_idx = self.world_to_voxel(pos)
            if self.is_valid(voxel_idx) and self.grid[tuple(voxel_idx)]:
                t_samples.append(t)
                t += self.step_size
            else:
                # Skip to next occupied voxel
                t = self.skip_empty_space(ray_o, ray_d, t)

            step += 1

        return torch.tensor(t_samples)

    def skip_empty_space(self, ray_o, ray_d, t_current):
        """Skip to next occupied voxel using DDA algorithm."""
        # Simplified: Jump by voxel size
        voxel_size = (self.aabb_max - self.aabb_min) / self.resolution
        return t_current + torch.min(voxel_size).item()
```

### Feature Grid Network

```python
class FeatureGridNeRF(nn.Module):
    def __init__(self, resolution=256, feature_dim=32):
        super().__init__()
        self.resolution = resolution
        self.feature_dim = feature_dim

        # Learnable feature grid
        self.features = nn.Parameter(
            torch.randn(resolution, resolution, resolution, feature_dim) * 0.1
        )

        # Small MLP to decode features
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim + 3, 64),  # +3 for position
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # σ + RGB
        )

    def interpolate_features(self, positions):
        """Trilinear interpolation of grid features."""
        # Normalize positions to [0, resolution-1]
        normalized = (positions + 1) / 2 * (self.resolution - 1)

        # Get integer and fractional parts
        floor = torch.floor(normalized).long()
        frac = normalized - floor

        # Clamp to valid range
        floor = torch.clamp(floor, 0, self.resolution - 2)

        # Get 8 corner features
        corners = []
        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    idx = floor + torch.tensor([dx, dy, dz], device=floor.device)
                    corner_features = self.features[idx[:, 0], idx[:, 1], idx[:, 2]]
                    corners.append(corner_features)

        # Trilinear interpolation
        # Weights for each corner
        wx = torch.stack([1 - frac[:, 0], frac[:, 0]], dim=1)
        wy = torch.stack([1 - frac[:, 1], frac[:, 1]], dim=1)
        wz = torch.stack([1 - frac[:, 2], frac[:, 2]], dim=1)

        interpolated = torch.zeros(
            len(positions), self.feature_dim,
            device=positions.device
        )

        for i, (dx, dy, dz) in enumerate(itertools.product([0,1], repeat=3)):
            weight = wx[:, dx] * wy[:, dy] * wz[:, dz]
            interpolated += weight[:, None] * corners[i]

        return interpolated

    def forward(self, positions, directions):
        # Interpolate features from grid
        features = self.interpolate_features(positions)

        # Decode with small MLP
        decoder_input = torch.cat([features, positions], dim=-1)
        output = self.decoder(decoder_input)

        density = F.relu(output[:, 0:1])
        color = torch.sigmoid(output[:, 1:4])

        return {"density": density, "color": color}
```

### Multi-Resolution Hash Encoding

```python
class HashEncoding(nn.Module):
    """Instant-NGP style hash encoding."""

    def __init__(
        self,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        growth_factor=2.0
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.growth_factor = growth_factor

        # Hash tables for each level
        self.hash_tables = nn.ParameterList([
            nn.Parameter(
                torch.randn(2 ** log2_hashmap_size, n_features_per_level) * 1e-4
            )
            for _ in range(n_levels)
        ])

    def hash_function(self, coords):
        """Spatial hash function."""
        # Simplified: Use prime numbers for hashing
        primes = torch.tensor([1, 2654435761, 805459861], device=coords.device)
        hashed = torch.sum(coords * primes, dim=-1)
        return hashed % (2 ** self.log2_hashmap_size)

    def forward(self, positions):
        """
        Args:
            positions: [N, 3] in range [-1, 1]

        Returns:
            encoded: [N, n_levels * n_features_per_level]
        """
        encoded_list = []

        for level in range(self.n_levels):
            resolution = self.base_resolution * (self.growth_factor ** level)

            # Scale position to current resolution
            scaled_pos = (positions + 1) / 2 * resolution  # [0, resolution]

            # Get voxel corners
            floor_pos = torch.floor(scaled_pos).long()
            frac_pos = scaled_pos - floor_pos

            # Hash each corner
            level_features = []
            for corner in itertools.product([0, 1], repeat=3):
                corner_coords = floor_pos + torch.tensor(
                    corner, device=positions.device
                )

                # Hash coordinates
                hash_idx = self.hash_function(corner_coords)

                # Lookup features
                corner_features = self.hash_tables[level][hash_idx]
                level_features.append(corner_features)

            # Trilinear interpolation
            # (Simplified - full implementation needs proper weight computation)
            interpolated = torch.stack(level_features).mean(dim=0)
            encoded_list.append(interpolated)

        return torch.cat(encoded_list, dim=-1)
```

## Code Walkthrough

Our implementation in `nexus/models/cv/nerf/fast_nerf.py`:

### Key Innovation: Cached Features

```python
# From fast_nerf.py (lines 14-22)

self.fast_layers = fast_config.get("fast_layers", [2, 4, 6])

self.fast_mlp = nn.Sequential(
    nn.Linear(self.hidden_size, self.hidden_size),
    nn.ReLU(),
    nn.Linear(self.hidden_size, self.hidden_size),
    nn.ReLU()
)
```

**Idea**: Add shortcut processing at certain layers for commonly queried regions.

### Adaptive Sampling

```python
# From fast_nerf.py (lines 81-93)

if bounded_positions.size(0) > 0:
    x = self.mlp[0](bounded_positions)
    for i, layer in enumerate(self.mlp[1:], 1):
        x = layer(x)
        if i in self.fast_layers:
            x = self.fast_mlp(x)  # Additional processing

    bounded_density = self.density_head(x)
    bounded_color = self.color_head(x)
```

## Optimization Tricks

### 1. Update Occupancy Grid Periodically

```python
# Every N iterations
if iteration % 1000 == 0:
    occupancy_grid.update(model, threshold=0.01)
```

### 2. Early Ray Termination

```python
# Stop if accumulated opacity exceeds threshold
accumulated_alpha = 0
for sample in samples:
    alpha = compute_alpha(sample)
    accumulated_alpha += alpha * (1 - accumulated_alpha)

    if accumulated_alpha > 0.99:
        break  # Early termination
```

### 3. Coarse-to-Fine Resolution

```python
# Start with low resolution grid, increase gradually
resolution_schedule = {
    0: 64,
    5000: 128,
    10000: 256,
    20000: 512
}
```

### 4. Spherical Harmonics Caching

```python
# Pre-compute SH basis for common directions
directions_uniform = fibonacci_sphere(n=1000)
sh_basis_cache = compute_sh_basis(directions_uniform)

# At inference: Nearest neighbor lookup
def fast_sh_lookup(direction):
    idx = nearest_neighbor(direction, directions_uniform)
    return sh_basis_cache[idx]
```

## Experiments & Results

### Speed Comparison

| Method | Training Time | Rendering Speed | Quality (PSNR) |
|--------|---------------|-----------------|----------------|
| NeRF | 1-2 days | 30 sec/frame | 31.0 |
| **Fast NeRF (Octree)** | **4-6 hours** | **3 sec/frame** | 30.5 |
| **Instant-NGP** | **5 min** | **0.1 sec/frame** | 30.8 |
| **Plenoxels** | **11 min** | **1 sec/frame** | 29.2 |

### Memory Trade-offs

| Method | Memory (Training) | Memory (Inference) |
|--------|-------------------|-------------------|
| NeRF | 2 GB | 5 MB (MLP weights) |
| Feature Grid (256³) | 8 GB | 2 GB |
| Hash Encoding | 4 GB | 100 MB |
| Octree | 6 GB | 500 MB |

## Common Pitfalls

### 1. Stale Occupancy Grid

```python
# Wrong: Never update grid
grid.update(model)  # Only once

# Correct: Update periodically as model learns
if iteration % 1000 == 0:
    grid.update(model)
```

### 2. Hash Collisions

```python
# Monitor hash collisions
collision_rate = count_collisions(hash_table)
if collision_rate > 0.5:
    warnings.warn("High hash collision rate, increase table size")
```

### 3. Over-Aggressive Pruning

```python
# Don't prune too early
if iteration < warmup_iterations:
    threshold = 0.0  # Keep all voxels
else:
    threshold = 0.01  # Start pruning
```

## References

### Key Papers

**Instant-NGP (2022)**:
```bibtex
@article{mueller2022instant,
  title={Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
  author={M{\"u}ller, Thomas and Evans, Alex and Schied, Christoph and Keller, Alexander},
  journal={ACM TOG},
  year={2022}
}
```

**TensoRF (2022)**:
```bibtex
@inproceedings{chen2022tensorf,
  title={TensoRF: Tensorial Radiance Fields},
  author={Chen, Anpei and Xu, Zexiang and Geiger, Andreas and Yu, Jingyi and Su, Hao},
  booktitle={ECCV},
  year={2022}
}
```

**Plenoxels (2022)**:
```bibtex
@inproceedings{fridovich2022plenoxels,
  title={Plenoxels: Radiance Fields without Neural Networks},
  author={Fridovich-Keil, Sara and Yu, Alex and Tancik, Matthew and Chen, Qinhong and Recht, Benjamin and Kanazawa, Angjoo},
  booktitle={CVPR},
  year={2022}
}
```

---

**Next**: See [Gaussian Splatting](./gaussian_splatting.md) for the fastest high-quality alternative.
