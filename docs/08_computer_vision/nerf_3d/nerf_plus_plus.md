# NeRF++: Modeling Unbounded Scenes

## Overview & Motivation

NeRF++, published in 2020 by Zhang et al., extends NeRF to handle **unbounded 360-degree scenes** such as outdoor environments. The original NeRF struggles with unbounded scenes because:

1. **Infinite extent**: Scene has no natural far bound
2. **Background modeling**: Distant objects (sky, mountains) require different treatment
3. **Sampling inefficiency**: Uniform sampling wastes computation on empty space

### Solution: Inverted Sphere Parameterization

NeRF++ splits the scene into:
- **Foreground**: Bounded region (standard NeRF)
- **Background**: Unbounded region (inverted sphere mapping)

```
If ||x|| ≤ r: Use standard NeRF
If ||x|| > r: Use inverted sphere parameterization
```

This maps infinite 3D space to a bounded domain that can be efficiently sampled.

## Theoretical Background

### Inverted Sphere Parameterization

For points outside radius r, map to bounded space:

```
x' = x / ||x||²

In 4D homogeneous coordinates:
(x, y, z) → (x/r², y/r², z/r², 1/r)

where r = ||x||
```

**Effect**: Points at infinity map to center, nearby background maps to outer shell.

### Foreground-Background Decomposition

**Foreground Model** (||x|| ≤ r_scene):
```
F_fg(x, d) → (σ_fg, c_fg)
Standard NeRF with bounded sampling
```

**Background Model** (||x|| > r_scene):
```
F_bg(x', d) → (σ_bg, c_bg)
where x' = inverted parameterization of x
```

### Volume Rendering Integration

Render each ray in two parts:

```
C_total = C_fg + T_fg · C_bg

where:
C_fg: Color from foreground
T_fg: Transmittance through foreground
C_bg: Color from background
```

## Mathematical Formulation

### Inverted Sphere Mapping

For point x with ||x|| = r > r_scene:

```
Position mapping:
x' = x / (||x|| - r_scene)²

Intuition:
- x at infinity → x' at r_scene
- x at r_scene → x' at infinity
```

### Density Scaling

Density must be scaled to account for space warping:

```
σ'(x') = σ(x) · |det(J)|

where J is Jacobian of transformation
```

For inverted sphere:
```
|det(J)| = (||x|| - r_scene)^(-3)
```

### Composite Rendering

```
C(r) = ∫[0 to r_scene] T_fg(t) σ_fg(t) c_fg(t) dt
     + T_fg(r_scene) ∫[r_scene to ∞] T_bg(t) σ_bg(t) c_bg(t) dt

where:
T_fg(r_scene): Transmittance through foreground
Second integral: Background contribution
```

## Implementation Details

### Inverted Sphere Transform

```python
def inverse_transform_sampling(
    positions: torch.Tensor,  # [N, 3]
    scene_bound: float = 4.0
) -> torch.Tensor:
    """
    Map unbounded positions to bounded space.

    Args:
        positions: 3D points outside scene_bound
        scene_bound: Radius of bounded foreground region

    Returns:
        transformed_positions: Mapped to bounded space
    """
    norm = torch.norm(positions, dim=-1, keepdim=True)  # [N, 1]

    # Normalize to unit sphere
    normalized = positions / norm  # [N, 3]

    # Apply inverted sphere mapping
    # x' = x / (||x|| - r)²
    distance_from_bound = norm - scene_bound
    inv_positions = normalized / (distance_from_bound ** 2)

    return inv_positions
```

### Foreground-Background Network

```python
class NeRFPlusPlusNetwork(nn.Module):
    def __init__(self, scene_bound=4.0):
        super().__init__()
        self.scene_bound = scene_bound

        # Foreground network (standard NeRF)
        self.fg_network = NeRFNetwork()

        # Background network (for unbounded region)
        self.bg_network = NeRFNetwork()

    def forward(
        self,
        positions: torch.Tensor,  # [N, 3]
        directions: torch.Tensor  # [N, 3]
    ) -> Dict[str, torch.Tensor]:
        # Determine which samples are in foreground vs background
        norms = torch.norm(positions, dim=-1)  # [N]
        fg_mask = norms <= self.scene_bound

        # Allocate outputs
        density = torch.zeros(len(positions), 1, device=positions.device)
        color = torch.zeros(len(positions), 3, device=positions.device)

        # Process foreground samples
        if fg_mask.any():
            fg_outputs = self.fg_network(
                positions[fg_mask],
                directions[fg_mask]
            )
            density[fg_mask] = fg_outputs["density"]
            color[fg_mask] = fg_outputs["color"]

        # Process background samples
        bg_mask = ~fg_mask
        if bg_mask.any():
            # Apply inverted sphere mapping
            bg_positions = self.inverse_transform_sampling(
                positions[bg_mask]
            )

            bg_outputs = self.bg_network(
                bg_positions,
                directions[bg_mask]
            )

            density[bg_mask] = bg_outputs["density"]
            color[bg_mask] = bg_outputs["color"]

        return {"density": density, "color": color}
```

### Composite Rendering

```python
def render_nerf_plusplus(
    model: NeRFPlusPlusNetwork,
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    near: float = 0.0,
    far: float = 1e10,  # Effectively infinite
    num_samples_fg: int = 64,
    num_samples_bg: int = 32
) -> Dict[str, torch.Tensor]:
    """Render with foreground-background decomposition."""

    # Sample foreground (0 to scene_bound)
    t_fg = torch.linspace(
        near,
        model.scene_bound,
        num_samples_fg,
        device=ray_origin.device
    )

    # Sample background (scene_bound to far)
    # Use disparity sampling for better distribution
    t_bg = 1.0 / torch.linspace(
        1.0 / model.scene_bound,
        1.0 / far,
        num_samples_bg,
        device=ray_origin.device
    )

    # Combine samples
    t_vals = torch.cat([t_fg, t_bg])
    t_vals, _ = torch.sort(t_vals)

    # Compute sample positions
    positions = ray_origin + ray_direction[:, None] * t_vals[None, :]

    # Evaluate network
    directions = ray_direction.unsqueeze(1).expand(-1, len(t_vals), -1)
    outputs = model(
        positions.reshape(-1, 3),
        directions.reshape(-1, 3)
    )

    # Volume rendering (standard)
    density = outputs["density"].reshape(len(ray_origin), -1, 1)
    color = outputs["color"].reshape(len(ray_origin), -1, 3)

    # ... rest of volume rendering code ...

    return {"rgb": rgb, "depth": depth}
```

## Code Walkthrough

Our implementation in `nexus/models/cv/nerf/nerf_plus_plus.py`:

### Key Components

```python
# From nerf_plus_plus.py (lines 8-31)

class NeRFPlusPlusNetwork(NeRFNetwork):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        nerf_pp_config = config.get("nerf++", {})
        self.scene_bound = nerf_pp_config.get("scene_bound", 4.0)

        # Background networks for unbounded regions
        self.background_density_net = nn.Sequential(
            nn.Linear(self.pos_encoder.get_output_dim(), self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1 + self.hidden_size)
        )

        self.background_color_net = nn.Sequential(
            nn.Linear(self.hidden_size + view_dependent_dim, self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 3),
            nn.Sigmoid()
        )
```

### Inverted Transform

```python
# From nerf_plus_plus.py (lines 32-37)

def _inverse_transform_sampling(self, positions: torch.Tensor) -> torch.Tensor:
    """Convert unbounded positions to bounded for background model."""
    norm = torch.norm(positions, dim=-1, keepdim=True)
    normalized_positions = positions / norm
    inv_positions = normalized_positions / (norm - self.scene_bound)
    return inv_positions
```

**Note**: This is a simplified version. Production code should handle edge cases at the boundary.

## Optimization Tricks

### 1. Disparity Sampling for Background

```python
# Instead of uniform t sampling
t_bg = torch.linspace(scene_bound, far, N)

# Use disparity (1/t) for better distribution
disparity = torch.linspace(1/scene_bound, 1/far, N)
t_bg = 1.0 / disparity

# Concentrates samples near scene_bound where detail matters
```

### 2. Separate Learning Rates

```python
optimizer = torch.optim.Adam([
    {'params': fg_network.parameters(), 'lr': 5e-4},
    {'params': bg_network.parameters(), 'lr': 1e-4}  # Lower for background
])
```

### 3. Progressive Training

```python
# Train foreground first, then add background
if iteration < 10000:
    loss = loss_foreground
else:
    loss = loss_foreground + loss_background
```

## Experiments & Results

### Quantitative Results (Outdoor Scenes)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|--------|--------|---------|
| **NeRF++** | **25.77** | **0.853** | **0.178** |
| NeRF | 22.41 | 0.768 | 0.284 |
| LLFF | 20.12 | 0.692 | 0.341 |

**+3.36 dB improvement** on unbounded scenes.

### Qualitative Improvements

- **Background quality**: Clearer sky, distant mountains
- **No artifacts at infinity**: Smooth transitions
- **360° consistency**: Handles full panoramas

### Scene Bound Selection

| Scene Type | Recommended r_scene |
|------------|---------------------|
| Indoor rooms | 2.0 - 3.0 |
| Outdoor objects | 3.0 - 5.0 |
| Cityscapes | 5.0 - 10.0 |
| Landscapes | 10.0 - 20.0 |

## Common Pitfalls

### 1. Incorrect Scene Bound

```python
# Too small: Background bleeds into foreground
scene_bound = 1.0  # ✗

# Too large: Inefficient foreground sampling
scene_bound = 100.0  # ✗

# Estimate from scene geometry
scene_bound = 1.2 * max_distance_to_object  # ✓
```

### 2. Forgetting Density Scaling

```python
# When using inverted parameterization, scale density
density_bg = density_bg * jacobian_determinant
```

### 3. Discontinuity at Boundary

```python
# Ensure smooth transition at r_scene
# Use soft boundary with sigmoid
weight = torch.sigmoid((norm - scene_bound) * 10)
output = (1 - weight) * fg_output + weight * bg_output
```

## References

### Primary Paper

```bibtex
@article{zhang2020nerf++,
  title={NeRF++: Analyzing and Improving Neural Radiance Fields},
  author={Zhang, Kai and Riegler, Gernot and Snavely, Noah and Koltun, Vladlen},
  journal={arXiv preprint arXiv:2010.07492},
  year={2020}
}
```

### Related Work

- **Mip-NeRF 360** (2022): Combines IPE with unbounded scenes
- **Block-NeRF** (2022): City-scale NeRF with spatial decomposition
- **BungeeNeRF** (2024): Multi-scale unbounded scenes

---

**Next**: Try [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) for the best unbounded scene quality.
