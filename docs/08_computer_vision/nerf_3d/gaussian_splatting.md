# 3D Gaussian Splatting

## Overview & Motivation

3D Gaussian Splatting, introduced at SIGGRAPH 2023 by Kerbl et al., represents a paradigm shift in neural 3D scene representation. Unlike NeRF's implicit representation with MLPs, Gaussian Splatting uses explicit 3D Gaussian primitives that can be rendered in real-time through differentiable rasterization.

### The Problem with NeRF

While NeRF achieved impressive quality, it had critical limitations:
- **Slow Rendering**: 10-30 seconds per frame (need to query MLP thousands of times per pixel)
- **Slow Training**: Hours to days for convergence
- **Hard to Edit**: Implicit representation makes manipulation difficult
- **Memory During Inference**: Must load entire MLP for rendering

### Gaussian Splatting's Solution

Represent scenes as collections of 3D Gaussians with learned parameters:

```
Scene = { (μᵢ, Σᵢ, cᵢ, αᵢ) | i = 1...N }

μᵢ ∈ ℝ³        # Position (mean)
Σᵢ ∈ ℝ³ˣ³     # Covariance (shape and orientation)
cᵢ ∈ ℝᴷ       # Color (spherical harmonics)
αᵢ ∈ [0,1]    # Opacity
```

**Key Innovations**:
1. **Explicit Representation**: Direct optimization of Gaussian parameters
2. **Differentiable Rasterization**: GPU-friendly tile-based rendering
3. **Adaptive Density Control**: Dynamic splitting and pruning of Gaussians
4. **Real-Time Performance**: 60+ FPS rendering with comparable quality to NeRF

## Theoretical Background

### 3D Gaussian Primitives

A 3D Gaussian is defined by:

```
G(x) = exp(-½(x - μ)ᵀ Σ⁻¹ (x - μ))

where:
μ: 3D center position
Σ: 3×3 covariance matrix (defines shape and orientation)
```

### Covariance Representation

To ensure positive semi-definite covariance, parameterize as:

```
Σ = R S Sᵀ Rᵀ

where:
R ∈ SO(3):  Rotation matrix (from quaternion q)
S ∈ ℝ³:     Scaling vector (diagonal matrix)
```

This gives us 7 parameters: 4 for quaternion + 3 for scale.

**Why this parameterization?**
- Guarantees valid covariance (positive semi-definite)
- Efficient gradient updates
- Intuitive controls (separate rotation and scale)

### 2D Projection for Rendering

Project 3D Gaussian to 2D screen space:

```
Σ' = J W Σ Wᵀ Jᵀ

where:
W: World-to-camera transformation
J: Jacobian of projective transformation
Σ': 2D covariance in screen space
```

The projected 2D Gaussian:

```
G'(x) = exp(-½(x - μ')ᵀ (Σ')⁻¹ (x - μ'))

μ': 2D projected center
Σ': 2×2 covariance in screen space
```

### Alpha Blending for Rendering

Given N Gaussians overlapping at pixel location x, sorted by depth:

```
C(x) = Σᵢ cᵢ αᵢ ∏ⱼ₌₁ⁱ⁻¹ (1 - αⱼ)

where:
cᵢ: Color of Gaussian i
αᵢ = α̂ᵢ · exp(-½(x - μᵢ')ᵀ (Σᵢ')⁻¹ (x - μᵢ'))
α̂ᵢ: Learned opacity parameter
```

This is exactly the volume rendering equation, but computed in 2D screen space.

### Tile-Based Rasterization

For efficiency, divide screen into 16×16 tiles:

1. **Frustum Culling**: Remove Gaussians outside view
2. **Tile Assignment**: Assign each Gaussian to overlapping tiles
3. **Per-Tile Sorting**: Sort Gaussians by depth within each tile
4. **Parallel Rendering**: Each tile rendered independently

**Complexity**: O(N log N) for sorting, O(N) for rendering

## Mathematical Formulation

### Complete Scene Representation

```
Scene Parameters Θ = {θᵢ}ᵢ₌₁ᴺ

where θᵢ = (μᵢ, qᵢ, sᵢ, αᵢ, {cᵢₗₘ})

μᵢ ∈ ℝ³           # Position
qᵢ ∈ S³           # Rotation quaternion (normalized)
sᵢ ∈ ℝ³₊          # Scale (positive)
αᵢ ∈ [0,1]        # Opacity
cᵢₗₘ ∈ ℝ          # Spherical harmonic coefficients
```

### Spherical Harmonics for Color

Instead of RGB, use spherical harmonics for view-dependent color:

```
c(d) = Σₗ₌₀ᴸ Σₘ₌₋ₗˡ cₗₘ Yₗₘ(d)

where:
d: View direction
Yₗₘ: Spherical harmonic basis functions
L: Maximum degree (typically 3)
```

**Benefits**:
- Continuous view-dependent effects
- Compact representation (16 coefficients for L=3)
- Fast evaluation (closed-form)

### Rendering Equation

For a pixel at position p with camera ray direction d:

```
C(p) = Σᵢ∈Vᵢsᵢble Tᵢ αᵢ cᵢ(d)

where:
Tᵢ = ∏ⱼ₌₁ⁱ⁻¹ (1 - αⱼ)  # Transmittance
αᵢ = α̂ᵢ · exp(-½(p - μᵢ')ᵀ (Σᵢ')⁻¹ (p - μᵢ'))
```

### Loss Function

```
L = Lphoto + λ₁ Lssim

Lphoto = ||C(p) - Ĉ(p)||²₂  # Photometric loss

Lssim = 1 - SSIM(C, Ĉ)     # Structural similarity
```

No explicit regularization needed due to adaptive density control.

## High-Level Intuition

### What are Gaussians?

Think of each Gaussian as a **soft, fuzzy ellipsoid**:
- **Center (μ)**: Where it's located in 3D
- **Rotation (R)**: Which way it's oriented
- **Scale (s)**: How stretched it is in each direction
- **Opacity (α)**: How transparent
- **Color (c)**: What color it appears from different angles

### Why Gaussians?

1. **Closed-Form Projection**: 3D Gaussian projects to 2D Gaussian (exact)
2. **Differentiable**: All operations have gradients
3. **Efficient**: Can be rasterized like traditional graphics
4. **Flexible**: Can represent surfaces, volumes, or anything in between

### The Rendering Process

```
1. For each Gaussian:
   - Project center to screen
   - Compute 2D covariance
   - Determine affected tiles

2. For each tile:
   - Sort Gaussians by depth
   - For each pixel:
     - Blend Gaussians front-to-back
     - Stop when accumulated opacity ≈ 1

3. Output rendered image
```

**Speed**: This is essentially traditional graphics rasterization, which GPUs excel at.

### Adaptive Density Control

The magic of Gaussian Splatting is **growing and pruning** Gaussians during training:

**Densification** (add Gaussians):
- **Split**: Large Gaussians with high gradients → split into 2 smaller ones
- **Clone**: Small Gaussians with high gradients → duplicate

**Pruning** (remove Gaussians):
- Low opacity (α < ε) → remove
- Too large after projection → remove

This allows the representation to adapt to scene complexity.

## Implementation Details

### Initialization

Start from Structure-from-Motion (SfM) point cloud:

```python
# Initialize from COLMAP/SfM points
positions = sfm_points.xyz           # [N, 3]
colors = sfm_points.rgb              # [N, 3]

# Initialize covariance as isotropic
mean_dist = compute_knn_distance(positions, k=3)
scales = log(mean_dist * ones(3))    # [N, 3]
rotations = [1, 0, 0, 0]             # Identity quaternion [N, 4]
opacities = inverse_sigmoid(0.1)      # Start semi-transparent
```

### Adaptive Density Control

**Every 100 iterations**:

```python
def densify_and_prune(gaussians, threshold_grad=0.0002):
    # Compute gradients of screen-space positions
    gradients = compute_2d_gradients(gaussians)

    # Split large Gaussians with high gradient
    large_mask = (gradients > threshold_grad) & (scales > scene_extent * 0.05)
    split_gaussians(gaussians[large_mask])

    # Clone small Gaussians with high gradient
    small_mask = (gradients > threshold_grad) & (scales <= scene_extent * 0.05)
    clone_gaussians(gaussians[small_mask])

    # Prune nearly transparent or too large Gaussians
    prune_mask = (opacities < 0.005) | (scales > scene_extent * 0.5)
    remove_gaussians(gaussians[prune_mask])
```

**Split Operation**:
```python
def split_gaussian(μ, Σ, c, α):
    # Sample two new positions from original Gaussian
    sample = sample_from_gaussian(μ, Σ)
    μ_new1 = μ + sample
    μ_new2 = μ - sample

    # Scale down covariance
    Σ_new = Σ / 1.6

    return [(μ_new1, Σ_new, c, α), (μ_new2, Σ_new, c, α)]
```

### Differentiable Rasterization

Custom CUDA kernels for forward and backward passes:

**Forward Pass**:
```cpp
// Pseudocode for tile-based rasterization
for each tile in parallel:
    // Load Gaussians affecting this tile
    shared_gaussians = load_tile_gaussians(tile_id)

    for each pixel in tile:
        accumulated_color = 0
        accumulated_alpha = 0

        // Front-to-back blending
        for gaussian in shared_gaussians:
            alpha = compute_alpha(pixel, gaussian)
            weight = alpha * (1 - accumulated_alpha)

            accumulated_color += weight * gaussian.color
            accumulated_alpha += weight

            if accumulated_alpha > 0.99:
                break  // Early stopping
```

**Backward Pass**:
- Automatic differentiation through entire pipeline
- Gradients w.r.t. all Gaussian parameters
- Efficient memory management with gradient checkpointing

### Training Hyperparameters

```yaml
# Optimization
optimizer: Adam
learning_rate_position: 1.6e-4
learning_rate_rotation: 1.0e-3
learning_rate_scale: 5.0e-3
learning_rate_opacity: 5.0e-2
learning_rate_sh: 2.5e-3

# Learning rate scheduling
exponential_decay:
  start: 1.0
  end: 0.01
  max_steps: 30000

# Loss weights
lambda_ssim: 0.2  # Weight for SSIM loss

# Densification
densify_grad_threshold: 0.0002
densify_every: 100
densify_until: 15000

# Pruning
opacity_cull_threshold: 0.005
prune_every: 100

# Training
iterations: 30000
time: 10-30 minutes on single GPU
```

### Spherical Harmonics Implementation

```python
def eval_sh(degree, sh_coeffs, directions):
    """
    Evaluate spherical harmonics.

    Args:
        degree: Maximum SH degree (0-3)
        sh_coeffs: [N, (degree+1)^2, 3] SH coefficients
        directions: [N, 3] View directions

    Returns:
        colors: [N, 3] RGB colors
    """
    result = SH_C0 * sh_coeffs[:, 0]  # L=0

    if degree > 0:
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]

        # L=1
        result += SH_C1 * sh_coeffs[:, 1] * y
        result += SH_C1 * sh_coeffs[:, 2] * z
        result += SH_C1 * sh_coeffs[:, 3] * x

    if degree > 1:
        # L=2 (5 coefficients)
        xx, yy, zz = x*x, y*y, z*z
        xy, yz, xz = x*y, y*z, x*z

        result += SH_C2[0] * sh_coeffs[:, 4] * xy
        result += SH_C2[1] * sh_coeffs[:, 5] * yz
        result += SH_C2[2] * sh_coeffs[:, 6] * (2*zz - xx - yy)
        result += SH_C2[3] * sh_coeffs[:, 7] * xz
        result += SH_C2[4] * sh_coeffs[:, 8] * (xx - yy)

    if degree > 2:
        # L=3 (7 coefficients)
        # Similar pattern...
        pass

    return result + 0.5  # Shift to [0, 1] range
```

## Code Walkthrough

Since Gaussian Splatting requires custom CUDA kernels, our Nexus implementation would focus on the high-level structure. Below is a conceptual implementation:

### Core Gaussian Class

```python
class GaussianModel:
    def __init__(self, sh_degree=3):
        self.sh_degree = sh_degree
        self.max_sh_degree = sh_degree

        # Learnable parameters
        self._xyz = torch.empty(0)         # [N, 3] Positions
        self._rotation = torch.empty(0)    # [N, 4] Quaternions
        self._scaling = torch.empty(0)     # [N, 3] Scales
        self._opacity = torch.empty(0)     # [N, 1] Opacities
        self._features_dc = torch.empty(0) # [N, 1, 3] SH degree 0
        self._features_rest = torch.empty(0) # [N, (sh_degree+1)^2-1, 3]

    def init_from_pcd(self, points, colors):
        """Initialize from point cloud."""
        N = points.shape[0]

        self._xyz = nn.Parameter(points.clone())
        self._rotation = nn.Parameter(torch.zeros(N, 4))
        self._rotation[:, 0] = 1.0  # Identity quaternion

        # Initialize scales based on kNN distances
        dists = compute_knn_distances(points, k=3)
        self._scaling = nn.Parameter(torch.log(dists.repeat(1, 3)))

        self._opacity = nn.Parameter(
            inverse_sigmoid(0.1 * torch.ones(N, 1))
        )

        # Initialize SH coefficients from RGB
        fused_color = RGB2SH(colors)
        features = torch.zeros((N, (self.sh_degree + 1) ** 2, 3))
        features[:, 0] = fused_color
        self._features_dc = nn.Parameter(features[:, :1])
        self._features_rest = nn.Parameter(features[:, 1:])

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_rotation(self):
        return F.normalize(self._rotation, dim=-1)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def get_covariance(self):
        """Compute 3D covariance from rotation and scaling."""
        # Build rotation matrix from quaternion
        R = build_rotation(self.get_rotation)  # [N, 3, 3]

        # Build scaling matrix
        S = torch.zeros((len(self._scaling), 3, 3), device=self._xyz.device)
        S[:, 0, 0] = self.get_scaling[:, 0]
        S[:, 1, 1] = self.get_scaling[:, 1]
        S[:, 2, 2] = self.get_scaling[:, 2]

        # Σ = R S S^T R^T
        RS = torch.bmm(R, S)
        covariance = torch.bmm(RS, RS.transpose(1, 2))

        return covariance

    def densify_and_split(self, grads, grad_threshold, scene_extent):
        """Split large Gaussians with high gradient."""
        selected_mask = torch.norm(grads, dim=-1) >= grad_threshold
        selected_mask &= torch.max(self.get_scaling, dim=1).values > scene_extent * 0.05

        # Sample split positions
        stds = self.get_scaling[selected_mask]
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.get_rotation[selected_mask])

        # Create two new Gaussians per selected
        new_xyz_1 = self._xyz[selected_mask] + torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        new_xyz_2 = self._xyz[selected_mask] - torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)

        new_scaling = self._scaling[selected_mask] - torch.log(torch.tensor(1.6))
        new_rotation = self._rotation[selected_mask]
        new_opacity = self._opacity[selected_mask]
        new_features_dc = self._features_dc[selected_mask]
        new_features_rest = self._features_rest[selected_mask]

        # Concatenate new Gaussians
        self._xyz = nn.Parameter(torch.cat([
            self._xyz[~selected_mask],
            new_xyz_1, new_xyz_2
        ]))
        # Similar for other parameters...

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Clone small Gaussians with high gradient."""
        selected_mask = torch.norm(grads, dim=-1) >= grad_threshold
        selected_mask &= torch.max(self.get_scaling, dim=1).values <= scene_extent * 0.05

        # Simply duplicate selected Gaussians
        new_xyz = self._xyz[selected_mask]
        # Concatenate with existing...

    def prune(self, min_opacity, max_screen_size):
        """Remove low opacity or too large Gaussians."""
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # Additional pruning criteria...

        # Keep only non-pruned
        self._xyz = nn.Parameter(self._xyz[~prune_mask])
        # Similar for other parameters...
```

### Rendering Loop

```python
def train_step(gaussians, viewpoint_camera, image_gt):
    """Single training step."""

    # Render
    rendered_image = render(gaussians, viewpoint_camera)

    # Compute loss
    loss_l1 = F.l1_loss(rendered_image, image_gt)
    loss_ssim = 1.0 - ssim(rendered_image, image_gt)
    loss = (1.0 - 0.2) * loss_l1 + 0.2 * loss_ssim

    # Backward
    loss.backward()

    # Densification step (every 100 iterations)
    if iteration % 100 == 0 and iteration < 15000:
        # Compute gradients of 2D positions
        grad_2d = compute_2d_position_gradients(gaussians)

        gaussians.densify_and_split(grad_2d, 0.0002, scene_extent)
        gaussians.densify_and_clone(grad_2d, 0.0002, scene_extent)
        gaussians.prune(0.005, max_screen_size)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()
```

## Optimization Tricks

### 1. Adaptive Learning Rates

Different parameters need different learning rates:

```python
optimizer = torch.optim.Adam([
    {'params': [gaussians._xyz], 'lr': 1.6e-4, 'name': 'xyz'},
    {'params': [gaussians._rotation], 'lr': 1.0e-3, 'name': 'rotation'},
    {'params': [gaussians._scaling], 'lr': 5.0e-3, 'name': 'scaling'},
    {'params': [gaussians._opacity], 'lr': 5.0e-2, 'name': 'opacity'},
    {'params': [gaussians._features_dc], 'lr': 2.5e-3, 'name': 'f_dc'},
    {'params': [gaussians._features_rest], 'lr': 2.5e-3 / 20, 'name': 'f_rest'},
])
```

### 2. Exponential Learning Rate Decay

```python
def get_expon_lr_func(
    lr_init, lr_final, max_steps,
    lr_delay_steps=0, lr_delay_mult=1.0
):
    def helper(step):
        if step < 0 or lr_init == lr_final:
            return lr_init
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    return helper
```

### 3. Opacity Reset

Periodically reset opacity to prevent collapse:

```python
if iteration % 3000 == 0:
    gaussians._opacity = nn.Parameter(
        torch.clamp(gaussians._opacity, max=inverse_sigmoid(0.01))
    )
```

### 4. Frustum Culling

Only render Gaussians visible to camera:

```python
def frustum_culling(gaussians, viewpoint):
    # Transform to camera space
    xyz_cam = transform_points(gaussians.get_xyz, viewpoint.world_to_cam)

    # Check if in frustum
    visible = (
        (xyz_cam[:, 2] > near) &
        (xyz_cam[:, 2] < far) &
        (xyz_cam[:, 0].abs() < xyz_cam[:, 2] * tan_fov_x) &
        (xyz_cam[:, 1].abs() < xyz_cam[:, 2] * tan_fov_y)
    )

    return visible
```

### 5. Tile-Based Memory Management

Process tiles in batches to fit GPU memory:

```python
def render_tiles_batched(tiles, gaussians, batch_size=8):
    outputs = []
    for i in range(0, len(tiles), batch_size):
        batch = tiles[i:i+batch_size]
        output = render_tile_batch(batch, gaussians)
        outputs.append(output)
    return torch.cat(outputs)
```

## Experiments & Results

### Quantitative Results (Mip-NeRF 360 Dataset)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FPS ↑ | Training Time |
|--------|--------|--------|---------|-------|---------------|
| **3DGS (Ours)** | **27.21** | **0.815** | **0.214** | **60+** | **30 min** |
| Mip-NeRF 360 | 27.69 | **0.792** | **0.237** | 0.1 | 48 hours |
| Instant-NGP | 25.59 | 0.694 | 0.344 | 10 | 5 min |
| Plenoxels | 23.08 | 0.626 | 0.463 | 1 | 11 min |

### Speed Comparison

| Resolution | NeRF | Instant-NGP | 3DGS |
|------------|------|-------------|------|
| 800×800 | 30 sec | 0.1 sec | 0.016 sec (60 FPS) |
| 1920×1080 | N/A | 0.3 sec | 0.033 sec (30 FPS) |

### Memory Footprint

- **Training**: 4-8 GB VRAM
- **Inference**: 100-500 MB (scene-dependent)
- **Number of Gaussians**: 100k - 5M (adaptive)

### Qualitative Strengths

1. **Real-time rendering** enables interactive applications
2. **Sharp details** comparable to or better than NeRF
3. **Fast training** allows rapid iteration
4. **Explicit representation** enables editing

### Limitations

1. **Memory scales with complexity**: Dense scenes need millions of Gaussians
2. **Artifacts on very shiny surfaces**: SH limited view-dependence
3. **Requires good initialization**: SfM point cloud quality matters
4. **Large file sizes**: Millions of Gaussians vs single MLP

## Common Pitfalls

### 1. Poor SfM Initialization

```python
# Problem: Sparse or inaccurate SfM reconstruction
# Solution: Use dense reconstruction or manual initialization

# Check SfM quality
if len(sfm_points) < 1000:
    print("Warning: Very sparse initialization")
    # Densify by random sampling in scene bounds
    additional_points = sample_random_points(scene_bounds, n=10000)
    sfm_points = torch.cat([sfm_points, additional_points])
```

### 2. Incorrect Quaternion Normalization

```python
# Wrong: Unnormalized quaternions lead to invalid rotations
rotation = self._rotation

# Correct: Always normalize
rotation = F.normalize(self._rotation, dim=-1)
```

### 3. Exploding Scales

```python
# Problem: Scales can grow unbounded during optimization
# Solution: Clamp scales periodically

if iteration % 100 == 0:
    with torch.no_grad():
        self._scaling = nn.Parameter(
            torch.clamp(self._scaling, max=np.log(scene_extent * 0.5))
        )
```

### 4. Degenerate Covariance Matrices

```python
# Add small epsilon to diagonal for numerical stability
def get_covariance(self):
    covariance = self.compute_covariance()
    # Add small value to prevent singular matrices
    eye = torch.eye(3, device=covariance.device)
    covariance = covariance + 1e-7 * eye[None, :, :]
    return covariance
```

### 5. Incorrect Alpha Blending Order

```python
# Must sort by depth BEFORE blending
depths = compute_depths(gaussians, viewpoint)
sorted_idx = torch.argsort(depths)

# Blend in front-to-back order
for idx in sorted_idx:
    alpha = compute_alpha(pixel, gaussians[idx])
    color += alpha * transmittance * gaussians[idx].color
    transmittance *= (1 - alpha)
```

### 6. Not Stopping Gradient Accumulation

```python
# When accumulated opacity reaches 1, stop
if transmittance < 0.01:  # Effectively opaque
    break  # Stop blending
```

## References

### Primary Paper

```bibtex
@article{kerbl20233d,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal={ACM Transactions on Graphics},
  volume={42},
  number={4},
  year={2023},
  publisher={ACM}
}
```

### Implementation Resources

- **Official Code**: https://github.com/graphdeco-inria/gaussian-splatting
- **Project Page**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **Unofficial PyTorch**: https://github.com/ashawkey/diff-gaussian-rasterization

### Follow-up Methods

- **SuGaR**: Surface-Aligned Gaussians
- **GaussianEditor**: Interactive editing framework
- **DreamGaussian**: Text-to-3D with Gaussians
- **Scaffold-GS**: Improved regularization
- **Mip-Splatting**: Anti-aliased rendering

### Related Techniques

- **Point-Based Rendering** (Classical graphics)
- **Pulsar** (Facebook: Differentiable point-based rendering)
- **Neural Point-Based Graphics** (NPBG)

---

**Next Steps**:
- See [SuGaR](./sugar.md) for better geometry reconstruction
- Try [GaussianEditor](./gaussian_editor.md) for editing capabilities
- Explore [DreamGaussian](./dream_gaussian.md) for generative 3D
