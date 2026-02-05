# SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction

## Overview & Motivation

SuGaR (Surface-Aligned Gaussians for Reconstruction) addresses a key limitation of standard 3D Gaussian Splatting: **poor geometric reconstruction**. While Gaussian Splatting excels at novel view synthesis, extracting clean meshes is challenging because Gaussians can float in space without forming coherent surfaces.

### The Geometry Problem

**Standard Gaussian Splatting**:
- Optimizes for rendering quality
- Gaussians can be anywhere (volume filling)
- Extracted meshes are noisy and incomplete

**SuGaR's Solution**:
- Regularize Gaussians to align with surfaces
- Hybrid representation: Mesh + Gaussians
- Clean, editable geometry

## Theoretical Background

### Surface Alignment Constraint

Encourage Gaussians to lie on a surface:

```
For each Gaussian at position μ:
  Find nearest surface point p
  Penalize distance ||μ - p||²
```

**Challenge**: We don't know the surface initially.

### Two-Stage Optimization

**Stage 1: Regularized Gaussian Splatting**
```
Add regularization terms:
L = L_render + λ_flat L_flat + λ_dense L_dense

L_flat: Encourage flat Gaussians (small thickness)
L_dense: Encourage high local density
```

**Stage 2: Mesh Binding**
```
Extract mesh from Gaussians
Bind Gaussians to mesh surface
Refine both jointly
```

### Flatness Regularization

```
L_flat = Σᵢ min(s_i^x, s_i^y, s_i^z)

Penalize smallest scale → Encourages flat ellipsoids
```

### Density Regularization

```
L_dense = Σᵢ -log(Σⱼ∈N(i) α_j)

Encourage high opacity in neighborhood
→ Forms continuous surfaces
```

## Mathematical Formulation

### Surface-Aligned Gaussian Parameterization

Each Gaussian defined by:
```
Position: μ ∈ Surface
Normal: n ∈ S²
Tangent scales: s_tangent ∈ ℝ²
Normal scale: s_normal ∈ ℝ (small)

Covariance: Σ = R(n) S S^T R(n)^T
where R(n): Rotation aligning z-axis to normal
      S: diag(s_tangent_1, s_tangent_2, s_normal)
```

### Mesh Extraction via Poisson Reconstruction

```
Given: Gaussians {μᵢ, nᵢ, αᵢ}

1. Compute oriented points: (μᵢ, nᵢ) weighted by αᵢ
2. Solve Poisson equation: ∇²f = ∇·n
3. Extract isosurface: {x | f(x) = iso_value}
```

### Gaussian-Mesh Binding

```
For each Gaussian i:
  Find closest triangle T on mesh
  Compute barycentric coordinates (u, v, w)

  Position: μᵢ = u·v₁ + v·v₂ + w·v₃ + offset·n_T
  Normal: nᵢ = n_T
```

Gaussians move with mesh during editing.

## Implementation Details

### Regularized Training

```python
def sugar_training_step(
    gaussians,
    viewpoint,
    target_image,
    iteration
):
    """SuGaR training with surface alignment."""

    # Standard rendering loss
    rendered = render_gaussians(gaussians, viewpoint)
    loss_render = F.mse_loss(rendered, target_image)

    # Flatness loss: Penalize smallest scale
    scales = gaussians.get_scaling()  # [N, 3]
    min_scales = torch.min(scales, dim=-1)[0]
    loss_flat = torch.mean(min_scales)

    # Density loss: Encourage local clustering
    positions = gaussians.get_xyz()  # [N, 3]
    opacities = gaussians.get_opacity()  # [N, 1]

    # Compute k-nearest neighbors
    dists, indices = knn(positions, k=16)
    neighbor_opacities = opacities[indices]  # [N, 16]

    # Encourage high neighbor opacity
    loss_dense = -torch.mean(torch.log(
        neighbor_opacities.sum(dim=-1) + 1e-5
    ))

    # Normal consistency loss
    normals = gaussians.compute_normals()
    neighbor_normals = normals[indices]
    loss_normal = torch.mean(1 - F.cosine_similarity(
        normals.unsqueeze(1),
        neighbor_normals,
        dim=-1
    ))

    # Weighting schedule
    if iteration < 7000:
        # Focus on rendering quality early
        w_flat, w_dense, w_normal = 0.01, 0.01, 0.0
    else:
        # Increase regularization
        w_flat, w_dense, w_normal = 0.1, 0.1, 0.05

    loss_total = (
        loss_render +
        w_flat * loss_flat +
        w_dense * loss_dense +
        w_normal * loss_normal
    )

    return loss_total, {
        'render': loss_render.item(),
        'flat': loss_flat.item(),
        'dense': loss_dense.item()
    }
```

### Mesh Extraction

```python
def extract_sugar_mesh(gaussians, resolution=512):
    """Extract mesh from surface-aligned Gaussians."""

    # 1. Get oriented points
    positions = gaussians.get_xyz().cpu().numpy()
    normals = gaussians.compute_normals().cpu().numpy()
    opacities = gaussians.get_opacity().cpu().numpy()

    # Weight by opacity (prune low opacity Gaussians)
    mask = opacities.squeeze() > 0.1
    positions = positions[mask]
    normals = normals[mask]
    opacities = opacities[mask]

    # 2. Poisson surface reconstruction
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=9,
        width=0,
        scale=1.1,
        linear_fit=False
    )

    # 3. Clean mesh (remove low density vertices)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # 4. Simplify mesh
    mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=100000
    )

    return mesh
```

### Gaussian-Mesh Binding

```python
class SuGaRHybridModel:
    """Hybrid representation: Mesh + surface-aligned Gaussians."""

    def __init__(self, gaussians, mesh):
        self.mesh = mesh
        self.gaussians = gaussians

        # Bind each Gaussian to closest triangle
        self.bind_gaussians_to_mesh()

    def bind_gaussians_to_mesh(self):
        """Bind Gaussians to mesh triangles."""
        positions = self.gaussians.get_xyz()

        # For each Gaussian, find closest triangle
        self.triangle_indices = []
        self.barycentric_coords = []
        self.normal_offsets = []

        vertices = torch.tensor(
            np.asarray(self.mesh.vertices),
            device=positions.device
        )
        triangles = torch.tensor(
            np.asarray(self.mesh.triangles),
            device=positions.device
        )

        for pos in positions:
            # Find closest triangle (simplified)
            closest_tri_idx, bary, offset = self.find_closest_triangle(
                pos, vertices, triangles
            )

            self.triangle_indices.append(closest_tri_idx)
            self.barycentric_coords.append(bary)
            self.normal_offsets.append(offset)

        self.triangle_indices = torch.tensor(self.triangle_indices)
        self.barycentric_coords = torch.stack(self.barycentric_coords)
        self.normal_offsets = torch.tensor(self.normal_offsets)

    def update_gaussian_positions(self):
        """Update Gaussian positions based on mesh."""
        vertices = torch.tensor(
            np.asarray(self.mesh.vertices),
            device=self.gaussians.get_xyz().device
        )
        triangles = torch.tensor(
            np.asarray(self.mesh.triangles),
            device=self.gaussians.get_xyz().device
        )

        # Get triangle vertices for each Gaussian
        tri_verts = vertices[triangles[self.triangle_indices]]  # [N, 3, 3]

        # Compute positions from barycentric coordinates
        bary = self.barycentric_coords.unsqueeze(-1)  # [N, 3, 1]
        surface_positions = torch.sum(tri_verts * bary, dim=1)  # [N, 3]

        # Compute triangle normals
        v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
        normals = torch.cross(v1 - v0, v2 - v0)
        normals = F.normalize(normals, dim=-1)

        # Add normal offset
        new_positions = surface_positions + self.normal_offsets[:, None] * normals

        # Update Gaussian positions
        self.gaussians._xyz.data = new_positions
```

## High-Level Intuition

### Why Surface Alignment Matters

**Standard Gaussians**: Like a cloud of fuzzy balls
- Can float anywhere
- No coherent surface
- Hard to edit

**SuGaR**: Like stickers on a balloon
- Aligned to surface
- Forms continuous sheet
- Easy to manipulate by moving balloon (mesh)

### The Hybrid Representation

```
Mesh: Provides structure and editability
  ├─ Vertices define geometry
  └─ Can apply standard 3D operations

Gaussians: Provide appearance details
  ├─ Bound to mesh surface
  ├─ Add texture and fine details
  └─ Render with Gaussian Splatting
```

## Optimization Tricks

### 1. Progressive Regularization

```python
def get_regularization_weights(iteration):
    """Increase regularization gradually."""
    if iteration < 3000:
        return 0.0, 0.0  # No regularization early
    elif iteration < 7000:
        return 0.01, 0.01  # Gentle
    else:
        return 0.1, 0.1  # Strong
```

### 2. Normal Estimation from Neighbors

```python
def compute_normals_from_neighbors(positions, k=16):
    """Estimate normals via PCA of local neighborhoods."""
    _, indices = knn(positions, k=k)

    normals = []
    for i, neighbors in enumerate(indices):
        # Get neighbor positions
        neighbor_pos = positions[neighbors]

        # Center
        centered = neighbor_pos - positions[i]

        # PCA (smallest eigenvector = normal)
        _, _, V = torch.pca_lowrank(centered, q=3)
        normal = V[:, -1]  # Last column (smallest eigenvalue)

        normals.append(normal)

    return torch.stack(normals)
```

### 3. Adaptive Iso-Value

```python
def find_optimal_isovalue(poisson_field, target_surface_area):
    """Find iso-value that gives desired surface area."""
    iso_values = np.linspace(
        poisson_field.min(),
        poisson_field.max(),
        num=20
    )

    best_iso = iso_values[0]
    best_diff = float('inf')

    for iso in iso_values:
        mesh = marching_cubes(poisson_field, iso)
        area = mesh.get_surface_area()

        diff = abs(area - target_surface_area)
        if diff < best_diff:
            best_diff = diff
            best_iso = iso

    return best_iso
```

## Experiments & Results

### Quantitative Comparison

| Method | PSNR ↑ | SSIM ↑ | Mesh Quality | Training Time |
|--------|--------|--------|--------------|---------------|
| 3DGS | 27.2 | 0.815 | Poor | 30 min |
| **SuGaR** | **26.8** | **0.809** | **Good** | **45 min** |
| NeuS | 24.1 | 0.752 | Good | 4 hours |
| VolSDF | 23.8 | 0.741 | Good | 6 hours |

Slight quality trade-off for much better geometry.

### Mesh Quality Metrics

| Method | Chamfer Distance ↓ | Normal Consistency ↑ |
|--------|-------------------|---------------------|
| 3DGS | 0.042 | 0.61 |
| **SuGaR** | **0.018** | **0.87** |
| NeRF | N/A | N/A |

SuGaR produces meshes 2-3x cleaner than standard Gaussian Splatting.

### Editing Capabilities

| Operation | 3DGS | SuGaR |
|-----------|------|-------|
| Mesh deformation | No | **Yes** |
| Texture editing | Hard | **Easy** |
| Boolean operations | No | **Yes** |
| Standard 3D tools | No | **Yes** |

## Common Pitfalls

### 1. Over-Regularization

```python
# Too much regularization → blurry rendering
if psnr < 24.0:
    print("Warning: Over-regularized, reduce weights")
    w_flat *= 0.5
    w_dense *= 0.5
```

### 2. Poor Normal Initialization

```python
# Ensure consistent normal orientation
normals = compute_normals(gaussians)
# Orient toward camera on average
avg_view_dir = compute_average_camera_direction(train_cameras)
flip_mask = (normals @ avg_view_dir) < 0
normals[flip_mask] *= -1
```

### 3. Mesh-Gaussian Misalignment

```python
# Re-bind periodically during refinement
if iteration % 1000 == 0:
    hybrid_model.bind_gaussians_to_mesh()
```

## References

### Primary Paper

```bibtex
@article{guedon2023sugar,
  title={SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering},
  author={Guédon, Antoine and Lepetit, Vincent},
  journal={arXiv preprint arXiv:2311.12775},
  year={2023}
}
```

### Related Work

- **3D Gaussian Splatting** (Kerbl et al., 2023): Base representation
- **NeuS** (Wang et al., 2021): SDF-based reconstruction
- **Poisson Reconstruction** (Kazhdan & Hoppe, 2013): Mesh extraction

---

**Next**: See [GaussianEditor](./gaussian_editor.md) for interactive 3D editing.
