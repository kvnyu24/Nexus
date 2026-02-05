# LRM: Large Reconstruction Model for Single Image to 3D

## Overview & Motivation

LRM (Large Reconstruction Model) achieves **5-second single-image to 3D reconstruction** using a large-scale transformer trained on massive 3D datasets. Unlike optimization-based methods (NeRF, Gaussian Splatting), LRM performs feed-forward inference without test-time optimization.

### The Single-Image 3D Problem

**Challenge**: Reconstruct full 3D from one image
- Extreme ill-posed problem (infinite solutions)
- Need strong priors about 3D geometry and appearance
- Traditional methods: Slow optimization (minutes to hours)

### LRM's Approach

1. **Large-Scale Training**: Train on millions of 3D objects (Objaverse)
2. **Transformer Architecture**: Learn powerful 3D priors
3. **Triplane Representation**: Efficient 3D output
4. **Feed-Forward Inference**: Single forward pass → 3D

**Result**: 5 seconds from image to renderable 3D

## Theoretical Background

### Triplane Representation

Represent 3D scene as three 2D feature planes:

```
Triplane: (P_xy, P_xz, P_yz)
Each plane: H × W × C feature map

For query point (x, y, z):
  f_xy = bilinear_sample(P_xy, (x, y))
  f_xz = bilinear_sample(P_xz, (x, z))
  f_yz = bilinear_sample(P_yz, (y, z))

  combined = f_xy + f_xz + f_yz  # or concat
```

**Benefits**:
- Compact: O(HWC) instead of O(H³C)
- Efficient queries: O(1) bilinear interpolation
- Differentiable: Can be optimized

### Transformer-Based Reconstruction

```
Input: Single RGB image I
Output: Triplane features T = (P_xy, P_xz, P_yz)

Architecture:
I → DINO features → Transformer → Triplane decoder → T
```

### Volume Rendering from Triplane

```
For ray r:
  Sample points {p_i}
  Query features {f_i} from triplane
  Decode to (σ_i, c_i) via small MLP
  Volume render: C = Σ T_i α_i c_i
```

## Mathematical Formulation

### Triplane Feature Query

For point p = (x, y, z):

```
Project to each plane:
  p_xy = (x, y)
  p_xz = (x, z)
  p_yz = (y, z)

Sample features:
  f_xy = BilinearSample(P_xy, p_xy)
  f_xz = BilinearSample(P_xz, p_xz)
  f_yz = BilinearSample(P_yz, p_yz)

Aggregate (sum):
  f(p) = f_xy + f_xz + f_yz
```

### Architecture Details

```
Input Image: I ∈ ℝ^(H×W×3)

1. Feature Extraction:
   F = DINO_encoder(I) ∈ ℝ^(N×D)  # N patches, D dims

2. Transformer Processing:
   F' = TransformerBlocks(F) ∈ ℝ^(N×D)

3. Triplane Decoding:
   P_xy = Upsample(Linear(F')) ∈ ℝ^(H_p×W_p×C_p)
   P_xz = Upsample(Linear(F')) ∈ ℝ^(H_p×W_p×C_p)
   P_yz = Upsample(Linear(F')) ∈ ℝ^(H_p×W_p×C_p)

4. Volume Rendering:
   For point p:
     f = query_triplane(p, (P_xy, P_xz, P_yz))
     (σ, c) = decoder_MLP(f)
```

### Training Loss

```
L = L_rgb + λ_mask L_mask + λ_depth L_depth

L_rgb: Photometric loss on rendered images
L_mask: Silhouette loss
L_depth: Depth consistency (if available)
```

## Implementation Details

### Triplane Implementation

```python
class Triplane(nn.Module):
    """Triplane 3D representation."""

    def __init__(self, resolution=256, channels=32):
        super().__init__()
        self.resolution = resolution
        self.channels = channels

        # Three learnable planes
        self.plane_xy = nn.Parameter(
            torch.randn(1, channels, resolution, resolution) * 0.01
        )
        self.plane_xz = nn.Parameter(
            torch.randn(1, channels, resolution, resolution) * 0.01
        )
        self.plane_yz = nn.Parameter(
            torch.randn(1, channels, resolution, resolution) * 0.01
        )

    def query(self, points):
        """
        Query triplane features at 3D points.

        Args:
            points: [B, N, 3] in range [-1, 1]

        Returns:
            features: [B, N, channels]
        """
        # Normalize points to [-1, 1] for grid_sample
        # points should already be in [-1, 1]

        # Project to each plane
        xy = points[..., [0, 1]]  # [B, N, 2]
        xz = points[..., [0, 2]]
        yz = points[..., [1, 2]]

        # Sample from planes using grid_sample
        # grid_sample expects [B, H, W, 2]
        f_xy = F.grid_sample(
            self.plane_xy,
            xy.unsqueeze(1),  # [B, 1, N, 2]
            align_corners=True,
            mode='bilinear'
        ).squeeze(2)  # [B, C, N]

        f_xz = F.grid_sample(
            self.plane_xz,
            xz.unsqueeze(1),
            align_corners=True,
            mode='bilinear'
        ).squeeze(2)

        f_yz = F.grid_sample(
            self.plane_yz,
            yz.unsqueeze(1),
            align_corners=True,
            mode='bilinear'
        ).squeeze(2)

        # Aggregate features (sum)
        features = f_xy + f_xz + f_yz  # [B, C, N]

        # Transpose to [B, N, C]
        features = features.transpose(1, 2)

        return features
```

### LRM Architecture

```python
class LRM(nn.Module):
    """Large Reconstruction Model."""

    def __init__(
        self,
        image_size=256,
        triplane_resolution=256,
        triplane_channels=32,
        transformer_dim=768,
        transformer_layers=12
    ):
        super().__init__()

        # Image encoder (DINO)
        self.image_encoder = torch.hub.load(
            'facebookresearch/dino:main',
            'dino_vitb16'
        )

        # Transformer for 3D reasoning
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=12,
                dim_feedforward=transformer_dim * 4
            ),
            num_layers=transformer_layers
        )

        # Triplane decoder
        self.triplane_decoder = nn.ModuleDict({
            'xy': nn.Sequential(
                nn.Linear(transformer_dim, triplane_channels * 64 * 64),
                nn.Unflatten(1, (triplane_channels, 64, 64)),
                nn.Upsample(size=triplane_resolution, mode='bilinear')
            ),
            'xz': nn.Sequential(
                nn.Linear(transformer_dim, triplane_channels * 64 * 64),
                nn.Unflatten(1, (triplane_channels, 64, 64)),
                nn.Upsample(size=triplane_resolution, mode='bilinear')
            ),
            'yz': nn.Sequential(
                nn.Linear(transformer_dim, triplane_channels * 64 * 64),
                nn.Unflatten(1, (triplane_channels, 64, 64)),
                nn.Upsample(size=triplane_resolution, mode='bilinear')
            )
        })

        # Small MLP to decode triplane features
        self.feature_decoder = nn.Sequential(
            nn.Linear(triplane_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # σ + RGB
        )

    def forward(self, image):
        """
        Args:
            image: [B, 3, H, W]

        Returns:
            triplane: Triplane object
        """
        # Extract image features
        with torch.no_grad():
            features = self.image_encoder(image)  # [B, N_patches, D]

        # Process with transformer
        features = self.transformer(features)  # [B, N, D]

        # Global pooling
        global_feature = features.mean(dim=1)  # [B, D]

        # Generate triplane
        plane_xy = self.triplane_decoder['xy'](global_feature)
        plane_xz = self.triplane_decoder['xz'](global_feature)
        plane_yz = self.triplane_decoder['yz'](global_feature)

        triplane = Triplane.from_tensors(plane_xy, plane_xz, plane_yz)

        return triplane

    def render(self, triplane, camera, num_samples=64):
        """Render from triplane."""
        # Generate rays
        rays_o, rays_d = camera.get_rays()

        # Sample points
        t_vals = torch.linspace(0, 1, num_samples, device=rays_o.device)
        z_vals = camera.near * (1 - t_vals) + camera.far * t_vals

        points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        # Flatten for querying
        B, H, W, N, _ = points.shape
        points_flat = points.reshape(B, -1, 3)

        # Query triplane
        features = triplane.query(points_flat)  # [B, H*W*N, C]

        # Decode to density and color
        output = self.feature_decoder(features)
        density = F.relu(output[..., 0:1])
        color = torch.sigmoid(output[..., 1:4])

        # Reshape
        density = density.reshape(B, H, W, N, 1)
        color = color.reshape(B, H, W, N, 3)

        # Volume rendering
        delta_z = z_vals[..., 1:] - z_vals[..., :-1]
        delta_z = torch.cat([
            delta_z,
            torch.full_like(delta_z[..., :1], 1e10)
        ], dim=-1)

        alpha = 1 - torch.exp(-density[..., 0] * delta_z)
        transmittance = torch.cumprod(
            torch.cat([
                torch.ones_like(alpha[..., :1]),
                1 - alpha + 1e-10
            ], dim=-1),
            dim=-1
        )[..., :-1]

        weights = alpha * transmittance
        rgb = (weights[..., None] * color).sum(dim=-2)

        return rgb
```

### Training Loop

```python
def train_lrm(model, dataset, num_epochs=100):
    """Train LRM on Objaverse dataset."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for batch in dataset:
            # Batch: {input_image, target_views, target_images}
            input_image = batch['input_image']  # [B, 3, H, W]
            target_views = batch['target_cameras']  # List of cameras
            target_images = batch['target_images']  # [B, N_views, 3, H, W]

            # Forward: Reconstruct triplane from single image
            triplane = model(input_image)

            # Render from novel views
            loss = 0
            for i, camera in enumerate(target_views):
                rendered = model.render(triplane, camera)
                target = target_images[:, i]

                loss += F.mse_loss(rendered, target)

            loss = loss / len(target_views)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
```

## High-Level Intuition

### Why LRM is Fast

**Traditional (NeRF, 3DGS)**:
```
Image → Optimize 3D representation (1000+ iterations)
       → Takes minutes
```

**LRM**:
```
Image → Pre-trained model → 3D representation (1 forward pass)
       → Takes 5 seconds
```

The model has "learned" 3D priors from millions of examples.

### The Triplane Advantage

```
3D Volume (256³): 16.7M voxels
Triplane (3×256²): 196k "voxels"

→ 85x memory reduction
→ Still captures 3D structure
```

## Optimization Tricks

### 1. Progressive Training

```python
# Start with low resolution, increase gradually
resolution_schedule = {
    0: 64,
    50000: 128,
    100000: 256
}
```

### 2. Mixed Precision Training

```python
# Essential for large models
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    triplane = model(image)
    loss = compute_loss(triplane, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Multi-View Consistency

```python
# Render from multiple views simultaneously
def multi_view_loss(triplane, cameras, target_images):
    losses = []
    for camera, target in zip(cameras, target_images):
        rendered = render(triplane, camera)
        losses.append(F.mse_loss(rendered, target))

    return torch.stack(losses).mean()
```

## Experiments & Results

### Quantitative Results

| Method | Time | PSNR ↑ | SSIM ↑ | 3D Consistency |
|--------|------|--------|--------|----------------|
| **LRM** | **5 sec** | 21.3 | 0.82 | **High** |
| Zero123 | 30 sec | 22.1 | 0.84 | Medium |
| DreamFusion | 60 min | 24.5 | 0.88 | High |
| NeRF (per-scene) | 120 min | **28.0** | **0.92** | High |

LRM trades some quality for massive speed gain.

### Generalization

- **Trained on**: Objaverse (800k+ 3D objects)
- **Generalizes to**: Real images, sketches, diverse objects
- **Limitations**: Struggles with very complex scenes

## Common Pitfalls

### 1. Coordinate System Alignment

```python
# Ensure input image and triplane use same coordinate system
# LRM typically uses OpenGL: +Y up, -Z forward, +X right
```

### 2. Triplane Resolution Trade-off

```python
# Higher resolution = better quality but more memory
# 128: Fast, lower quality
# 256: Balanced
# 512: High quality, slow
```

### 3. Camera Pose Estimation

```python
# LRM assumes canonical object pose
# May need to estimate pose from image first
pose = estimate_canonical_pose(image)
camera = Camera(pose)
```

## References

### Primary Paper

```bibtex
@article{hong2023lrm,
  title={LRM: Large Reconstruction Model for Single Image to 3D},
  author={Hong, Yicong and Zhang, Kai and Gu, Jiuxiang and Bi, Sai and Zhou, Yang and Liu, Difan and Liu, Feng and Sunkavalli, Kalyan and Bui, Trung and Tan, Hao},
  journal={arXiv preprint arXiv:2311.04400},
  year={2023}
}
```

### Related Work

- **Triplane** (Chan et al., 2022): Efficient 3D representation
- **Objaverse** (Deitke et al., 2023): Large-scale 3D dataset
- **Zero-1-to-3** (Liu et al., 2023): View synthesis
- **DINO** (Caron et al., 2021): Image features

---

**Next**: For higher quality (but slower), see [ProlificDreamer](./prolific_dreamer.md).
