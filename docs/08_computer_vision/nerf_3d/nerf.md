# Neural Radiance Fields (NeRF)

## Overview & Motivation

Neural Radiance Fields (NeRF) revolutionized 3D computer vision by introducing a novel way to represent and render 3D scenes using neural networks. Published at ECCV 2020 by Mildenhall et al., NeRF enables photorealistic novel view synthesis from a sparse set of input images.

### The Problem
Traditional 3D representations face fundamental challenges:
- **Discrete representations** (voxels, meshes): Limited resolution, memory-intensive
- **View-dependent effects**: Difficult to model reflections, transparency, specularity
- **Optimization**: Hard to optimize directly from images
- **Continuity**: Aliasing artifacts at limited resolutions

### NeRF's Solution
Represent scenes as continuous 5D functions mapping 3D coordinates and viewing direction to color and density:

```
F_θ: (x, y, z, θ, φ) → (r, g, b, σ)
```

**Key Insight**: A fully-connected neural network can implicitly encode scene geometry and appearance, optimized through differentiable volume rendering.

## Theoretical Background

### Volume Rendering Foundations

Volume rendering accumulates color and opacity along camera rays passing through a 3D volume.

#### Classical Volume Rendering Equation
The color observed along a ray `r(t) = o + td` is:

```
C(r) = ∫[t_n to t_f] T(t) · σ(r(t)) · c(r(t), d) dt
```

where:
- **C(r)**: Final pixel color
- **t_n, t_f**: Near and far bounds
- **T(t)**: Transmittance (light reaching point t from camera)
- **σ(r(t))**: Volume density (probability of ray termination)
- **c(r(t), d)**: Emitted radiance (view-dependent color)

#### Transmittance
Accumulated transparency from near plane to point t:

```
T(t) = exp(-∫[t_n to t] σ(r(s)) ds)
```

**Physical Interpretation**: Probability that ray travels from t_n to t without hitting any particle.

#### Discrete Approximation
For practical implementation, discretize the integral:

```
Ĉ(r) = Σ[i=1 to N] T_i · (1 - exp(-σ_i δ_i)) · c_i

where:
T_i = exp(-Σ[j=1 to i-1] σ_j δ_j)
δ_i = t_{i+1} - t_i
```

### Why Volume Rendering?

1. **Differentiability**: Entire pipeline is differentiable
2. **Semi-transparency**: Naturally handles glass, fog, fur
3. **Anti-aliasing**: Continuous representation prevents jagged edges
4. **Occlusion**: Automatically handled through transmittance

## Mathematical Formulation

### Scene Representation

NeRF approximates the continuous 5D radiance field using an MLP:

```
F_θ: (x, d) → (c, σ)

where:
x = (x, y, z) ∈ ℝ³  # 3D position
d = (θ, φ) ∈ S²     # viewing direction (unit sphere)
c = (r, g, b) ∈ [0,1]³  # RGB color
σ ∈ ℝ⁺              # volume density
```

### Network Architecture

The MLP consists of two parts:

1. **Density Network** (view-invariant):
```
x → γ(x) → [MLP layers] → [σ, h]

where:
γ(x): Positional encoding
h: 256-dim feature vector
σ: Volume density
```

2. **Color Network** (view-dependent):
```
[h, γ(d)] → [MLP layers] → c

where:
γ(d): Encoded viewing direction
c: RGB color
```

**Design Rationale**:
- Geometry (σ) is view-invariant
- Appearance (c) depends on viewing angle (reflections, specularity)

### Positional Encoding

MLPs struggle with high-frequency signals. Positional encoding maps inputs to higher dimensions:

```
γ(p) = [p, sin(2^0πp), cos(2^0πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]
```

**For positions**: L = 10 (input: 3D → output: 63D)
**For directions**: L = 4 (input: 3D → output: 27D)

**Mathematical Justification**:
- Fourier features enable learning high-frequency functions
- Similar to positional encoding in Transformers
- Maps input to frequency domain

### Hierarchical Volume Sampling

To improve efficiency, NeRF uses two networks:

1. **Coarse Network**: Samples N_c points uniformly
2. **Fine Network**: Samples N_f additional points based on coarse weights

**Importance Sampling**:
```
Weights: w_i = T_i · (1 - exp(-σ_i δ_i))
PDF: p̂(t) = w_i / Σ[j] w_j
```

Sample from inverse CDF for more samples in high-density regions.

### Loss Function

Photometric reconstruction loss (mean squared error):

```
L = Σ[r∈R] [ ||Ĉ_c(r) - C(r)||² + ||Ĉ_f(r) - C(r)||² ]

where:
R: Set of rays in training batch
Ĉ_c: Coarse network prediction
Ĉ_f: Fine network prediction
C: Ground truth color
```

Both networks are supervised to prevent coarse network degradation.

## High-Level Intuition

### What is NeRF Learning?

Think of NeRF as learning a **"volumetric lightfield"**:

1. **Geometry through Density**:
   - High density σ = surface or solid object
   - Low density σ = empty space or air
   - Smooth transitions = semi-transparent materials

2. **Appearance through Color**:
   - Color changes with view direction = reflections
   - Color constant across views = diffuse surface
   - Learned implicitly from images

### The Training Process

1. **Random Ray Sampling**: Sample pixels from training images
2. **Point Sampling**: Sample points along each ray
3. **Network Evaluation**: Query MLP for color and density
4. **Volume Rendering**: Integrate along ray to get pixel color
5. **Gradient Descent**: Minimize difference with ground truth

**Key Insight**: The network learns to "explain" the training images by finding a 3D representation that renders consistently across views.

### Why Does It Work?

- **Multi-view Consistency**: Same 3D point must explain multiple views
- **Continuous Optimization**: Gradient descent finds optimal representation
- **Inductive Bias**: MLP smoothness acts as regularization
- **View-Dependent Modeling**: Separate networks for geometry and appearance

## Implementation Details

### Network Architecture Specifications

```python
# Positional Encoding
L_pos = 10  # Position encoding frequencies
L_dir = 4   # Direction encoding frequencies

# Position encoding dimension
pos_dim = 3 + 3 * 2 * L_pos = 63

# Direction encoding dimension
dir_dim = 3 + 3 * 2 * L_dir = 27

# Main MLP (8 layers)
Layer 1: Linear(63, 256) + ReLU
Layer 2-4: Linear(256, 256) + ReLU
Skip connection at layer 5: concat(layer4_output, encoded_position)
Layer 5: Linear(256+63, 256) + ReLU
Layer 6-8: Linear(256, 256) + ReLU

# Density Head
Linear(256, 1) + ReLU  → σ
Linear(256, 256)       → feature vector

# Color Head
concat(features, encoded_direction)
Linear(256+27, 128) + ReLU
Linear(128, 3) + Sigmoid  → RGB
```

### Sampling Strategy

**Coarse Sampling** (N_c = 64 points):
```python
t = linspace(near, far, N_c)
# Add stratified noise for training
t = t + uniform(0, (far-near)/N_c)
```

**Fine Sampling** (N_f = 128 points):
```python
# Compute normalized weights from coarse network
weights = alpha * transmittance
pdf = weights / sum(weights)

# Inverse transform sampling
cdf = cumsum(pdf)
u = uniform(0, 1, N_f)
t_fine = inverse_cdf(cdf, u)

# Combine coarse and fine samples
t_all = sort(concat(t_coarse, t_fine))
```

### Training Hyperparameters

```yaml
# Optimization
optimizer: Adam
learning_rate: 5e-4
lr_decay: exponential (250k steps)
batch_size: 4096 rays

# Sampling
N_coarse: 64
N_fine: 128
near: 2.0
far: 6.0

# Regularization
weight_decay: 0.0
positional_encoding_decay: None (fixed)

# Training
iterations: 200k - 300k
time: 1-2 days on single GPU
```

### Ray Generation

**Pinhole Camera Model**:
```python
def get_rays(H, W, focal, c2w):
    """
    H, W: Image height and width
    focal: Focal length in pixels
    c2w: Camera-to-world transformation matrix (4x4)
    """
    # Pixel coordinates
    i, j = meshgrid(arange(W), arange(H))

    # Camera coordinates (centered at principal point)
    dirs = stack([
        (i - W*.5) / focal,
        -(j - H*.5) / focal,  # Negative for image y-axis
        -ones_like(i)
    ], dim=-1)

    # Transform to world coordinates
    rays_d = sum(dirs[..., None, :] * c2w[:3,:3], dim=-1)
    rays_o = broadcast_to(c2w[:3,-1], rays_d.shape)

    return rays_o, rays_d
```

## Code Walkthrough

### Core Implementation: `nexus/models/cv/nerf/nerf.py`

#### 1. NeRF Network Class

```python
class NeRFNetwork(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Configuration
        self.pos_encoding_dims = config.get("pos_encoding_dims", 10)
        self.dir_encoding_dims = config.get("dir_encoding_dims", 4)
        self.hidden_dim = config.get("hidden_dim", 256)
```

**Design Pattern**: Inherits from `NexusModule` for consistent configuration and logging.

#### 2. Positional Encoding

```python
# From nexus/components/embeddings.py
self.position_encoder = PositionalEncoding(self.pos_encoding_dims)
self.direction_encoder = PositionalEncoding(self.dir_encoding_dims)

# Input dimensions after encoding
pos_channels = 3 * 2 * self.pos_encoding_dims  # 60 dims
dir_channels = 3 * 2 * self.dir_encoding_dims  # 24 dims
```

**Note**: Original paper uses 3 + 2L*3 (includes raw input), but effect is minimal.

#### 3. MLP Architecture

```python
# Main processing network
self.mlp = nn.Sequential(
    nn.Linear(pos_channels, self.hidden_dim),
    nn.ReLU(),
    nn.Linear(self.hidden_dim, self.hidden_dim),
    nn.ReLU(),
    nn.Linear(self.hidden_dim, self.hidden_dim),
    nn.ReLU(),
    nn.Linear(self.hidden_dim, self.hidden_dim),
    nn.ReLU()
)

# Density prediction (view-invariant)
self.density_head = nn.Sequential(
    nn.Linear(self.hidden_dim, 1),
    nn.ReLU()  # Ensures non-negative density
)

# Color prediction (view-dependent)
self.color_head = nn.Sequential(
    nn.Linear(self.hidden_dim + dir_channels, self.hidden_dim // 2),
    nn.ReLU(),
    nn.Linear(self.hidden_dim // 2, 3),
    nn.Sigmoid()  # Clamp to [0, 1]
)
```

**Simplified Architecture**: Omits skip connections for clarity. Production code should include them.

#### 4. Forward Pass

```python
def forward(
    self,
    positions: torch.Tensor,      # [N, 3]
    directions: torch.Tensor       # [N, 3]
) -> Dict[str, torch.Tensor]:
    # Encode inputs
    pos_encoded = self.position_encoder(positions)   # [N, 63]
    dir_encoded = self.direction_encoder(directions) # [N, 27]

    # Main network processes position
    features = self.mlp(pos_encoded)  # [N, 256]

    # Predict density (view-invariant)
    density = self.density_head(features)  # [N, 1]

    # Predict color (view-dependent)
    color_input = torch.cat([features, dir_encoded], dim=-1)
    color = self.color_head(color_input)  # [N, 3]

    return {"density": density, "color": color}
```

#### 5. Volume Rendering

```python
def render_rays(
    self,
    ray_origins: torch.Tensor,      # [R, 3]
    ray_directions: torch.Tensor,   # [R, 3]
    near: float,
    far: float,
    num_samples: int = 64,
    noise_std: float = 0.0
) -> Dict[str, torch.Tensor]:

    # Generate sample points along rays
    t_vals = torch.linspace(near, far, num_samples,
                           device=ray_origins.device)
    z_vals = t_vals[None, :].expand(ray_origins.shape[0], -1)  # [R, N]

    # Add noise for training (stratified sampling)
    if noise_std > 0:
        z_vals = z_vals + torch.randn_like(z_vals) * noise_std

    # Compute 3D sample positions
    # r(t) = o + td
    sample_points = (ray_origins[..., None, :] +
                    ray_directions[..., None, :] *
                    z_vals[..., :, None])  # [R, N, 3]

    # Prepare for network evaluation
    directions = ray_directions[:, None].expand(-1, num_samples, -1)
    sample_points_flat = sample_points.reshape(-1, 3)
    directions_flat = directions.reshape(-1, 3)

    # Evaluate network
    outputs = self(sample_points_flat, directions_flat)
    density = outputs["density"].reshape(-1, num_samples, 1)  # [R, N, 1]
    color = outputs["color"].reshape(-1, num_samples, 3)      # [R, N, 3]
```

#### 6. Alpha Compositing

```python
    # Compute distances between samples
    delta_z = z_vals[..., 1:] - z_vals[..., :-1]
    delta_z = torch.cat([
        delta_z,
        torch.tensor([1e10], device=delta_z.device).expand(delta_z.shape[0], 1)
    ], dim=-1)  # [R, N]

    # Compute alpha (opacity)
    # α = 1 - exp(-σδ)
    alpha = 1 - torch.exp(-density.squeeze(-1) * delta_z)  # [R, N]

    # Compute transmittance T = exp(-Σ σδ)
    # = ∏(1 - α)
    transmittance = torch.cumprod(
        torch.cat([
            torch.ones((alpha.shape[0], 1), device=alpha.device),
            1 - alpha + 1e-10  # Numerical stability
        ], dim=-1),
        dim=-1
    )[:, :-1]  # [R, N]

    # Compute weights w = T * α
    weights = alpha * transmittance  # [R, N]

    # Final color: C = Σ w_i * c_i
    rgb = (weights[..., None] * color).sum(dim=1)  # [R, 3]

    # Depth: D = Σ w_i * t_i
    depth = (weights * z_vals).sum(dim=1)  # [R]

    return {
        "rgb": rgb,
        "depth": depth,
        "weights": weights,
        "z_vals": z_vals
    }
```

**Key Points**:
- `1e10` for last delta ensures far samples contribute properly
- `1e-10` added to avoid log(0) in transmittance
- Weights sum to ≈1 (normalized by transmittance)

### Usage Example

```python
import torch
from nexus.models.cv.nerf import NeRFNetwork

# Initialize model
config = {
    "pos_encoding_dims": 10,
    "dir_encoding_dims": 4,
    "hidden_dim": 256
}
model = NeRFNetwork(config).cuda()

# Generate camera rays
H, W = 400, 400
focal = 400.0
c2w = torch.eye(4).cuda()  # Identity camera

# Sample one ray for demonstration
ray_o = torch.tensor([[0., 0., 0.]]).cuda()
ray_d = torch.tensor([[0., 0., -1.]]).cuda()  # Looking down -Z

# Render ray
outputs = model.render_rays(
    ray_origins=ray_o,
    ray_directions=ray_d,
    near=2.0,
    far=6.0,
    num_samples=128
)

print(f"RGB: {outputs['rgb']}")
print(f"Depth: {outputs['depth']}")
```

## Optimization Tricks

### 1. Learning Rate Scheduling

```python
# Exponential decay
initial_lr = 5e-4
decay_rate = 0.1
decay_steps = 250000

def get_lr(step):
    return initial_lr * (decay_rate ** (step / decay_steps))
```

### 2. Chunk Processing for Memory

```python
def render_rays_chunked(rays_o, rays_d, chunk_size=1024*32):
    """Render rays in chunks to avoid OOM."""
    all_outputs = []

    for i in range(0, rays_o.shape[0], chunk_size):
        chunk_outputs = model.render_rays(
            rays_o[i:i+chunk_size],
            rays_d[i:i+chunk_size]
        )
        all_outputs.append(chunk_outputs)

    # Concatenate results
    return {
        k: torch.cat([out[k] for out in all_outputs], dim=0)
        for k in all_outputs[0].keys()
    }
```

### 3. Batched Network Queries

```python
# Instead of querying network N times per ray
# Query once for all samples across all rays
N_rays, N_samples = 4096, 192
total_samples = N_rays * N_samples

# Flatten spatial dimensions
positions = sample_points.reshape(total_samples, 3)
directions = ray_dirs.reshape(total_samples, 3)

# Single forward pass
outputs = model(positions, directions)
```

### 4. White Background for Synthetic Data

```python
# Add white background for better convergence
rgb = rgb + (1 - weights.sum(-1, keepdim=True))
```

### 5. Coarse-to-Fine Curriculum

```python
# Start with fewer samples, increase gradually
N_samples = min(64 + step // 1000, 192)
```

## Experiments & Results

### Datasets

1. **Synthetic NeRF Dataset**:
   - 8 objects with known camera poses
   - 100 train views, 200 test views
   - White background
   - Resolution: 800×800

2. **Real-World Scenes**:
   - Forward-facing captures (LLFF dataset)
   - 20-60 training images
   - Handheld cellphone captures

### Quantitative Results (Synthetic)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|--------|--------|---------|
| **NeRF (Ours)** | **31.01** | **0.947** | **0.081** |
| Neural Volumes | 26.05 | 0.893 | 0.160 |
| SRN | 22.26 | 0.846 | 0.170 |
| LLFF | 24.88 | 0.911 | 0.114 |

**PSNR**: Peak Signal-to-Noise Ratio (higher = better)
**SSIM**: Structural Similarity Index (higher = better)
**LPIPS**: Learned Perceptual Image Patch Similarity (lower = better)

### Qualitative Observations

**Strengths**:
- Photorealistic novel views
- Correct view-dependent effects (reflections on Ship)
- Sharp details at all depths
- Smooth interpolation between views

**Limitations**:
- Slow rendering (~30 seconds per 800×800 image)
- Long training time (~1-2 days)
- Struggles with very shiny/specular objects
- Requires accurate camera poses

### Ablation Studies

**Positional Encoding**:
- Without encoding: PSNR drops by ~10dB
- Low frequency (L=4): Blurry results
- High frequency (L=10): Sharp details

**Hierarchical Sampling**:
- Without fine network: -2dB PSNR
- Saves ~50% computation
- Critical for efficiency

**View Dependence**:
- Without direction input: Flat, diffuse appearance
- Cannot model reflections
- -1-2dB on realistic scenes

## Common Pitfalls

### 1. Incorrect Camera Conventions

```python
# Common mistake: Wrong coordinate system
# OpenGL: +Y up, -Z forward, +X right ✓
# OpenCV: +Y down, +Z forward, +X right ✗

# Solution: Verify camera coordinate system
# NeRF typically uses OpenGL convention
```

### 2. Ray Direction Normalization

```python
# Wrong: Unnormalized rays
rays_d = directions  # May have arbitrary magnitude

# Correct: Unit vectors
rays_d = F.normalize(directions, dim=-1)
```

### 3. Density Activation

```python
# Wrong: No activation (can be negative)
density = self.density_head(features)

# Correct: ReLU to ensure non-negative
density = F.relu(self.density_head(features))

# Better: Softplus for smoothness
density = F.softplus(self.density_head(features))
```

### 4. Numerical Instability in Transmittance

```python
# Wrong: Can cause NaN due to numerical errors
transmittance = torch.cumprod(1 - alpha, dim=-1)

# Correct: Add small epsilon
transmittance = torch.cumprod(1 - alpha + 1e-10, dim=-1)
```

### 5. Forgetting Gradient Detachment

```python
# When sampling fine points based on coarse weights:
# Correct: Detach coarse weights
weights_coarse = weights_coarse.detach()
t_fine = sample_pdf(z_vals, weights_coarse, N_fine)

# Otherwise: Gradient flows through sampling
```

### 6. Training on Too Few Views

```python
# Minimum recommended: 50-100 views for complex scenes
# For simple objects: 20-30 views sufficient
# Solution: Add regularization or use fewer frequencies
```

### 7. Wrong Sampling Bounds

```python
# Scene-dependent near/far is critical
# Too small: Miss parts of scene
# Too large: Waste samples in empty space

# Solution: Estimate from COLMAP points or depth maps
near = min(scene_depths) * 0.9
far = max(scene_depths) * 1.1
```

## References

### Primary Paper
```bibtex
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and
          Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren},
  booktitle={ECCV},
  year={2020}
}
```

### Related Work
- **Neural Volumes** (Lombardi et al., 2019): Template-based neural rendering
- **Scene Representation Networks** (Sitzmann et al., 2019): Similar implicit representation
- **LLFF** (Mildenhall et al., 2019): Multi-plane images for view synthesis

### Implementation Resources
- Official Code: https://github.com/bmild/nerf
- PyTorch Implementation: https://github.com/yenchenlin/nerf-pytorch
- Nerfstudio: https://docs.nerf.studio/ (modern framework)

### Follow-up Papers
- **Mip-NeRF** (Barron et al., 2021): Anti-aliasing
- **NeRF++** (Zhang et al., 2020): Unbounded scenes
- **Instant-NGP** (Müller et al., 2022): Real-time training
- **3D Gaussian Splatting** (Kerbl et al., 2023): Real-time rendering

### Tutorials and Talks
- [CVPR 2020 Best Paper Talk](https://www.youtube.com/watch?v=JuH79E8rdKc)
- [Two Minute Papers Explanation](https://www.youtube.com/watch?v=WSfEfZ0ilw4)
- [Yannic Kilcher Analysis](https://www.youtube.com/watch?v=CRlN-cYFxTk)

---

**Next Steps**:
- Explore [Fast NeRF](./fast_nerf.md) for acceleration techniques
- See [Mip-NeRF](./mip_nerf.md) for anti-aliasing improvements
- Try [Gaussian Splatting](./gaussian_splatting.md) for real-time rendering
