# DreamGaussian: Generative 3D from 2D Diffusion Priors

## Overview & Motivation

DreamGaussian (2023) enables **fast text-to-3D and image-to-3D generation** by combining 2D diffusion models with 3D Gaussian Splatting. Unlike previous methods requiring hours of optimization, DreamGaussian produces high-quality 3D assets in minutes.

### The Text-to-3D Challenge

Previous approaches (DreamFusion, Magic3D):
- Use NeRF as 3D representation
- Optimize via Score Distillation Sampling (SDS)
- Require 1-2 hours per object
- Often produce over-saturated colors ("Janus problem")

### DreamGaussian's Innovation

1. **Gaussian Splatting** instead of NeRF for faster rendering
2. **Improved distillation** with variance reduction
3. **Mesh extraction** from Gaussians for editing
4. **Two-stage pipeline**: Coarse generation → Fine texturing

**Result**: High-quality 3D in ~10 minutes

## Theoretical Background

### Score Distillation Sampling (SDS)

Core idea: Use a pre-trained 2D diffusion model as a prior for 3D:

```
Goal: Generate 3D θ such that rendered images look realistic

Diffusion model pϕ(x): Learned distribution over images
3D parameters θ: Gaussian positions, colors, etc.
```

**SDS Loss**:
```
∇θ L_SDS = E_t,ε [ w(t) (ε_ϕ(x_t; y, t) - ε) ∂x/∂θ ]

where:
x = render(θ): Rendered image from 3D
x_t: Noisy version at timestep t
ε_ϕ: Noise prediction from diffusion model
y: Text condition ("a cat")
w(t): Weighting function
```

**Intuition**: Make rendered views look like they came from the diffusion model.

### Gaussian-Based SDS

Traditional SDS with NeRF:
```
x = NeRF_render(θ) → Slow (seconds per view)
```

DreamGaussian with Gaussians:
```
x = Gaussian_render(θ) → Fast (milliseconds per view)
```

**Impact**: Can optimize with more views per iteration → better geometry

### Variational Score Distillation (VSD)

Standard SDS problem: High variance, slow convergence

VSD improvement:
```
∇θ L_VSD = E_t,ε,c [ w(t) (ε_ϕ(x_t; y, t, c) - ε_θ(x_t; t, c)) ∂x/∂θ ]

where ε_θ: Learned variance reducer (small LoRA model)
```

**Benefit**: Lower variance → faster, more stable optimization

## Mathematical Formulation

### Two-Stage Pipeline

**Stage 1: Coarse Geometry** (2-5 minutes)
```
Initialize: Random Gaussians
For iteration in [0, N_coarse]:
  Sample camera pose
  Render image x = G(θ)
  Compute SDS loss
  Update θ via gradient descent
```

**Stage 2: Texture Refinement** (5-10 minutes)
```
Fix Gaussian positions (geometry)
Optimize colors and opacities
Use higher resolution diffusion model
Add mesh-based UV texturing
```

### Mesh Extraction

Convert Gaussians to mesh for editing:

```
1. Extract surface: Density threshold
2. Poisson reconstruction: Smooth mesh
3. UV unwrapping: Texture atlas
4. Texture baking: Render Gaussians onto UV map
```

### Regularization Terms

```
L_total = L_SDS + λ_opacity L_opacity + λ_scale L_scale

L_opacity: Σ (1 - α_i)²
  → Encourage near-binary opacity

L_scale: Σ ||s_i||²
  → Prevent extremely large Gaussians
```

## Implementation Details

### Gaussian Initialization

```python
def initialize_gaussians(num_gaussians=10000, radius=1.0):
    """Initialize random Gaussians in sphere."""
    # Random positions on sphere surface
    theta = torch.rand(num_gaussians) * 2 * np.pi
    phi = torch.arccos(2 * torch.rand(num_gaussians) - 1)

    positions = radius * torch.stack([
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi)
    ], dim=-1)

    # Small random scales
    scales = torch.ones(num_gaussians, 3) * 0.01

    # Random rotations
    rotations = torch.randn(num_gaussians, 4)
    rotations = F.normalize(rotations, dim=-1)

    # Random colors (will be optimized)
    colors = torch.rand(num_gaussians, 3)

    # Low initial opacity
    opacities = torch.ones(num_gaussians, 1) * 0.1

    return GaussianModel(positions, scales, rotations, colors, opacities)
```

### SDS Loss Implementation

```python
def sds_loss(
    diffusion_model,
    rendered_image,  # [H, W, 3]
    text_prompt,
    guidance_scale=7.5,
    t_min=0.02,
    t_max=0.98
):
    """
    Compute Score Distillation Sampling loss.

    Args:
        diffusion_model: Pre-trained diffusion model (e.g., Stable Diffusion)
        rendered_image: Image rendered from 3D Gaussians
        text_prompt: Text description
        guidance_scale: CFG strength
        t_min, t_max: Timestep sampling range

    Returns:
        loss: SDS loss value
    """
    # Sample random timestep
    t = torch.rand(1) * (t_max - t_min) + t_min
    t_idx = int(t * diffusion_model.num_timesteps)

    # Add noise to rendered image
    noise = torch.randn_like(rendered_image)
    noisy_image = diffusion_model.add_noise(rendered_image, noise, t_idx)

    # Predict noise (with and without text condition)
    with torch.no_grad():
        # Conditional prediction
        noise_pred_cond = diffusion_model.predict_noise(
            noisy_image,
            t_idx,
            text_prompt
        )

        # Unconditional prediction
        noise_pred_uncond = diffusion_model.predict_noise(
            noisy_image,
            t_idx,
            ""  # Empty prompt
        )

        # Classifier-free guidance
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

    # SDS gradient
    # ∇_θ = (noise_pred - noise) * ∂rendered_image/∂θ
    # PyTorch autograd handles ∂rendered_image/∂θ
    grad = noise_pred - noise

    # Use as pseudo-gradient (stop gradient on pred)
    loss = torch.sum(grad.detach() * rendered_image)

    return loss
```

### Training Loop

```python
def train_dreamgaussian(
    text_prompt,
    num_iterations=1000,
    lr=0.01,
    guidance_scale=100
):
    """Train 3D Gaussians from text."""

    # Initialize
    gaussians = initialize_gaussians(num_gaussians=5000)
    diffusion_model = load_stable_diffusion()

    optimizer = torch.optim.Adam([
        {'params': gaussians.positions, 'lr': lr * 0.01},
        {'params': gaussians.scales, 'lr': lr},
        {'params': gaussians.rotations, 'lr': lr},
        {'params': gaussians.colors, 'lr': lr * 0.1},
        {'params': gaussians.opacities, 'lr': lr * 0.05}
    ])

    for iteration in range(num_iterations):
        # Sample random camera
        camera = sample_random_camera(
            radius=2.0,
            fov=60,
            resolution=512
        )

        # Render
        rendered = render_gaussians(gaussians, camera)

        # Compute losses
        loss_sds = sds_loss(
            diffusion_model,
            rendered,
            text_prompt,
            guidance_scale=guidance_scale
        )

        # Regularization
        loss_opacity = torch.mean((1 - gaussians.opacities) ** 2)
        loss_scale = torch.mean(gaussians.scales ** 2)

        loss = loss_sds + 0.1 * loss_opacity + 0.01 * loss_scale

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Densification (periodically)
        if iteration % 100 == 0:
            gaussians = densify_gaussians(gaussians, threshold=0.01)

        if iteration % 50 == 0:
            print(f"Iter {iteration}: Loss = {loss.item():.4f}")

    return gaussians
```

### Mesh Extraction

```python
def extract_mesh_from_gaussians(gaussians, resolution=256):
    """Convert Gaussians to textured mesh."""

    # 1. Create density field
    density_field = gaussians_to_density_field(gaussians, resolution)

    # 2. Extract isosurface (marching cubes)
    vertices, faces = marching_cubes(
        density_field,
        threshold=0.5
    )

    # 3. UV unwrapping
    uvs = compute_uv_unwrap(vertices, faces)

    # 4. Texture baking: Render Gaussians onto UV map
    texture_map = torch.zeros(1024, 1024, 3)

    for camera_angle in sample_sphere_cameras(n=64):
        rendered = render_gaussians(gaussians, camera_angle)
        project_onto_texture_map(
            rendered,
            camera_angle,
            vertices,
            faces,
            uvs,
            texture_map
        )

    mesh = Mesh(vertices, faces, uvs, texture_map)
    return mesh
```

## High-Level Intuition

### The Dream Process

Think of DreamGaussian as a sculptor guided by a knowledgeable art critic (diffusion model):

1. **Start with clay** (random Gaussians)
2. **Show sculpture from random angle** (render view)
3. **Critic says**: "This should look more like X" (SDS gradient)
4. **Sculptor adjusts** (gradient descent on Gaussians)
5. **Repeat** thousands of times from many angles

Eventually, the sculpture looks good from all angles.

### Why Gaussians?

Compared to NeRF:
- **10-100x faster rendering**: Can use more views per iteration
- **Explicit representation**: Easier to manipulate and edit
- **Mesh extraction**: Can export to standard formats

## Optimization Tricks

### 1. Progressive Timestep Annealing

```python
def get_timestep_range(iteration, total_iterations):
    """Start with high noise, gradually reduce."""
    progress = iteration / total_iterations
    t_max = 0.98 - 0.6 * progress  # 0.98 → 0.38
    t_min = 0.02
    return t_min, t_max
```

### 2. View-Dependent Prompting

```python
def get_view_dependent_prompt(base_prompt, elevation, azimuth):
    """Adjust prompt based on view angle."""
    if elevation > 60:
        return base_prompt + ", top view"
    elif elevation < 20:
        return base_prompt + ", bottom view"
    elif 170 < azimuth < 190:
        return base_prompt + ", back view"
    else:
        return base_prompt + ", front view"
```

### 3. Adaptive Guidance Scale

```python
def get_guidance_scale(iteration):
    """Higher guidance early, lower later."""
    if iteration < 500:
        return 100  # Strong guidance
    elif iteration < 1000:
        return 50
    else:
        return 20  # Refine details
```

### 4. Camera Pose Sampling

```python
def sample_training_camera(iteration, focus_front=True):
    """Sample camera with bias toward front views early."""
    if focus_front and iteration < 300:
        # Front hemisphere
        azimuth = np.random.uniform(-60, 60)
        elevation = np.random.uniform(0, 60)
    else:
        # Full sphere
        azimuth = np.random.uniform(-180, 180)
        elevation = np.random.uniform(-30, 90)

    radius = np.random.uniform(1.5, 2.5)
    return Camera(azimuth, elevation, radius)
```

## Experiments & Results

### Quantitative Metrics (Text-to-3D)

| Method | Time | CLIP Score ↑ | User Preference ↑ |
|--------|------|--------------|-------------------|
| **DreamGaussian** | **10 min** | **0.28** | **72%** |
| DreamFusion | 90 min | 0.25 | 45% |
| Magic3D | 45 min | 0.26 | 52% |
| ProlificDreamer | 120 min | **0.29** | **78%** |

DreamGaussian: Best speed-quality tradeoff

### Image-to-3D Results

| Method | Time | Novel View Quality |
|--------|------|-------------------|
| Zero-1-to-3 | 30 min | Medium |
| **DreamGaussian** | **5 min** | **High** |
| One-2-3-45 | 10 min | Medium-High |

### Qualitative Strengths

- Fast iteration for creative workflows
- Good geometry from most prompts
- Exportable meshes for game engines
- Stable optimization

### Limitations

- Can still have multi-face problem (Janus)
- Quality depends on diffusion model
- Some prompts don't work well
- Requires GPU with 16GB+ VRAM

## Common Pitfalls

### 1. Janus Problem (Multi-Face)

```python
# Mitigation: Penalize symmetric patterns
def anti_janus_loss(rendered_front, rendered_back):
    """Encourage front and back to be different."""
    similarity = F.cosine_similarity(
        rendered_front.flatten(),
        rendered_back.flatten(),
        dim=0
    )
    return torch.relu(similarity - 0.3)  # Penalize high similarity
```

### 2. Collapsed Geometry

```python
# Monitor Gaussian spread
mean_scale = torch.mean(gaussians.scales)
if mean_scale < 0.001:
    warnings.warn("Gaussians collapsed, increase scale regularization")
```

### 3. Over-Saturation

```python
# Clamp colors during optimization
with torch.no_grad():
    gaussians.colors.clamp_(0.1, 0.9)  # Avoid extreme values
```

### 4. Poor Initialization

```python
# Better init: Use structure from different model
if has_reference_image:
    # Extract depth, initialize Gaussians on surface
    depth = estimate_depth(reference_image)
    gaussians = initialize_from_depth(depth)
```

## References

### Primary Paper

```bibtex
@article{tang2023dreamgaussian,
  title={DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation},
  author={Tang, Jiaxiang and Ren, Jiawei and Zhou, Hang and Liu, Ziwei and Zeng, Gang},
  journal={arXiv preprint arXiv:2309.16653},
  year={2023}
}
```

### Related Work

- **DreamFusion** (Poole et al., 2022): Original SDS method
- **Magic3D** (Lin et al., 2023): Two-stage text-to-3D
- **ProlificDreamer** (Wang et al., 2023): VSD for better quality
- **Stable Diffusion** (Rombach et al., 2022): 2D diffusion prior

### Code & Resources

- Official code: https://github.com/dreamgaussian/dreamgaussian
- Online demo: https://dreamgaussian.github.io/
- Stable Diffusion: https://github.com/Stability-AI/stablediffusion

---

**Next Steps**:
- See [ProlificDreamer](./prolific_dreamer.md) for higher quality (but slower)
- Try [LRM](./lrm.md) for single-image 3D reconstruction
