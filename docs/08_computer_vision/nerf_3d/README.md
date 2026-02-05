# 3D Computer Vision: Neural Radiance Fields and Gaussian Splatting

## Overview

This directory contains comprehensive documentation on modern 3D computer vision techniques, focusing on neural representations for novel view synthesis, 3D reconstruction, and scene understanding.

## The Evolution of Neural 3D Representations

### Traditional 3D Representations

Before neural methods, 3D scenes were represented using:
- **Point Clouds**: Discrete 3D points with optional attributes (color, normals)
- **Meshes**: Vertices connected by edges forming polygonal surfaces
- **Voxel Grids**: 3D discretization of space (memory-intensive)
- **Multi-View Stereo (MVS)**: Reconstructing geometry from multiple images

**Limitations**:
- Memory inefficient (especially voxels)
- Discrete representations with limited resolution
- Difficult to optimize end-to-end
- Poor handling of view-dependent effects

### The Neural Revolution: NeRF (2020)

Neural Radiance Fields (NeRF) revolutionized 3D vision by representing scenes as continuous functions learned by neural networks:

```
F_θ: (x, y, z, θ, φ) → (r, g, b, σ)
```

**Key Innovations**:
1. **Continuous Representation**: Infinite resolution through coordinate-based MLPs
2. **Differentiable Rendering**: End-to-end optimization with volume rendering
3. **View-Dependent Effects**: Modeling reflections, specularity, transparency
4. **Implicit Geometry**: No explicit surface representation needed

**Impact**: Enabled photorealistic novel view synthesis from sparse images, sparking an explosion of research.

### From NeRF to Gaussian Splatting (2020-2023)

The evolution can be understood through three main axes:

#### 1. Speed and Efficiency
- **NeRF (2020)**: Hours of training, seconds per frame rendering
- **Fast NeRF Variants (2021)**: Caching, factorization, octrees
- **Instant-NGP (2022)**: Hash encoding, sub-minute training
- **3D Gaussian Splatting (2023)**: Real-time rendering (>30 FPS)

#### 2. Quality and Representation
- **Mip-NeRF (2021)**: Anti-aliasing through cone tracing
- **NeRF++ (2020)**: Unbounded scene modeling
- **Zip-NeRF (2023)**: Combined anti-aliasing and regularization
- **SuGaR (2023)**: Surface-aligned Gaussians for better geometry

#### 3. Controllability and Editing
- **Early NeRF**: Static scene representation
- **GaussianEditor (2023)**: Direct 3D editing capabilities
- **DreamGaussian (2023)**: Text-to-3D generation
- **LRM (2023)**: Single-image 3D reconstruction

### The Gaussian Splatting Revolution (2023)

3D Gaussian Splatting represents a paradigm shift from implicit to explicit representations:

**NeRF Approach** (Implicit):
- Scene = MLP network weights
- Rendering = Thousands of network queries per ray
- Optimization = Gradient descent on network parameters

**Gaussian Splatting Approach** (Explicit):
- Scene = Collection of 3D Gaussians with parameters (position, covariance, color, opacity)
- Rendering = Rasterization-based splatting (GPU-friendly)
- Optimization = Direct gradient descent on Gaussian parameters

**Why Gaussians Won**:
1. **Real-Time Rendering**: 100-1000x faster than NeRF
2. **Differentiable Rasterization**: GPU-optimized forward/backward pass
3. **Explicit Representation**: Easier editing and manipulation
4. **Quality**: Comparable or better than NeRF for many scenes
5. **Memory Efficient**: Adaptive density based on scene complexity

## Documentation Structure

### Core Methods

#### NeRF Family (Implicit Representations)
1. **[NeRF](./nerf.md)** - Original Neural Radiance Fields
   - Foundation of neural 3D representations
   - Volume rendering with MLPs
   - Positional encoding and hierarchical sampling

2. **[Fast NeRF](./fast_nerf.md)** - Acceleration Techniques
   - Caching and factorization
   - Neural sparse voxel octrees
   - Efficient sampling strategies

3. **[Mip-NeRF](./mip_nerf.md)** - Anti-Aliasing via Cone Tracing
   - Integrated positional encoding
   - Multi-scale representation
   - Superior rendering quality

4. **[NeRF++](./nerf_plus_plus.md)** - Unbounded Scene Modeling
   - Inverted sphere parameterization
   - Foreground-background decomposition
   - 360-degree outdoor scenes

5. **[Zip-NeRF](./zip_nerf.md)** - State-of-the-Art NeRF
   - Multi-resolution hash encoding
   - Anti-aliasing and regularization
   - Best quality-speed tradeoff

#### Gaussian Splatting Family (Explicit Representations)
6. **[3D Gaussian Splatting](./gaussian_splatting.md)** - Real-Time Novel View Synthesis
   - Explicit 3D Gaussian primitives
   - Differentiable rasterization
   - Adaptive density control

7. **[SuGaR](./sugar.md)** - Surface-Aligned Gaussians
   - Regularization for surface alignment
   - Extracting explicit meshes
   - Better geometry reconstruction

8. **[GaussianEditor](./gaussian_editor.md)** - 3D Scene Editing
   - Semantic-aware editing
   - Interactive manipulation
   - Consistent multi-view editing

#### Generative and Reconstruction Methods
9. **[DreamGaussian](./dream_gaussian.md)** - Text/Image to 3D
   - Fast 3D generation from 2D priors
   - Combining diffusion and Gaussians
   - Mesh extraction and refinement

10. **[LRM](./lrm.md)** - Large Reconstruction Model
    - Single-image to 3D reconstruction
    - Transformer-based architecture
    - Generalizable across objects

11. **[ProlificDreamer](./prolific_dreamer.md)** - High-Quality Text-to-3D
    - Variational Score Distillation
    - Superior geometry and texture
    - Multi-view consistent generation

## Key Concepts Across Methods

### 1. Volume Rendering Equation
The foundation of NeRF-based methods:

```
C(r) = ∫ T(t)σ(r(t))c(r(t), d) dt
```

where:
- `C(r)` is the rendered color along ray `r`
- `T(t) = exp(-∫σ(r(s))ds)` is transmittance (accumulated transparency)
- `σ(r(t))` is volume density at point `r(t)`
- `c(r(t), d)` is view-dependent color

### 2. Positional Encoding
Mapping low-dimensional coordinates to high-dimensional space:

```
γ(p) = [sin(2^0πp), cos(2^0πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]
```

Enables MLPs to learn high-frequency details.

### 3. Hierarchical Sampling
Two-stage sampling strategy:
- **Coarse Network**: Uniform sampling to identify important regions
- **Fine Network**: Importance sampling focusing on high-density areas

Reduces computational cost while maintaining quality.

### 4. Gaussian Representation
Each 3D Gaussian is parameterized by:
- **Position** μ ∈ ℝ³: Center in 3D space
- **Covariance** Σ ∈ ℝ³ˣ³: 3D shape and orientation
- **Color** c ∈ ℝ³: RGB appearance (or spherical harmonics)
- **Opacity** α ∈ [0,1]: Transparency

Projected to 2D for efficient rasterization.

### 5. Differentiable Rendering
Both NeRF and Gaussian Splatting use differentiable rendering:
- **NeRF**: Differentiable volume rendering through numerical integration
- **Gaussians**: Differentiable splatting through custom CUDA kernels

Enables end-to-end optimization from 2D images.

## Comparison Matrix

| Method | Training Time | Rendering Speed | Quality | Memory | Editing | Use Case |
|--------|---------------|-----------------|---------|--------|---------|----------|
| NeRF | Hours | Slow (FPS: 0.1) | High | Low | Hard | Research baseline |
| Fast NeRF | 1-2 hours | Medium (FPS: 1-10) | High | Medium | Hard | Balanced quality/speed |
| Mip-NeRF | Hours | Slow | Very High | Low | Hard | Best quality NeRF |
| NeRF++ | Hours | Slow | High | Low | Hard | Unbounded scenes |
| Zip-NeRF | 1-2 hours | Medium | Very High | Medium | Hard | SOTA NeRF variant |
| Gaussian Splatting | Minutes | Real-time (FPS: 60+) | High | Medium-High | Easy | Production, real-time |
| SuGaR | 30-60 min | Real-time | High | Medium | Medium | When geometry matters |
| GaussianEditor | Minutes | Real-time | High | Medium | Very Easy | Interactive editing |
| DreamGaussian | Minutes | Real-time | Medium | Low | Medium | Quick 3D generation |
| LRM | Seconds | Real-time | Medium | High | Medium | Single-image 3D |
| ProlificDreamer | Hours | N/A | Very High | Medium | Medium | High-quality generation |

## When to Use What?

### Use NeRF when:
- Prioritizing rendering quality over speed
- Working with complex view-dependent effects
- Memory is constrained
- Research and experimentation

### Use Gaussian Splatting when:
- Real-time rendering is required
- Interactive editing is needed
- Training time must be minimized
- Deployment to production systems

### Use Generative Methods when:
- Creating 3D content from text/images
- Single-view 3D reconstruction
- No multi-view images available
- Fast prototyping of 3D assets

## Implementation Overview

Our implementation in `nexus/models/cv/` provides:

### NeRF Module (`nexus/models/cv/nerf/`)
```
nerf/
├── nerf.py              # Base NeRF implementation
├── nerf_plus_plus.py    # Unbounded scene extension
├── fast_nerf.py         # Acceleration techniques
├── mipnerf.py           # Anti-aliased rendering
├── renderer.py          # Volume rendering utilities
├── networks.py          # MLP architectures
└── hierarchical.py      # Hierarchical sampling
```

### Key Classes
- `NeRFNetwork`: Base MLP with positional encoding
- `NeRFPlusPlusNetwork`: Foreground-background decomposition
- `FastNeRFNetwork`: Cached and factorized rendering
- `MipNeRFNetwork`: Integrated positional encoding
- `NeRFRenderer`: Volume rendering implementation

## Research Timeline

```
2020: NeRF, NeRF++
      └─ Foundation of neural 3D representations

2021: Mip-NeRF, Fast NeRF variants
      └─ Quality improvements and acceleration

2022: Instant-NGP, TensoRF
      └─ Hybrid representations, dramatic speedups

2023: 3D Gaussian Splatting, Zip-NeRF
      └─ Real-time rendering, SOTA quality

2023: DreamGaussian, LRM, ProlificDreamer
      └─ Generative 3D from 2D priors

2024: GaussianEditor, SuGaR
      └─ Editing and geometric improvements
```

## Common Challenges and Solutions

### 1. Training Instability
- **Problem**: NeRF optimization can be unstable
- **Solutions**: Learning rate scheduling, weight regularization, coarse-to-fine training

### 2. View-Dependent Artifacts
- **Problem**: Floaters, inconsistencies across views
- **Solutions**: Multi-view consistency losses, depth regularization, pruning

### 3. Rendering Speed
- **Problem**: NeRF rendering is computationally expensive
- **Solutions**: Neural acceleration, caching, Gaussian splatting, neural sparse voxels

### 4. Limited Training Views
- **Problem**: Overfitting with sparse inputs
- **Solutions**: Regularization, depth priors, semantic guidance, generative priors

### 5. Dynamic Scenes
- **Problem**: Both NeRF and Gaussians assume static scenes
- **Solutions**: Deformation fields, 4D representations, temporal consistency

## Getting Started

1. **Start with NeRF**: Understand the foundation
   ```python
   from nexus.models.cv.nerf import NeRFNetwork

   config = {
       "pos_encoding_dims": 10,
       "dir_encoding_dims": 4,
       "hidden_dim": 256
   }
   model = NeRFNetwork(config)
   ```

2. **Explore Gaussian Splatting**: For production use cases
   - See `gaussian_splatting.md` for implementation details
   - Understand differentiable rasterization
   - Learn adaptive density control

3. **Try Generative Methods**: For content creation
   - DreamGaussian for quick prototypes
   - LRM for single-image reconstruction
   - ProlificDreamer for high quality

## References and Resources

### Seminal Papers
- **NeRF**: [Mildenhall et al., ECCV 2020](https://arxiv.org/abs/2003.08934)
- **Gaussian Splatting**: [Kerbl et al., SIGGRAPH 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **Mip-NeRF**: [Barron et al., ICCV 2021](https://arxiv.org/abs/2103.13415)

### Code Resources
- Official NeRF: https://github.com/bmild/nerf
- Official Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
- Nerfstudio: https://docs.nerf.studio/ (unified framework)

### Datasets
- **Synthetic**: NeRF Synthetic, ShapeNet
- **Real-World**: Mip-NeRF 360, Tanks and Temples
- **Unbounded**: Mip-NeRF 360, Free datasets

## Contributing

When adding new methods:
1. Follow the documentation template (see individual method docs)
2. Include mathematical formulations
3. Provide code walkthroughs with references to implementation
4. Add comparisons with existing methods
5. Include experimental results and ablations

## Future Directions

Active areas of research:
- **Dynamic 3D**: Handling moving scenes and deformations
- **Generalization**: Single-forward-pass reconstruction
- **Compression**: Reducing memory footprint for deployment
- **Physics Integration**: Simulating physical interactions
- **Relighting**: Separating lighting and materials
- **Large-Scale Scenes**: City-level reconstruction

---

Explore individual method documentation for deep dives into each technique.
