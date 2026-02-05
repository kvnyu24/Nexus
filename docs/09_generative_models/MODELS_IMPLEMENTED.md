# Implemented Generative Models Documentation

This document provides an overview of all generative models with comprehensive documentation.

## Documentation Structure

Each model documentation includes:
1. **Overview & Motivation** - Why this model matters
2. **Theoretical Background** - Core principles and theory
3. **Mathematical Formulation** - Loss functions, equations
4. **High-Level Intuition** - Conceptual understanding
5. **Implementation Details** - Architecture and config
6. **Code Walkthrough** - Key implementation sections
7. **Optimization Tricks** - Training improvements
8. **Experiments & Results** - Benchmarks and comparisons
9. **Common Pitfalls** - Issues and solutions
10. **References** - Original papers and resources

## Completed Documentation

### Main Categories
- [x] **Main README** (`README.md`) - Overview of all generative models
- [x] **Diffusion Models** (`diffusion/README.md`) - Comprehensive diffusion guide
- [x] **GANs** (`gans.md`) - All GAN variants
- [x] **VAE** (`vae.md`) - Variational autoencoders
- [x] **Audio/Video** (`audio_video/README.md`) - Temporal generation

### Individual Model Documentation

#### Diffusion Models (diffusion/)
- [x] **Base Diffusion** (`base_diffusion.md`) - DDPM foundation
- [ ] **Conditional Diffusion** - Conditioning mechanisms
- [ ] **Stable Diffusion** - Latent diffusion models
- [ ] **UNet** - Architecture for diffusion
- [ ] **DiT** - Diffusion transformers
- [ ] **MMDiT** - Multimodal transformers (SD3/FLUX)
- [ ] **Consistency Models** - Single-step generation
- [ ] **LCM** - Latent consistency models
- [ ] **Flow Matching** - Continuous normalizing flows
- [ ] **Rectified Flow** - Straight probability paths
- [ ] **PixArt-alpha** - Efficient high-res generation

#### Audio/Video Models (audio_video/)
- [ ] **CogVideoX** - Text-to-video generation
- [ ] **VideoPoet** - LLM for video
- [ ] **VALL-E** - Neural codec TTS
- [ ] **Voicebox** - Flow-based speech
- [ ] **SoundStorm** - Parallel audio generation
- [ ] **MusicGen** - Text-to-music
- [ ] **NaturalSpeech 3** - Factorized diffusion TTS

## Implementation Paths

All implementations are in `/Users/kevinyu/Projects/Nexus/nexus/models/`:

```
nexus/models/
├── diffusion/           # Diffusion model implementations
│   ├── base_diffusion.py
│   ├── conditional_diffusion.py
│   ├── stable_diffusion.py
│   ├── unet.py
│   ├── dit.py
│   ├── mmdit.py
│   ├── consistency_model.py
│   ├── flow_matching.py
│   ├── rectified_flow.py
│   └── pixart_alpha.py
├── video/               # Video generation
│   ├── cogvideox.py
│   └── videopoet.py
├── audio/               # Audio generation
│   ├── valle.py
│   ├── voicebox.py
│   ├── soundstorm.py
│   ├── musicgen.py
│   └── naturalspeech3.py
├── gan/                 # GAN models
│   ├── base_gan.py
│   ├── conditional_gan.py
│   ├── cycle_gan.py
│   └── wgan.py
└── cv/vae/              # VAE models
    └── vae.py
```

## Quick Start Guide

### For Diffusion Models
1. Start with [Base Diffusion](./diffusion/base_diffusion.md) to understand DDPM
2. Learn [Conditional Diffusion](./diffusion/README.md) for control
3. Study [Stable Diffusion](./diffusion/README.md) for latent space
4. Explore [Fast Sampling](./diffusion/README.md) methods

### For GANs
1. Read [GANs documentation](./gans.md)
2. Start with base GAN implementation
3. Try WGAN-GP for stable training
4. Experiment with conditional variants

### For VAEs
1. Study [VAE documentation](./vae.md)
2. Understand beta-VAE trade-offs
3. Try different architectures (MLP vs Conv)
4. Experiment with disentanglement

### For Audio/Video
1. Review [Audio/Video README](./audio_video/README.md)
2. Start with codec-based models (VALL-E)
3. Try diffusion-based (NaturalSpeech 3)
4. Explore video generation (CogVideoX)

## Documentation Templates

For models without individual docs, refer to:
- **Category READMEs** for comprehensive overviews
- **Implementation files** for code details
- **Main README** for general concepts

Each implementation file includes:
- Detailed docstrings
- Architecture descriptions
- Key innovations explained
- Usage examples in comments

## Contributing

To add documentation for remaining models:

1. Follow the 10-section structure listed above
2. Include code examples from implementations
3. Add mathematical formulations where relevant
4. Reference original papers
5. Provide practical optimization tips
6. Document common pitfalls and solutions

## References

See individual documentation files for complete reference lists.

Key resources:
- [Lil'Log Blog](https://lilianweng.github.io/) - Excellent overviews
- [Hugging Face Diffusion Course](https://github.com/huggingface/diffusion-models-class)
- [Papers with Code](https://paperswithcode.com/) - Implementations and benchmarks

---

*Last updated: 2026-02-06*
