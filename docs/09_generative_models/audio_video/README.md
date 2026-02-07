# Audio & Video Generation Models

## Overview

Comprehensive documentation for temporal generative models that handle audio, speech, music, and video synthesis. These models extend diffusion and autoregressive techniques to the temporal domain, enabling high-quality generation of time-series data with complex dependencies.

## Contents

### Video Generation

1. **[CogVideoX](./cogvideox.md)**
   - Expert transformer for text-to-video generation
   - 3D causal attention (spatial + temporal)
   - Progressive training strategy
   - Accepted at ICLR 2025

2. **[VideoPoet](./videopoet.md)**
   - Large language model approach to video
   - Unified tokenization for video, audio, text
   - Multi-task pre-training
   - Zero-shot capabilities

### Audio & Speech Synthesis

3. **[VALL-E](./valle.md)**
   - Neural codec language modeling for TTS
   - Zero-shot voice cloning from 3-second prompt
   - Autoregressive + non-autoregressive architecture
   - EnCodec discrete tokens

4. **[Voicebox](./voicebox.md)**
   - Non-autoregressive speech generation
   - Flow matching for speech synthesis
   - In-context learning for voice styles
   - Fast, high-quality generation

5. **[SoundStorm](./soundstorm.md)**
   - Parallel audio generation
   - Confidence-based iterative decoding
   - Maskgit-style generation
   - 2-second audio in 0.5 seconds

6. **[MusicGen](./musicgen.md)**
   - Text-to-music generation
   - Controllable music synthesis
   - Multi-stream transformer
   - Melody conditioning

7. **[NaturalSpeech 3](./naturalspeech3.md)**
   - Factorized diffusion for speech
   - Disentangled prosody and content
   - Neural codec integration
   - State-of-the-art quality

## Key Concepts

### Temporal Modeling Challenges

**Video-specific:**
- **Spatiotemporal coherence**: Maintaining consistency across frames
- **Long-range dependencies**: Modeling motion over time
- **Memory constraints**: High-dimensional data (B, T, C, H, W)
- **Compression**: VAE for spatial, tokenization for temporal

**Audio-specific:**
- **High sample rates**: 16-48 kHz raw audio
- **Long sequences**: Seconds of audio = thousands of timesteps
- **Discrete representations**: Neural codecs (EnCodec, SoundStream)
- **Multi-scale structure**: Prosody, phonemes, acoustics

### Architecture Paradigms

| Approach | Examples | Pros | Cons |
|----------|----------|------|------|
| **Autoregressive** | VALL-E, VideoPoet | High quality, flexible | Slow sampling |
| **Non-autoregressive** | Voicebox, SoundStorm | Fast sampling | Training complexity |
| **Diffusion** | NaturalSpeech 3, CogVideoX | SOTA quality | Slow, many steps |
| **Hybrid** | VALL-E (AR+NAR) | Balanced | Architecture complexity |

### Neural Audio Codecs

**Purpose**: Compress raw audio to discrete tokens

**Examples:**
- **EnCodec**: 8 codebooks, 75 tokens/sec at 24kHz
- **SoundStream**: Similar to EnCodec, used in AudioLM
- **DAC**: Descript Audio Codec

**Benefits:**
- 100-300x compression
- Discrete tokens enable language modeling
- Residual vector quantization (RVQ) for quality

### Video Tokenization

**Spatial Compression:**
```
Image (3, 256, 256) -> VAE -> Latent (4, 32, 32)
```

**Temporal Compression:**
```
Video (T, 4, 32, 32) -> Transformer -> Tokens (T', D)
```

**Full Pipeline:**
```
Video -> 3D VAE -> Latent Video -> Diffusion Transformer -> Generated Latent -> 3D VAE Decoder -> Video
```

## Training Strategies

### Progressive Training

**For Video (CogVideoX):**
1. **Stage 1**: Image generation (T=1)
2. **Stage 2**: Short videos (T=16)
3. **Stage 3**: Long videos (T=64)

**Benefits:**
- Faster convergence
- Better motion modeling
- Reduced memory requirements

### Multi-Task Pre-training

**For VideoPoet:**
- Video generation
- Video inpainting
- Video outpainting
- Video-to-audio
- Stylization

**Benefits:**
- Shared representations
- Zero-shot transfer
- Better generalization

### Two-Stage Training

**For VALL-E:**
1. **AR Stage**: Generate first codebook level
2. **NAR Stage**: Generate remaining codebook levels

**Benefits:**
- Balance quality (AR) and speed (NAR)
- Coarse-to-fine generation
- Scalability

## Conditioning Mechanisms

### Text Conditioning

**T5/CLIP Embeddings:**
```python
text_emb = t5_encoder(text_tokens)  # (B, L, D)
cross_attention(video_tokens, text_emb)
```

**Classifier-Free Guidance:**
```python
output = uncond_output + scale * (cond_output - uncond_output)
```

### Audio Prompting (VALL-E, Voicebox)

**Acoustic Prompting:**
```python
# Encode reference audio
prompt_tokens = codec.encode(prompt_audio)

# Generate conditioned on prompt
generated = model(text, prompt_tokens)
```

### Melody Conditioning (MusicGen)

**Chroma Features:**
```python
melody_features = extract_chroma(melody_audio)
music = musicgen(text, melody_features)
```

## Evaluation Metrics

### Video Quality

**Quantitative:**
- **FVD (Fréchet Video Distance)**: Distribution similarity
- **IS (Inception Score)**: Quality and diversity
- **FID (per-frame)**: Image quality
- **CLIP Score**: Text-video alignment

**Qualitative:**
- Temporal consistency
- Motion realism
- Object permanence
- Scene transitions

### Audio Quality

**Objective Metrics:**
- **MOS (Mean Opinion Score)**: Human evaluation (1-5)
- **PESQ**: Perceptual Evaluation of Speech Quality
- **STOI**: Short-Time Objective Intelligibility
- **SECS**: Speaker Encoder Cosine Similarity

**Perceptual:**
- Naturalness
- Prosody
- Speaker similarity
- Audio quality

### Music Quality

- **FAD (Fréchet Audio Distance)**: Distribution similarity
- **Melodic coherence**: Musical structure
- **Harmonic consistency**: Chord progressions
- **Rhythmic accuracy**: Beat alignment

## Optimization Tricks

### 1. Gradient Checkpointing

Essential for long sequences:

```python
from torch.utils.checkpoint import checkpoint

def forward_block(block, x):
    if self.training:
        return checkpoint(block, x)
    return block(x)
```

Memory savings: 2-4x

### 2. Flash Attention

Efficient attention for long sequences:

```python
from flash_attn import flash_attn_func

# Standard attention: O(N^2) memory
attn = scaled_dot_product_attention(q, k, v)

# Flash attention: O(N) memory
attn = flash_attn_func(q, k, v)
```

Speedup: 2-3x, Memory: 5-10x less

### 3. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(video)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4. Temporal Chunking

Process video in chunks:

```python
chunk_size = 16  # frames
for i in range(0, T, chunk_size):
    chunk = video[:, i:i+chunk_size]
    process(chunk)
```

### 5. Codec Compression

Use neural codecs to reduce sequence length:

```python
# Raw audio: 24000 samples/sec
# EnCodec: 75 tokens/sec (320x compression)
tokens = codec.encode(audio)  # (B, T_raw) -> (B, 8, T_compressed)
```

## Common Pitfalls

### 1. Temporal Inconsistency

**Problem**: Flickering, jittering between frames

**Solutions:**
- Temporal attention layers
- Frame differencing loss
- Optical flow consistency
- Higher frame rate training

### 2. Memory Issues

**Problem**: OOM errors with long sequences

**Solutions:**
- Gradient checkpointing
- Reduce batch size
- Temporal chunking
- Lower resolution training

### 3. Slow Sampling

**Problem**: Generation takes too long

**Solutions:**
- Distillation to fewer steps
- Non-autoregressive models
- Cached KV for autoregressive
- Parallel generation (SoundStorm)

### 4. Audio Artifacts

**Problem**: Pops, clicks, robotic voice

**Solutions:**
- Higher codec quality
- Post-filtering
- Better neural vocoder
- Sufficient context length

### 5. Text-Video Misalignment

**Problem**: Generated video doesn't match text

**Solutions:**
- Higher guidance scale
- Better text encoder (T5 vs CLIP)
- Multi-stage generation
- Reward model fine-tuning

## Model Comparison

### Video Generation

| Model | Resolution | FPS | Length | FVD | Speed | Notes |
|-------|-----------|-----|--------|-----|-------|-------|
| **CogVideoX** | 720p | 8 | 6s | 82 | Medium | SOTA quality |
| **VideoPoet** | 1080p | 24 | 10s | 95 | Slow | Multi-modal |
| **Make-A-Video** | 512p | 16 | 5s | 118 | Fast | Image-based |
| **Imagen Video** | 1280p | 24 | 5.3s | 74 | Slow | Best quality |

### Audio Generation

| Model | Type | Quality (MOS) | Speed | Zero-Shot | Notes |
|-------|------|--------------|-------|-----------|-------|
| **VALL-E** | TTS | 4.2 | Medium | Yes | Voice cloning |
| **Voicebox** | TTS | 4.5 | Fast | Yes | Flow matching |
| **SoundStorm** | TTS | 4.1 | Very Fast | No | Parallel decoding |
| **MusicGen** | Music | 4.3 | Medium | Partial | Melody control |
| **NaturalSpeech 3** | TTS | 4.6 | Slow | Yes | SOTA quality |

## Code Structure

```
nexus/models/
├── video/
│   ├── cogvideox.py          # Text-to-video transformer
│   └── videopoet.py          # LLM for video
└── audio/
    ├── valle.py              # Neural codec TTS
    ├── voicebox.py           # Flow-based TTS
    ├── soundstorm.py         # Parallel audio generation
    ├── musicgen.py           # Text-to-music
    └── naturalspeech3.py     # Factorized diffusion TTS
```

## References

### Video Generation

1. **Hong et al., "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer" (2024)**
   - https://arxiv.org/abs/2408.06072

2. **Kondratyuk et al., "VideoPoet: A Large Language Model for Zero-Shot Video Generation" (2023)**
   - https://arxiv.org/abs/2312.14125

### Audio & Speech

3. **Wang et al., "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (VALL-E, 2023)**
   - https://arxiv.org/abs/2301.02111

4. **Le et al., "Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale" (2023)**
   - https://arxiv.org/abs/2306.15687

5. **Borsos et al., "SoundStorm: Efficient Parallel Audio Generation" (2023)**
   - https://arxiv.org/abs/2305.09636

6. **Copet et al., "Simple and Controllable Music Generation" (MusicGen, 2023)**
   - https://arxiv.org/abs/2306.05284

7. **Ju et al., "NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models" (2024)**
   - https://arxiv.org/abs/2403.03100

## Next Steps

1. **Study CogVideoX** for state-of-the-art video generation
2. **Implement VALL-E** for zero-shot voice cloning
3. **Try Voicebox** for fast, high-quality TTS
4. **Explore MusicGen** for controllable music synthesis

---

*Implementations in `Nexus/nexus/models/video/` and `Nexus/nexus/models/audio/`*
