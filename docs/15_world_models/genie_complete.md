# Genie: Generative Interactive Environments

## Overview & Motivation

Genie is a foundation world model trained on 200k hours of internet videos that can generate interactive, playable 2D environments from a single image prompt. Unlike traditional world models that require action labels, Genie learns latent action representations directly from video, enabling training on vast unlabeled video datasets.

### Key Innovation

**Action-free world model training**:
- Learns latent actions from video alone (no action labels)
- Generates playable environments from single images
- Spatiotemporal transformer architecture
- Trained on internet-scale video data (200k hours)
- Can create interactive game-like environments

### Problem Statement

Traditional world models face significant limitations:
- **Require action labels**: Need expensive human annotations or access to game/simulator internals
- **Limited training data**: Constrained to environments with available action information
- **Not generalizable**: Cannot leverage vast amounts of unlabeled video on the internet
- **Domain-specific**: Trained for specific games or tasks

Genie solves these through:
1. **Unsupervised action discovery**: Learns latent action space from video transitions alone
2. **Internet-scale training**: Trains on 200k hours of unlabeled platform gameplay videos
3. **Controllable generation**: Maps discovered actions to user controls for interactive play
4. **Zero-shot transfer**: Generates new environments from single image prompts

## Theoretical Background

### Latent Action Model

Genie learns a world model with latent actions discovered from video:

**Standard world model** (requires actions):
```
s_t+1 = f(s_t, a_t)  # a_t is observed/labeled
```

**Genie** (action-free):
```
a_t^latent = infer(s_t, s_t+1)  # Infer latent action from transition
ŝ_t+1 = f(s_t, a_t^latent)      # Predict next frame
```

The key insight is that consistent patterns in video transitions reveal an underlying action structure, even without labels.
