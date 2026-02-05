# IDQL: Implicit Diffusion Q-Learning

**Paper**: [Efficient Diffusion Policies For Offline Reinforcement Learning](https://arxiv.org/abs/2304.10573) (Hansen-Estruch et al., NeurIPS 2023)

**Status**: Reference documentation (implementation pending)

## Overview

IDQL combines IQL's expectile-based value learning with **diffusion policies** to represent complex, multi-modal action distributions.

### Motivation

**IQL**: Uses Gaussian policies, struggles with multi-modal behavior
**IDQL**: Uses diffusion models, naturally represents multi-modality

Example: A robot can pick up an object with left OR right hand (two distinct modes). Gaussian policy averages → picks with middle (fails). Diffusion policy → learns both modes.

## Architecture

### Components

1. **Value network V(s)**: Expectile regression (from IQL)
2. **Q-networks Q(s,a)**: Twin critics (from IQL)
3. **Diffusion policy π(a|s)**: Replaces Gaussian policy

### Diffusion Policy

```
Forward process (noise):  a_t = √(α_t) a_{t-1} + √(1-α_t) ε

Reverse process (denoise): a_{t-1} = denoise_net(a_t, t, s)

Generation: a_0 ~ N(0,I) → denoise T steps → a_clean
```

## Key Modifications from IQL

### 1. Policy Architecture

```python
# IQL: Gaussian policy
mean, log_std = policy_net(state)
action = mean + std * noise

# IDQL: Diffusion policy
action_T = torch.randn_like(action)
for t in reversed(range(T)):
    action_{t-1} = denoise(action_t, t, state)
action = action_0
```

### 2. Training Objective

```python
# IQL: Advantage-weighted BC
weights = exp(beta * (Q(s,a) - V(s)))
loss = -(weights * log π(a|s)).mean()

# IDQL: Advantage-weighted diffusion
weights = exp(beta * (Q(s,a) - V(s)))
loss = weights * ||noise - predicted_noise||²
```

The diffusion model learns to denoise actions, weighted by advantage.

## Implementation Sketch

```python
class IDQLAgent:
    def __init__(self, config):
        # Same as IQL
        self.v_network = ValueNetwork(...)
        self.q_network = TwinQNetwork(...)

        # Different: Diffusion policy
        self.diffusion_policy = DiffusionPolicy(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            num_steps=100,  # Diffusion steps
            beta_schedule="linear"
        )

    def update_policy(self, batch):
        """Train diffusion policy with advantage weighting."""
        states, actions = batch["states"], batch["actions"]

        # Compute advantages (same as IQL)
        advantages = self.q_network.q_min(states, actions) - self.v_network(states)
        weights = torch.exp(self.beta * advantages).clamp(max=100.0)

        # Sample diffusion timesteps
        t = torch.randint(0, self.num_steps, (batch_size,))

        # Add noise to actions (forward diffusion)
        noise = torch.randn_like(actions)
        noisy_actions = self.q_sample(actions, t, noise)

        # Predict noise (reverse diffusion)
        predicted_noise = self.diffusion_policy(noisy_actions, t, states)

        # Advantage-weighted denoising loss
        loss = (weights.detach() * (noise - predicted_noise).pow(2)).mean()

        return loss
```

## Advantages

1. **Multi-modal**: Learns complex, multi-modal policies
2. **Expressiveness**: More expressive than Gaussian
3. **No mode collapse**: Diffusion naturally avoids averaging

## Disadvantages

1. **Slow inference**: Requires T denoising steps (e.g., 100)
2. **More complex**: Harder to implement and debug
3. **Training cost**: More parameters and computation

## When to Use IDQL

**Best for**:
- Multi-modal behavior (e.g., robot grasping)
- Complex action distributions
- Offline datasets with diverse strategies

**Avoid when**:
- Need fast inference
- Unimodal behavior (IQL is simpler)
- Limited compute

## Key Hyperparameters

```python
config = {
    # IQL hyperparameters
    "expectile": 0.7,
    "temperature": 3.0,

    # Diffusion hyperparameters
    "num_diffusion_steps": 100,    # More = better quality, slower
    "beta_schedule": "linear",     # Noise schedule
    "clip_sample": True,           # Clip actions during generation
}
```

## References

```bibtex
@inproceedings{hansen2023idql,
  title={Efficient Diffusion Policies For Offline Reinforcement Learning},
  author={Hansen-Estruch, Philippe and Kostrikov, Ilya and Janner, Michael and Kuba, Jakub Grudzien and Levine, Sergey},
  booktitle={NeurIPS},
  year={2023}
}
```

**Related**:
- IQL: Base algorithm
- Diffusion-QL (Wang et al., 2023): Alternative diffusion offline RL
- Decision Diffuser (Ajay et al., 2022): Sequence modeling with diffusion
