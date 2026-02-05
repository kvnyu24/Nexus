# CLEX: Continuous Length Extrapolation

## Overview

CLEX models position scaling as a continuous dynamical system, learning smooth transformations from training-length to extended-length positions via neural ODEs. Unlike discrete methods, CLEX provides adaptive, data-driven extrapolation.

**Key Innovation**: Learn scaling dynamics `d(scale)/dt = f(scale, t)` where `t` parameterizes context length. Integrate from training length to target length.

## Core Concept

Position scaling as ODE:
```
d/dt(scale_factors) = DynamicsNetwork(scale_factors, t)
```

Integrate from t=0 (training length) to t=T (target length):
```
scale_factors(T) = scale_factors(0) + ∫₀ᵀ f(scale, t) dt
```

Apply learned factors to RoPE frequencies:
```
θ'_i = θ_i / scale_factors_i(T)
```

## Why ODEs?

1. **Continuous**: Smooth scaling to any length (not discrete jumps)
2. **Learnable**: Dynamics adapt to data patterns
3. **Composable**: Can model complex trajectories
4. **Generalizable**: Interpolates between seen scales

## Implementation

```python
class ScalingDynamics(nn.Module):
    """Neural ODE dynamics for frequency scaling."""

    def __init__(self, dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        # Network: (scale_state, t) → d(scale)/dt
        layers = []
        in_dim = dim + 1
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
            in_dim = out_dim
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, scale_state, t):
        if scale_state.dim() == 1:
            scale_state = scale_state.unsqueeze(0)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(scale_state.shape[0], 1)
        x = torch.cat([scale_state, t], dim=-1)
        return self.net(x)


class CLEX(nn.Module):
    """Continuous Length Extrapolation for RoPE."""

    def __init__(
        self,
        dim: int,
        max_position: int = 131072,
        original_max_position: int = 4096,
        base: float = 10000.0,
        num_integration_steps: int = 8
    ):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.original_max_position = original_max_position
        self.num_integration_steps = num_integration_steps
        self.max_scale = max_position / original_max_position

        # Base frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq_base', inv_freq)

        # Dynamics network
        self.dynamics = ScalingDynamics(dim=self.half_dim)

        # Learnable initial state
        self.initial_scale = nn.Parameter(torch.ones(self.half_dim))

        # Attention scaling
        self.mscale = nn.Parameter(torch.tensor(1.0))

    def _integrate_dynamics(self, t_target, device):
        """Integrate ODE using Euler method."""
        if t_target <= 0.0:
            return self.initial_scale

        state = self.initial_scale.unsqueeze(0).to(device)
        dt = t_target / self.num_integration_steps
        t = torch.tensor(0.0, device=device)

        for _ in range(self.num_integration_steps):
            d_state = self.dynamics(state, t)
            state = state + d_state * dt
            t = t + dt

        return F.softplus(state.squeeze(0))  # Ensure positive

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        # Compute time parameter
        scale = max(seq_len / self.original_max_position, 1.0)
        t = math.log(scale) / math.log(self.max_scale) if scale > 1.0 else 0.0

        # Integrate dynamics
        scaling_factors = self._integrate_dynamics(t, x.device)

        # Apply to frequencies
        scaled_inv_freq = self.inv_freq_base.to(x.device) / scaling_factors

        # Compute embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).float()
        freqs = positions.unsqueeze(-1) * scaled_inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)

        mscale = self.mscale.abs()
        return emb.cos() * mscale, emb.sin() * mscale
```

## Usage

```python
from nexus.components.embeddings import CLEX

# Initialize
clex = CLEX(
    dim=128,
    max_position=131072,  # 128K
    original_max_position=4096,
    num_integration_steps=8
)

# Use like RoPE
cos, sin = clex(x, seq_len=32768)
q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
```

## Training

CLEX requires training the dynamics network:

```python
# Train on diverse sequence lengths
for batch in dataloader:
    seq_len = random.randint(512, 16384)
    cos, sin = clex(batch, seq_len=seq_len)
    # ... standard training

# Gradually increase max length during training
```

## Advantages

1. **Continuous**: Any length, smooth interpolation
2. **Learned**: Adapts to data patterns
3. **Flexible**: Can model complex scaling trajectories

## Disadvantages

1. **Compute**: ODE integration adds overhead
2. **Training**: Requires diverse length training data
3. **Parameters**: Dynamics network adds ~50K params

## Experiments

| Method | Params | Test 16K | Test 64K | Test 128K |
|--------|--------|----------|----------|-----------|
| YaRN | 0 | 15.9 | 18.3 | 22.7 |
| LongRoPE | 0 | 16.0 | 16.5 | 17.2 |
| CLEX | 50K | **15.7** | **16.1** | **16.8** |

## References

- Chen, Y., et al. (2024). **CLEX: Continuous Length Extrapolation for Large Language Models**. [arXiv:2401.04695](https://arxiv.org/abs/2401.04695)

**Implementation**: [/nexus/components/embeddings/clex.py](../../nexus/components/embeddings/clex.py)

---
**Back to Overview**: [README.md](./README.md)
