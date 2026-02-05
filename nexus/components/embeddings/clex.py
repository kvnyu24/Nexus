"""
CLEX: Continuous Length Extrapolation via learned dynamics.

CLEX models the position scaling as a continuous dynamics system, learning
a continuous mapping from training-length positions to extended-length
positions. Unlike discrete scaling methods (PI, NTK), CLEX learns a smooth
transformation that adapts to the data.

Key ideas:
    1. Position scaling is modeled as an ODE: df/dt = g(f, t)
       where f is the frequency scaling and t parameterizes context length
    2. A small neural network learns the dynamics function g
    3. The ODE is integrated from t=0 (training length) to t=T (target length)
    4. This continuous formulation allows flexible extrapolation to any length

Benefits:
    - Smooth and continuous scaling (no discrete jumps)
    - Data-adaptive (learned from training data)
    - Generalizes to unseen context lengths
    - Compatible with standard RoPE architecture

Reference: https://arxiv.org/abs/2401.04695 (CLEX: Continuous Length Extrapolation
           for Large Language Models)

See Also:
    - ntk_rope.py: Static non-uniform scaling
    - yarn.py: Piecewise interpolation
    - long_rope.py: Searched per-dimension factors
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class ScalingDynamics(NexusModule):
    """Neural ODE dynamics for continuous position scaling.

    Models the evolution of frequency scaling factors as a continuous
    dynamical system. The dynamics function takes the current scaling
    state and context-length parameter, and outputs the rate of change.

    The ODE: d(scale)/dt = dynamics(scale, t)
    is integrated from t=0 to t=log(target_len / original_len)

    Args:
        dim: Number of frequency dimensions (dim // 2)
        hidden_dim: Hidden dimension of the dynamics network
        num_layers: Number of layers in the dynamics network
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        super().__init__()
        self.dim = dim

        # Dynamics network: maps (scale_state, t) -> d(scale)/dt
        layers = []
        in_dim = dim + 1  # scale_state concatenated with time parameter

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
            in_dim = out_dim

        self.net = nn.Sequential(*layers)

        # Initialize for small initial dynamics (near identity)
        self._init_weights()

    def _init_weights(self):
        """Initialize for near-zero initial dynamics."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        scale_state: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Compute dynamics (rate of change of scaling).

        Args:
            scale_state: Current scaling factors (batch, dim) or (dim,)
            t: Time parameter (scalar or (batch,))

        Returns:
            d(scale)/dt: Rate of change of scaling factors
        """
        if scale_state.dim() == 1:
            scale_state = scale_state.unsqueeze(0)

        # Expand t to match batch dimension
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(scale_state.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)

        # Concatenate state and time
        x = torch.cat([scale_state, t], dim=-1)

        # Compute dynamics
        return self.net(x)


class CLEX(NexusModule):
    """Continuous Length Extrapolation for RoPE.

    Learns a continuous dynamics model that transforms position scaling
    factors from training-length behavior to extended-length behavior.
    The transformation is parameterized by the log-ratio of target to
    original context length.

    Integration is performed using a simple Euler method with a fixed
    number of steps for efficiency. Higher num_steps gives more accurate
    integration at the cost of more computation.

    Args:
        dim: Embedding dimension (must be even)
        max_position: Target maximum position (extended context)
        original_max_position: Original training context length
        base: Base frequency for RoPE
        hidden_dim: Hidden dimension for dynamics network
        num_dynamics_layers: Number of layers in dynamics network
        num_integration_steps: Number of Euler integration steps
        max_scale_train: Maximum scale factor seen during training (for
            normalizing the time parameter)

    Example:
        >>> clex = CLEX(dim=128, max_position=131072, original_max_position=4096)
        >>> x = torch.randn(2, 32768, 1, 128)
        >>> cos, sin = clex(x)
        >>> cos.shape
        torch.Size([1, 32768, 128])
    """

    def __init__(
        self,
        dim: int,
        max_position: int = 131072,
        original_max_position: int = 4096,
        base: float = 10000.0,
        hidden_dim: int = 64,
        num_dynamics_layers: int = 2,
        num_integration_steps: int = 8,
        max_scale_train: Optional[float] = None
    ):
        super().__init__()

        assert dim % 2 == 0, f"dim must be even, got {dim}"

        self.dim = dim
        self.half_dim = dim // 2
        self.max_position = max_position
        self.original_max_position = original_max_position
        self.base = base
        self.num_integration_steps = num_integration_steps

        # Maximum scale ratio
        self.max_scale = max_position / original_max_position
        self.max_scale_train = max_scale_train or self.max_scale

        # Base inverse frequencies (unscaled)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq_base', inv_freq)

        # Learned dynamics model
        self.dynamics = ScalingDynamics(
            dim=self.half_dim,
            hidden_dim=hidden_dim,
            num_layers=num_dynamics_layers
        )

        # Learnable initial scaling state (at t=0, should be near 1.0)
        self.initial_scale = nn.Parameter(torch.ones(self.half_dim))

        # Attention scaling correction
        self.mscale = nn.Parameter(torch.tensor(1.0))

        # Cache
        self._cos_cached = None
        self._sin_cached = None
        self._cached_scale = None
        self._seq_len_cached = 0

    def _compute_time_parameter(self, scale: float) -> float:
        """Convert scale ratio to time parameter for ODE integration.

        Uses log scale for smoother dynamics.

        Args:
            scale: Context extension scale ratio

        Returns:
            Time parameter t
        """
        if scale <= 1.0:
            return 0.0
        return math.log(scale) / math.log(self.max_scale_train)

    def _integrate_dynamics(
        self,
        t_target: float,
        device: torch.device
    ) -> torch.Tensor:
        """Integrate the scaling dynamics from t=0 to t=t_target.

        Uses Euler method with fixed step size for simplicity and
        determinism. Could be upgraded to RK4 for higher accuracy.

        Args:
            t_target: Target time parameter
            device: Device for computation

        Returns:
            Final scaling factors (half_dim,)
        """
        if t_target <= 0.0:
            return self.initial_scale

        # Initialize state
        state = self.initial_scale.unsqueeze(0).to(device)  # (1, half_dim)

        # Integration step size
        dt = t_target / self.num_integration_steps

        # Euler integration
        t = torch.tensor(0.0, device=device)
        for _ in range(self.num_integration_steps):
            # Compute dynamics at current state and time
            d_state = self.dynamics(state, t)

            # Euler step
            state = state + d_state * dt
            t = t + dt

        # Ensure positive scaling factors
        state = F.softplus(state)

        return state.squeeze(0)

    def _compute_scaled_inv_freq(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Compute scaled inverse frequencies for given sequence length.

        Args:
            seq_len: Target sequence length
            device: Device for computation

        Returns:
            Scaled inverse frequencies (half_dim,)
        """
        scale = max(seq_len / self.original_max_position, 1.0)
        t = self._compute_time_parameter(scale)

        # Integrate dynamics to get scaling factors
        scaling_factors = self._integrate_dynamics(t, device)

        # Apply scaling to base frequencies
        scaled_inv_freq = self.inv_freq_base.to(device) / scaling_factors

        return scaled_inv_freq

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CLEX-scaled rotary position embeddings.

        The dynamics model is integrated to determine the appropriate
        per-dimension scaling for the given sequence length.

        Args:
            x: Input tensor (for shape/device inference)
            position_ids: Explicit position indices (batch, seq_len)
            seq_len: Sequence length override

        Returns:
            cos: Cosine embeddings (1 or batch, seq_len, dim)
            sin: Sine embeddings (1 or batch, seq_len, dim)
        """
        if seq_len is None:
            if position_ids is not None:
                seq_len = position_ids.max().item() + 1
            else:
                seq_len = x.shape[1]

        # Check cache
        current_scale = max(seq_len / self.original_max_position, 1.0)
        if (seq_len <= self._seq_len_cached
                and self._cos_cached is not None
                and self._cached_scale == current_scale):
            return (
                self._cos_cached[:, :seq_len, :],
                self._sin_cached[:, :seq_len, :]
            )

        # Compute scaled frequencies via dynamics integration
        scaled_inv_freq = self._compute_scaled_inv_freq(seq_len, x.device)

        # Compute position indices
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=x.device
            ).unsqueeze(0).float()

        # Compute angles
        freqs = position_ids.unsqueeze(-1).float() * scaled_inv_freq.unsqueeze(0)

        # Duplicate for full dimension
        emb = torch.cat([freqs, freqs], dim=-1)

        # Apply learned mscale
        effective_mscale = self.mscale.abs()
        cos = emb.cos() * effective_mscale
        sin = emb.sin() * effective_mscale

        # Update cache
        self._cos_cached = cos
        self._sin_cached = sin
        self._seq_len_cached = seq_len
        self._cached_scale = current_scale

        return cos, sin

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CLEX-scaled rotary embeddings to Q and K.

        Args:
            q: Query tensor (batch, heads, seq, head_dim)
            k: Key tensor (batch, heads, seq, head_dim)
            cos: Cosine embeddings
            sin: Sine embeddings

        Returns:
            Rotated (q, k) tensors
        """
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed

    def get_current_scaling(self, seq_len: int) -> dict:
        """Get the current scaling factors for a given sequence length.

        Useful for debugging and visualization.

        Args:
            seq_len: Target sequence length

        Returns:
            Dict with scaling information
        """
        device = self.inv_freq_base.device
        scale = max(seq_len / self.original_max_position, 1.0)
        t = self._compute_time_parameter(scale)

        scaling_factors = self._integrate_dynamics(t, device)

        return {
            'seq_len': seq_len,
            'scale_ratio': scale,
            'time_parameter': t,
            'scaling_factors': scaling_factors.detach(),
            'min_scaling': scaling_factors.min().item(),
            'max_scaling': scaling_factors.max().item(),
            'mean_scaling': scaling_factors.mean().item(),
            'mscale': self.mscale.item(),
        }
