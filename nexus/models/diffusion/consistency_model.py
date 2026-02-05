"""
Consistency Models for fast generative sampling.

Implements consistency models that learn a function mapping any point on
a diffusion ODE trajectory directly to the trajectory's endpoint (the
clean data). This enables single-step generation while maintaining the
ability to trade compute for quality via multi-step sampling.

Key insight: A consistency function f(x_t, t) should satisfy the
self-consistency property: f(x_t, t) = f(x_{t'}, t') for any t, t'
on the same ODE trajectory. In particular, f(x_t, t) = x_0 for all t.

Key components:
- ConsistencyFunction: Neural network parameterization with boundary
  condition enforcement (f(x, epsilon) = x).
- ConsistencyTraining: Trains from scratch using the consistency loss
  without requiring a pre-trained diffusion model.
- ConsistencyDistillation: Distills a pre-trained diffusion model into
  a consistency model.

References:
    "Consistency Models"
    Song et al., 2023 (https://arxiv.org/abs/2303.01469)

    "Improved Techniques for Training Consistency Models"
    Song & Dhariwal, 2023 (https://arxiv.org/abs/2310.14189)
"""

from typing import Dict, Any, Optional, Tuple, List
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule


def _ema_update(target: nn.Module, source: nn.Module, decay: float) -> None:
    """Update target network parameters with exponential moving average.

    target_param = decay * target_param + (1 - decay) * source_param

    Args:
        target: Target (EMA) network.
        source: Source (online) network.
        decay: EMA decay rate (0 = copy, 1 = no update).
    """
    with torch.no_grad():
        for p_target, p_source in zip(target.parameters(), source.parameters()):
            p_target.data.mul_(decay).add_(p_source.data, alpha=1.0 - decay)


def _karras_schedule(
    num_timesteps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> torch.Tensor:
    """Compute the Karras noise schedule used in consistency models.

    Generates a sequence of noise levels (sigma values) that are
    approximately equally spaced in the perceptual metric defined
    by the Karras schedule.

    sigma_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho

    Args:
        num_timesteps: Number of discrete noise levels.
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.
        rho: Schedule shape parameter (higher = more spacing at high noise).

    Returns:
        Decreasing sequence of sigma values of shape (num_timesteps,).
    """
    ramp = torch.linspace(0, 1, num_timesteps)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


class TimestepEmbedding(nn.Module):
    """Fourier feature embedding for continuous sigma values.

    Maps scalar noise levels to high-dimensional feature vectors
    using sinusoidal embeddings followed by an MLP.

    Args:
        hidden_dim: Output dimension.
        frequency_dim: Dimension of the Fourier features.
    """

    def __init__(self, hidden_dim: int, frequency_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_dim = frequency_dim

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma: Noise levels of shape (B,).

        Returns:
            Embeddings of shape (B, hidden_dim).
        """
        half_dim = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=sigma.device, dtype=torch.float32) / half_dim
        )
        # Use log-sigma for better numerical conditioning
        args = sigma[:, None].float().log() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


class ConsistencyFunction(NexusModule):
    """Neural network parameterization of the consistency function.

    The consistency function f_theta(x, sigma) maps a noisy sample x
    at noise level sigma to the denoised sample x_0. It must satisfy
    the boundary condition f(x, sigma_min) = x (identity at minimum noise).

    This is enforced via skip connection parameterization:
        f(x, sigma) = c_skip(sigma) * x + c_out(sigma) * F_theta(x, sigma)

    where c_skip and c_out are chosen so that c_skip(sigma_min) = 1
    and c_out(sigma_min) = 0.

    Args:
        config: Dictionary containing model hyperparameters.
            - hidden_dim (int): Network hidden dimension. Default: 512.
            - num_layers (int): Number of residual blocks. Default: 8.
            - in_channels (int): Number of input channels. Default: 4.
            - sigma_min (float): Minimum noise level. Default: 0.002.
            - sigma_max (float): Maximum noise level. Default: 80.0.
            - sigma_data (float): Data standard deviation. Default: 0.5.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_dim = config.get("hidden_dim", 512)
        self.num_layers = config.get("num_layers", 8)
        self.in_channels = config.get("in_channels", 4)
        self.sigma_min = config.get("sigma_min", 0.002)
        self.sigma_max = config.get("sigma_max", 80.0)
        self.sigma_data = config.get("sigma_data", 0.5)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(self.hidden_dim)

        # Input projection
        self.input_proj = nn.Conv2d(self.in_channels, self.hidden_dim, 3, padding=1)

        # Residual blocks with timestep conditioning
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(
                ResBlock(self.hidden_dim, self.hidden_dim)
            )

        # Time conditioning projection for each block
        self.time_projs = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(32, self.hidden_dim),
            nn.SiLU(),
            nn.Conv2d(self.hidden_dim, self.in_channels, 3, padding=1),
        )
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        """Skip connection coefficient.

        c_skip(sigma) = sigma_data^2 / (sigma^2 + sigma_data^2)

        At sigma_min (nearly 0), c_skip -> 1 (identity).
        At large sigma, c_skip -> 0 (network output dominates).
        """
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        """Output scaling coefficient.

        c_out(sigma) = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)

        At sigma_min (nearly 0), c_out -> 0 (no network contribution).
        """
        return sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        """Input scaling coefficient for preconditioning.

        c_in(sigma) = 1 / sqrt(sigma^2 + sigma_data^2)
        """
        return 1.0 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the consistency function.

        Computes f(x, sigma) = c_skip(sigma) * x + c_out(sigma) * F_theta(c_in(sigma) * x, sigma)

        Args:
            x: Noisy input of shape (B, C, H, W).
            sigma: Noise levels of shape (B,).

        Returns:
            Dictionary with:
                - "denoised": Predicted clean sample of shape (B, C, H, W).
                - "network_output": Raw network output before skip connection.
        """
        # Preconditioning coefficients
        c_skip = self.c_skip(sigma).view(-1, 1, 1, 1)
        c_out = self.c_out(sigma).view(-1, 1, 1, 1)
        c_in = self.c_in(sigma).view(-1, 1, 1, 1)

        # Scale input
        x_scaled = c_in * x

        # Time embedding
        t_emb = self.time_embed(sigma)

        # Network forward pass
        h = self.input_proj(x_scaled)
        for block, time_proj in zip(self.blocks, self.time_projs):
            h = block(h, time_proj(t_emb))

        network_output = self.output_proj(h)

        # Apply skip connection parameterization
        denoised = c_skip * x + c_out * network_output

        return {
            "denoised": denoised,
            "network_output": network_output,
        }


class ResBlock(nn.Module):
    """Residual block with timestep conditioning via additive bias.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (B, C, H, W).
            t_emb: Time embedding of shape (B, C).

        Returns:
            Output features of shape (B, C_out, H, W).
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # Add time conditioning
        h = h + t_emb[:, :, None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return self.skip(x) + h


class ConsistencyTraining(NexusModule):
    """Consistency Training (CT): train consistency models from scratch.

    Does not require a pre-trained diffusion model. Instead, uses
    the consistency condition directly as a training signal:
        L = d(f_theta(x + sigma_{n+1} * eps, sigma_{n+1}),
              f_{theta^-}(x + sigma_n * eps', sigma_n))

    where f_{theta^-} is an EMA (exponential moving average) copy of
    the model parameters (the "target" network).

    Training schedule:
    - The number of discrete noise levels N(k) increases during training
    - The EMA decay mu(k) also follows a schedule
    - These schedules are crucial for training stability

    Reference: "Consistency Models" (https://arxiv.org/abs/2303.01469)

    Args:
        config: Dictionary containing hyperparameters.
            - sigma_min (float): Minimum noise level. Default: 0.002.
            - sigma_max (float): Maximum noise level. Default: 80.0.
            - sigma_data (float): Data standard deviation. Default: 0.5.
            - num_timesteps (int): Maximum discrete noise levels. Default: 40.
            - initial_timesteps (int): Initial noise levels (increases over training). Default: 2.
            - ema_decay (float): Base EMA decay rate. Default: 0.999.
            - rho (float): Karras schedule parameter. Default: 7.0.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.sigma_min = config.get("sigma_min", 0.002)
        self.sigma_max = config.get("sigma_max", 80.0)
        self.sigma_data = config.get("sigma_data", 0.5)
        self.num_timesteps = config.get("num_timesteps", 40)
        self.initial_timesteps = config.get("initial_timesteps", 2)
        self.ema_decay = config.get("ema_decay", 0.999)
        self.rho = config.get("rho", 7.0)

        # Online model
        self.model = ConsistencyFunction(config)

        # Target (EMA) model
        self.target_model = copy.deepcopy(self.model)
        self.target_model.requires_grad_(False)

        # Training step counter
        self.register_buffer("training_step", torch.tensor(0, dtype=torch.long))

    def get_current_num_timesteps(self, total_training_steps: int) -> int:
        """Compute the current number of noise levels based on training progress.

        N(k) starts at initial_timesteps and increases to num_timesteps
        over the course of training.

        Args:
            total_training_steps: Total number of training steps planned.

        Returns:
            Current number of noise levels.
        """
        progress = min(self.training_step.item() / max(total_training_steps, 1), 1.0)
        current_n = int(
            math.ceil(
                self.initial_timesteps
                + progress * (self.num_timesteps - self.initial_timesteps)
            )
        )
        return max(current_n, 2)  # Need at least 2 levels

    def get_current_ema_decay(self, current_n: int) -> float:
        """Compute EMA decay rate based on current number of timesteps.

        mu = exp(s_0 * log(mu_0) / current_n)
        where s_0 = initial_timesteps and mu_0 = base EMA decay.

        Args:
            current_n: Current number of noise levels.

        Returns:
            Current EMA decay rate.
        """
        return math.exp(
            self.initial_timesteps * math.log(self.ema_decay) / current_n
        )

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the online model.

        Args:
            x: Noisy input of shape (B, C, H, W).
            sigma: Noise levels of shape (B,).

        Returns:
            Dictionary with denoised output.
        """
        return self.model(x, sigma, **kwargs)

    def compute_loss(
        self,
        x_start: torch.Tensor,
        total_training_steps: int = 100000,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute the consistency training loss.

        Samples adjacent noise levels from the schedule, creates noisy
        samples at both levels, and enforces consistency between the
        online and target model outputs.

        Args:
            x_start: Clean data of shape (B, C, H, W).
            total_training_steps: Total planned training steps.
            **kwargs: Additional conditioning.

        Returns:
            Dictionary with:
                - "loss": Scalar consistency loss.
                - "online_denoised": Online model output.
                - "target_denoised": Target model output.
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        # Get current schedule parameters
        current_n = self.get_current_num_timesteps(total_training_steps)
        current_decay = self.get_current_ema_decay(current_n)

        # Compute sigma schedule
        sigmas = _karras_schedule(
            current_n, self.sigma_min, self.sigma_max, self.rho
        ).to(device)

        # Sample adjacent noise level indices
        # n is the index into sigmas, with n ranging from 0 to current_n - 2
        n = torch.randint(0, current_n - 1, (batch_size,), device=device)

        sigma_n = sigmas[n]      # Lower noise level
        sigma_n1 = sigmas[n + 1]   # This is actually lower (schedule is decreasing)
        # Note: Karras schedule is in decreasing order, so sigmas[0] > sigmas[-1]
        # sigma_n is the higher noise level, sigma_n+1 is lower

        # Sample noise
        noise = torch.randn_like(x_start)

        # Create noisy samples at adjacent noise levels
        x_n = x_start + sigma_n.view(-1, 1, 1, 1) * noise
        x_n1 = x_start + sigma_n1.view(-1, 1, 1, 1) * noise

        # Online model prediction at higher noise level
        online_output = self.model(x_n, sigma_n, **kwargs)
        online_denoised = online_output["denoised"]

        # Target model prediction at lower noise level
        with torch.no_grad():
            target_output = self.target_model(x_n1, sigma_n1, **kwargs)
            target_denoised = target_output["denoised"]

        # Consistency loss (LPIPS or MSE)
        loss = F.mse_loss(online_denoised, target_denoised.detach())

        # Weighting by noise level (optional, helps training)
        weight = 1.0 / (sigma_n - sigma_n1 + 1e-8)
        loss = (loss * weight.view(-1, 1, 1, 1)).mean()

        # Update EMA target model
        _ema_update(self.target_model, self.model, current_decay)

        # Increment step counter
        self.training_step += 1

        return {
            "loss": loss,
            "online_denoised": online_denoised,
            "target_denoised": target_denoised,
            "current_n": current_n,
            "current_decay": current_decay,
        }

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 1,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using the consistency model.

        Single-step generation:
            x_0 = f(sigma_max * eps, sigma_max)  where eps ~ N(0, I)

        Multi-step generation iteratively denoises and re-adds noise
        at decreasing noise levels for improved quality.

        Args:
            shape: Shape of samples (B, C, H, W).
            num_steps: Number of generation steps (1 for single-step).
            device: Target device.
            **kwargs: Additional conditioning.

        Returns:
            Generated samples of shape matching `shape`.
        """
        if device is None:
            device = next(self.parameters()).device

        # Start from pure noise at maximum noise level
        x = torch.randn(*shape, device=device) * self.sigma_max

        if num_steps == 1:
            # Single-step generation
            sigma = torch.full((shape[0],), self.sigma_max, device=device)
            output = self.model(x, sigma, **kwargs)
            return output["denoised"]

        # Multi-step generation
        sigmas = _karras_schedule(
            num_steps + 1, self.sigma_min, self.sigma_max, self.rho
        ).to(device)

        for i in range(num_steps):
            sigma = sigmas[i]
            sigma_batch = torch.full((shape[0],), sigma.item(), device=device)

            # Denoise
            output = self.model(x, sigma_batch, **kwargs)
            x_denoised = output["denoised"]

            if i < num_steps - 1:
                # Add noise at the next (lower) noise level
                sigma_next = sigmas[i + 1]
                noise = torch.randn_like(x)
                x = x_denoised + sigma_next * noise
            else:
                x = x_denoised

        return x


class ConsistencyDistillation(NexusModule):
    """Consistency Distillation (CD): distill a pre-trained diffusion model.

    Uses a pre-trained score/noise prediction network (teacher) to
    provide one-step denoising targets, then trains a consistency model
    (student) to match these targets across adjacent noise levels.

    The teacher provides a more accurate estimate of x_0 by performing
    one denoising step, giving stronger training signal than CT.

    Training procedure:
    1. Sample noise level sigma_n from the schedule
    2. Create noisy sample: x_n = x_0 + sigma_n * eps
    3. Use teacher to estimate x_hat from x_n at sigma_{n+1} (one DDIM step)
    4. Train student: f_theta(x_n, sigma_n) = f_{theta^-}(x_hat, sigma_{n+1})

    Reference: "Consistency Models" (https://arxiv.org/abs/2303.01469)

    Args:
        config: Dictionary containing hyperparameters.
            - sigma_min (float): Minimum noise level. Default: 0.002.
            - sigma_max (float): Maximum noise level. Default: 80.0.
            - sigma_data (float): Data standard deviation. Default: 0.5.
            - num_timesteps (int): Number of discrete noise levels. Default: 40.
            - ema_decay (float): EMA decay for target network. Default: 0.9999.
            - rho (float): Karras schedule parameter. Default: 7.0.
        teacher: Pre-trained diffusion model (predicts noise given (x, sigma)).
    """

    def __init__(self, config: Dict[str, Any], teacher: nn.Module):
        super().__init__(config)

        self.sigma_min = config.get("sigma_min", 0.002)
        self.sigma_max = config.get("sigma_max", 80.0)
        self.sigma_data = config.get("sigma_data", 0.5)
        self.num_timesteps = config.get("num_timesteps", 40)
        self.ema_decay = config.get("ema_decay", 0.9999)
        self.rho = config.get("rho", 7.0)

        # Teacher model (frozen pre-trained diffusion model)
        self.teacher = teacher
        self.teacher.requires_grad_(False)

        # Student (online) consistency model
        self.model = ConsistencyFunction(config)

        # Target (EMA) consistency model
        self.target_model = copy.deepcopy(self.model)
        self.target_model.requires_grad_(False)

        # Pre-compute sigma schedule
        sigmas = _karras_schedule(
            self.num_timesteps, self.sigma_min, self.sigma_max, self.rho
        )
        self.register_buffer("sigmas", sigmas)

    @torch.no_grad()
    def _teacher_denoise_step(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> torch.Tensor:
        """Use the teacher model to take one denoising step (DDIM-style).

        Estimates x at a lower noise level using the teacher's noise prediction.

        Args:
            x: Noisy input at noise level sigma.
            sigma: Current noise level.
            sigma_next: Target (lower) noise level.

        Returns:
            Denoised estimate at sigma_next.
        """
        # Teacher predicts noise
        teacher_output = self.teacher(x, sigma)
        if isinstance(teacher_output, dict):
            noise_pred = teacher_output.get("noise_pred", teacher_output.get("prediction", next(iter(teacher_output.values()))))
        else:
            noise_pred = teacher_output

        # Estimate x_0 from noise prediction
        x0_pred = x - sigma.view(-1, 1, 1, 1) * noise_pred

        # DDIM step to sigma_next
        x_next = x0_pred + sigma_next.view(-1, 1, 1, 1) * noise_pred

        return x_next

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the online consistency model.

        Args:
            x: Noisy input of shape (B, C, H, W).
            sigma: Noise levels of shape (B,).

        Returns:
            Dictionary with denoised output.
        """
        return self.model(x, sigma, **kwargs)

    def compute_loss(
        self,
        x_start: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute the consistency distillation loss.

        Uses the teacher to provide denoising targets and enforces
        consistency between online and target model predictions.

        Args:
            x_start: Clean data of shape (B, C, H, W).
            **kwargs: Additional conditioning.

        Returns:
            Dictionary with:
                - "loss": Scalar distillation loss.
                - "online_denoised": Student (online) output.
                - "target_denoised": Target (EMA) output.
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        # Sample adjacent noise level indices
        n = torch.randint(0, self.num_timesteps - 1, (batch_size,), device=device)

        sigma_n = self.sigmas[n]
        sigma_n1 = self.sigmas[n + 1]

        # Sample noise and create noisy input at sigma_n
        noise = torch.randn_like(x_start)
        x_n = x_start + sigma_n.view(-1, 1, 1, 1) * noise

        # Teacher one-step denoising: x_n at sigma_n -> x_hat at sigma_{n+1}
        x_hat = self._teacher_denoise_step(x_n, sigma_n, sigma_n1)

        # Online model at (x_n, sigma_n)
        online_output = self.model(x_n, sigma_n, **kwargs)
        online_denoised = online_output["denoised"]

        # Target model at (x_hat, sigma_{n+1})
        with torch.no_grad():
            target_output = self.target_model(x_hat, sigma_n1, **kwargs)
            target_denoised = target_output["denoised"]

        # Distillation loss
        loss = F.mse_loss(online_denoised, target_denoised.detach())

        # Update EMA target
        _ema_update(self.target_model, self.model, self.ema_decay)

        return {
            "loss": loss,
            "online_denoised": online_denoised,
            "target_denoised": target_denoised,
        }

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 1,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples (same interface as ConsistencyTraining).

        Args:
            shape: Shape of samples (B, C, H, W).
            num_steps: Number of generation steps.
            device: Target device.
            **kwargs: Additional conditioning.

        Returns:
            Generated samples.
        """
        if device is None:
            device = next(self.parameters()).device

        x = torch.randn(*shape, device=device) * self.sigma_max

        if num_steps == 1:
            sigma = torch.full((shape[0],), self.sigma_max, device=device)
            output = self.model(x, sigma, **kwargs)
            return output["denoised"]

        # Multi-step sampling
        sigmas = _karras_schedule(
            num_steps + 1, self.sigma_min, self.sigma_max, self.rho
        ).to(device)

        for i in range(num_steps):
            sigma = sigmas[i]
            sigma_batch = torch.full((shape[0],), sigma.item(), device=device)

            output = self.model(x, sigma_batch, **kwargs)
            x_denoised = output["denoised"]

            if i < num_steps - 1:
                sigma_next = sigmas[i + 1]
                noise = torch.randn_like(x)
                x = x_denoised + sigma_next * noise
            else:
                x = x_denoised

        return x
