"""
Rectified Flow for generative modeling.

Implements the rectified flow framework, which learns a velocity field
along straight-line (linear interpolation) paths between the noise
and data distributions. The key insight is that linear paths are the
shortest paths in Euclidean space, leading to efficient transport.

The reflow procedure iteratively straightens learned trajectories by
using the model's own generated pairs (x_0, x_1) as new training data,
enabling progressively fewer-step generation.

Key components:
- RectifiedFlowTrainer: Trains a velocity field along linear interpolation
  paths between noise and data.
- ReflowProcedure: Iteratively straightens flow trajectories by
  re-pairing generated noise-data pairs.

References:
    "Flow Straight and Fast: Learning to Generate and Transfer Data
     with Rectified Flows"
    Liu et al., 2023 (https://arxiv.org/abs/2209.03003)

    "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
    Esser et al., 2024 (https://arxiv.org/abs/2403.03206)
"""

from typing import Dict, Any, Optional, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule


class RectifiedFlowTrainer(NexusModule):
    """Trains a neural network to predict the velocity field of a rectified flow.

    The training objective learns a velocity field v_theta such that the ODE
        dx/dt = v_theta(x, t)
    transports samples from noise (t=0) to data (t=1) along approximately
    straight-line paths.

    Training procedure:
    1. Sample data x_1 and noise x_0 ~ N(0, I)
    2. Sample time t ~ U[0, 1]
    3. Compute interpolation: x_t = (1 - t) * x_0 + t * x_1
    4. Predict velocity: v_pred = network(x_t, t)
    5. Regress against target velocity: v_target = x_1 - x_0
    6. Loss = ||v_pred - v_target||^2

    The key property is that the target velocity is constant along
    each linear path, making the regression problem well-conditioned.

    Reference: "Flow Straight and Fast" (https://arxiv.org/abs/2209.03003)

    Args:
        config: Dictionary containing hyperparameters.
            - num_timesteps (int): Discretization steps for sampling. Default: 1000.
            - sigma_min (float): Minimum noise scale. Default: 0.0.
        network: Neural network backbone that accepts (x_t, t, **kwargs)
            and returns velocity predictions.
    """

    def __init__(self, config: Dict[str, Any], network: nn.Module):
        super().__init__(config)

        self.network = network
        self.num_timesteps = config.get("num_timesteps", 1000)
        self.sigma_min = config.get("sigma_min", 0.0)

    def interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Linear interpolation between noise and data.

        Computes x_t = (1 - t) * x_0 + t * x_1, the straight-line
        path from noise to data.

        Args:
            x_0: Noise samples of shape (B, ...).
            x_1: Data samples of shape (B, ...).
            t: Time values of shape (B, 1, ...) in [0, 1].

        Returns:
            Interpolated samples x_t of shape (B, ...).
        """
        return (1.0 - t) * x_0 + t * x_1

    def target_velocity(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the target velocity along the linear path.

        For straight-line transport, the velocity is constant:
            v = x_1 - x_0

        Args:
            x_0: Noise samples of shape (B, ...).
            x_1: Data samples of shape (B, ...).

        Returns:
            Target velocity of shape (B, ...).
        """
        return x_1 - x_0

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Predict velocity using the neural network.

        Args:
            x_t: Current state of shape (B, ...).
            t: Time values of shape (B,) or scalar.
            **kwargs: Additional conditioning for the network.

        Returns:
            Predicted velocity of shape (B, ...).
        """
        if isinstance(t, (int, float)):
            t = torch.full(
                (x_t.shape[0],), t, device=x_t.device, dtype=x_t.dtype
            )
        output = self.network(x_t, t, **kwargs)
        if isinstance(output, dict):
            return output.get("prediction", output.get("sample", next(iter(output.values()))))
        return output

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: predict velocity field.

        Args:
            x_t: Noisy/interpolated samples of shape (B, ...).
            t: Time values of shape (B,).
            **kwargs: Additional conditioning.

        Returns:
            Dictionary with "velocity": predicted velocity field.
        """
        velocity = self.predict_velocity(x_t, t, **kwargs)
        return {"velocity": velocity}

    def compute_loss(
        self,
        x_1: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute the rectified flow training loss.

        Samples random time, computes linear interpolation, and regresses
        predicted velocity against the constant target velocity x_1 - x_0.

        Args:
            x_1: Data samples of shape (B, ...).
            x_0: Noise samples of shape (B, ...). If None, sampled from N(0, I).
            **kwargs: Additional conditioning for the network.

        Returns:
            Dictionary with:
                - "loss": Scalar MSE loss.
                - "velocity_pred": Predicted velocity.
                - "velocity_target": Target velocity.
                - "x_t": Interpolated sample.
                - "t": Sampled time values.
        """
        batch_size = x_1.shape[0]
        device = x_1.device

        if x_0 is None:
            x_0 = torch.randn_like(x_1)

        # Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # Reshape for broadcasting
        dims = [1] * (x_1.dim() - 1)
        t_expanded = t.view(batch_size, *dims)

        # Compute interpolation and target velocity
        x_t = self.interpolate(x_0, x_1, t_expanded)
        v_target = self.target_velocity(x_0, x_1)

        # Optionally add small noise for regularization
        if self.sigma_min > 0:
            x_t = x_t + self.sigma_min * torch.randn_like(x_t)

        # Predict velocity
        v_pred = self.predict_velocity(x_t, t, **kwargs)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        return {
            "loss": loss,
            "velocity_pred": v_pred,
            "velocity_target": v_target,
            "x_t": x_t,
            "t": t,
        }

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_steps: Optional[int] = None,
        x_0: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using Euler integration of the learned ODE.

        Integrates dx/dt = v_theta(x, t) from t=0 (noise) to t=1 (data)
        using uniform time steps.

        Args:
            shape: Shape of samples to generate.
            num_steps: Number of Euler steps. Default: self.num_timesteps.
            x_0: Initial noise. If None, sampled from N(0, I).
            device: Target device.
            **kwargs: Additional conditioning.

        Returns:
            Generated samples of shape matching `shape`.
        """
        num_steps = num_steps or self.num_timesteps
        if device is None:
            device = next(self.parameters()).device

        if x_0 is None:
            x_0 = torch.randn(*shape, device=device)

        x = x_0
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = i * dt
            t_tensor = torch.full(
                (shape[0],), t, device=device, dtype=torch.float32
            )
            v = self.predict_velocity(x, t_tensor, **kwargs)
            x = x + dt * v

        return x

    @torch.no_grad()
    def sample_few_step(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 1,
        x_0: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples with very few steps (works best after reflow).

        After sufficient reflow iterations, the learned flow becomes
        nearly straight, enabling single-step or few-step generation.

        Args:
            shape: Shape of samples.
            num_steps: Number of steps (1 for single-step generation).
            x_0: Initial noise.
            device: Target device.
            **kwargs: Additional conditioning.

        Returns:
            Generated samples.
        """
        return self.sample(shape, num_steps=num_steps, x_0=x_0, device=device, **kwargs)


class ReflowProcedure(NexusModule):
    """Iteratively straightens rectified flow trajectories.

    The reflow procedure works by:
    1. Using the current flow model to generate noise-data pairs:
       Given x_0 ~ N(0, I), generate x_1 = ODE(x_0, t=0->1)
    2. Training a new flow model on these synthetic pairs (x_0, x_1)
    3. Repeating, producing progressively straighter trajectories

    After k reflow iterations, the flow trajectories become approximately
    k times straighter, enabling generation with 1/k fewer steps.

    This is analogous to progressive distillation but operates in the
    continuous flow framework.

    Reference: "Flow Straight and Fast" (https://arxiv.org/abs/2209.03003)

    Args:
        config: Dictionary containing hyperparameters.
            - num_timesteps (int): ODE integration steps for pair generation.
            - reflow_steps (int): Steps used when generating reflow pairs.
            - sigma_min (float): Noise regularization.
        source_model: Pre-trained RectifiedFlowTrainer to generate pairs.
        target_network: New neural network to train on reflowed pairs.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        source_model: RectifiedFlowTrainer,
        target_network: nn.Module,
    ):
        super().__init__(config)

        self.source_model = source_model
        self.source_model.freeze()  # Freeze source model
        self.target_trainer = RectifiedFlowTrainer(config, target_network)
        self.reflow_steps = config.get("reflow_steps", 100)

    @torch.no_grad()
    def generate_reflow_pairs(
        self,
        batch_size: int,
        shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate noise-data pairs using the source flow model.

        Samples noise x_0 ~ N(0, I) and integrates the source model's
        ODE to generate corresponding x_1 samples, creating new
        training pairs for the target model.

        Args:
            batch_size: Number of pairs to generate.
            shape: Shape of each sample (excluding batch dimension).
            device: Target device.
            **kwargs: Additional conditioning for the source model.

        Returns:
            Tuple of (x_0, x_1) where x_0 is the noise and x_1 is
            the generated data from the source model.
        """
        if device is None:
            device = next(self.source_model.parameters()).device

        full_shape = (batch_size, *shape)
        x_0 = torch.randn(*full_shape, device=device)

        # Generate x_1 by integrating the source model
        x_1 = self.source_model.sample(
            full_shape,
            num_steps=self.reflow_steps,
            x_0=x_0.clone(),
            device=device,
            **kwargs,
        )

        return x_0, x_1

    def compute_loss(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute the reflow training loss on pre-generated pairs.

        Takes noise-data pairs (either from generate_reflow_pairs or
        a pre-cached dataset) and trains the target model.

        Args:
            x_0: Noise samples of shape (B, ...).
            x_1: Corresponding data samples of shape (B, ...).
            **kwargs: Additional conditioning.

        Returns:
            Dictionary with loss and diagnostics (same as RectifiedFlowTrainer).
        """
        return self.target_trainer.compute_loss(x_1, x_0=x_0, **kwargs)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the target (reflowed) model.

        Args:
            x_t: Input state of shape (B, ...).
            t: Time values of shape (B,).
            **kwargs: Additional conditioning.

        Returns:
            Dictionary with "velocity": predicted velocity.
        """
        return self.target_trainer(x_t, t, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_steps: Optional[int] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using the target (reflowed) model.

        Args:
            shape: Shape of samples to generate.
            num_steps: Number of ODE steps.
            device: Target device.
            **kwargs: Additional conditioning.

        Returns:
            Generated samples.
        """
        return self.target_trainer.sample(
            shape, num_steps=num_steps, device=device, **kwargs
        )

    def compute_straightness(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        num_points: int = 10,
        **kwargs,
    ) -> torch.Tensor:
        """Measure the straightness of learned trajectories.

        Computes the ratio of displacement to path length. A value of
        1.0 indicates a perfectly straight path. Lower values indicate
        curvature in the learned trajectory.

        Straightness = ||x_1 - x_0|| / sum(||x_{t+1} - x_t||)

        Args:
            x_0: Starting noise of shape (B, ...).
            x_1: Target data of shape (B, ...).
            num_points: Number of points to evaluate along the trajectory.
            **kwargs: Additional conditioning.

        Returns:
            Straightness score (scalar tensor, 1.0 = perfectly straight).
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Compute displacement (straight-line distance)
        displacement = (x_1 - x_0).reshape(batch_size, -1).norm(dim=-1)

        # Compute path length by integrating along the trajectory
        x = x_0.clone()
        path_length = torch.zeros(batch_size, device=device)
        dt = 1.0 / num_points

        for i in range(num_points):
            t = i * dt
            t_tensor = torch.full((batch_size,), t, device=device)
            v = self.target_trainer.predict_velocity(x, t_tensor, **kwargs)
            step = dt * v
            path_length += step.reshape(batch_size, -1).norm(dim=-1)
            x = x + step

        straightness = displacement / (path_length + 1e-8)
        return straightness.mean()
