"""
Flow Matching for generative modeling.

Implements Conditional Flow Matching (CFM) and Optimal Transport
Conditional Flow Matching (OT-CFM) for training continuous normalizing
flows (CNFs) in a simulation-free manner. Instead of learning a score
function, flow matching directly regresses a velocity field that
transports samples from a noise distribution to the data distribution.

Key components:
- ConditionalFlowMatcher: Standard Gaussian conditional flow matching
- OTPFlowMatcher: Optimal transport conditional flow matching with
  mini-batch OT coupling for straighter transport paths
- FlowMatchingModel: Wraps any neural network backbone for flow matching
  training and ODE-based sampling (Euler, Heun, RK45)

References:
    "Flow Matching for Generative Modeling"
    Lipman et al., 2023 (https://arxiv.org/abs/2210.02747)

    "Improving and Generalizing Flow-Based Generative Models
     with Minibatch Optimal Transport"
    Tong et al., 2023 (https://arxiv.org/abs/2302.00482)
"""

from typing import Dict, Any, Optional, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule


class ConditionalFlowMatcher:
    """Conditional Flow Matching (CFM) with Gaussian probability paths.

    Defines conditional probability paths p_t(x | x_1) that interpolate
    between a standard Gaussian prior p_0 = N(0, I) and a Dirac delta
    centered at data point x_1. The conditional vector field u_t(x | x_1)
    generates these paths and serves as the regression target.

    For the Gaussian path:
        mu_t(x_1) = t * x_1
        sigma_t = 1 - (1 - sigma_min) * t

    The conditional velocity field is:
        u_t(x | x_1) = (x_1 - (1 - sigma_min) * x) / (1 - (1 - sigma_min) * t)

    Args:
        sigma_min: Minimum standard deviation at t=1. Default: 1e-5.
    """

    def __init__(self, sigma_min: float = 1e-5):
        self.sigma_min = sigma_min

    def compute_mu_t(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute the conditional mean at time t.

        Args:
            x_0: Source samples (noise) of shape (B, ...).
            x_1: Target samples (data) of shape (B, ...).
            t: Time values of shape (B, 1, ...) in [0, 1].

        Returns:
            Conditional mean mu_t of shape (B, ...).
        """
        return t * x_1 + (1.0 - t) * x_0

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the conditional standard deviation at time t.

        For the standard Gaussian path, sigma_t = 1 - (1 - sigma_min) * t.
        At t=0 we have sigma_0=1 (pure noise), at t=1 we have sigma_1=sigma_min.

        Args:
            t: Time values of shape (B, 1, ...) in [0, 1].

        Returns:
            Conditional sigma_t.
        """
        del t  # sigma is constant in the simplest case
        return self.sigma_min

    def sample_path(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the conditional probability path p_t(x | x_0, x_1).

        Draws x_t ~ N(mu_t, sigma_t^2 I) along the interpolation path.

        Args:
            x_0: Source samples (noise) of shape (B, ...).
            x_1: Target samples (data) of shape (B, ...).
            t: Time values of shape (B, 1, ...) in [0, 1].

        Returns:
            Samples x_t of shape (B, ...).
        """
        mu_t = self.compute_mu_t(x_0, x_1, t)
        sigma_t = self.compute_sigma_t(t)
        epsilon = torch.randn_like(x_0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_velocity(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the conditional velocity field u_t(x | x_0, x_1).

        For the linear interpolation path, the target velocity is simply:
            u_t = x_1 - x_0

        This is the regression target for training.

        Args:
            x_0: Source samples (noise) of shape (B, ...).
            x_1: Target samples (data) of shape (B, ...).
            t: Time values of shape (B, 1, ...) in [0, 1].

        Returns:
            Target velocity of shape (B, ...).
        """
        _ = t  # Velocity is constant for linear paths
        return x_1 - (1.0 - self.sigma_min) * x_0

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random time values uniformly from [0, 1].

        Args:
            batch_size: Number of samples.
            device: Target device.

        Returns:
            Time values of shape (B, 1).
        """
        return torch.rand(batch_size, 1, device=device)


class OTPFlowMatcher(ConditionalFlowMatcher):
    """Optimal Transport Conditional Flow Matching (OT-CFM).

    Extends standard CFM by using mini-batch optimal transport to
    find better couplings between source and target samples. Instead
    of independently pairing noise and data samples, OT-CFM solves
    a discrete OT problem within each mini-batch to produce couplings
    that result in straighter, more efficient transport paths.

    This leads to:
    - Faster convergence during training
    - Straighter learned trajectories
    - Better few-step generation quality

    Args:
        sigma_min: Minimum standard deviation at t=1. Default: 1e-5.
        ot_reg: Entropic regularization for Sinkhorn OT. Default: 0.05.
        ot_iterations: Number of Sinkhorn iterations. Default: 50.
    """

    def __init__(
        self,
        sigma_min: float = 1e-5,
        ot_reg: float = 0.05,
        ot_iterations: int = 50,
    ):
        super().__init__(sigma_min=sigma_min)
        self.ot_reg = ot_reg
        self.ot_iterations = ot_iterations

    def compute_ot_coupling(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute optimal transport coupling using Sinkhorn algorithm.

        Finds the permutation of x_0 that minimizes the total squared
        Euclidean distance to x_1, producing straighter transport paths.

        Args:
            x_0: Source samples of shape (B, ...).
            x_1: Target samples of shape (B, ...).

        Returns:
            Tuple of (reordered_x_0, x_1) with OT-optimal pairing.
        """
        B = x_0.shape[0]
        x_0_flat = x_0.reshape(B, -1)
        x_1_flat = x_1.reshape(B, -1)

        # Compute pairwise squared distances
        cost = torch.cdist(x_0_flat, x_1_flat, p=2).pow(2)

        # Sinkhorn iterations for entropic OT
        log_a = torch.zeros(B, device=x_0.device)
        log_b = torch.zeros(B, device=x_0.device)

        M = -cost / self.ot_reg

        for _ in range(self.ot_iterations):
            # Row normalization
            log_a = -torch.logsumexp(M + log_b.unsqueeze(0), dim=1)
            # Column normalization
            log_b = -torch.logsumexp(M + log_a.unsqueeze(1), dim=0)

        # Compute transport plan
        log_plan = M + log_a.unsqueeze(1) + log_b.unsqueeze(0)
        plan = torch.exp(log_plan)

        # Use argmax coupling (hard assignment)
        indices = plan.argmax(dim=0)
        x_0_reordered = x_0[indices]

        return x_0_reordered, x_1

    def sample_path(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the OT-coupled conditional probability path.

        First computes OT coupling, then samples along the path.

        Args:
            x_0: Source samples of shape (B, ...).
            x_1: Target samples of shape (B, ...).
            t: Time values of shape (B, 1, ...) in [0, 1].

        Returns:
            Samples x_t with OT-optimal coupling.
        """
        x_0_ot, x_1 = self.compute_ot_coupling(x_0, x_1)
        return super().sample_path(x_0_ot, x_1, t)

    def compute_conditional_velocity(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity field with OT coupling.

        Args:
            x_0: Source samples of shape (B, ...).
            x_1: Target samples of shape (B, ...).
            t: Time values in [0, 1].

        Returns:
            OT-coupled target velocity.
        """
        x_0_ot, x_1 = self.compute_ot_coupling(x_0, x_1)
        return super().compute_conditional_velocity(x_0_ot, x_1, t)


def _euler_step(
    model_fn: Callable,
    x: torch.Tensor,
    t: float,
    dt: float,
    **kwargs,
) -> torch.Tensor:
    """Single Euler integration step: x_{t+dt} = x_t + dt * v(x_t, t)."""
    v = model_fn(x, t, **kwargs)
    return x + dt * v


def _heun_step(
    model_fn: Callable,
    x: torch.Tensor,
    t: float,
    dt: float,
    **kwargs,
) -> torch.Tensor:
    """Single Heun (improved Euler / trapezoidal) integration step.

    Performs a predictor-corrector step for second-order accuracy:
        x_predict = x_t + dt * v(x_t, t)
        x_{t+dt} = x_t + dt/2 * (v(x_t, t) + v(x_predict, t+dt))
    """
    v1 = model_fn(x, t, **kwargs)
    x_predict = x + dt * v1
    v2 = model_fn(x_predict, t + dt, **kwargs)
    return x + 0.5 * dt * (v1 + v2)


def _rk45_step(
    model_fn: Callable,
    x: torch.Tensor,
    t: float,
    dt: float,
    **kwargs,
) -> torch.Tensor:
    """Single Runge-Kutta 4th order (RK4) integration step.

    Classical 4-stage RK method for higher-order accuracy:
        k1 = v(x, t)
        k2 = v(x + dt/2 * k1, t + dt/2)
        k3 = v(x + dt/2 * k2, t + dt/2)
        k4 = v(x + dt * k3, t + dt)
        x_{t+dt} = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    k1 = model_fn(x, t, **kwargs)
    k2 = model_fn(x + 0.5 * dt * k1, t + 0.5 * dt, **kwargs)
    k3 = model_fn(x + 0.5 * dt * k2, t + 0.5 * dt, **kwargs)
    k4 = model_fn(x + dt * k3, t + dt, **kwargs)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


ODE_SOLVERS = {
    "euler": _euler_step,
    "heun": _heun_step,
    "rk45": _rk45_step,
}


class FlowMatchingModel(NexusModule):
    """Wraps a neural network for flow matching training and sampling.

    Provides a unified interface for:
    1. Computing the flow matching training loss
    2. Generating samples via ODE integration
    3. Supporting multiple ODE solvers (Euler, Heun, RK45)
    4. Both standard CFM and OT-CFM training

    The wrapped network should accept (x_t, t, **conditioning) and
    return a velocity prediction of the same shape as x_t.

    Architecture:
        - Any backbone network (DiT, MMDiT, U-Net, etc.)
        - ConditionalFlowMatcher or OTPFlowMatcher for path definition
        - ODE solver for sample generation

    Reference: "Flow Matching for Generative Modeling"
               Lipman et al., 2023 (https://arxiv.org/abs/2210.02747)

    Args:
        config: Dictionary containing model hyperparameters.
            - sigma_min (float): Minimum noise level. Default: 1e-5.
            - solver (str): ODE solver type ('euler', 'heun', 'rk45'). Default: 'euler'.
            - num_steps (int): Number of ODE integration steps. Default: 50.
            - use_ot (bool): Whether to use OT-CFM. Default: False.
            - ot_reg (float): Sinkhorn regularization. Default: 0.05.
            - ot_iterations (int): Sinkhorn iterations. Default: 50.
    """

    def __init__(self, config: Dict[str, Any], network: nn.Module):
        super().__init__(config)

        self.network = network
        self.sigma_min = config.get("sigma_min", 1e-5)
        self.solver_name = config.get("solver", "euler")
        self.num_steps = config.get("num_steps", 50)
        self.use_ot = config.get("use_ot", False)

        if self.solver_name not in ODE_SOLVERS:
            raise ValueError(
                f"Unknown solver '{self.solver_name}'. "
                f"Supported: {list(ODE_SOLVERS.keys())}"
            )

        # Initialize the appropriate flow matcher
        if self.use_ot:
            self.flow_matcher = OTPFlowMatcher(
                sigma_min=self.sigma_min,
                ot_reg=config.get("ot_reg", 0.05),
                ot_iterations=config.get("ot_iterations", 50),
            )
        else:
            self.flow_matcher = ConditionalFlowMatcher(sigma_min=self.sigma_min)

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Predict the velocity field at (x_t, t).

        Args:
            x_t: Current state of shape (B, ...).
            t: Time values of shape (B,) or scalar.
            **kwargs: Additional conditioning arguments for the network.

        Returns:
            Predicted velocity of shape (B, ...).
        """
        if isinstance(t, (int, float)):
            t = torch.full(
                (x_t.shape[0],), t, device=x_t.device, dtype=x_t.dtype
            )
        output = self.network(x_t, t, **kwargs)
        # Support both dict and tensor outputs from the backbone
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
            x_t: Noisy samples of shape (B, ...).
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
        """Compute the flow matching training loss.

        Samples random time t ~ U[0,1], computes x_t along the
        conditional path, and regresses the velocity field against
        the target conditional velocity.

        Args:
            x_1: Data samples of shape (B, ...).
            x_0: Noise samples of shape (B, ...). If None, sampled from N(0,I).
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

        # Sample time
        t = self.flow_matcher.sample_time(batch_size, device)

        # Reshape t for broadcasting
        dims = [1] * (x_1.dim() - 1)
        t_expanded = t.view(batch_size, *dims)

        # Compute target velocity and sample along path
        if self.use_ot:
            # OT coupling reorders x_0 internally
            x_0_ot, _ = self.flow_matcher.compute_ot_coupling(x_0, x_1)
            x_t = self.flow_matcher.compute_mu_t(x_0_ot, x_1, t_expanded)
            x_t = x_t + self.flow_matcher.compute_sigma_t(t_expanded) * torch.randn_like(x_t)
            velocity_target = x_1 - (1.0 - self.sigma_min) * x_0_ot
        else:
            x_t = self.flow_matcher.sample_path(x_0, x_1, t_expanded)
            velocity_target = self.flow_matcher.compute_conditional_velocity(x_0, x_1, t_expanded)

        # Predict velocity
        velocity_pred = self.predict_velocity(x_t, t.squeeze(-1), **kwargs)

        # MSE loss
        loss = F.mse_loss(velocity_pred, velocity_target)

        return {
            "loss": loss,
            "velocity_pred": velocity_pred,
            "velocity_target": velocity_target,
            "x_t": x_t,
            "t": t,
        }

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_steps: Optional[int] = None,
        solver: Optional[str] = None,
        x_0: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples by integrating the learned ODE from t=0 to t=1.

        Numerically integrates dx/dt = v_theta(x, t) from the noise
        distribution (t=0) to the data distribution (t=1).

        Args:
            shape: Shape of samples to generate (B, C, H, W) or (B, D).
            num_steps: Number of integration steps. Default: self.num_steps.
            solver: ODE solver name. Default: self.solver_name.
            x_0: Initial noise. If None, sampled from N(0, I).
            device: Target device.
            **kwargs: Additional conditioning arguments for the network.

        Returns:
            Generated samples of shape matching `shape`.
        """
        num_steps = num_steps or self.num_steps
        solver = solver or self.solver_name
        solver_fn = ODE_SOLVERS[solver]

        if device is None:
            device = next(self.parameters()).device

        if x_0 is None:
            x_0 = torch.randn(*shape, device=device)
        x = x_0

        dt = 1.0 / num_steps
        timesteps = torch.linspace(0.0, 1.0 - dt, num_steps, device=device)

        for t_val in timesteps:
            def model_fn(x_in, t_in, **kw):
                return self.predict_velocity(x_in, t_in, **kw)

            x = solver_fn(model_fn, x, t_val.item(), dt, **kwargs)

        return x

    @torch.no_grad()
    def sample_trajectory(
        self,
        shape: Tuple[int, ...],
        num_steps: Optional[int] = None,
        solver: Optional[str] = None,
        x_0: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples and return the full trajectory.

        Useful for visualization and analysis of the transport path.

        Args:
            shape: Shape of samples to generate.
            num_steps: Number of integration steps.
            solver: ODE solver name.
            x_0: Initial noise.
            device: Target device.
            **kwargs: Additional conditioning.

        Returns:
            Trajectory tensor of shape (num_steps + 1, *shape).
        """
        num_steps = num_steps or self.num_steps
        solver = solver or self.solver_name
        solver_fn = ODE_SOLVERS[solver]

        if device is None:
            device = next(self.parameters()).device

        if x_0 is None:
            x_0 = torch.randn(*shape, device=device)

        trajectory = [x_0.clone()]
        x = x_0
        dt = 1.0 / num_steps
        timesteps = torch.linspace(0.0, 1.0 - dt, num_steps, device=device)

        for t_val in timesteps:
            def model_fn(x_in, t_in, **kw):
                return self.predict_velocity(x_in, t_in, **kw)

            x = solver_fn(model_fn, x, t_val.item(), dt, **kwargs)
            trajectory.append(x.clone())

        return torch.stack(trajectory, dim=0)
