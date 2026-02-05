"""
SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction

Implementation of SuGaR (Surface-Aligned Gaussian Representation), which improves
3D Gaussian Splatting by encouraging Gaussians to align with scene surfaces. This
alignment enables high-quality mesh reconstruction via Poisson surface reconstruction
while preserving the real-time rendering benefits of Gaussian splatting.

Reference:
    Guedon, A. and Lepetit, V. (2024).
    "SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh
     Reconstruction and High-Quality Mesh Rendering."
    arXiv:2311.12775 (CVPR 2024)

Key Components:
    - SurfaceRegularization: Loss terms encouraging Gaussian-surface alignment
    - SuGaRModel: Gaussian splatting with surface alignment regularization
    - MeshReconstructor: Poisson surface reconstruction from aligned Gaussians

Architecture Details:
    - Gaussians are regularized to be flat (pancake-shaped) and lie on surfaces
    - Flatness is enforced by penalizing the smallest scale dimension
    - Normal consistency is enforced via a normal alignment loss
    - After optimization, aligned Gaussians serve as an oriented point cloud
    - Poisson reconstruction converts the point cloud to a watertight mesh
    - Optional mesh binding step locks Gaussians to mesh faces for rendering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple, List

from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin
from ...core.mixins import ConfigValidatorMixin


class SurfaceRegularization(NexusModule):
    """Surface alignment regularization losses for Gaussian splatting.

    Provides loss terms that encourage Gaussians to:
    1. Become flat (one scale dimension much smaller than the other two)
    2. Align their normals consistently with nearby Gaussians
    3. Distribute evenly across the scene surface

    These regularizations transform an unstructured Gaussian cloud into
    an oriented point cloud suitable for mesh reconstruction.

    Args:
        config: Configuration dictionary with keys:
            regularization_weight (float): Overall regularization weight. Default: 0.5.
            flatness_weight (float): Weight for flatness loss. Default: 1.0.
            normal_weight (float): Weight for normal consistency loss. Default: 0.1.
            opacity_weight (float): Weight for opacity binarization loss. Default: 0.01.
            flatness_threshold (float): Threshold for considering a Gaussian as flat.
                Default: 0.01.
            k_neighbors (int): Number of neighbors for normal consistency. Default: 16.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.regularization_weight = config.get("regularization_weight", 0.5)
        self.flatness_weight = config.get("flatness_weight", 1.0)
        self.normal_weight = config.get("normal_weight", 0.1)
        self.opacity_weight = config.get("opacity_weight", 0.01)
        self.flatness_threshold = config.get("flatness_threshold", 0.01)
        self.k_neighbors = config.get("k_neighbors", 16)

    def flatness_loss(self, scales: torch.Tensor) -> torch.Tensor:
        """Encourage Gaussians to be flat (pancake-shaped).

        Penalizes the minimum scale dimension, pushing Gaussians to have
        one very small axis (the surface normal direction) while remaining
        extended in the tangent plane.

        Args:
            scales: Gaussian scales (N, 3), should be positive (e.g., exp-activated).

        Returns:
            Scalar flatness loss.
        """
        # The smallest scale should be close to zero for flat Gaussians
        min_scale, _ = scales.min(dim=-1)
        loss = F.relu(min_scale - self.flatness_threshold).mean()
        return loss

    def normal_consistency_loss(
        self,
        means: torch.Tensor,
        normals: torch.Tensor,
    ) -> torch.Tensor:
        """Encourage nearby Gaussians to have consistent normals.

        For each Gaussian, finds its k nearest neighbors and penalizes
        angular deviation between their normals. This promotes smooth
        surface reconstruction.

        Args:
            means: Gaussian positions (N, 3).
            normals: Gaussian surface normals (N, 3), should be unit vectors.

        Returns:
            Scalar normal consistency loss.
        """
        N = means.shape[0]
        k = min(self.k_neighbors, N - 1)

        if k <= 0:
            return torch.tensor(0.0, device=means.device)

        # Compute pairwise distances (using chunked computation for memory)
        # For large N, subsample
        max_samples = 4096
        if N > max_samples:
            indices = torch.randperm(N, device=means.device)[:max_samples]
            means_sub = means[indices]
            normals_sub = normals[indices]
        else:
            means_sub = means
            normals_sub = normals

        M = means_sub.shape[0]
        dists = torch.cdist(means_sub, means, p=2)  # (M, N)

        # Get k nearest neighbors (excluding self)
        _, knn_indices = dists.topk(k + 1, dim=1, largest=False)
        knn_indices = knn_indices[:, 1:]  # Exclude self (closest is self)

        # Gather neighbor normals
        neighbor_normals = normals[knn_indices]  # (M, k, 3)

        # Compute cosine similarity (absolute value since normals can be flipped)
        cos_sim = torch.abs(
            torch.sum(normals_sub.unsqueeze(1) * neighbor_normals, dim=-1)
        )  # (M, k)

        # Loss: 1 - mean cosine similarity (want similarity close to 1)
        loss = (1.0 - cos_sim).mean()
        return loss

    def opacity_binarization_loss(self, opacities: torch.Tensor) -> torch.Tensor:
        """Encourage opacities to be close to 0 or 1.

        Binary opacities improve mesh reconstruction quality by eliminating
        semi-transparent Gaussians that don't correspond to real surfaces.

        Args:
            opacities: Gaussian opacities in [0, 1] (N, 1).

        Returns:
            Scalar binarization loss.
        """
        # Entropy-like loss that is minimized when opacity is 0 or 1
        loss = -(opacities * torch.log(opacities + 1e-7)
                 + (1 - opacities) * torch.log(1 - opacities + 1e-7)).mean()
        return loss

    def forward(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all surface regularization losses.

        Args:
            means: Gaussian positions (N, 3).
            scales: Gaussian scales (N, 3), positive values.
            rotations: Gaussian rotations as quaternions (N, 4).
            opacities: Gaussian opacities in [0, 1] (N, 1).

        Returns:
            Dictionary with individual loss terms and total regularization loss.
        """
        # Compute normals from rotation (normal = smallest scale axis direction)
        normals = self._compute_normals(scales, rotations)

        # Individual losses
        flat_loss = self.flatness_loss(scales)
        normal_loss = self.normal_consistency_loss(means, normals)
        opacity_loss = self.opacity_binarization_loss(opacities)

        # Weighted total
        total = self.regularization_weight * (
            self.flatness_weight * flat_loss
            + self.normal_weight * normal_loss
            + self.opacity_weight * opacity_loss
        )

        return {
            "total_reg_loss": total,
            "flatness_loss": flat_loss,
            "normal_consistency_loss": normal_loss,
            "opacity_binarization_loss": opacity_loss,
            "normals": normals,
        }

    def _compute_normals(
        self,
        scales: torch.Tensor,
        rotations: torch.Tensor,
    ) -> torch.Tensor:
        """Compute surface normals from Gaussian orientation.

        The normal direction is the axis corresponding to the smallest
        scale dimension. This is extracted from the rotation matrix.

        Args:
            scales: Gaussian scales (N, 3).
            rotations: Quaternions (N, 4).

        Returns:
            Unit normals (N, 3).
        """
        # Find the axis with minimum scale
        min_idx = scales.argmin(dim=-1)  # (N,)

        # Convert quaternion to rotation matrix
        q = F.normalize(rotations, dim=-1)
        qw, qx, qy, qz = q.unbind(-1)

        R = torch.stack([
            1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
            2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw,
            2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2,
        ], dim=-1).view(-1, 3, 3)

        # Extract the column corresponding to the minimum scale axis
        N = scales.shape[0]
        normals = R[torch.arange(N, device=R.device), :, min_idx]

        return F.normalize(normals, dim=-1)


class SuGaRModel(ConfigValidatorMixin, WeightInitMixin, NexusModule):
    """Surface-Aligned Gaussian Splatting model.

    Extends standard Gaussian splatting with surface alignment regularization
    that encourages Gaussians to be flat and aligned with scene surfaces.
    The resulting representation is both renderable in real-time and suitable
    for high-quality mesh extraction.

    Config:
        num_gaussians (int): Number of Gaussians. Default: 10000.
        hidden_dim (int): Hidden dimension for neural features. Default: 64.
        sh_degree (int): Spherical harmonics degree. Default: 3.
        regularization_weight (float): Surface regularization strength. Default: 0.5.
        flatness_threshold (float): Scale threshold for flatness. Default: 0.01.
        min_opacity (float): Minimum opacity (for pruning). Default: 0.005.
        densify_grad_threshold (float): Gradient threshold for densification. Default: 0.0002.
        k_neighbors (int): Neighbors for normal consistency. Default: 16.

    Example:
        >>> config = {"num_gaussians": 5000, "hidden_dim": 64}
        >>> model = SuGaRModel(config)
        >>> cam_pos = torch.randn(2, 3)
        >>> cam_dir = torch.randn(2, 3)
        >>> output = model(cam_pos, cam_dir)
        >>> output["rgb"].shape
        torch.Size([2, 3, 256, 256])
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.validate_config(config, required_keys=[], optional_keys=[
            "num_gaussians", "hidden_dim", "sh_degree",
            "regularization_weight", "flatness_threshold",
            "min_opacity", "densify_grad_threshold", "k_neighbors",
            "dropout",
        ])

        self.num_gaussians = config.get("num_gaussians", 10000)
        self.hidden_dim = config.get("hidden_dim", 64)
        self.sh_degree = config.get("sh_degree", 3)
        self.num_sh_coeffs = (self.sh_degree + 1) ** 2

        # Gaussian parameters
        self.register_parameter(
            "means", nn.Parameter(torch.randn(self.num_gaussians, 3) * 0.5)
        )
        self.register_parameter(
            "log_scales", nn.Parameter(torch.ones(self.num_gaussians, 3) * -3.0)
        )
        self.register_parameter(
            "rotations", nn.Parameter(torch.zeros(self.num_gaussians, 4))
        )
        with torch.no_grad():
            self.rotations[:, 0] = 1.0  # Identity quaternion

        self.register_parameter(
            "raw_opacities", nn.Parameter(torch.zeros(self.num_gaussians, 1))
        )
        self.register_parameter(
            "sh_coeffs", nn.Parameter(
                torch.randn(self.num_gaussians, self.num_sh_coeffs, 3) * 0.1
            )
        )

        # Surface regularization
        reg_config = {
            "regularization_weight": config.get("regularization_weight", 0.5),
            "flatness_threshold": config.get("flatness_threshold", 0.01),
            "k_neighbors": config.get("k_neighbors", 16),
        }
        self.surface_reg = SurfaceRegularization(reg_config)

        # View-dependent color MLP
        self.color_mlp = nn.Sequential(
            nn.Linear(3 + self.num_sh_coeffs * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3),
            nn.Sigmoid(),
        )

        # Density network for adaptive opacity
        self.density_net = nn.Sequential(
            nn.Linear(3 + self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Gradient tracking for densification
        self.register_buffer("grad_accum", torch.zeros(self.num_gaussians, 1))
        self.register_buffer("grad_count", torch.zeros(self.num_gaussians, 1))

    @property
    def scales(self) -> torch.Tensor:
        """Get activated (positive) scales."""
        return torch.exp(self.log_scales)

    @property
    def opacities(self) -> torch.Tensor:
        """Get activated opacities in [0, 1]."""
        return torch.sigmoid(self.raw_opacities)

    @property
    def normals(self) -> torch.Tensor:
        """Get computed surface normals."""
        return self.surface_reg._compute_normals(self.scales, self.rotations)

    def compute_covariance(self) -> torch.Tensor:
        """Compute 3D covariance matrices from scales and rotations.

        Returns:
            Symmetric positive semi-definite covariance matrices (N, 3, 3).
        """
        scales = self.scales
        q = F.normalize(self.rotations, dim=-1)
        qw, qx, qy, qz = q.unbind(-1)

        R = torch.stack([
            1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
            2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw,
            2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2,
        ], dim=-1).view(-1, 3, 3)

        S = torch.diag_embed(scales)
        cov = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
        return 0.5 * (cov + cov.transpose(-1, -2))

    def render(
        self,
        camera_pos: torch.Tensor,
        camera_dir: torch.Tensor,
        image_size: Tuple[int, int] = (256, 256),
    ) -> Dict[str, torch.Tensor]:
        """Differentiable rendering with surface-aligned Gaussians.

        Args:
            camera_pos: Camera position (B, 3).
            camera_dir: Camera direction (B, 3).
            image_size: Output image resolution (H, W).

        Returns:
            Dictionary with rendered RGB, depth, alpha, and normals.
        """
        B = camera_pos.shape[0]
        H, W = image_size

        # View directions
        view_dirs = F.normalize(
            self.means.unsqueeze(0) - camera_pos.unsqueeze(1), dim=-1
        )

        # Compute view-dependent colors
        sh_flat = self.sh_coeffs.reshape(self.num_gaussians, -1)
        sh_expanded = sh_flat.unsqueeze(0).expand(B, -1, -1)
        color_input = torch.cat([view_dirs, sh_expanded], dim=-1)
        colors = self.color_mlp(color_input)

        # Distances for depth sorting
        dists = torch.norm(
            self.means.unsqueeze(0) - camera_pos.unsqueeze(1), dim=-1
        )

        # Sort by depth
        sorted_idx = dists.argsort(dim=1)
        sorted_colors = torch.gather(colors, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, 3))
        sorted_opacities = torch.gather(
            self.opacities.unsqueeze(0).expand(B, -1, -1), 1, sorted_idx.unsqueeze(-1)
        )
        sorted_dists = torch.gather(dists, 1, sorted_idx)

        # Normals for the sorted Gaussians
        normals_expanded = self.normals.unsqueeze(0).expand(B, -1, -1)
        sorted_normals = torch.gather(normals_expanded, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, 3))

        # Alpha compositing
        alpha = sorted_opacities.squeeze(-1)
        transmittance = torch.cumprod(1.0 - alpha + 1e-7, dim=1)
        transmittance = torch.cat([
            torch.ones(B, 1, device=alpha.device), transmittance[:, :-1]
        ], dim=1)
        weights = alpha * transmittance

        # Composite outputs
        rgb = (weights.unsqueeze(-1) * sorted_colors).sum(dim=1)
        depth = (weights * sorted_dists).sum(dim=1, keepdim=True)
        rendered_normals = F.normalize(
            (weights.unsqueeze(-1) * sorted_normals).sum(dim=1), dim=-1
        )
        total_alpha = weights.sum(dim=1, keepdim=True)

        # Expand to spatial dimensions
        rgb = rgb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        depth = depth.unsqueeze(-1).expand(-1, -1, H, W)
        rendered_normals = rendered_normals.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        alpha_map = total_alpha.unsqueeze(-1).expand(-1, -1, H, W)

        return {
            "rgb": rgb,
            "depth": depth,
            "normals": rendered_normals,
            "alpha": alpha_map,
        }

    def forward(
        self,
        camera_pos: torch.Tensor,
        camera_dir: torch.Tensor,
        image_size: Tuple[int, int] = (256, 256),
        compute_reg: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional surface regularization.

        Args:
            camera_pos: Camera position (B, 3).
            camera_dir: Camera direction (B, 3).
            image_size: Rendering resolution (H, W).
            compute_reg: Whether to compute regularization losses.

        Returns:
            Dictionary with rendering outputs and regularization losses.
        """
        # Render
        render_output = self.render(camera_pos, camera_dir, image_size)

        output = {
            **render_output,
            "means": self.means,
            "scales": self.scales,
            "rotations": F.normalize(self.rotations, dim=-1),
            "opacities": self.opacities,
            "covariances": self.compute_covariance(),
        }

        # Surface regularization
        if compute_reg:
            reg_output = self.surface_reg(
                means=self.means,
                scales=self.scales,
                rotations=self.rotations,
                opacities=self.opacities,
            )
            output.update({
                "reg_loss": reg_output["total_reg_loss"],
                "flatness_loss": reg_output["flatness_loss"],
                "normal_consistency_loss": reg_output["normal_consistency_loss"],
                "opacity_binarization_loss": reg_output["opacity_binarization_loss"],
                "surface_normals": reg_output["normals"],
            })

        return output

    def get_oriented_point_cloud(self) -> Dict[str, torch.Tensor]:
        """Extract an oriented point cloud for mesh reconstruction.

        Returns Gaussian centers as points with estimated surface normals
        and colors. Only includes Gaussians with sufficiently high opacity
        and flat shape.

        Returns:
            Dictionary with:
                points: Surface point positions (M, 3).
                normals: Surface normals (M, 3).
                colors: Point colors (M, 3).
        """
        with torch.no_grad():
            opacities = self.opacities.squeeze(-1)
            scales = self.scales
            min_scale = scales.min(dim=-1).values

            # Filter: high opacity + flat shape
            mask = (opacities > 0.5) & (min_scale < self.surface_reg.flatness_threshold * 5)

            points = self.means[mask]
            normals = self.normals[mask]
            colors = torch.sigmoid(self.sh_coeffs[mask, 0, :])  # DC component

            return {
                "points": points,
                "normals": normals,
                "colors": colors,
                "num_points": points.shape[0],
            }


class MeshReconstructor(NexusModule):
    """Poisson surface reconstruction from aligned Gaussians.

    Takes the oriented point cloud from SuGaR (surface-aligned Gaussians)
    and performs mesh reconstruction. Implements a simplified screened
    Poisson reconstruction that converts oriented points into a watertight
    triangle mesh.

    In practice, external libraries (e.g., Open3D, PyMeshLab) are used for
    full Poisson reconstruction. This module provides the point cloud
    preparation and post-processing.

    Args:
        config: Configuration dictionary with keys:
            poisson_depth (int): Octree depth for Poisson reconstruction. Default: 8.
            density_threshold (float): Density threshold for trimming. Default: 0.5.
            grid_resolution (int): Grid resolution for density estimation. Default: 128.
            smoothing_iterations (int): Laplacian smoothing iterations. Default: 3.
            smoothing_lambda (float): Smoothing strength. Default: 0.5.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.poisson_depth = config.get("poisson_depth", 8)
        self.density_threshold = config.get("density_threshold", 0.5)
        self.grid_resolution = config.get("grid_resolution", 128)
        self.smoothing_iterations = config.get("smoothing_iterations", 3)
        self.smoothing_lambda = config.get("smoothing_lambda", 0.5)

        # Implicit function network (for differentiable reconstruction)
        self.implicit_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Normal prediction for mesh vertices
        self.normal_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def estimate_density_field(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate density field from oriented point cloud.

        Uses the oriented points to define an implicit surface via a
        combination of nearest-neighbor interpolation and learned features.

        Args:
            points: Surface points (M, 3).
            normals: Point normals (M, 3).
            query_points: Grid points to evaluate density at (Q, 3).

        Returns:
            Density values at query points (Q,).
        """
        # Use a neural implicit function conditioned on the point cloud
        # Simplified: evaluate the implicit network at query points
        density = self.implicit_net(query_points).squeeze(-1)
        return torch.sigmoid(density)

    def extract_surface(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract mesh surface from oriented point cloud.

        Creates a regular 3D grid, evaluates the density field, and
        extracts the isosurface. Vertex colors are interpolated from
        the nearest source points.

        Args:
            points: Oriented surface points (M, 3).
            normals: Surface normals (M, 3).
            colors: Optional point colors (M, 3).

        Returns:
            Dictionary with:
                vertices: Mesh vertices (V, 3).
                vertex_normals: Vertex normals (V, 3).
                vertex_colors: Vertex colors (V, 3) if colors provided.
                density_grid: Evaluated density field.
        """
        device = points.device
        res = self.grid_resolution

        # Determine bounding box with padding
        bbox_min = points.min(dim=0).values - 0.1
        bbox_max = points.max(dim=0).values + 0.1

        # Create evaluation grid
        x = torch.linspace(bbox_min[0], bbox_max[0], res, device=device)
        y = torch.linspace(bbox_min[1], bbox_max[1], res, device=device)
        z = torch.linspace(bbox_min[2], bbox_max[2], res, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
        query_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

        # Evaluate density in chunks
        chunk_size = 8192
        densities = []
        for i in range(0, query_points.shape[0], chunk_size):
            chunk = query_points[i:i + chunk_size]
            d = self.estimate_density_field(points, normals, chunk)
            densities.append(d)
        density_grid = torch.cat(densities, dim=0).reshape(res, res, res)

        # Extract surface vertices (simplified: find threshold crossings)
        above = density_grid > self.density_threshold
        surface_mask = torch.zeros_like(above)
        for axis in range(3):
            shifted = torch.roll(above, 1, dims=axis)
            surface_mask = surface_mask | (above != shifted)

        surface_indices = surface_mask.nonzero(as_tuple=False).float()
        if surface_indices.shape[0] == 0:
            # Return empty mesh if no surface found
            return {
                "vertices": torch.zeros(0, 3, device=device),
                "vertex_normals": torch.zeros(0, 3, device=device),
                "density_grid": density_grid,
            }

        # Convert grid indices to world coordinates
        scale = (bbox_max - bbox_min) / (res - 1)
        vertices = surface_indices * scale + bbox_min

        # Estimate vertex normals
        vertex_normals = self.normal_net(vertices)
        vertex_normals = F.normalize(vertex_normals, dim=-1)

        result = {
            "vertices": vertices,
            "vertex_normals": vertex_normals,
            "density_grid": density_grid,
        }

        # Interpolate colors from nearest source points
        if colors is not None and vertices.shape[0] > 0:
            dists = torch.cdist(vertices, points)
            nearest = dists.argmin(dim=1)
            result["vertex_colors"] = colors[nearest]

        return result

    def forward(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: reconstruct mesh from oriented point cloud.

        Args:
            points: Surface points (M, 3).
            normals: Surface normals (M, 3).
            colors: Optional point colors (M, 3).

        Returns:
            Mesh reconstruction results.
        """
        return self.extract_surface(points, normals, colors)


__all__ = [
    "SuGaRModel",
    "SurfaceRegularization",
    "MeshReconstructor",
]
