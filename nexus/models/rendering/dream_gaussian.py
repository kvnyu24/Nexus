"""
DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation

Implementation of DreamGaussian, a framework for efficient 3D content generation
that combines 3D Gaussian Splatting with score distillation sampling (SDS) from
2D diffusion models. The pipeline generates 3D assets in minutes by first
optimizing Gaussian representations via SDS, then extracting and refining meshes.

Reference:
    Tang, J., Ren, J., Zhou, H., et al. (2023).
    "DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation."
    arXiv:2309.16653 (ICLR 2024)

Key Components:
    - GaussianGenerator: Generates 3D Gaussians from score distillation sampling
    - MeshExtractor: Extracts triangle mesh from Gaussian density field via marching cubes
    - TextureRefiner: UV-space texture refinement network for final quality
    - DreamGaussian: Full pipeline combining generation, extraction, and refinement

Pipeline:
    1. Initialize random Gaussians in 3D space
    2. Render from random viewpoints, compute SDS loss against diffusion model
    3. Optimize Gaussian parameters (positions, colors, opacities, scales)
    4. Apply densification/pruning to improve representation quality
    5. Extract mesh via marching cubes on the density field
    6. Refine UV texture map using a dedicated refinement network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple, List

from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin


class GaussianGenerator(NexusModule):
    """3D Gaussian generation via score distillation sampling.

    Maintains and optimizes a set of 3D Gaussians that represent a scene.
    Each Gaussian is parameterized by position, covariance (via scale and
    rotation quaternion), opacity, and color (via spherical harmonics).
    Includes densification and pruning strategies to adaptively refine
    the representation.

    Args:
        config: Configuration dictionary with keys:
            num_gaussians (int): Initial number of Gaussians. Default: 10000.
            sh_degree (int): Spherical harmonics degree for colors. Default: 3.
            densify_grad_threshold (float): Gradient threshold for densification.
                Default: 0.0002.
            min_opacity (float): Minimum opacity for pruning. Default: 0.01.
            max_screen_size (float): Max screen size for split pruning. Default: 20.0.
            hidden_dim (int): Hidden dim for feature MLP. Default: 64.
            percent_dense (float): Percent of Gaussians to densify. Default: 0.01.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.num_gaussians = config.get("num_gaussians", 10000)
        self.sh_degree = config.get("sh_degree", 3)
        self.densify_grad_threshold = config.get("densify_grad_threshold", 0.0002)
        self.min_opacity = config.get("min_opacity", 0.01)
        self.max_screen_size = config.get("max_screen_size", 20.0)
        self.hidden_dim = config.get("hidden_dim", 64)
        self.percent_dense = config.get("percent_dense", 0.01)

        # Number of SH coefficients per channel
        self.num_sh_coeffs = (self.sh_degree + 1) ** 2

        # Gaussian parameters
        self.register_parameter(
            "means", nn.Parameter(torch.randn(self.num_gaussians, 3) * 0.5)
        )
        self.register_parameter(
            "scales", nn.Parameter(torch.ones(self.num_gaussians, 3) * -2.0)  # log-scale
        )
        self.register_parameter(
            "rotations", nn.Parameter(torch.zeros(self.num_gaussians, 4))
        )
        # Initialize quaternion w-component to 1 (identity rotation)
        with torch.no_grad():
            self.rotations[:, 0] = 1.0

        self.register_parameter(
            "opacities", nn.Parameter(torch.zeros(self.num_gaussians, 1))  # pre-sigmoid
        )
        self.register_parameter(
            "sh_coeffs", nn.Parameter(
                torch.zeros(self.num_gaussians, self.num_sh_coeffs, 3)
            )
        )

        # Densification tracking buffers
        self.register_buffer(
            "xyz_gradient_accum", torch.zeros(self.num_gaussians, 1)
        )
        self.register_buffer(
            "denom", torch.zeros(self.num_gaussians, 1)
        )

        # Feature MLP for view-dependent effects
        self.feature_mlp = nn.Sequential(
            nn.Linear(3 + self.num_sh_coeffs * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3),
            nn.Sigmoid(),
        )

    def get_activated_params(self) -> Dict[str, torch.Tensor]:
        """Get Gaussian parameters with appropriate activations applied.

        Returns:
            Dictionary with activated parameters:
                means: 3D positions (N, 3).
                scales: Positive scales via exp (N, 3).
                rotations: Normalized quaternions (N, 4).
                opacities: Opacities in [0, 1] via sigmoid (N, 1).
                colors: Base colors from DC SH component (N, 3).
        """
        return {
            "means": self.means,
            "scales": torch.exp(self.scales),
            "rotations": F.normalize(self.rotations, dim=-1),
            "opacities": torch.sigmoid(self.opacities),
            "colors": torch.sigmoid(self.sh_coeffs[:, 0, :]),  # DC component as base color
        }

    def compute_covariance(self) -> torch.Tensor:
        """Compute 3D covariance matrices from scale and rotation parameters.

        Uses the decomposition: Sigma = R * S * S^T * R^T where R is the
        rotation matrix from the quaternion and S is the diagonal scale matrix.

        Returns:
            Covariance matrices of shape (N, 3, 3).
        """
        scales = torch.exp(self.scales)
        rotations = F.normalize(self.rotations, dim=-1)

        # Quaternion to rotation matrix
        qw, qx, qy, qz = rotations.unbind(-1)
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
        """Differentiable rendering of Gaussians from a given viewpoint.

        Projects 3D Gaussians to 2D, computes view-dependent colors, and
        performs alpha compositing to produce the final image.

        Args:
            camera_pos: Camera position (B, 3).
            camera_dir: Camera look-at direction (B, 3).
            image_size: Output image resolution (H, W).

        Returns:
            Dictionary with:
                rgb: Rendered RGB image (B, 3, H, W).
                depth: Rendered depth map (B, 1, H, W).
                alpha: Alpha/opacity map (B, 1, H, W).
        """
        B = camera_pos.shape[0]
        H, W = image_size
        params = self.get_activated_params()

        # Compute view directions for each Gaussian
        view_dirs = F.normalize(
            params["means"].unsqueeze(0) - camera_pos.unsqueeze(1), dim=-1
        )  # (B, N, 3)

        # Compute view-dependent colors using SH and feature MLP
        sh_flat = self.sh_coeffs.reshape(self.num_gaussians, -1)  # (N, num_sh*3)
        sh_flat_expanded = sh_flat.unsqueeze(0).expand(B, -1, -1)
        feature_input = torch.cat([view_dirs, sh_flat_expanded], dim=-1)
        colors = self.feature_mlp(feature_input)  # (B, N, 3)

        # Simplified splatting: project Gaussians to image plane
        # Compute distances to camera
        dists = torch.norm(
            params["means"].unsqueeze(0) - camera_pos.unsqueeze(1), dim=-1
        )  # (B, N)

        # Sort by depth (front-to-back)
        sorted_indices = dists.argsort(dim=1)

        # Gather sorted values
        sorted_colors = torch.gather(
            colors, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 3)
        )
        sorted_opacities = torch.gather(
            params["opacities"].unsqueeze(0).expand(B, -1, -1),
            1,
            sorted_indices.unsqueeze(-1),
        )
        sorted_dists = torch.gather(dists, 1, sorted_indices)

        # Alpha compositing (simplified: uniform spatial contribution)
        alpha = sorted_opacities.squeeze(-1)
        transmittance = torch.cumprod(1.0 - alpha + 1e-7, dim=1)
        transmittance = torch.cat([
            torch.ones(B, 1, device=alpha.device),
            transmittance[:, :-1]
        ], dim=1)
        weights = alpha * transmittance  # (B, N)

        # Composite color and depth
        rgb = (weights.unsqueeze(-1) * sorted_colors).sum(dim=1)  # (B, 3)
        depth = (weights * sorted_dists).sum(dim=1, keepdim=True)  # (B, 1)
        total_alpha = weights.sum(dim=1, keepdim=True)

        # Reshape to image (simplified: tile for full image)
        rgb = rgb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        depth = depth.unsqueeze(-1).expand(-1, -1, H, W)
        alpha_map = total_alpha.unsqueeze(-1).expand(-1, -1, H, W)

        return {
            "rgb": rgb,
            "depth": depth,
            "alpha": alpha_map,
        }

    @torch.no_grad()
    def densify_and_prune(self, grad_threshold: Optional[float] = None) -> int:
        """Densify under-reconstructed regions and prune low-opacity Gaussians.

        Densification strategy:
        - Clone: duplicate Gaussians with large gradients but small scale
        - Split: split large Gaussians with large gradients into smaller ones

        Pruning removes Gaussians with opacity below the threshold.

        Args:
            grad_threshold: Override for the densification gradient threshold.

        Returns:
            Number of Gaussians after densification and pruning.
        """
        threshold = grad_threshold or self.densify_grad_threshold

        # Compute mean gradient magnitudes
        grads = self.xyz_gradient_accum / (self.denom + 1e-7)
        grads = grads.squeeze(-1)

        # Find Gaussians to densify
        mask_densify = grads > threshold
        scales = torch.exp(self.scales)
        max_scale = scales.max(dim=-1).values

        # Clone small Gaussians
        mask_clone = mask_densify & (max_scale < self.percent_dense)
        num_cloned = mask_clone.sum().item()

        # Split large Gaussians
        mask_split = mask_densify & (max_scale >= self.percent_dense)
        num_split = mask_split.sum().item()

        # Prune low-opacity Gaussians
        opacities = torch.sigmoid(self.opacities).squeeze(-1)
        mask_prune = opacities < self.min_opacity

        # For simplicity, just track the counts (full implementation would
        # resize parameter tensors)
        total_new = self.num_gaussians + num_cloned + num_split - mask_prune.sum().item()

        # Reset accumulation buffers
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()

        return max(total_new, 0)

    def forward(
        self,
        camera_pos: torch.Tensor,
        camera_dir: torch.Tensor,
        image_size: Tuple[int, int] = (256, 256),
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: render from given viewpoint.

        Args:
            camera_pos: Camera position (B, 3).
            camera_dir: Camera direction (B, 3).
            image_size: Output resolution.

        Returns:
            Rendering outputs including RGB, depth, alpha, and Gaussian parameters.
        """
        render_output = self.render(camera_pos, camera_dir, image_size)
        params = self.get_activated_params()
        covariances = self.compute_covariance()

        return {
            **render_output,
            "means": params["means"],
            "scales": params["scales"],
            "rotations": params["rotations"],
            "opacities": params["opacities"],
            "covariances": covariances,
        }


class MeshExtractor(NexusModule):
    """Extracts triangle mesh from Gaussian density field via marching cubes.

    Evaluates the density field defined by the Gaussians on a regular 3D grid,
    then applies the marching cubes algorithm to extract an isosurface as a
    triangle mesh. The mesh provides a cleaner geometric representation
    suitable for downstream applications like UV mapping and texture refinement.

    Args:
        config: Configuration dictionary with keys:
            grid_resolution (int): Resolution of the evaluation grid. Default: 128.
            density_threshold (float): Isosurface threshold. Default: 0.5.
            grid_range (float): Spatial extent of the grid in each axis. Default: 1.0.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.grid_resolution = config.get("grid_resolution", 128)
        self.density_threshold = config.get("density_threshold", 0.5)
        self.grid_range = config.get("grid_range", 1.0)

        # Density evaluation network
        self.density_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def evaluate_density(
        self,
        query_points: torch.Tensor,
        gaussian_means: torch.Tensor,
        gaussian_covariances: torch.Tensor,
        gaussian_opacities: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate density at query points from Gaussian representation.

        Computes the density at each query point as a weighted sum of
        Gaussian contributions based on Mahalanobis distance.

        Args:
            query_points: Points to evaluate (M, 3).
            gaussian_means: Gaussian centers (N, 3).
            gaussian_covariances: Gaussian covariance matrices (N, 3, 3).
            gaussian_opacities: Gaussian opacities (N, 1).

        Returns:
            Density values at query points (M,).
        """
        M = query_points.shape[0]
        N = gaussian_means.shape[0]

        # Compute pairwise differences
        diff = query_points.unsqueeze(1) - gaussian_means.unsqueeze(0)  # (M, N, 3)

        # Compute inverse covariance (with regularization)
        cov = gaussian_covariances + 1e-4 * torch.eye(3, device=gaussian_covariances.device)
        cov_inv = torch.linalg.inv(cov)  # (N, 3, 3)

        # Mahalanobis distance
        mahal = torch.einsum("mni,nij,mnj->mn", diff, cov_inv, diff)  # (M, N)

        # Gaussian kernel evaluation
        weights = torch.exp(-0.5 * mahal)  # (M, N)
        weights = weights * gaussian_opacities.squeeze(-1).unsqueeze(0)

        # Sum contributions
        density = weights.sum(dim=1)  # (M,)
        return density.clamp(0, 1)

    def extract_mesh(
        self,
        gaussian_means: torch.Tensor,
        gaussian_covariances: torch.Tensor,
        gaussian_opacities: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extract mesh from Gaussian density field.

        Creates a 3D grid, evaluates density at each point, and extracts
        the isosurface. Returns vertices and vertex normals estimated from
        the density gradient.

        Args:
            gaussian_means: Gaussian centers (N, 3).
            gaussian_covariances: Covariance matrices (N, 3, 3).
            gaussian_opacities: Opacities (N, 1).

        Returns:
            Dictionary with:
                vertices: Mesh vertices (V, 3).
                density_grid: Evaluated density field (res, res, res).
                grid_points: Grid coordinates (res^3, 3).
        """
        res = self.grid_resolution
        r = self.grid_range
        device = gaussian_means.device

        # Create evaluation grid
        coords = torch.linspace(-r, r, res, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing="ij")
        grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

        # Evaluate density in chunks (for memory efficiency)
        chunk_size = 4096
        densities = []
        for i in range(0, grid_points.shape[0], chunk_size):
            chunk = grid_points[i:i + chunk_size]
            d = self.evaluate_density(
                chunk, gaussian_means, gaussian_covariances, gaussian_opacities,
            )
            densities.append(d)
        density_grid = torch.cat(densities, dim=0).reshape(res, res, res)

        # Find vertices at isosurface crossings (simplified marching cubes)
        # In practice, use an external marching cubes library; here we identify
        # surface voxels where the density crosses the threshold
        above = density_grid > self.density_threshold
        surface_mask = torch.zeros_like(above)

        # Check neighbors for threshold crossings
        for axis in range(3):
            shifted = torch.roll(above, 1, dims=axis)
            surface_mask = surface_mask | (above != shifted)

        # Extract surface point coordinates
        surface_indices = surface_mask.nonzero(as_tuple=False).float()
        vertices = surface_indices / (res - 1) * (2 * r) - r

        return {
            "vertices": vertices,
            "density_grid": density_grid,
            "grid_points": grid_points,
        }

    def forward(
        self,
        gaussian_means: torch.Tensor,
        gaussian_covariances: torch.Tensor,
        gaussian_opacities: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: extract mesh from Gaussians.

        Args:
            gaussian_means: Gaussian positions (N, 3).
            gaussian_covariances: Covariance matrices (N, 3, 3).
            gaussian_opacities: Opacity values (N, 1).

        Returns:
            Mesh extraction results.
        """
        return self.extract_mesh(
            gaussian_means, gaussian_covariances, gaussian_opacities,
        )


class TextureRefiner(WeightInitMixin, NexusModule):
    """UV-space texture refinement network.

    After mesh extraction, refines the texture in UV space using a
    convolutional network. Takes an initial coarse texture map (obtained
    by projecting Gaussian colors onto the UV atlas) and refines it with
    a U-Net-like architecture for higher quality and consistency.

    Args:
        config: Configuration dictionary with keys:
            texture_resolution (int): UV texture map resolution. Default: 1024.
            in_channels (int): Input texture channels. Default: 3.
            base_channels (int): Base channel count for the U-Net. Default: 64.
            num_levels (int): Number of U-Net downsampling levels. Default: 4.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.texture_resolution = config.get("texture_resolution", 1024)
        self.in_channels = config.get("in_channels", 3)
        base_ch = config.get("base_channels", 64)
        num_levels = config.get("num_levels", 4)

        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = self.in_channels
        encoder_channels = []

        for i in range(num_levels):
            out_ch = base_ch * (2 ** i)
            self.encoders.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            encoder_channels.append(out_ch)
            self.pools.append(nn.MaxPool2d(2))
            in_ch = out_ch

        # Bottleneck
        bottleneck_ch = base_ch * (2 ** num_levels)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, bottleneck_ch, 3, padding=1),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_ch, bottleneck_ch, 3, padding=1),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )

        # Decoder (upsampling path)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        in_ch = bottleneck_ch

        for i in range(num_levels - 1, -1, -1):
            out_ch = encoder_channels[i]
            self.upconvs.append(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2))
            self.decoders.append(nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),  # *2 for skip connection
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch

        # Final output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[0], self.in_channels, 1),
            nn.Sigmoid(),
        )

        self.init_weights_vision()

    def forward(
        self,
        coarse_texture: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Refine coarse texture map.

        Args:
            coarse_texture: Initial texture map (B, 3, H, W).

        Returns:
            Dictionary with:
                refined_texture: Refined texture map (B, 3, H, W).
                residual: Refinement residual (B, 3, H, W).
        """
        # Encoder
        encoder_features = []
        x = coarse_texture
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            encoder_features.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        for i, (upconv, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = encoder_features[-(i + 1)]
            # Handle size mismatch from pooling
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        refined = self.output_conv(x)
        residual = refined - coarse_texture

        return {
            "refined_texture": refined,
            "residual": residual,
        }


class DreamGaussian(WeightInitMixin, NexusModule):
    """DreamGaussian: Full pipeline for 3D content generation.

    Combines Gaussian generation via score distillation, mesh extraction,
    and texture refinement into a unified pipeline. The model operates in
    three stages:
    1. Gaussian optimization: generate 3D Gaussians using SDS loss
    2. Mesh extraction: convert Gaussians to mesh via marching cubes
    3. Texture refinement: refine UV texture with a convolutional network

    Config:
        num_gaussians (int): Number of initial Gaussians. Default: 10000.
        sh_degree (int): Spherical harmonics degree. Default: 3.
        densify_grad_threshold (float): Densification gradient threshold. Default: 0.0002.
        grid_resolution (int): Marching cubes grid resolution. Default: 128.
        density_threshold (float): Isosurface density threshold. Default: 0.5.
        texture_resolution (int): UV texture resolution. Default: 1024.
        sds_weight (float): Weight for SDS loss. Default: 1.0.
        rgb_weight (float): Weight for RGB reconstruction loss. Default: 1.0.
        opacity_reg_weight (float): Weight for opacity regularization. Default: 0.01.

    Example:
        >>> config = {"num_gaussians": 5000, "sh_degree": 3}
        >>> model = DreamGaussian(config)
        >>> cam_pos = torch.randn(2, 3)
        >>> cam_dir = torch.randn(2, 3)
        >>> output = model(cam_pos, cam_dir, stage="gaussian")
        >>> output["rgb"].shape
        torch.Size([2, 3, 256, 256])
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Loss weights
        self.sds_weight = config.get("sds_weight", 1.0)
        self.rgb_weight = config.get("rgb_weight", 1.0)
        self.opacity_reg_weight = config.get("opacity_reg_weight", 0.01)

        # Stage 1: Gaussian generation
        self.gaussian_generator = GaussianGenerator(config)

        # Stage 2: Mesh extraction
        mesh_config = {
            "grid_resolution": config.get("grid_resolution", 128),
            "density_threshold": config.get("density_threshold", 0.5),
            "grid_range": config.get("grid_range", 1.0),
        }
        self.mesh_extractor = MeshExtractor(mesh_config)

        # Stage 3: Texture refinement
        texture_config = {
            "texture_resolution": config.get("texture_resolution", 1024),
            "in_channels": 3,
            "base_channels": config.get("texture_base_channels", 64),
            "num_levels": config.get("texture_num_levels", 4),
        }
        self.texture_refiner = TextureRefiner(texture_config)

    def forward(
        self,
        camera_pos: torch.Tensor,
        camera_dir: torch.Tensor,
        stage: str = "gaussian",
        image_size: Tuple[int, int] = (256, 256),
        coarse_texture: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the specified pipeline stage.

        Args:
            camera_pos: Camera position (B, 3).
            camera_dir: Camera direction (B, 3).
            stage: Pipeline stage ("gaussian", "mesh", "texture", or "full").
            image_size: Rendering resolution (H, W).
            coarse_texture: Coarse texture for refinement stage (B, 3, H_tex, W_tex).

        Returns:
            Dictionary with outputs appropriate to the requested stage.
        """
        if stage == "gaussian":
            return self._forward_gaussian(camera_pos, camera_dir, image_size)
        elif stage == "mesh":
            return self._forward_mesh()
        elif stage == "texture":
            assert coarse_texture is not None, "coarse_texture required for texture stage"
            return self._forward_texture(coarse_texture)
        elif stage == "full":
            return self._forward_full(camera_pos, camera_dir, image_size)
        else:
            raise ValueError(f"Unknown stage: {stage}. Choose from 'gaussian', 'mesh', 'texture', 'full'.")

    def _forward_gaussian(
        self,
        camera_pos: torch.Tensor,
        camera_dir: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Stage 1: Gaussian rendering and optimization."""
        render_output = self.gaussian_generator(camera_pos, camera_dir, image_size)

        # Compute opacity regularization loss
        params = self.gaussian_generator.get_activated_params()
        opacity_reg = (params["opacities"] * (1 - params["opacities"])).mean()

        return {
            **render_output,
            "opacity_reg": opacity_reg,
            "stage": "gaussian",
        }

    def _forward_mesh(self) -> Dict[str, torch.Tensor]:
        """Stage 2: Mesh extraction from Gaussians."""
        params = self.gaussian_generator.get_activated_params()
        covariances = self.gaussian_generator.compute_covariance()

        mesh_output = self.mesh_extractor(
            gaussian_means=params["means"],
            gaussian_covariances=covariances,
            gaussian_opacities=params["opacities"],
        )

        return {
            **mesh_output,
            "stage": "mesh",
        }

    def _forward_texture(
        self,
        coarse_texture: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Stage 3: Texture refinement."""
        texture_output = self.texture_refiner(coarse_texture)
        return {
            **texture_output,
            "stage": "texture",
        }

    def _forward_full(
        self,
        camera_pos: torch.Tensor,
        camera_dir: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Full pipeline: Gaussian rendering + mesh extraction."""
        gaussian_output = self._forward_gaussian(camera_pos, camera_dir, image_size)
        mesh_output = self._forward_mesh()

        return {
            "gaussian": gaussian_output,
            "mesh": mesh_output,
            "stage": "full",
        }


__all__ = [
    "DreamGaussian",
    "GaussianGenerator",
    "MeshExtractor",
    "TextureRefiner",
]
