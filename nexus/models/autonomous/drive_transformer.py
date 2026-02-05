"""
DriveTransformer: Unified Transformer for Scalable End-to-End Autonomous Driving.

DriveTransformer presents a scalable, unified transformer architecture that handles
all autonomous driving tasks through a single model. It processes multi-modal inputs
(cameras, LiDAR, maps) and produces all outputs (detection, tracking, forecasting,
planning) using a shared transformer backbone with task-specific heads.

Key innovations:
- Unified transformer backbone for all driving tasks
- Multi-modal sensor fusion (camera + LiDAR + HD maps)
- Scalable architecture that improves with model size and data
- Temporal modeling with recurrent state for online driving
- End-to-end differentiable from sensors to actions
- Supports both imitation learning and reinforcement learning

The architecture uses:
- Sensor encoders to tokenize multi-modal inputs
- Core transformer for reasoning and feature extraction
- Task decoders for perception, prediction, and planning outputs
- Recurrent memory for temporal consistency across frames

Paper: "Scaling End-to-End Autonomous Driving with Unified Transformers"
       Anonymous, 2025 (under review)
       Inspired by scaling trends in vision and language transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from nexus.core.base import NexusModule
import math


class SensorTokenizer(NexusModule):
    """Tokenize multi-modal sensor inputs into unified token sequences.

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 512)
        self.patch_size = config.get('patch_size', 16)
        self.num_cameras = config.get('num_cameras', 6)

        # Camera image tokenizer (ViT-style)
        self.camera_tokenizer = nn.Sequential(
            nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size),
            nn.Flatten(2),  # Flatten spatial dimensions
        )

        # Optional: LiDAR point cloud tokenizer
        self.use_lidar = config.get('use_lidar', False)
        if self.use_lidar:
            self.lidar_tokenizer = nn.Sequential(
                nn.Linear(4, 128),  # (x, y, z, intensity)
                nn.ReLU(inplace=True),
                nn.Linear(128, self.embed_dim)
            )

        # Optional: HD map tokenizer
        self.use_map = config.get('use_map', False)
        if self.use_map:
            self.map_tokenizer = nn.Sequential(
                nn.Linear(2, 128),  # (x, y) map coordinates
                nn.ReLU(inplace=True),
                nn.Linear(128, self.embed_dim)
            )

        # Modality embeddings
        self.camera_embed = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        if self.use_lidar:
            self.lidar_embed = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        if self.use_map:
            self.map_embed = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # Camera view embeddings
        self.view_embeds = nn.Parameter(torch.randn(self.num_cameras, 1, self.embed_dim))

        # Positional embeddings (learnable)
        self.max_seq_len = config.get('max_tokens', 2048)
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_len, self.embed_dim))

    def forward(self,
                camera_images: torch.Tensor,
                lidar_points: Optional[torch.Tensor] = None,
                map_data: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Tokenize multi-modal sensor inputs.

        Args:
            camera_images: Multi-view images (B, N_cams, C, H, W)
            lidar_points: LiDAR point cloud (B, N_points, 4)
            map_data: HD map polylines (B, N_map_pts, 2)

        Returns:
            Tuple of:
                - tokens: Unified token sequence (B, N_tokens, embed_dim)
                - metadata: Token metadata (types, positions, etc.)
        """
        B = camera_images.shape[0]
        all_tokens = []
        token_types = []

        # Tokenize camera images
        for cam_idx in range(self.num_cameras):
            img = camera_images[:, cam_idx]  # (B, C, H, W)
            tokens = self.camera_tokenizer(img)  # (B, embed_dim, n_patches)
            tokens = tokens.permute(0, 2, 1)  # (B, n_patches, embed_dim)

            # Add modality and view embeddings
            tokens = tokens + self.camera_embed + self.view_embeds[cam_idx]

            all_tokens.append(tokens)
            token_types.extend(['camera'] * tokens.shape[1])

        # Tokenize LiDAR if available
        if self.use_lidar and lidar_points is not None:
            lidar_tokens = self.lidar_tokenizer(lidar_points)
            lidar_tokens = lidar_tokens + self.lidar_embed
            all_tokens.append(lidar_tokens)
            token_types.extend(['lidar'] * lidar_tokens.shape[1])

        # Tokenize map if available
        if self.use_map and map_data is not None:
            map_tokens = self.map_tokenizer(map_data)
            map_tokens = map_tokens + self.map_embed
            all_tokens.append(map_tokens)
            token_types.extend(['map'] * map_tokens.shape[1])

        # Concatenate all tokens
        tokens = torch.cat(all_tokens, dim=1)  # (B, N_total_tokens, embed_dim)

        # Add positional embeddings
        N_tokens = tokens.shape[1]
        if N_tokens > self.max_seq_len:
            # Interpolate positional embeddings if sequence is too long
            pos_embed = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=N_tokens,
                mode='linear'
            ).permute(0, 2, 1)
        else:
            pos_embed = self.pos_embed[:, :N_tokens]

        tokens = tokens + pos_embed

        metadata = {
            'token_types': token_types,
            'num_tokens': N_tokens
        }

        return tokens, metadata


class RecurrentMemory(nn.Module):
    """Recurrent memory module for temporal state across frames.

    Args:
        embed_dim: Feature dimension
        memory_size: Number of memory tokens
    """

    def __init__(self, embed_dim: int, memory_size: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_size = memory_size

        # Learnable initial memory
        self.init_memory = nn.Parameter(torch.randn(1, memory_size, embed_dim))

        # Memory update gate
        self.update_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        self.memory_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self,
                current_features: torch.Tensor,
                prev_memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Update memory with current features.

        Args:
            current_features: Current frame features (B, N, embed_dim)
            prev_memory: Previous memory state (B, memory_size, embed_dim)

        Returns:
            Updated memory (B, memory_size, embed_dim)
        """
        B = current_features.shape[0]

        if prev_memory is None:
            prev_memory = self.init_memory.expand(B, -1, -1)

        # Aggregate current features (mean pooling)
        current_summary = current_features.mean(dim=1, keepdim=True)  # (B, 1, embed_dim)
        current_summary = current_summary.expand(-1, self.memory_size, -1)

        # Compute update gate
        concat = torch.cat([prev_memory, current_summary], dim=-1)
        gate = self.update_gate(concat)

        # Update memory
        new_content = self.memory_proj(current_features.mean(dim=1, keepdim=True))
        new_content = new_content.expand(-1, self.memory_size, -1)

        updated_memory = gate * new_content + (1 - gate) * prev_memory

        return updated_memory


class UnifiedTransformerBackbone(NexusModule):
    """Unified transformer backbone for all driving tasks.

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 512)
        self.num_layers = config.get('num_layers', 12)
        self.num_heads = config.get('num_heads', 8)
        self.ff_dim = config.get('ff_dim', self.embed_dim * 4)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability at scale
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Layer normalization
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, tokens: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process tokens through transformer.

        Args:
            tokens: Input tokens (B, N, embed_dim)
            attention_mask: Optional attention mask

        Returns:
            Processed features (B, N, embed_dim)
        """
        features = self.transformer(tokens, src_key_padding_mask=attention_mask)
        features = self.norm(features)
        return features


class TaskDecoder(NexusModule):
    """Generic task decoder with cross-attention to features.

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 512)
        self.num_queries = config.get('num_queries', 100)
        self.num_layers = config.get('decoder_layers', 3)

        # Task-specific learnable queries
        self.queries = nn.Parameter(torch.randn(1, self.num_queries, self.embed_dim))

        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, features: torch.Tensor,
                memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode task-specific outputs.

        Args:
            features: Backbone features (B, N, embed_dim)
            memory: Optional recurrent memory (B, M, embed_dim)

        Returns:
            Task queries (B, num_queries, embed_dim)
        """
        B = features.shape[0]

        # Concatenate features and memory if available
        if memory is not None:
            context = torch.cat([features, memory], dim=1)
        else:
            context = features

        # Initialize queries
        queries = self.queries.expand(B, -1, -1)

        # Decode
        decoded = self.decoder(queries, context)
        decoded = self.norm(decoded)

        return decoded


class DriveTransformer(NexusModule):
    """Unified Transformer for Scalable End-to-End Autonomous Driving.

    A single transformer model that handles all autonomous driving tasks
    through multi-modal sensor fusion and task-specific decoders.

    Args:
        config: Configuration dictionary with keys:
            - embed_dim (int): Model dimension. Default 512
            - num_layers (int): Transformer layers. Default 12
            - num_heads (int): Attention heads. Default 8
            - patch_size (int): Image patch size. Default 16
            - num_cameras (int): Number of cameras. Default 6
            - use_lidar (bool): Use LiDAR input. Default False
            - use_map (bool): Use HD map input. Default False
            - use_memory (bool): Use recurrent memory. Default True
            - memory_size (int): Memory token count. Default 128
            - num_detection_queries (int): Detection queries. Default 300
            - num_motion_modes (int): Motion prediction modes. Default 6
            - future_steps (int): Motion horizon. Default 12
            - planning_horizon (int): Planning horizon. Default 6

    References:
        Inspired by:
        - "Scaling End-to-End Autonomous Driving" (2025)
        - "Planning-oriented Autonomous Driving" (UniAD, CVPR 2023)
        - Vision Transformer scaling principles
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 512)

        # Sensor tokenizer
        self.tokenizer = SensorTokenizer(config)

        # Unified transformer backbone
        self.backbone = UnifiedTransformerBackbone(config)

        # Recurrent memory for temporal consistency
        self.use_memory = config.get('use_memory', True)
        if self.use_memory:
            memory_size = config.get('memory_size', 128)
            self.memory_module = RecurrentMemory(self.embed_dim, memory_size)

        # Task decoders
        detection_config = {
            **config,
            'num_queries': config.get('num_detection_queries', 300),
            'decoder_layers': 3
        }
        self.detection_decoder = TaskDecoder(detection_config)

        motion_config = {
            **config,
            'num_queries': config.get('num_detection_queries', 300),
            'decoder_layers': 3
        }
        self.motion_decoder = TaskDecoder(motion_config)

        planning_config = {
            **config,
            'num_queries': config.get('num_planning_modes', 3),
            'decoder_layers': 3
        }
        self.planning_decoder = TaskDecoder(planning_config)

        # Task prediction heads
        self.detection_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 10)  # (x, y, z, l, w, h, sin, cos, vx, vy)
        )

        self.class_head = nn.Linear(self.embed_dim, 10)

        num_modes = config.get('num_motion_modes', 6)
        future_steps = config.get('future_steps', 12)
        self.motion_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, num_modes * future_steps * 2)
        )
        self.motion_conf_head = nn.Linear(self.embed_dim, num_modes)

        planning_horizon = config.get('planning_horizon', 6)
        self.planning_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, planning_horizon * 3)  # (x, y, yaw)
        )
        self.planning_cost_head = nn.Linear(self.embed_dim, 1)

        # Store config
        self.num_motion_modes = num_modes
        self.future_steps = future_steps
        self.planning_horizon = planning_horizon
        self.num_detection_queries = config.get('num_detection_queries', 300)

    def forward(self,
                camera_images: torch.Tensor,
                lidar_points: Optional[torch.Tensor] = None,
                map_data: Optional[torch.Tensor] = None,
                prev_memory: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for end-to-end driving.

        Args:
            camera_images: Multi-view images (B, N_cams, C, H, W)
            lidar_points: Optional LiDAR points (B, N_pts, 4)
            map_data: Optional HD map (B, N_map, 2)
            prev_memory: Previous frame memory state

        Returns:
            Dictionary containing:
                - detections: 3D bounding boxes
                - classes: Object classes
                - trajectories: Predicted agent trajectories
                - traj_confidences: Trajectory mode probabilities
                - ego_plan: Ego trajectory plan
                - planning_costs: Cost per planning mode
                - memory: Updated memory state for next frame
        """
        # Stage 1: Tokenize multi-modal inputs
        tokens, metadata = self.tokenizer(camera_images, lidar_points, map_data)

        # Stage 2: Process through unified transformer
        features = self.backbone(tokens)

        # Stage 3: Update recurrent memory
        if self.use_memory:
            memory = self.memory_module(features, prev_memory)
        else:
            memory = None

        # Stage 4: Task-specific decoding
        # Detection
        detection_queries = self.detection_decoder(features, memory)
        detections = self.detection_head(detection_queries)
        classes = self.class_head(detection_queries)

        # Motion forecasting
        motion_queries = self.motion_decoder(features, memory)
        B, N, _ = motion_queries.shape
        trajectories = self.motion_head(motion_queries)
        trajectories = trajectories.view(B, N, self.num_motion_modes, self.future_steps, 2)
        traj_confs = self.motion_conf_head(motion_queries)
        traj_confs = F.softmax(traj_confs, dim=-1)

        # Ego planning
        planning_queries = self.planning_decoder(features, memory)
        ego_plans = self.planning_head(planning_queries)
        B_plan, N_plan, _ = planning_queries.shape
        ego_plans = ego_plans.view(B_plan, N_plan, self.planning_horizon, 3)
        planning_costs = self.planning_cost_head(planning_queries).squeeze(-1)

        return {
            'detections': detections,
            'classes': classes,
            'trajectories': trajectories,
            'traj_confidences': traj_confs,
            'ego_plan': ego_plans,
            'planning_costs': planning_costs,
            'memory': memory,  # Pass to next frame
            'features': features,  # For analysis
        }

    def compute_loss(self,
                     outputs: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task training losses.

        Args:
            outputs: Model predictions
            targets: Ground truth labels

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Detection loss
        if 'gt_boxes' in targets:
            det_loss = F.smooth_l1_loss(outputs['detections'], targets['gt_boxes'])
            losses['detection_loss'] = det_loss

        # Classification loss
        if 'gt_classes' in targets:
            cls_loss = F.cross_entropy(
                outputs['classes'].view(-1, outputs['classes'].shape[-1]),
                targets['gt_classes'].view(-1),
                ignore_index=-1
            )
            losses['classification_loss'] = cls_loss

        # Motion forecasting loss
        if 'gt_trajectories' in targets:
            pred_trajs = outputs['trajectories']
            gt_trajs = targets['gt_trajectories']

            # Winner-takes-all: best mode
            errors = torch.norm(pred_trajs - gt_trajs.unsqueeze(2), dim=-1).mean(dim=-1)
            best_mode = errors.argmin(dim=-1)

            # Regression loss on best mode
            B, N = best_mode.shape
            batch_idx = torch.arange(B, device=best_mode.device).unsqueeze(1).expand(B, N)
            agent_idx = torch.arange(N, device=best_mode.device).unsqueeze(0).expand(B, N)

            motion_loss = F.smooth_l1_loss(
                pred_trajs[batch_idx, agent_idx, best_mode],
                gt_trajs
            )
            losses['motion_loss'] = motion_loss

        # Planning loss (imitation learning)
        if 'gt_ego_plan' in targets:
            ego_plans = outputs['ego_plan']
            gt_plan = targets['gt_ego_plan']

            # Best mode matching
            plan_errors = torch.norm(ego_plans - gt_plan.unsqueeze(1), dim=-1).mean(dim=-1)
            best_plan_mode = plan_errors.argmin(dim=-1)

            planning_loss = F.smooth_l1_loss(
                ego_plans[torch.arange(len(best_plan_mode)), best_plan_mode],
                gt_plan
            )
            losses['planning_loss'] = planning_loss

        # Total loss
        losses['total_loss'] = sum(losses.values())

        return losses


__all__ = [
    'DriveTransformer',
    'SensorTokenizer',
    'RecurrentMemory',
    'UnifiedTransformerBackbone',
    'TaskDecoder'
]
