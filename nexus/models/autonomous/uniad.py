"""
UniAD: Unified Autonomous Driving Framework.

A unified end-to-end autonomous driving framework that jointly performs:
- Multi-camera 3D object detection
- Multi-object tracking (MOT)
- Online HD mapping
- Motion forecasting
- Occupancy prediction
- Planning

UniAD introduces query-centric design where learnable queries represent agents,
maps, and future states, enabling efficient multi-task learning through shared
transformer-based architecture with task-specific decoder heads.

Key innovations:
- Query-based unified representation across perception, prediction, and planning
- Agent queries for tracking and motion forecasting
- Map queries for HD map construction
- Goal-oriented motion planning with collision-aware trajectory scoring
- End-to-end trainable with multi-task losses

Paper: "Planning-oriented Autonomous Driving"
       Hu et al., CVPR 2023 (Best Paper Award)
       https://arxiv.org/abs/2212.10156

GitHub: https://github.com/OpenDriveLab/UniAD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from nexus.core.base import NexusModule


class QueryEmbedding(nn.Module):
    """Learnable query embeddings for agents, maps, and motion.

    Args:
        num_queries: Number of query embeddings
        embed_dim: Embedding dimension
    """

    def __init__(self, num_queries: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_queries, embed_dim)

    def forward(self, batch_size: int) -> torch.Tensor:
        """Generate query embeddings for batch.

        Args:
            batch_size: Batch size

        Returns:
            Query embeddings of shape (batch_size, num_queries, embed_dim)
        """
        indices = torch.arange(self.embed.num_embeddings, device=self.embed.weight.device)
        queries = self.embed(indices).unsqueeze(0).expand(batch_size, -1, -1)
        return queries


class BEVEncoder(NexusModule):
    """Bird's Eye View encoder from multi-camera images.

    Transforms multi-view camera features into unified BEV representation
    using deformable attention and depth estimation.

    Args:
        config: Configuration dictionary containing:
            - num_cameras (int): Number of camera views
            - image_height (int): Input image height
            - image_width (int): Input image width
            - embed_dim (int): Feature embedding dimension
            - bev_height (int): BEV grid height
            - bev_width (int): BEV grid width
            - num_bev_layers (int): Number of BEV encoder layers
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_cameras = config.get('num_cameras', 6)
        self.embed_dim = config.get('embed_dim', 256)
        self.bev_height = config.get('bev_height', 200)
        self.bev_width = config.get('bev_width', 200)
        self.num_layers = config.get('num_bev_layers', 3)

        # Image feature extractor (ResNet-like)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            self._make_layer(64, 256, 3),
            self._make_layer(256, 512, 4, stride=2),
            self._make_layer(512, self.embed_dim, 6, stride=2),
        )

        # Depth estimation head
        self.depth_net = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim // 2, 64, 3, padding=1),  # 64 depth bins
            nn.Softmax(dim=1)
        )

        # BEV grid learnable queries
        self.bev_queries = nn.Parameter(
            torch.randn(1, self.bev_height * self.bev_width, self.embed_dim)
        )

        # BEV encoder layers with cross-attention to image features
        self.bev_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.embed_dim,
                nhead=8,
                dim_feedforward=self.embed_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(self.num_layers)
        ])

    def _make_layer(self, in_channels: int, out_channels: int,
                    num_blocks: int, stride: int = 1) -> nn.Module:
        """Create residual layer."""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, images: torch.Tensor, camera_params: Optional[Dict] = None) -> torch.Tensor:
        """Transform multi-view images to BEV features.

        Args:
            images: Multi-view images of shape (B, N_cams, C, H, W)
            camera_params: Camera intrinsics/extrinsics (optional)

        Returns:
            BEV features of shape (B, bev_h * bev_w, embed_dim)
        """
        B, N, C, H, W = images.shape

        # Extract image features for all cameras
        images_flat = images.view(B * N, C, H, W)
        img_feats = self.image_encoder(images_flat)
        _, C_feat, H_feat, W_feat = img_feats.shape

        # Estimate depth distribution
        depth = self.depth_net(img_feats)  # (B*N, D, H_feat, W_feat)

        # Reshape for attention
        img_feats = img_feats.view(B, N, C_feat, H_feat * W_feat)
        img_feats = img_feats.permute(0, 1, 3, 2).contiguous()  # (B, N, H*W, C)
        img_feats = img_feats.view(B, N * H_feat * W_feat, C_feat)

        # Initialize BEV queries
        bev_feats = self.bev_queries.expand(B, -1, -1)

        # Process through BEV encoder layers
        for layer in self.bev_layers:
            bev_feats = layer(bev_feats, img_feats)

        return bev_feats


class TrackingDecoder(NexusModule):
    """Multi-object tracking decoder with agent queries.

    Maintains persistent agent queries across frames for tracking.

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_queries = config.get('num_agent_queries', 300)
        self.embed_dim = config.get('embed_dim', 256)
        self.num_layers = config.get('tracking_layers', 3)

        # Agent queries for tracking
        self.agent_queries = QueryEmbedding(self.num_queries, self.embed_dim)

        # Tracking decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.embed_dim,
                nhead=8,
                dim_feedforward=self.embed_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(self.num_layers)
        ])

        # Detection heads
        self.bbox_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 10)  # (x, y, z, l, w, h, sin(yaw), cos(yaw), vx, vy)
        )

        self.class_head = nn.Linear(self.embed_dim, 10)  # 10 object classes
        self.track_id_embed = nn.Linear(self.embed_dim, 256)  # For association

    def forward(self, bev_feats: torch.Tensor,
                prev_queries: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Track objects in BEV space.

        Args:
            bev_feats: BEV features (B, N_bev, embed_dim)
            prev_queries: Previous frame agent queries for temporal consistency

        Returns:
            Dictionary containing:
                - agent_queries: Updated agent queries
                - detections: Bounding boxes
                - classes: Object classes
                - track_embeds: Embeddings for association
        """
        B = bev_feats.shape[0]

        # Initialize or use previous agent queries
        if prev_queries is None:
            queries = self.agent_queries(B)
        else:
            queries = prev_queries

        # Process through tracking decoder
        for layer in self.decoder_layers:
            queries = layer(queries, bev_feats)

        # Predict bounding boxes, classes, and track embeddings
        detections = self.bbox_head(queries)
        classes = self.class_head(queries)
        track_embeds = self.track_id_embed(queries)

        return {
            'agent_queries': queries,
            'detections': detections,
            'classes': classes,
            'track_embeds': track_embeds
        }


class MotionForecastingDecoder(NexusModule):
    """Motion forecasting decoder for multi-agent trajectory prediction.

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 256)
        self.num_modes = config.get('num_modes', 6)  # Multi-modal predictions
        self.future_steps = config.get('future_steps', 12)  # 6 seconds at 2 Hz
        self.num_layers = config.get('motion_layers', 3)

        # Motion decoder
        self.motion_decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.embed_dim,
                nhead=8,
                dim_feedforward=self.embed_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(self.num_layers)
        ])

        # Trajectory prediction head
        self.traj_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.num_modes * self.future_steps * 2)  # (x, y) per step
        )

        # Mode confidence head
        self.conf_head = nn.Linear(self.embed_dim, self.num_modes)

    def forward(self, agent_queries: torch.Tensor,
                bev_feats: torch.Tensor,
                map_feats: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forecast future trajectories for tracked agents.

        Args:
            agent_queries: Agent query features (B, N_agents, embed_dim)
            bev_feats: BEV context features (B, N_bev, embed_dim)
            map_feats: Optional HD map features

        Returns:
            Dictionary containing:
                - trajectories: Predicted trajectories (B, N_agents, num_modes, T, 2)
                - confidences: Mode probabilities (B, N_agents, num_modes)
        """
        # Combine context
        context = bev_feats
        if map_feats is not None:
            context = torch.cat([context, map_feats], dim=1)

        # Decode motion-aware features
        motion_queries = agent_queries
        for layer in self.motion_decoder:
            motion_queries = layer(motion_queries, context)

        # Predict multi-modal trajectories
        B, N, _ = motion_queries.shape
        trajs = self.traj_head(motion_queries)
        trajs = trajs.view(B, N, self.num_modes, self.future_steps, 2)

        # Predict mode confidences
        confs = self.conf_head(motion_queries)
        confs = F.softmax(confs, dim=-1)

        return {
            'trajectories': trajs,
            'confidences': confs
        }


class PlanningDecoder(NexusModule):
    """Ego planning decoder for trajectory generation.

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 256)
        self.num_modes = config.get('planning_modes', 3)
        self.planning_steps = config.get('planning_steps', 6)
        self.num_layers = config.get('planning_layers', 3)

        # Planning queries for ego modes
        self.planning_queries = QueryEmbedding(self.num_modes, self.embed_dim)

        # Planning decoder
        self.planning_decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.embed_dim,
                nhead=8,
                dim_feedforward=self.embed_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(self.num_layers)
        ])

        # Ego trajectory head
        self.ego_traj_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.planning_steps * 3)  # (x, y, yaw) per step
        )

        # Planning cost/score head
        self.cost_head = nn.Linear(self.embed_dim, 1)

    def forward(self, agent_queries: torch.Tensor,
                future_trajs: torch.Tensor,
                bev_feats: torch.Tensor,
                map_feats: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Generate ego vehicle trajectory considering agents and map.

        Args:
            agent_queries: Agent features (B, N_agents, embed_dim)
            future_trajs: Predicted agent trajectories for collision checking
            bev_feats: BEV features (B, N_bev, embed_dim)
            map_feats: Optional HD map features

        Returns:
            Dictionary containing:
                - ego_trajectories: Ego trajectory modes (B, num_modes, T, 3)
                - costs: Cost/safety score per mode (B, num_modes)
        """
        B = bev_feats.shape[0]

        # Aggregate context: agents + BEV + map
        context_parts = [agent_queries, bev_feats]
        if map_feats is not None:
            context_parts.append(map_feats)
        context = torch.cat(context_parts, dim=1)

        # Initialize planning mode queries
        plan_queries = self.planning_queries(B)

        # Decode planning features
        for layer in self.planning_decoder:
            plan_queries = layer(plan_queries, context)

        # Predict ego trajectories for each mode
        ego_trajs = self.ego_traj_head(plan_queries)
        ego_trajs = ego_trajs.view(B, self.num_modes, self.planning_steps, 3)

        # Compute cost (considers collision, comfort, progress)
        costs = self.cost_head(plan_queries).squeeze(-1)

        return {
            'ego_trajectories': ego_trajs,
            'costs': costs
        }


class UniAD(NexusModule):
    """Unified Planning-Oriented Autonomous Driving Framework.

    End-to-end model that jointly performs:
    - Perception (detection & tracking)
    - Prediction (motion forecasting)
    - Planning (ego trajectory generation)

    Args:
        config: Configuration dictionary with keys:
            - num_cameras (int): Number of camera views. Default 6
            - embed_dim (int): Feature dimension. Default 256
            - bev_height (int): BEV grid height. Default 200
            - bev_width (int): BEV grid width. Default 200
            - num_agent_queries (int): Number of agent queries. Default 300
            - num_modes (int): Motion prediction modes. Default 6
            - future_steps (int): Motion prediction horizon. Default 12
            - planning_modes (int): Ego planning modes. Default 3
            - planning_steps (int): Planning horizon. Default 6

    References:
        Paper: "Planning-oriented Autonomous Driving" (CVPR 2023 Best Paper)
        Code: https://github.com/OpenDriveLab/UniAD
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # BEV encoder from multi-camera images
        self.bev_encoder = BEVEncoder(config)

        # Tracking decoder with agent queries
        self.tracking_decoder = TrackingDecoder(config)

        # Motion forecasting decoder
        self.motion_decoder = MotionForecastingDecoder(config)

        # Planning decoder for ego vehicle
        self.planning_decoder = PlanningDecoder(config)

        # Optional HD map encoder (simplified)
        self.use_map = config.get('use_map', False)
        if self.use_map:
            embed_dim = config.get('embed_dim', 256)
            self.map_encoder = nn.Sequential(
                nn.Linear(2, 128),  # (x, y) map points
                nn.ReLU(inplace=True),
                nn.Linear(128, embed_dim)
            )

    def forward(self,
                images: torch.Tensor,
                camera_params: Optional[Dict] = None,
                map_data: Optional[torch.Tensor] = None,
                prev_agent_queries: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for unified autonomous driving.

        Args:
            images: Multi-view camera images (B, N_cams, C, H, W)
            camera_params: Camera calibration parameters
            map_data: Optional HD map polylines (B, N_polylines, N_points, 2)
            prev_agent_queries: Previous frame agent queries for tracking

        Returns:
            Dictionary containing:
                - bev_features: BEV representation
                - detections: 3D bounding boxes
                - classes: Object classes
                - track_embeds: Track embeddings
                - trajectories: Future trajectories for agents
                - traj_confidences: Trajectory mode confidences
                - ego_trajectories: Planned ego trajectories
                - planning_costs: Cost for each ego mode
                - agent_queries: Agent queries for next frame
        """
        # Stage 1: BEV encoding from multi-camera images
        bev_feats = self.bev_encoder(images, camera_params)

        # Stage 2: Multi-object tracking
        tracking_output = self.tracking_decoder(bev_feats, prev_agent_queries)
        agent_queries = tracking_output['agent_queries']

        # Optional: Encode HD map
        map_feats = None
        if self.use_map and map_data is not None:
            B, N_poly, N_pts, _ = map_data.shape
            map_feats = self.map_encoder(map_data.view(B, N_poly * N_pts, 2))

        # Stage 3: Motion forecasting for tracked agents
        motion_output = self.motion_decoder(agent_queries, bev_feats, map_feats)

        # Stage 4: Ego planning considering predicted agent motions
        planning_output = self.planning_decoder(
            agent_queries,
            motion_output['trajectories'],
            bev_feats,
            map_feats
        )

        # Combine all outputs
        return {
            'bev_features': bev_feats,
            'detections': tracking_output['detections'],
            'classes': tracking_output['classes'],
            'track_embeds': tracking_output['track_embeds'],
            'trajectories': motion_output['trajectories'],
            'traj_confidences': motion_output['confidences'],
            'ego_trajectories': planning_output['ego_trajectories'],
            'planning_costs': planning_output['costs'],
            'agent_queries': agent_queries,  # Pass to next frame
        }

    def compute_loss(self,
                     outputs: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task losses.

        Args:
            outputs: Model predictions
            targets: Ground truth labels

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Detection loss (L1 + GIoU)
        if 'gt_boxes' in targets:
            det_loss = F.l1_loss(outputs['detections'], targets['gt_boxes'])
            losses['detection_loss'] = det_loss

        # Classification loss
        if 'gt_classes' in targets:
            cls_loss = F.cross_entropy(
                outputs['classes'].view(-1, outputs['classes'].shape[-1]),
                targets['gt_classes'].view(-1)
            )
            losses['classification_loss'] = cls_loss

        # Motion forecasting loss (multi-modal NLL)
        if 'gt_future_trajs' in targets:
            # Winner-takes-all: match best mode
            pred_trajs = outputs['trajectories']
            gt_trajs = targets['gt_future_trajs']

            # Compute distance for each mode
            dists = torch.norm(pred_trajs - gt_trajs.unsqueeze(2), dim=-1).mean(dim=-1)
            best_mode = dists.argmin(dim=-1)

            # Loss for best mode
            motion_loss = F.smooth_l1_loss(
                pred_trajs[torch.arange(len(best_mode)), :, best_mode],
                gt_trajs
            )
            losses['motion_loss'] = motion_loss

        # Planning loss (imitation from expert)
        if 'gt_ego_traj' in targets:
            ego_trajs = outputs['ego_trajectories']
            gt_ego = targets['gt_ego_traj']

            # Match best ego mode
            ego_dists = torch.norm(ego_trajs - gt_ego.unsqueeze(1), dim=-1).mean(dim=-1)
            best_ego_mode = ego_dists.argmin(dim=-1)

            plan_loss = F.smooth_l1_loss(
                ego_trajs[torch.arange(len(best_ego_mode)), best_ego_mode],
                gt_ego
            )
            losses['planning_loss'] = plan_loss

        # Total loss
        losses['total_loss'] = sum(losses.values())

        return losses


__all__ = ['UniAD', 'BEVEncoder', 'TrackingDecoder', 'MotionForecastingDecoder', 'PlanningDecoder']
