"""
VAD: Vectorized Autonomous Driving.

VAD represents the driving scene using vectorized formats for lanes, agents,
and ego trajectories, enabling efficient end-to-end learning. Unlike raster-based
approaches, VAD directly predicts and reasons over structured vector representations,
improving interpretability and downstream task performance.

Key innovations:
- Vectorized scene representation (polylines for lanes, bounding boxes for agents)
- Hierarchical query design: scene-level, agent-level, and point-level queries
- Vector-based map construction using polyline queries
- End-to-end motion planning with vectorized goal-conditioned trajectory generation
- Differentiable vector matching for training with structured outputs

The model uses a unified transformer architecture with specialized decoders
for each vectorized output type.

Paper: "Vectorized Scene Representation for Efficient Autonomous Driving"
       Zhou et al., ICCV 2023
       https://arxiv.org/abs/2303.12077

GitHub: https://github.com/hustvl/VAD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from nexus.core.base import NexusModule
import math


class VectorEncoder(nn.Module):
    """Encode vector primitives (points, lines) into embeddings.

    Args:
        input_dim: Input coordinate dimension (2 for 2D, 3 for 3D)
        embed_dim: Output embedding dimension
    """

    def __init__(self, input_dim: int = 2, embed_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coordinates to embeddings.

        Args:
            coords: Coordinate tensor (..., input_dim)

        Returns:
            Embeddings of shape (..., embed_dim)
        """
        return self.encoder(coords)


class PolylineEncoder(NexusModule):
    """Encode polylines as sequences of connected points.

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 256)
        self.num_layers = config.get('polyline_layers', 2)

        # Point encoder
        self.point_encoder = VectorEncoder(input_dim=2, embed_dim=self.embed_dim)

        # Polyline aggregation with self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.polyline_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

    def forward(self, polylines: torch.Tensor,
                polyline_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode polylines.

        Args:
            polylines: Polyline points (B, N_polylines, N_points, 2)
            polyline_masks: Valid point masks (B, N_polylines, N_points)

        Returns:
            Polyline embeddings (B, N_polylines, embed_dim)
        """
        B, N_poly, N_pts, _ = polylines.shape

        # Encode points
        polylines_flat = polylines.view(B * N_poly, N_pts, 2)
        point_embeds = self.point_encoder(polylines_flat)

        # Aggregate points per polyline
        if polyline_masks is not None:
            masks_flat = polyline_masks.view(B * N_poly, N_pts)
            # Create attention mask
            attn_mask = ~masks_flat.bool()
        else:
            attn_mask = None

        polyline_embeds = self.polyline_encoder(
            point_embeds,
            src_key_padding_mask=attn_mask
        )

        # Pool to single vector per polyline (mean pooling over valid points)
        if polyline_masks is not None:
            valid_counts = polyline_masks.view(B * N_poly, N_pts).sum(dim=1, keepdim=True).clamp(min=1)
            masked_embeds = polyline_embeds * polyline_masks.view(B * N_poly, N_pts, 1)
            polyline_feats = masked_embeds.sum(dim=1) / valid_counts
        else:
            polyline_feats = polyline_embeds.mean(dim=1)

        polyline_feats = polyline_feats.view(B, N_poly, self.embed_dim)
        return polyline_feats


class VectorMapDecoder(NexusModule):
    """Decoder for vectorized HD map construction.

    Predicts map elements as polylines with semantic labels.

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 256)
        self.num_queries = config.get('num_map_queries', 100)
        self.num_points_per_line = config.get('points_per_line', 20)
        self.num_classes = config.get('num_map_classes', 3)  # lane, boundary, crossing
        self.num_layers = config.get('map_decoder_layers', 3)

        # Map element queries
        self.map_queries = nn.Parameter(torch.randn(1, self.num_queries, self.embed_dim))

        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        # Prediction heads
        self.polyline_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.num_points_per_line * 2)  # (x, y) per point
        )

        self.class_head = nn.Linear(self.embed_dim, self.num_classes)
        self.confidence_head = nn.Linear(self.embed_dim, 1)

    def forward(self, bev_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode vectorized map elements.

        Args:
            bev_features: BEV feature map (B, N_bev, embed_dim)

        Returns:
            Dictionary containing:
                - polylines: Predicted map polylines (B, N_queries, N_points, 2)
                - classes: Map element classes (B, N_queries, num_classes)
                - confidences: Detection confidence (B, N_queries)
        """
        B = bev_features.shape[0]

        # Initialize map queries
        queries = self.map_queries.expand(B, -1, -1)

        # Decode with attention to BEV
        decoded = self.decoder(queries, bev_features)

        # Predict polylines
        polylines = self.polyline_head(decoded)
        polylines = polylines.view(B, self.num_queries, self.num_points_per_line, 2)

        # Predict classes and confidence
        classes = self.class_head(decoded)
        confidences = self.confidence_head(decoded).squeeze(-1)
        confidences = torch.sigmoid(confidences)

        return {
            'polylines': polylines,
            'classes': classes,
            'confidences': confidences
        }


class VectorAgentDecoder(NexusModule):
    """Decoder for vectorized agent detection and tracking.

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 256)
        self.num_queries = config.get('num_agent_queries', 200)
        self.num_classes = config.get('num_agent_classes', 10)
        self.num_layers = config.get('agent_decoder_layers', 3)

        # Agent queries
        self.agent_queries = nn.Parameter(torch.randn(1, self.num_queries, self.embed_dim))

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        # Prediction heads
        self.bbox_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 8)  # (x, y, l, w, sin(yaw), cos(yaw), vx, vy)
        )

        self.class_head = nn.Linear(self.embed_dim, self.num_classes)
        self.confidence_head = nn.Linear(self.embed_dim, 1)

    def forward(self, bev_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode vectorized agents.

        Args:
            bev_features: BEV feature map (B, N_bev, embed_dim)

        Returns:
            Dictionary containing:
                - bboxes: Agent bounding boxes (B, N_queries, 8)
                - classes: Agent classes (B, N_queries, num_classes)
                - confidences: Detection confidence (B, N_queries)
                - agent_features: Agent feature embeddings (B, N_queries, embed_dim)
        """
        B = bev_features.shape[0]

        # Initialize agent queries
        queries = self.agent_queries.expand(B, -1, -1)

        # Decode
        decoded = self.decoder(queries, bev_features)

        # Predictions
        bboxes = self.bbox_head(decoded)
        classes = self.class_head(decoded)
        confidences = self.confidence_head(decoded).squeeze(-1)
        confidences = torch.sigmoid(confidences)

        return {
            'bboxes': bboxes,
            'classes': classes,
            'confidences': confidences,
            'agent_features': decoded
        }


class VectorMotionDecoder(NexusModule):
    """Decoder for vectorized trajectory prediction.

    Predicts future trajectories as sequences of waypoints.

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 256)
        self.num_modes = config.get('num_motion_modes', 6)
        self.num_future_steps = config.get('num_future_steps', 12)
        self.num_layers = config.get('motion_decoder_layers', 3)

        # Motion decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        # Trajectory prediction head (multi-modal)
        self.traj_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.num_modes * self.num_future_steps * 2)
        )

        self.mode_prob_head = nn.Linear(self.embed_dim, self.num_modes)

    def forward(self,
                agent_features: torch.Tensor,
                map_features: torch.Tensor,
                bev_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict vectorized trajectories for agents.

        Args:
            agent_features: Agent embeddings (B, N_agents, embed_dim)
            map_features: Map polyline embeddings (B, N_map, embed_dim)
            bev_features: BEV context (B, N_bev, embed_dim)

        Returns:
            Dictionary containing:
                - trajectories: Future trajectories (B, N_agents, num_modes, T, 2)
                - mode_probs: Mode probabilities (B, N_agents, num_modes)
        """
        # Combine context
        context = torch.cat([map_features, bev_features], dim=1)

        # Decode motion
        motion_features = self.decoder(agent_features, context)

        # Predict multi-modal trajectories
        B, N, _ = motion_features.shape
        trajs = self.traj_head(motion_features)
        trajs = trajs.view(B, N, self.num_modes, self.num_future_steps, 2)

        # Predict mode probabilities
        mode_probs = self.mode_prob_head(motion_features)
        mode_probs = F.softmax(mode_probs, dim=-1)

        return {
            'trajectories': trajs,
            'mode_probs': mode_probs
        }


class VectorPlanningDecoder(NexusModule):
    """Decoder for vectorized ego trajectory planning.

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 256)
        self.num_modes = config.get('num_planning_modes', 3)
        self.num_waypoints = config.get('num_waypoints', 6)
        self.num_layers = config.get('planning_decoder_layers', 3)

        # Planning mode queries
        self.planning_queries = nn.Parameter(
            torch.randn(1, self.num_modes, self.embed_dim)
        )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        # Planning heads
        self.waypoint_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.num_waypoints * 3)  # (x, y, yaw)
        )

        self.cost_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self,
                agent_features: torch.Tensor,
                agent_trajectories: torch.Tensor,
                map_features: torch.Tensor,
                bev_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Plan ego trajectory considering all scene elements.

        Args:
            agent_features: Agent embeddings (B, N_agents, embed_dim)
            agent_trajectories: Predicted agent future paths
            map_features: Map embeddings (B, N_map, embed_dim)
            bev_features: BEV context (B, N_bev, embed_dim)

        Returns:
            Dictionary containing:
                - waypoints: Ego trajectory waypoints (B, num_modes, num_waypoints, 3)
                - costs: Trajectory cost/score (B, num_modes)
        """
        B = bev_features.shape[0]

        # Aggregate all context
        context = torch.cat([agent_features, map_features, bev_features], dim=1)

        # Initialize planning queries
        queries = self.planning_queries.expand(B, -1, -1)

        # Decode planning
        planning_features = self.decoder(queries, context)

        # Predict waypoints
        waypoints = self.waypoint_head(planning_features)
        waypoints = waypoints.view(B, self.num_modes, self.num_waypoints, 3)

        # Compute costs
        costs = self.cost_head(planning_features).squeeze(-1)

        return {
            'waypoints': waypoints,
            'costs': costs
        }


class VAD(NexusModule):
    """Vectorized Autonomous Driving Framework.

    End-to-end autonomous driving using fully vectorized representations
    for maps, agents, and trajectories.

    Args:
        config: Configuration dictionary with keys:
            - embed_dim (int): Feature embedding dimension. Default 256
            - num_cameras (int): Number of camera views. Default 6
            - bev_height (int): BEV grid height. Default 200
            - bev_width (int): BEV grid width. Default 200
            - num_map_queries (int): Number of map element queries. Default 100
            - num_agent_queries (int): Number of agent queries. Default 200
            - num_motion_modes (int): Motion prediction modes. Default 6
            - num_future_steps (int): Future prediction steps. Default 12
            - num_planning_modes (int): Planning trajectory modes. Default 3
            - num_waypoints (int): Waypoints per trajectory. Default 6

    References:
        Paper: "Vectorized Scene Representation for Efficient Autonomous Driving" (ICCV 2023)
        Code: https://github.com/hustvl/VAD
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 256)

        # BEV encoder (simplified - can be replaced with proper BEV encoder)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(3 * config.get('num_cameras', 6), 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.embed_dim, 3, padding=1),
        )

        # Vectorized decoders
        self.map_decoder = VectorMapDecoder(config)
        self.agent_decoder = VectorAgentDecoder(config)

        # Polyline encoder for map elements
        self.polyline_encoder = PolylineEncoder(config)

        # Motion and planning decoders
        self.motion_decoder = VectorMotionDecoder(config)
        self.planning_decoder = VectorPlanningDecoder(config)

        # BEV feature flattening
        self.bev_height = config.get('bev_height', 200)
        self.bev_width = config.get('bev_width', 200)

    def forward(self,
                images: torch.Tensor,
                camera_params: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for vectorized autonomous driving.

        Args:
            images: Multi-camera images (B, N_cams, C, H, W)
            camera_params: Optional camera parameters

        Returns:
            Dictionary containing all vectorized predictions:
                - map_polylines: Predicted HD map as polylines
                - map_classes: Map element classes
                - agent_bboxes: Detected agent bounding boxes
                - agent_classes: Agent classes
                - trajectories: Predicted agent trajectories
                - ego_waypoints: Planned ego waypoints
                - planning_costs: Costs for ego modes
        """
        B, N, C, H, W = images.shape

        # Extract BEV features (simplified projection)
        # In practice, use proper BEV encoder like in UniAD
        images_flat = images.view(B, N * C, H, W)
        bev_map = F.interpolate(images_flat, size=(self.bev_height, self.bev_width))
        bev_feats = self.bev_encoder(bev_map)

        # Flatten BEV for transformer
        bev_feats_flat = bev_feats.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        # Stage 1: Vectorized map construction
        map_output = self.map_decoder(bev_feats_flat)
        map_polylines = map_output['polylines']

        # Encode map polylines
        map_features = self.polyline_encoder(map_polylines)

        # Stage 2: Vectorized agent detection
        agent_output = self.agent_decoder(bev_feats_flat)
        agent_features = agent_output['agent_features']

        # Stage 3: Vectorized motion forecasting
        motion_output = self.motion_decoder(
            agent_features,
            map_features,
            bev_feats_flat
        )

        # Stage 4: Vectorized ego planning
        planning_output = self.planning_decoder(
            agent_features,
            motion_output['trajectories'],
            map_features,
            bev_feats_flat
        )

        return {
            'map_polylines': map_output['polylines'],
            'map_classes': map_output['classes'],
            'map_confidences': map_output['confidences'],
            'agent_bboxes': agent_output['bboxes'],
            'agent_classes': agent_output['classes'],
            'agent_confidences': agent_output['confidences'],
            'trajectories': motion_output['trajectories'],
            'trajectory_probs': motion_output['mode_probs'],
            'ego_waypoints': planning_output['waypoints'],
            'planning_costs': planning_output['costs'],
        }

    def compute_loss(self,
                     outputs: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor],
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """Compute vectorized losses with Hungarian matching.

        Args:
            outputs: Model predictions
            targets: Ground truth in vectorized format
            config: Optional loss configuration

        Returns:
            Dictionary of losses
        """
        losses = {}
        loss_weights = config.get('loss_weights', {}) if config else {}

        # Map polyline loss (Chamfer distance + classification)
        if 'gt_map_polylines' in targets:
            # Simplified - in practice use Hungarian matching
            map_loss = F.smooth_l1_loss(
                outputs['map_polylines'],
                targets['gt_map_polylines']
            )
            losses['map_loss'] = map_loss * loss_weights.get('map', 1.0)

            # Map classification loss
            if 'gt_map_classes' in targets:
                map_cls_loss = F.cross_entropy(
                    outputs['map_classes'].view(-1, outputs['map_classes'].shape[-1]),
                    targets['gt_map_classes'].view(-1)
                )
                losses['map_cls_loss'] = map_cls_loss * loss_weights.get('map_cls', 0.5)

        # Agent detection loss
        if 'gt_agent_bboxes' in targets:
            agent_loss = F.smooth_l1_loss(
                outputs['agent_bboxes'],
                targets['gt_agent_bboxes']
            )
            losses['agent_loss'] = agent_loss * loss_weights.get('agent', 2.0)

        # Motion forecasting loss
        if 'gt_trajectories' in targets:
            # Winner-takes-all for multi-modal
            pred_trajs = outputs['trajectories']
            gt_trajs = targets['gt_trajectories']

            # Find best mode
            errors = torch.norm(
                pred_trajs - gt_trajs.unsqueeze(2),
                dim=-1
            ).mean(dim=-1)
            best_mode = errors.argmin(dim=-1)

            # Loss on best mode
            B, N = best_mode.shape
            batch_idx = torch.arange(B, device=best_mode.device).unsqueeze(1).expand(B, N)
            agent_idx = torch.arange(N, device=best_mode.device).unsqueeze(0).expand(B, N)

            motion_loss = F.smooth_l1_loss(
                pred_trajs[batch_idx, agent_idx, best_mode],
                gt_trajs
            )
            losses['motion_loss'] = motion_loss * loss_weights.get('motion', 1.5)

        # Ego planning loss
        if 'gt_ego_waypoints' in targets:
            ego_waypoints = outputs['ego_waypoints']
            gt_ego = targets['gt_ego_waypoints']

            # Find best mode
            ego_errors = torch.norm(
                ego_waypoints - gt_ego.unsqueeze(1),
                dim=-1
            ).mean(dim=-1)
            best_ego_mode = ego_errors.argmin(dim=-1)

            planning_loss = F.smooth_l1_loss(
                ego_waypoints[torch.arange(len(best_ego_mode)), best_ego_mode],
                gt_ego
            )
            losses['planning_loss'] = planning_loss * loss_weights.get('planning', 2.0)

        # Total loss
        losses['total_loss'] = sum(losses.values())

        return losses


__all__ = [
    'VAD',
    'VectorEncoder',
    'PolylineEncoder',
    'VectorMapDecoder',
    'VectorAgentDecoder',
    'VectorMotionDecoder',
    'VectorPlanningDecoder'
]
