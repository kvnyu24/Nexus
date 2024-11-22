import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ....core.base import NexusModule
from .point_cloud import PointCloudProcessor
from .mesh_generator import MeshGenerator
from .texture_mapper import TextureMapper

class CityReconstructionModel(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core configuration
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_heads = config.get("num_heads", 8)
        self.dropout = config.get("dropout", 0.1)
        
        # Initialize components
        self.point_cloud_processor = PointCloudProcessor(config)
        self.mesh_generator = MeshGenerator(config)
        self.texture_mapper = TextureMapper(config)
        
        # Feature extraction backbone (reuse existing FPN architecture)
        self.backbone = self._build_backbone(config)
        
        # City-specific attention module
        self.city_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
    def _build_backbone(self, config: Dict[str, Any]) -> nn.Module:
        # Reuse FPN backbone architecture
        from ..mask_rcnn.backbone import FPNBackbone
        return FPNBackbone(config)
        
    def forward(
        self,
        images: torch.Tensor,
        camera_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.backbone(images)
        
        # Generate point cloud
        point_cloud = self.point_cloud_processor(
            features,
            camera_params=camera_params
        )
        
        # Generate mesh
        mesh = self.mesh_generator(point_cloud)
        
        # Apply textures
        textured_mesh = self.texture_mapper(
            mesh,
            features=features,
            images=images
        )
        
        return {
            "point_cloud": point_cloud,
            "mesh": mesh,
            "textured_mesh": textured_mesh,
            "features": features
        } 