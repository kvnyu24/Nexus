import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ....core.base import NexusModule

class MeshGenerator(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core configuration
        self.hidden_dim = config.get("hidden_dim", 256)
        self.feature_dim = config.get("feature_dim", 128)
        self.num_graph_conv_layers = config.get("num_graph_conv_layers", 3)
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(3, self.feature_dim),  # 3D point coordinates
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.hidden_dim)
        )
        
        # Graph convolution layers for mesh refinement
        self.graph_conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_dim)
            ) for _ in range(self.num_graph_conv_layers)
        ])
        
        # Face generation layers
        self.face_generator = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Edge refinement
        self.edge_refiner = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3)  # 3D edge displacement
        )
        
    def _compute_adjacency(self, points: torch.Tensor, k: int = 8) -> torch.Tensor:
        # Compute k-nearest neighbors for each point
        dist = torch.cdist(points, points)
        _, indices = torch.topk(dist, k=k, dim=-1, largest=False)
        return indices
        
    def forward(
        self,
        points: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_points, _ = points.shape
        
        # Extract point features
        point_features = self.feature_extractor(points)
        if features is not None:
            point_features = point_features + features
            
        # Graph convolution
        adjacency = self._compute_adjacency(points)
        for conv_layer in self.graph_conv_layers:
            # Aggregate neighboring features
            neighbor_features = torch.gather(
                point_features.unsqueeze(2).expand(-1, -1, adjacency.size(1), -1),
                1,
                adjacency.unsqueeze(-1).expand(-1, -1, -1, point_features.size(-1))
            )
            
            # Update features
            point_features = conv_layer(
                torch.cat([point_features, neighbor_features.mean(dim=2)], dim=-1)
            )
            
        # Generate faces
        face_features = torch.cat([
            point_features.unsqueeze(2).expand(-1, -1, num_points, -1),
            point_features.unsqueeze(1).expand(-1, num_points, -1, -1)
        ], dim=-1)
        face_probabilities = self.face_generator(face_features)
        
        # Refine edges
        edge_features = torch.cat([
            point_features.unsqueeze(2).expand(-1, -1, num_points, -1),
            point_features.unsqueeze(1).expand(-1, num_points, -1, -1)
        ], dim=-1)
        edge_refinements = self.edge_refiner(edge_features)
        
        return {
            "vertices": points + edge_refinements.mean(dim=2),
            "face_probabilities": face_probabilities,
            "point_features": point_features
        } 