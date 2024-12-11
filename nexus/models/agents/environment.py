import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from ...core.base import NexusModule
import numpy as np
import torch.nn.functional as F

class ObjectRegistry:
    def __init__(self, config: Dict[str, Any]):
        self.hidden_dim = config["hidden_dim"]
        self.max_objects = config.get("max_objects", 1000)
        self.object_types = config.get("object_types", ["item", "furniture", "structure", "agent", "resource"])
        
        # Object state storage with enhanced tracking
        self.objects = {}
        self.object_embeddings = nn.Parameter(
            torch.randn(self.max_objects, self.hidden_dim)
        )
        self.type_embeddings = nn.Embedding(len(self.object_types), self.hidden_dim)
        
        # Track object states and interactions
        self.interaction_history = {}
        self.state_history = {}
        self.last_update_time = {}
        
    def register_object(
        self,
        object_id: str,
        object_type: str,
        position: torch.Tensor,
        properties: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        if object_type not in self.object_types:
            raise ValueError(f"Invalid object type: {object_type}")
            
        if len(self.objects) >= self.max_objects:
            raise RuntimeError("Maximum object limit reached")
            
        self.objects[object_id] = {
            "type": object_type,
            "position": position,
            "properties": properties,
            "metadata": metadata or {},
            "embedding_idx": len(self.objects),
            "active": True
        }
        
        self.interaction_history[object_id] = []
        self.state_history[object_id] = []
        self.last_update_time[object_id] = 0
        
    def get_object_embedding(self, object_id: str) -> torch.Tensor:
        if object_id not in self.objects:
            raise KeyError(f"Object not found: {object_id}")
            
        obj = self.objects[object_id]
        if not obj["active"]:
            raise ValueError(f"Object {object_id} is no longer active")
            
        type_embed = self.type_embeddings(
            torch.tensor(self.object_types.index(obj["type"]))
        )
        
        # Combine embeddings with learned weights
        base_embed = self.object_embeddings[obj["embedding_idx"]]
        combined = 0.7 * base_embed + 0.3 * type_embed
        
        # Add positional encoding
        pos_encoding = self._positional_encoding(obj["position"])
        return combined + 0.1 * pos_encoding
        
    def _positional_encoding(self, pos: torch.Tensor) -> torch.Tensor:
        # Implement sinusoidal positional encoding
        dim = self.hidden_dim
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pos_enc = torch.zeros(dim)
        pos_enc[0::2] = torch.sin(pos[0] * div_term)
        pos_enc[1::2] = torch.cos(pos[0] * div_term)
        return pos_enc

class VirtualEnvironment(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_dim = config["hidden_dim"]
        self.grid_size = config.get("grid_size", (32, 32))
        self.max_agents = config.get("max_agents", 100)
        
        # Environment components
        self.object_registry = ObjectRegistry(config)
        
        # Enhanced spatial encoding with multi-scale features
        scales = [1, 2, 4]
        self.position_embeddings = nn.ModuleList([
            nn.Parameter(torch.randn(
                self.grid_size[0] // scale,
                self.grid_size[1] // scale,
                self.hidden_dim
            )) for scale in scales
        ])
        
        # Environment state processing with attention
        self.state_processor = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        self.state_attention = nn.MultiheadAttention(
            self.hidden_dim, 
            num_heads=8,
            dropout=0.1
        )
        
        # Physics simulation parameters
        self.collision_threshold = config.get("collision_threshold", 0.5)
        self.interaction_radius = config.get("interaction_radius", 2.0)
        self.friction = config.get("friction", 0.1)
        self.elasticity = config.get("elasticity", 0.5)
        
    def get_local_state(
        self,
        position: torch.Tensor,
        radius: Optional[float] = None,
        include_objects: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Get the local environment state around a position"""
        if radius is None:
            radius = self.interaction_radius
            
        # Get grid positions within radius
        x, y = position[..., 0], position[..., 1]
        grid_x = torch.arange(self.grid_size[0], device=position.device)
        grid_y = torch.arange(self.grid_size[1], device=position.device)
        
        distances = torch.sqrt(
            (grid_x.unsqueeze(1) - x.unsqueeze(-1)) ** 2 +
            (grid_y.unsqueeze(1) - y.unsqueeze(-1)) ** 2
        )
        
        mask = distances <= radius
        
        # Combine multi-scale embeddings
        local_embeddings = []
        for pos_embed in self.position_embeddings:
            scaled_mask = F.interpolate(
                mask.float().unsqueeze(0),
                size=pos_embed.shape[:2],
                mode='nearest'
            ).bool()[0]
            local_embeddings.append(pos_embed[scaled_mask])
            
        combined_embeddings = torch.cat(local_embeddings, dim=0)
        
        # Add nearby objects if requested
        nearby_objects = {}
        if include_objects:
            for obj_id, obj in self.object_registry.objects.items():
                if obj["active"]:
                    obj_dist = torch.norm(obj["position"] - position)
                    if obj_dist <= radius:
                        nearby_objects[obj_id] = {
                            "distance": obj_dist,
                            "embedding": self.object_registry.get_object_embedding(obj_id)
                        }
        
        return {
            "local_state": combined_embeddings,
            "distances": distances[mask],
            "mask": mask,
            "nearby_objects": nearby_objects
        }
        
    def step(
        self,
        agent_actions: torch.Tensor,
        agent_interactions: torch.Tensor,
        dt: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Update environment state based on agent actions"""
        # Process agent actions with momentum
        new_positions = self._apply_actions(agent_actions, dt)
        
        # Handle collisions with improved physics
        valid_positions = self._resolve_collisions(new_positions)
        
        # Update environment state with attention
        state_features = torch.cat([
            self.position_embeddings[0].view(-1, self.hidden_dim),
            agent_interactions.mean(dim=0).unsqueeze(0).expand(
                self.grid_size[0] * self.grid_size[1],
                self.hidden_dim
            ),
            valid_positions.mean(dim=0).unsqueeze(0).expand(
                self.grid_size[0] * self.grid_size[1], 
                self.hidden_dim
            )
        ], dim=-1)
        
        state_update = self.state_processor(state_features)
        
        # Apply self-attention for global consistency
        state_update = state_update.view(-1, self.hidden_dim).unsqueeze(0)
        state_update, _ = self.state_attention(
            state_update, 
            state_update,
            state_update
        )
        state_update = state_update.squeeze(0)
        
        return {
            "new_positions": valid_positions,
            "state_embedding": state_update.view(
                self.grid_size[0],
                self.grid_size[1],
                self.hidden_dim
            ),
            "collision_mask": self._get_collision_mask(valid_positions)
        }
        
    def _apply_actions(
        self,
        actions: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """Convert actions to new positions with momentum"""
        with torch.no_grad():
            # Enhanced movement with 8 directions
            action_vectors = F.one_hot(actions, 8)
            movements = torch.tensor([
                [0, 1],    # up
                [0, -1],   # down
                [-1, 0],   # left
                [1, 0],    # right
                [1, 1],    # up-right
                [-1, 1],   # up-left
                [1, -1],   # down-right
                [-1, -1]   # down-left
            ], device=actions.device, dtype=torch.float)
            
            # Apply friction and time scaling
            position_updates = torch.matmul(
                action_vectors.float(),
                movements
            ) * (1 - self.friction) * dt
            
            return position_updates
            
    def _resolve_collisions(self, positions: torch.Tensor) -> torch.Tensor:
        """Resolve collisions between agents and objects with improved physics"""
        distances = torch.cdist(positions, positions)
        collisions = distances < self.collision_threshold
        
        # Zero out self-collisions
        collisions.fill_diagonal_(False)
        
        # Calculate collision response with elasticity
        collision_forces = torch.zeros_like(positions)
        collision_mask = collisions.any(dim=1)
        
        if collision_mask.any():
            colliding_positions = positions[collision_mask]
            collision_pairs = positions[collisions]
            
            # Elastic collision response
            relative_vel = collision_pairs - colliding_positions.unsqueeze(1)
            collision_response = (
                relative_vel * self.elasticity +
                (relative_vel.sign() * self.collision_threshold)
            )
            
            collision_forces[collision_mask] = collision_response.mean(dim=1)
        
        return positions + collision_forces
        
    def _get_collision_mask(self, positions: torch.Tensor) -> torch.Tensor:
        """Generate mask indicating collision areas"""
        collision_map = torch.zeros(self.grid_size, device=positions.device)
        positions_grid = positions.long()
        
        # Mark collision areas
        for pos in positions_grid:
            x, y = pos
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                collision_map[x, y] = 1.0
                
        return collision_map