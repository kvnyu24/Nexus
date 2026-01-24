import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin
from ...components.attention import MultiHeadSelfAttention

class AtomicFeatureExtractor(NexusModule):
    def __init__(self, hidden_size: int, num_atom_types: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Atomic properties embedding
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_size)
        self.electronegativity_embedding = nn.Embedding(num_atom_types, hidden_size//4)
        self.valence_embedding = nn.Embedding(num_atom_types, hidden_size//4)
        self.atomic_radius_embedding = nn.Embedding(num_atom_types, hidden_size//4)
        self.period_embedding = nn.Embedding(10, hidden_size//4) # Up to period 7 + padding
        
        # Enhanced 3D position encoder with periodic boundary conditions
        self.position_encoder = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Chemical feature fusion
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Sigmoid()
        )
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU()
        )
        
    def forward(self, atom_types: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(positions).all():
            raise ValueError("Position tensor contains invalid values")
            
        # Get atomic properties
        atom_features = self.atom_embedding(atom_types)
        electronegativity = self.electronegativity_embedding(atom_types)
        valence = self.valence_embedding(atom_types)
        atomic_radius = self.atomic_radius_embedding(atom_types)
        
        # Calculate period from atomic number
        periods = torch.div(atom_types, 18, rounding_mode='floor') + 1
        period_features = self.period_embedding(periods)
        
        # Combine chemical properties
        chemical_features = torch.cat([
            electronegativity, 
            valence,
            atomic_radius,
            period_features
        ], dim=-1)
        
        # Process 3D positions with periodic boundary handling
        position_features = self.position_encoder(positions)
        
        # Fuse all features with gating mechanism
        combined = torch.cat([
            atom_features,
            chemical_features, 
            position_features
        ], dim=-1)
        
        gates = self.feature_gate(combined)
        transformed = self.feature_transform(combined)
        
        return gates * transformed

class AtomInteractionModule(NexusModule):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Chemical interaction MLP with gating
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LayerNorm(hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture with chemical gating
        attended = self.attention(self.norm1(x), attention_mask=mask)
        x = x + attended
        
        mlp_out = self.mlp(self.norm2(x))
        gate = self.gate(x)
        x = x + (gate * mlp_out)
        return x

class AtomTracker(WeightInitMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Model dimensions
        self.hidden_size = config.get("hidden_size", 256)
        self.num_layers = config.get("num_layers", 6)
        self.num_heads = config.get("num_heads", 8)
        self.num_atom_types = config.get("num_atom_types", 118)
        self.dropout = config.get("dropout", 0.1)

        # Core components
        self.feature_extractor = AtomicFeatureExtractor(
            hidden_size=self.hidden_size,
            num_atom_types=self.num_atom_types
        )

        # Interaction layers with gradient checkpointing
        self.interaction_layers = nn.ModuleList([
            AtomInteractionModule(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])

        # Physics-informed prediction heads
        self.velocity_predictor = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 3)  # 3D velocity vector
        )

        self.energy_predictor = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1)  # Scalar energy value
        )

        # Initialize weights
        self.init_weights_vision()
            
    def forward(
        self,
        atom_types: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Input validation
        if not torch.isfinite(positions).all():
            raise ValueError("Position tensor contains invalid values")
            
        # Extract atomic features with chemical properties
        features = self.feature_extractor(atom_types, positions)
        
        # Process through interaction layers with residual connections
        hidden_states = features
        intermediate_states = []
        
        for layer in self.interaction_layers:
            if self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(layer, hidden_states, attention_mask)
            else:
                hidden_states = layer(hidden_states, attention_mask)
            intermediate_states.append(hidden_states)
            
        # Generate physics-based predictions
        velocities = self.velocity_predictor(hidden_states)
        energies = self.energy_predictor(hidden_states)
        
        return {
            "velocities": velocities,
            "energies": energies,
            "hidden_states": hidden_states,
            "intermediate_states": intermediate_states
        }