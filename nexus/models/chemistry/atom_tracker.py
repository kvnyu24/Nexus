import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from ...core.base import NexusModule
from ...components.attention import MultiHeadSelfAttention

class AtomicFeatureExtractor(NexusModule):
    def __init__(self, hidden_size: int, num_atom_types: int):
        super().__init__()
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_size)
        self.position_encoder = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.feature_combiner = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, atom_types: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        atom_features = self.atom_embedding(atom_types)
        position_features = self.position_encoder(positions)
        return self.feature_combiner(torch.cat([atom_features, position_features], dim=-1))

class AtomInteractionModule(NexusModule):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attended = self.attention(self.norm1(x), attention_mask=mask)
        x = x + attended
        x = x + self.mlp(self.norm2(x))
        return x

class AtomTracker(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model dimensions
        self.hidden_size = config.get("hidden_size", 256)
        self.num_layers = config.get("num_layers", 6)
        self.num_heads = config.get("num_heads", 8)
        self.num_atom_types = config.get("num_atom_types", 118)  # Default to all elements
        
        # Core components
        self.feature_extractor = AtomicFeatureExtractor(
            hidden_size=self.hidden_size,
            num_atom_types=self.num_atom_types
        )
        
        # Interaction layers
        self.interaction_layers = nn.ModuleList([
            AtomInteractionModule(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads
            ) for _ in range(self.num_layers)
        ])
        
        # Prediction heads
        self.velocity_predictor = nn.Linear(self.hidden_size, 3)
        self.energy_predictor = nn.Linear(self.hidden_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(
        self,
        atom_types: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Extract atomic features
        features = self.feature_extractor(atom_types, positions)
        
        # Process through interaction layers
        hidden_states = features
        intermediate_states = []
        
        for layer in self.interaction_layers:
            hidden_states = layer(hidden_states, attention_mask)
            intermediate_states.append(hidden_states)
            
        # Generate predictions
        velocities = self.velocity_predictor(hidden_states)
        energies = self.energy_predictor(hidden_states)
        
        return {
            "velocities": velocities,
            "energies": energies,
            "hidden_states": hidden_states,
            "intermediate_states": intermediate_states
        } 