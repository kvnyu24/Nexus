from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from ...core.base import NexusModule
from .layers import EnhancedGNNLayer
from torch_scatter import scatter_mean, scatter_max

class EnhancedGNN(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config
        self._validate_config(config)
        
        # Core dimensions
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_layers = config.get("num_layers", 6)
        
        # Input projection
        self.input_proj = nn.Linear(
            config.get("input_dim", self.hidden_dim),
            self.hidden_dim
        )
        
        # GNN layers
        self.layers = nn.ModuleList([
            EnhancedGNNLayer(config)
            for _ in range(self.num_layers)
        ])
        
        # Global pooling
        self.global_pool = config.get("global_pool", "mean")
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim, config.get("output_dim", self.hidden_dim))
        )
        
        # Feature bank (following EnhancedReID pattern)
        self.register_buffer(
            "feature_bank",
            torch.zeros(
                config.get("bank_size", 10000),
                self.hidden_dim
            )
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = ["input_dim", "output_dim"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
    def update_feature_bank(self, features: torch.Tensor):
        """Update feature bank following EnhancedReID pattern"""
        batch_size = features.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.feature_bank.size(0):
            ptr = 0
            
        self.feature_bank[ptr:ptr + batch_size] = features.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.feature_bank.size(0)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Initial projection
        h = self.input_proj(x)
        
        # Process through GNN layers
        attention_weights = []
        for layer in self.layers:
            outputs = layer(h, edge_index, edge_attr, batch)
            h = outputs["node_features"]
            attention_weights.append(outputs["attention_weights"])
            
        # Global pooling
        if batch is not None:
            if self.global_pool == "mean":
                global_features = scatter_mean(h, batch, dim=0)
            elif self.global_pool == "max":
                global_features = scatter_max(h, batch, dim=0)[0]
            else:
                raise ValueError(f"Unknown pooling type: {self.global_pool}")
        else:
            global_features = h.mean(dim=0, keepdim=True)
            
        # Update feature bank
        self.update_feature_bank(global_features)
        
        # Final projection
        output = self.output_proj(global_features)
        
        return {
            "output": output,
            "node_features": h,
            "global_features": global_features,
            "attention_weights": attention_weights
        } 