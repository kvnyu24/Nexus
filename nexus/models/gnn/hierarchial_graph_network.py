from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule
from .base_gnn import BaseGNNLayer
from .attention import GraphAttention
from .message_passing import AdaptiveMessagePassingLayer
from ...visualization.hierarchical import HierarchicalVisualizer
from torch_scatter import scatter_mean, scatter_max, scatter_add

class HierarchicalGraphNetwork(NexusModule):
    """
    A hierarchical graph neural network that combines multiple layer types and pooling strategies
    with feature banking and visualization capabilities.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate and store core configuration
        self._validate_config(config)
        self._init_core_dimensions(config)
        
        # Enhanced input processing
        self.input_processor = self._build_input_processor(config)
        
        # Hierarchical layer stack
        self.layer_stack = self._build_layer_stack(config)
        
        # Advanced pooling mechanisms
        self.global_pools = self._build_pooling_operators()
        self.pool_type = config.get("global_pool", "mean")
        self.adaptive_pool = config.get("adaptive_pool", False)
        if self.adaptive_pool:
            self.pool_mixer = nn.Parameter(torch.ones(4) / 4)
        
        # Feature transformation and output generation
        self.feature_processor = self._build_feature_processor(config)
        
        # Memory bank with configurable update strategy
        self._init_memory_bank(config)
        
        # Visualization and monitoring
        self.visualizer = HierarchicalVisualizer(config)
        self.track_gradients = config.get("track_gradients", False)
        
    def _init_core_dimensions(self, config: Dict[str, Any]) -> None:
        """Initialize core network dimensions with validation"""
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_layers = config.get("num_layers", 6)
        self.num_heads = config.get("num_heads", 8)
        
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("Hidden dimension must be divisible by number of heads")
            
    def _build_input_processor(self, config: Dict[str, Any]) -> nn.Module:
        """Construct enhanced input processing pipeline"""
        return nn.Sequential(
            nn.Linear(config.get("input_dim", self.hidden_dim), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim, eps=config.get("layer_norm_eps", 1e-5)),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.GELU()
        )
        
    def _build_layer_stack(self, config: Dict[str, Any]) -> nn.ModuleList:
        """Construct hierarchical layer stack with type-specific configurations"""
        layer_types = config.get("layer_types", ["adaptive"] * self.num_layers)
        layer_configs = config.get("layer_configs", [{}] * self.num_layers)
        
        layers = nn.ModuleList()
        for layer_type, layer_config in zip(layer_types, layer_configs):
            combined_config = {**config, **layer_config}
            
            if layer_type == "adaptive":
                layers.append(AdaptiveMessagePassingLayer(combined_config))
            elif layer_type == "attention":
                layers.append(GraphAttention(combined_config))
            elif layer_type == "base":
                layers.append(BaseGNNLayer(combined_config))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
                
        return layers
        
    def _build_pooling_operators(self) -> nn.ModuleDict:
        """Construct advanced pooling operators"""
        return nn.ModuleDict({
            "mean": lambda x, b: scatter_mean(x, b, dim=0),
            "max": lambda x, b: scatter_max(x, b, dim=0)[0],
            "sum": lambda x, b: scatter_add(x, b, dim=0),
            "attention": nn.MultiheadAttention(
                self.hidden_dim, 
                self.num_heads,
                dropout=0.1
            )
        })
        
    def _build_feature_processor(self, config: Dict[str, Any]) -> nn.Module:
        """Construct feature processing and output generation pipeline"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim * 2, config.get("output_dim", self.hidden_dim)),
            nn.LayerNorm(config.get("output_dim", self.hidden_dim))
        )
        
    def _init_memory_bank(self, config: Dict[str, Any]) -> None:
        """Initialize memory bank with configurable parameters"""
        self.bank_size = config.get("bank_size", 10000)
        self.register_buffer("feature_bank", torch.zeros(self.bank_size, self.hidden_dim))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        self.momentum = config.get("bank_momentum", 0.99)
        self.bank_temperature = config.get("bank_temperature", 0.07)
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters"""
        required = ["input_dim", "output_dim"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
                
    def update_feature_bank(self, features: torch.Tensor) -> None:
        """Update feature bank with momentum and temperature scaling"""
        batch_size = features.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.bank_size:
            ptr = 0
            
        # Temperature-scaled momentum update
        scaled_features = F.normalize(features, dim=-1) / self.bank_temperature
        self.feature_bank[ptr:ptr + batch_size] = (
            self.momentum * self.feature_bank[ptr:ptr + batch_size] +
            (1 - self.momentum) * scaled_features.detach()
        )
        self.bank_ptr[0] = (ptr + batch_size) % self.bank_size
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass with enhanced feature processing and hierarchical structure
        """
        # Initial feature processing
        h = self.input_processor(x)
        
        # Hierarchical layer processing
        attention_weights = []
        intermediate_features = []
        
        for i, layer in enumerate(self.layer_stack):
            outputs = layer(h, edge_index, edge_attr, batch)
            h = outputs["node_features"] + h if i > 0 else outputs["node_features"]
            
            if self.track_gradients:
                h.register_hook(lambda grad: self.visualizer.log_gradient(f"layer_{i}", grad))
                
            attention_weights.append(outputs.get("attention_weights", None))
            intermediate_features.append(h)
            
        # Advanced pooling with adaptive weights
        if batch is not None:
            if self.pool_type == "attention":
                h_reshaped = h.unsqueeze(0)
                global_features, attn_weights = self.global_pools["attention"](
                    h_reshaped, h_reshaped, h_reshaped
                )
                global_features = global_features.squeeze(0)
            elif self.adaptive_pool:
                pool_weights = F.softmax(self.pool_mixer, dim=0)
                global_features = sum(
                    w * pool_fn(h, batch)
                    for w, (_, pool_fn) in zip(pool_weights, self.global_pools.items())
                    if isinstance(pool_fn, type(lambda: None))
                )
            else:
                global_features = self.global_pools[self.pool_type](h, batch)
        else:
            global_features = h.mean(dim=0, keepdim=True)
            
        # Update memory bank
        self.update_feature_bank(global_features)
        
        # Generate final output
        output = self.feature_processor(global_features)
        
        result = {
            "output": output,
            "node_features": h,
            "global_features": global_features,
            "intermediate_features": intermediate_features,
            "feature_bank": self.feature_bank
        }
        
        if return_attention:
            result["attention_weights"] = attention_weights
            
        return result