from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...components.attention import CrossAttention
from ..cv.vit import VisionTransformer as InternViT6B
from .processor import NVLMProcessor

class NVLMMixin:
    """Mixin class for adding NVLM capabilities to any NexusModule"""
    
    def init_nvlm(self, config: Dict[str, Any]) -> None:
        """Initialize NVLM components"""
        # Vision encoder setup
        self.vision_encoder = InternViT6B(pretrained=True)
        self.freeze_vision = config.get("freeze_vision", True)
        if self.freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
                
        # Core dimensions
        self.hidden_size = config.get("hidden_size", 256)
        self.max_tiles = config.get("max_tiles", 7)
        self.num_heads = config.get("num_heads", 8)
        
        # Visual processing components
        self.downsample = nn.Sequential(
            nn.Linear(self.vision_encoder.hidden_size * 4, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        self.tile_embeddings = nn.Embedding(self.max_tiles, self.hidden_size)
        
        # Processor for handling visual features
        self.processor = NVLMProcessor()
        
        # Architecture setup
        self.arch_type = config.get("architecture", "cross")
        if self.arch_type == "cross":
            self._init_cross_attention(config)
        elif self.arch_type == "decoder":
            self._init_decoder_only(config)
        else:
            raise ValueError(f"Unknown architecture type: {self.arch_type}")
            
        # Optional components
        self.use_vision_pooling = config.get("use_vision_pooling", True)
        if self.use_vision_pooling:
            self.vision_pool = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 1)
            )
    
    def _init_cross_attention(self, config: Dict[str, Any]) -> None:
        """Initialize cross-attention specific components"""
        num_layers = config.get("num_cross_layers", 10)
        self.cross_attention = nn.ModuleList([
            CrossAttention(config)
            for _ in range(num_layers)
        ])
        self.cross_layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(config.get("dropout", 0.1))
    
    def _init_decoder_only(self, config: Dict[str, Any]) -> None:
        """Initialize decoder-only specific components"""
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.LayerNorm(self.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )