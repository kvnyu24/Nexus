from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from .modules import CrossAttentionLayer
from ..cv.vit import VisionTransformer as InternViT6B
from .processor import NVLMProcessor

class NVLMMixin:
    """Mixin class for adding NVLM capabilities to any NexusModule"""
    
    def init_nvlm(self, config: Dict[str, Any]) -> None:
        """Initialize NVLM components"""
        self.vision_encoder = InternViT6B(pretrained=True)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        self.hidden_size = config.get("hidden_size", 256)
        self.max_tiles = config.get("max_tiles", 7)
        
        # Downsampling and tile embeddings
        self.downsample = nn.Linear(self.vision_encoder.hidden_size * 4, self.hidden_size)
        self.tile_embeddings = nn.Embedding(self.max_tiles, self.hidden_size)
        
        # Architecture-specific components
        self.arch_type = config.get("architecture", "decoder")
        if self.arch_type == "cross":
            self._init_cross_attention(config)
        else:
            self._init_decoder_only(config)
    
    def _init_cross_attention(self, config: Dict[str, Any]) -> None:
        """Initialize cross-attention specific components"""
        self.cross_attention = nn.ModuleList([
            CrossAttentionLayer(config)
            for _ in range(config.get("num_cross_layers", 10))
        ])
    
    def _init_decoder_only(self, config: Dict[str, Any]) -> None:
        """Initialize decoder-only specific components"""
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        ) 