import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from nexus.core.base import NexusModule

class NVLMProcessor(NexusModule):
    """Helper class for processing NVLM inputs and outputs"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.tile_size = config.get("tile_size", 224)
        self.overlap = config.get("overlap", 32)
        self.min_tiles = config.get("min_tiles", 4)
        self.max_tiles = config.get("max_tiles", 7)
        
    def process_visual_features(
        self,
        images: torch.Tensor,
        vision_encoder: NexusModule,
        downsample: NexusModule,
        tile_embeddings: nn.Embedding,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process images through vision encoder and prepare features
        
        Args:
            images: Input images of shape (B, C, H, W)
            vision_encoder: Vision encoder module
            downsample: Feature dimensionality reduction module
            tile_embeddings: Learnable tile position embeddings
            attention_mask: Optional attention mask for tiles
            
        Returns:
            Tuple of:
                - Processed visual features (B, num_tiles, hidden_dim)
                - Attention mask for tiles (B, num_tiles)
        """
        # Extract tiles with dynamic tiling
        tiles, num_tiles = self._extract_tiles(images)
        B = tiles.shape[0]
        
        # Process tiles in chunks to handle memory
        chunk_size = 32
        visual_features = []
        
        for i in range(0, num_tiles, chunk_size):
            chunk = tiles[:, i:i+chunk_size].reshape(-1, *tiles.shape[2:])
            
            # Get features through vision encoder
            with torch.cuda.amp.autocast(enabled=True):
                chunk_features = vision_encoder(chunk)
                chunk_features = downsample(chunk_features)
            
            # Reshape back to batch
            chunk_features = chunk_features.reshape(B, -1, chunk_features.shape[-1])
            visual_features.append(chunk_features)
            
        visual_features = torch.cat(visual_features, dim=1)
        
        # Add learned tile position embeddings
        positions = torch.arange(num_tiles, device=tiles.device)
        pos_embeddings = tile_embeddings(positions)
        visual_features = visual_features + pos_embeddings.unsqueeze(0)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((B, num_tiles), device=tiles.device)
            
        return visual_features, attention_mask
    
    def _extract_tiles(self, image: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Extract tiles from high resolution image using dynamic tiling
        
        Args:
            image: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of:
                - Extracted tiles tensor (B, num_tiles, C, tile_size, tile_size) 
                - Number of tiles
        """
        B, C, H, W = image.shape
        
        # Calculate optimal number of tiles
        num_tiles_h = max(self.min_tiles, min(self.max_tiles, H // (self.tile_size - self.overlap)))
        num_tiles_w = max(self.min_tiles, min(self.max_tiles, W // (self.tile_size - self.overlap)))
        
        stride_h = (H - self.tile_size) // (num_tiles_h - 1) if num_tiles_h > 1 else 0
        stride_w = (W - self.tile_size) // (num_tiles_w - 1) if num_tiles_w > 1 else 0
        
        tiles = F.unfold(
            image,
            kernel_size=(self.tile_size, self.tile_size),
            stride=(stride_h, stride_w)
        )
        
        tiles = tiles.reshape(B, C, self.tile_size, self.tile_size, -1)
        tiles = tiles.permute(0, 4, 1, 2, 3)
        
        return tiles, tiles.shape[1]