import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule

class DETR(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model dimensions
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_classes = config.get("num_classes", 91)
        self.num_queries = config.get("num_queries", 100)
        
        # Backbone
        self.backbone = self._build_backbone(config)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=self.hidden_dim,
            nhead=config.get("num_heads", 8),
            num_encoder_layers=config.get("num_encoder_layers", 6),
            num_decoder_layers=config.get("num_decoder_layers", 6),
            dim_feedforward=config.get("dim_feedforward", 2048),
            dropout=config.get("dropout", 0.1)
        )
        
        # Object queries
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        
        # Output heads
        self.class_head = nn.Linear(self.hidden_dim, self.num_classes + 1)  # +1 for no-object
        self.bbox_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 4)  # (cx, cy, w, h)
        )
        
    def forward(
        self,
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.backbone(images)
        
        # Prepare inputs for transformer
        src = self.input_proj(features)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, src.shape[1], 1)
        
        # Transformer forward pass
        memory = self.transformer.encoder(src)
        hs = self.transformer.decoder(query_embed, memory)
        
        # Predict classes and boxes
        outputs_class = self.class_head(hs)
        outputs_coord = self.bbox_head(hs).sigmoid()
        
        return {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "aux_outputs": self._get_aux_outputs(outputs_class, outputs_coord)
        } 