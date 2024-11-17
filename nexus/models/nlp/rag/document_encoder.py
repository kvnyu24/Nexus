from typing import Dict, Any
import torch
import torch.nn as nn
from ....core.base import NexusModule
from ....components.attention import FlashAttention

class DocumentEncoder(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_size = config["hidden_size"]
        self.num_layers = config.get("encoder_layers", 4)
        self.num_heads = config.get("num_heads", 8)
        self.intermediate_size = config.get("intermediate_size", 2048)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config["vocab_size"], self.hidden_size)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config["max_seq_length"], self.hidden_size)
        )
        
        # Efficient attention layers
        self.layers = nn.ModuleList([
            FlashAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                batch_size=1  # Optimized for document encoding
            ) for _ in range(self.num_layers)
        ])
        
        self.norm = nn.LayerNorm(self.hidden_size)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        embeddings = self.token_embedding(input_ids)
        embeddings = embeddings + self.position_embedding[:, :input_ids.size(1), :]
        
        # Apply transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        # Pool document embeddings
        document_embedding = self.norm(hidden_states.mean(dim=1))
        return document_embedding 