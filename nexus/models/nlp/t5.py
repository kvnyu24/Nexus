import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule
from ...components.attention import FlashAttention

class T5EncoderBlock(NexusModule):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = FlashAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        norm_x = self.layer_norm1(x)
        x = x + self.attention(norm_x)
        norm_x = self.layer_norm2(x)
        x = x + self.feed_forward(norm_x)
        return x

class T5(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        
        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            T5EncoderBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                dropout=config.get("dropout", 0.1)
            ) for _ in range(self.num_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            T5EncoderBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                dropout=config.get("dropout", 0.1)
            ) for _ in range(self.num_layers)
        ])
        
        # Output
        self.output_proj = nn.Linear(self.hidden_size, self.vocab_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode
        encoder_states = self.token_embedding(input_ids)
        for layer in self.encoder_layers:
            encoder_states = layer(encoder_states)
            
        # Decode
        decoder_states = self.token_embedding(decoder_input_ids)
        for layer in self.decoder_layers:
            decoder_states = layer(decoder_states)
            
        # Generate logits
        logits = self.output_proj(decoder_states)
        
        return {
            "logits": logits,
            "encoder_states": encoder_states,
            "decoder_states": decoder_states
        }


# Alias for backwards compatibility
EnhancedT5 = T5
