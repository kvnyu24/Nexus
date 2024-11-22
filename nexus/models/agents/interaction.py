import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from ...core.base import NexusModule

class InteractionModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_dim = config["hidden_dim"]
        self.num_heads = config.get("num_heads", 8)
        self.dropout = config.get("dropout", 0.1)
        
        # Social attention mechanism
        self.social_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Interaction MLP
        self.interaction_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Interaction type prediction
        self.interaction_classifier = nn.Linear(self.hidden_dim, config["num_interaction_types"])
        
    def forward(
        self,
        agent_state: torch.Tensor,
        other_agents_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Compute social attention
        attended_states, attention_weights = self.social_attention(
            agent_state.unsqueeze(0),
            other_agents_states.unsqueeze(0),
            other_agents_states.unsqueeze(0),
            key_padding_mask=attention_mask
        )
        
        # Combine agent state with attended social context
        combined = torch.cat([
            agent_state,
            attended_states.squeeze(0)
        ], dim=-1)
        
        # Process through interaction MLP
        interaction_features = self.interaction_mlp(combined)
        
        # Predict interaction types
        interaction_logits = self.interaction_classifier(interaction_features)
        
        return {
            "interaction_logits": interaction_logits,
            "interaction_features": interaction_features,
            "attention_weights": attention_weights
        }

class DialogueManager(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_dim = config["hidden_dim"]
        self.vocab_size = config["vocab_size"]
        self.max_seq_length = config.get("max_seq_length", 512)
        
        # Dialogue state encoder
        self.dialogue_encoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            dropout=config.get("dropout", 0.1),
            batch_first=True
        )
        
        # Response generator
        self.response_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=config.get("num_heads", 8)
            ),
            num_layers=config.get("num_decoder_layers", 3)
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size)
        
    def forward(
        self,
        dialogue_history: torch.Tensor,
        agent_state: torch.Tensor,
        social_context: Optional[torch.Tensor] = None,
        response_prefix: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode dialogue history
        dialogue_encoding, (h_n, c_n) = self.dialogue_encoder(dialogue_history)
        
        # Combine with agent state and social context
        if social_context is not None:
            context = torch.cat([agent_state, social_context], dim=-1)
        else:
            context = agent_state
            
        # Generate response
        if response_prefix is not None:
            decoder_output = self.response_decoder(
                response_prefix,
                dialogue_encoding,
                tgt_mask=self.generate_square_subsequent_mask(response_prefix.size(1))
            )
        else:
            decoder_output = dialogue_encoding
            
        # Project to vocabulary
        logits = self.output_proj(decoder_output)
        
        return {
            "logits": logits,
            "dialogue_encoding": dialogue_encoding,
            "hidden_state": h_n
        }
        
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        return mask 