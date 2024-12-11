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
        self.max_agents = config.get("max_agents", 100)
        
        # Enhanced social attention with multiple layers
        self.social_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            ) for _ in range(3)
        ])
        
        # Layer normalization and residual connections
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(3)
        ])
        
        # Enhanced interaction MLP with skip connections
        self.interaction_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Interaction type and intensity prediction
        self.interaction_classifier = nn.Linear(self.hidden_dim, config["num_interaction_types"])
        self.interaction_intensity = nn.Linear(self.hidden_dim, 1)
        
        # Temporal context
        self.temporal_gru = nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            batch_first=True
        )
        
    def forward(
        self,
        agent_state: torch.Tensor,
        other_agents_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temporal_state: Optional[torch.Tensor] = None,
        interaction_history: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = agent_state.size(0)
        
        # Input validation
        if other_agents_states.size(-1) != self.hidden_dim:
            raise ValueError(f"Expected other_agents_states dim {self.hidden_dim}, got {other_agents_states.size(-1)}")
            
        # Multi-layer social attention with residual connections
        attended_states = agent_state.unsqueeze(0)
        attention_weights_list = []
        
        for layer_idx, (attention, norm) in enumerate(zip(self.social_attention_layers, self.layer_norms)):
            attended_output, attention_weights = attention(
                attended_states,
                other_agents_states.unsqueeze(0),
                other_agents_states.unsqueeze(0),
                key_padding_mask=attention_mask
            )
            attended_states = norm(attended_output + attended_states)
            attention_weights_list.append(attention_weights)
            
        # Process temporal context if provided
        if temporal_state is not None and interaction_history is not None:
            temporal_encoding, temporal_state = self.temporal_gru(interaction_history, temporal_state)
            temporal_context = temporal_encoding[:, -1]
        else:
            temporal_context = torch.zeros_like(agent_state)
            
        # Combine agent state with attended social context and temporal context
        combined = torch.cat([
            agent_state,
            attended_states.squeeze(0),
            temporal_context
        ], dim=-1)
        
        # Process through enhanced interaction MLP
        interaction_features = self.interaction_mlp(combined)
        
        # Predict interaction types and intensity
        interaction_logits = self.interaction_classifier(interaction_features)
        intensity_scores = torch.sigmoid(self.interaction_intensity(interaction_features))
        
        # Compute aggregated attention for visualization
        mean_attention = torch.stack(attention_weights_list).mean(0)
        
        return {
            "interaction_logits": interaction_logits,
            "interaction_features": interaction_features,
            "attention_weights": mean_attention,
            "intensity_scores": intensity_scores,
            "temporal_state": temporal_state,
            "layer_attention_weights": attention_weights_list
        }

class DialogueManager(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_dim = config["hidden_dim"]
        self.vocab_size = config["vocab_size"]
        self.max_seq_length = config.get("max_seq_length", 512)
        self.num_heads = config.get("num_heads", 8)
        
        # Enhanced dialogue encoder with bidirectional LSTM
        self.dialogue_encoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=3,
            dropout=config.get("dropout", 0.1),
            bidirectional=True,
            batch_first=True
        )
        
        # Context fusion layer
        self.context_fusion = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=config.get("dropout", 0.1)
        )
        
        # Enhanced response generator
        self.response_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=4 * self.hidden_dim,
                dropout=config.get("dropout", 0.1)
            ),
            num_layers=config.get("num_decoder_layers", 4),
            norm=nn.LayerNorm(self.hidden_dim)
        )
        
        # Output projection with tied embeddings
        self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size)
        
        # Emotion and style classifiers
        self.emotion_classifier = nn.Linear(self.hidden_dim, config.get("num_emotions", 8))
        self.style_classifier = nn.Linear(self.hidden_dim, config.get("num_styles", 4))
        
    def forward(
        self,
        dialogue_history: torch.Tensor,
        agent_state: torch.Tensor,
        social_context: Optional[torch.Tensor] = None,
        response_prefix: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        style_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Input validation
        if dialogue_history.size(-1) != self.hidden_dim:
            raise ValueError(f"Expected dialogue_history dim {self.hidden_dim}, got {dialogue_history.size(-1)}")
            
        # Encode dialogue history
        dialogue_encoding, (h_n, c_n) = self.dialogue_encoder(dialogue_history)
        
        # Combine bidirectional states
        dialogue_encoding = dialogue_encoding.view(dialogue_encoding.size(0), dialogue_encoding.size(1), 2, -1)
        dialogue_encoding = dialogue_encoding.sum(dim=2)
        
        # Fuse with agent state and social context
        if social_context is not None:
            context = torch.cat([agent_state, social_context], dim=-1)
            context_encoding, _ = self.context_fusion(
                dialogue_encoding.transpose(0, 1),
                context.unsqueeze(0),
                context.unsqueeze(0)
            )
            dialogue_encoding = context_encoding.transpose(0, 1)
            
        # Generate response with enhanced decoder
        if response_prefix is not None:
            decoder_output = self.response_decoder(
                response_prefix,
                dialogue_encoding,
                tgt_mask=self.generate_square_subsequent_mask(response_prefix.size(1)).to(response_prefix.device)
            )
        else:
            decoder_output = dialogue_encoding
            
        # Project to vocabulary
        logits = self.output_proj(decoder_output)
        
        # Emotion and style classification
        emotion_logits = self.emotion_classifier(decoder_output.mean(dim=1))
        style_logits = self.style_classifier(decoder_output.mean(dim=1))
        
        return {
            "logits": logits,
            "dialogue_encoding": dialogue_encoding,
            "hidden_state": h_n,
            "emotion_logits": emotion_logits,
            "style_logits": style_logits,
            "cell_state": c_n
        }
        
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        return mask