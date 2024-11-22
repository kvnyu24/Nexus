import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from ...core.base import NexusModule

class EnhancedLSTM(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model configuration
        self.hidden_size = config["hidden_size"]
        self.num_layers = config.get("num_layers", 2)
        self.vocab_size = config["vocab_size"]
        self.dropout = config.get("dropout", 0.1)
        self.bidirectional = config.get("bidirectional", False)
        
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 and 1")
            
        # Token embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            padding_idx=0  # Add padding token support
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # Layer normalization
        output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.layer_norm = nn.LayerNorm(output_size)
        
        # Output projection
        self.output = nn.Linear(
            output_size,
            self.vocab_size,
            bias=False
        )
        
        # Tie weights between embedding and output layer
        self.output.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            module.weight.data[0].zero_()  # Initialize padding embedding to zeros
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            
    def _validate_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        hidden_state: Optional[torch.Tensor]
    ) -> None:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2-dimensional (batch_size, seq_len)")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have same shape as input_ids")
        if hidden_state is not None:
            expected_shape = (
                self.num_layers * (2 if self.bidirectional else 1),
                input_ids.size(0),
                self.hidden_size
            )
            if hidden_state.shape != expected_shape:
                raise ValueError(f"hidden_state must have shape {expected_shape}")
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Validate inputs
        self._validate_input(input_ids, attention_mask, hidden_state)
        
        # Get embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Initialize LSTM states if not provided
        if hidden_state is None:
            device = input_ids.device
            batch_size = input_ids.size(0)
            num_directions = 2 if self.bidirectional else 1
            hidden_state = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
                device=device
            )
            cell_state = torch.zeros_like(hidden_state)
        else:
            # Split hidden_state into (h0, c0) if provided
            hidden_state, cell_state = hidden_state, hidden_state.clone()
        
        # Pack padded sequence if attention mask is provided
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).clamp(min=1)  # Prevent 0 lengths
            try:
                packed_embeddings = nn.utils.rnn.pack_padded_sequence(
                    embeddings,
                    lengths.cpu(),
                    batch_first=True,
                    enforce_sorted=False
                )
                
                # Process through LSTM
                packed_output, (hidden_state, cell_state) = self.lstm(
                    packed_embeddings,
                    (hidden_state, cell_state)
                )
                
                # Unpack sequence
                output, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_output,
                    batch_first=True,
                    padding_value=0.0
                )
            except Exception:
                # Fallback if packing fails
                output, (hidden_state, cell_state) = self.lstm(
                    embeddings,
                    (hidden_state, cell_state)
                )
        else:
            output, (hidden_state, cell_state) = self.lstm(
                embeddings,
                (hidden_state, cell_state)
            )
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        # Generate logits
        logits = self.output(output)
        
        return {
            "logits": logits,
            "hidden_states": output,
            "last_hidden_state": hidden_state,
            "cell_state": cell_state
        }
