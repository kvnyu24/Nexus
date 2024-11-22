from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from base_rnn import BaseRNN

class LSTM(BaseRNN):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # LSTM specific layers
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
        
        # Initialize weights
        self.apply(self._init_weights)
        
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
            cell_state = hidden_state.clone()
        
        # Process through LSTM with attention mask handling
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            try:
                packed_embeddings = nn.utils.rnn.pack_padded_sequence(
                    embeddings,
                    lengths.cpu(),
                    batch_first=True,
                    enforce_sorted=False
                )
                packed_output, (hidden_state, cell_state) = self.lstm(
                    packed_embeddings,
                    (hidden_state, cell_state)
                )
                output, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_output,
                    batch_first=True,
                    padding_value=0.0
                )
            except Exception:
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