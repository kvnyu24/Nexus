import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from .base_rnn import BaseRNN

class GRU(BaseRNN):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # Output projection
        output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.output_proj = nn.Linear(output_size, self.output_size)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(config.get("dropout", 0.1))
        
        # Optional residual connection
        self.use_residual = config.get("use_residual", True)
        if self.use_residual:
            self.residual_proj = nn.Linear(self.input_size, self.output_size)
            
        # Activation
        self.activation = nn.GELU()
        
    def forward(
        self,
        input_sequence: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = input_sequence.size(0)
        
        # Store original input for residual
        residual = input_sequence if self.use_residual else None
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)
            
        # Apply input dropout
        input_sequence = self.dropout(input_sequence)
            
        # Handle attention mask if provided
        if attention_mask is not None:
            # Ensure mask is boolean
            attention_mask = attention_mask.bool()
            lengths = attention_mask.sum(dim=1).cpu()
            input_sequence = nn.utils.rnn.pack_padded_sequence(
                input_sequence, lengths, batch_first=True, enforce_sorted=False
            )
        
        # Process through GRU
        output, hidden_state = self.gru(input_sequence, hidden_state)
        
        # Unpack if using attention mask
        if attention_mask is not None:
            output, lengths = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True
            )
            # Apply attention mask
            output = output * attention_mask.unsqueeze(-1)
        
        # Apply layer norm and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        # Project and activate
        output = self.output_proj(output)
        output = self.activation(output)
        
        # Add residual connection if enabled
        if self.use_residual:
            residual = self.residual_proj(residual)
            output = output + residual
        
        return {
            "output": output,
            "hidden_state": hidden_state,
            "attention_mask": attention_mask
        }