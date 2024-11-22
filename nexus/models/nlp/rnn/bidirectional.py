import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .base_rnn import BaseRNN

class BidirectionalRNN(BaseRNN):
    def __init__(self, config: Dict[str, Any]):
        # Force bidirectional for this implementation
        config["bidirectional"] = True
        super().__init__(config)
        
        # RNN layer
        rnn_type = config.get("rnn_type", "lstm").lower()
        rnn_class = {
            "lstm": nn.LSTM,
            "gru": nn.GRU,
            "rnn": nn.RNN
        }.get(rnn_type)
        
        if rnn_class is None:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
            
        self.rnn = rnn_class(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Merge bidirectional states
        self.merge_mode = config.get("merge_mode", "concat")
        if self.merge_mode not in ["concat", "sum", "mean", "weighted"]:
            raise ValueError(f"Unsupported merge mode: {self.merge_mode}")
            
        merged_size = self.hidden_size * 2 if self.merge_mode == "concat" else self.hidden_size
        
        # Add layer normalization
        self.layer_norm = nn.LayerNorm(merged_size)
        
        # Add weighted merge option
        if self.merge_mode == "weighted":
            self.direction_weights = nn.Parameter(torch.ones(2))
            
        # Output projection with activation
        self.output_proj = nn.Sequential(
            nn.Linear(merged_size, merged_size),
            nn.ReLU(),
            nn.Linear(merged_size, self.output_size)
        )
        
    def _merge_directions(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Split directions
        forward, backward = torch.chunk(hidden_states, 2, dim=-1)
        
        if self.merge_mode == "concat":
            return torch.cat([forward, backward], dim=-1)
        elif self.merge_mode == "sum":
            return forward + backward
        elif self.merge_mode == "weighted":
            weights = torch.softmax(self.direction_weights, dim=0)
            return weights[0] * forward + weights[1] * backward
        else:  # mean
            return (forward + backward) / 2
            
    def forward(
        self,
        input_sequence: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = input_sequence.size(0)
        
        # Handle attention mask if provided
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            input_sequence = nn.utils.rnn.pack_padded_sequence(
                input_sequence, lengths, batch_first=True, enforce_sorted=False
            )
            
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)
            
        # Process through RNN
        output, hidden_state = self.rnn(input_sequence, hidden_state)
        
        # Unpack if using attention mask
        if attention_mask is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True
            )
        
        # Merge bidirectional states
        merged_output = self._merge_directions(output)
        
        # Apply layer normalization
        normalized_output = self.layer_norm(merged_output)
        
        # Project to output size
        final_output = self.output_proj(normalized_output)
        
        return {
            "output": final_output,
            "hidden_state": hidden_state,
            "merged_states": merged_output,
            "normalized_states": normalized_output
        }