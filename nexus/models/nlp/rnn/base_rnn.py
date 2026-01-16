from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from ....core.base import NexusModule
from ....core.initialization import WeightInitMixin

class BaseRNN(WeightInitMixin, NexusModule):
    """
    Base RNN class that implements common functionality for RNN-based models.
    Serves as a foundation for LSTM, GRU, and other RNN variants.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core configuration
        self.hidden_size = config["hidden_size"]
        self.num_layers = config.get("num_layers", 2)
        self.vocab_size = config["vocab_size"]
        self.dropout = config.get("dropout", 0.1)
        self.bidirectional = config.get("bidirectional", False)
        
        # Validate configuration
        self._validate_config()
        
        # Common components
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            padding_idx=0
        )
        
        # Output size accounting for bidirectional
        self.output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_size)
        
        # Output projection
        self.output = nn.Linear(self.output_size, self.vocab_size, bias=False)
        
        # Tie weights between embedding and output layer
        self.output.weight = self.token_embedding.weight

        # Initialize weights using WeightInitMixin (llm preset uses normal distribution with std=0.02)
        self.apply_weight_init(preset='llm')

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 and 1")

    def _validate_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        hidden_state: Optional[torch.Tensor]
    ) -> None:
        """Validate input tensors."""
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
                
    def _init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden states."""
        num_directions = 2 if self.bidirectional else 1
        return torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Base forward pass implementation with common RNN processing logic.
        
        Args:
            input_ids: Input token ids of shape (batch_size, seq_len)
            attention_mask: Optional mask of shape (batch_size, seq_len)
            hidden_state: Optional initial hidden state
            
        Returns:
            Dictionary containing model outputs
        """
        # Validate inputs
        self._validate_input(input_ids, attention_mask, hidden_state)
        
        # Get embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(input_ids.size(0), input_ids.device)
        
        # Handle attention mask and prepare packed sequence if needed
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            try:
                packed_embeddings = nn.utils.rnn.pack_padded_sequence(
                    embeddings,
                    lengths.cpu(),
                    batch_first=True,
                    enforce_sorted=False
                )
                # Process through RNN (to be implemented by subclass)
                packed_output, final_state = self._forward_rnn(
                    packed_embeddings, 
                    hidden_state
                )
                # Unpack sequence
                output, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_output,
                    batch_first=True,
                    padding_value=0.0
                )
            except Exception:
                # Fallback if packing fails
                output, final_state = self._forward_rnn(embeddings, hidden_state)
        else:
            # Process without packing
            output, final_state = self._forward_rnn(embeddings, hidden_state)
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        # Generate logits
        logits = self.output(output)
        
        return {
            "logits": logits,
            "hidden_states": output,
            "last_hidden_state": final_state
        }

    def _forward_rnn(
        self,
        embeddings: torch.Tensor,
        hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Abstract method for RNN-specific forward pass.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("_forward_rnn must be implemented by subclass")
        
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> torch.Tensor:
        """
        Generate sequence using the RNN model.
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Initialize sequence with input_ids
        generated = input_ids
        
        # Initialize hidden state
        hidden = self._init_hidden(batch_size, device)
        
        # Generate tokens
        for _ in range(max_length - input_ids.size(1)):
            # Get model output
            outputs = self.forward(generated, hidden_state=hidden)
            logits = outputs["logits"][:, -1, :] / temperature
            
            # Sample or greedy select next token
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
            # Append new token
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update hidden state
            hidden = outputs["last_hidden_state"]
            
        return generated