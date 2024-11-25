import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule
from .state_transition import StateTransitionModule
from .memory_bank import MemoryBank

class NeuralStateMachine(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config
        self._validate_config(config)
        
        # Core dimensions
        self.hidden_dim = config["hidden_dim"]
        self.num_states = config["num_states"]
        self.input_dim = config["input_dim"]
        
        # State embeddings
        self.state_embeddings = nn.Parameter(
            torch.randn(self.num_states, self.hidden_dim)
        )
        nn.init.orthogonal_(self.state_embeddings)
        
        # Input encoder (following EnhancedRL pattern)
        self.input_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # State transition module
        self.transition = StateTransitionModule(
            hidden_dim=self.hidden_dim,
            num_states=self.num_states
        )
        
        # Memory bank for state tracking
        self.memory_bank = MemoryBank(
            hidden_dim=self.hidden_dim,
            bank_size=config.get("bank_size", 10000)
        )
        
        # Output network (following EnhancedRL pattern)
        self.output_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, config["output_dim"])
        )
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = [
            "hidden_dim",
            "num_states",
            "input_dim",
            "output_dim"
        ]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
    
    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = inputs.size(0)
        
        # Initialize state if not provided
        if initial_state is None:
            state_logits = torch.zeros(
                batch_size, self.num_states,
                device=inputs.device
            )
            state_logits[:, 0] = 1.0  # Start at first state
        else:
            state_logits = initial_state
            
        # Process sequence
        all_states = []
        all_outputs = []
        
        for t in range(inputs.size(1)):
            # Get current state embedding
            state_embed = torch.matmul(
                state_logits,
                self.state_embeddings
            )
            
            # Encode input
            input_embed = self.input_encoder(inputs[:, t])
            
            # Update memory bank
            self.memory_bank.update(state_embed)
            
            # Generate output
            output = self.output_net(state_embed)
            all_outputs.append(output)
            
            # Predict next state if not last timestep
            if t < inputs.size(1) - 1:
                next_state_logits = self.transition(
                    state_embed=state_embed,
                    input_embed=input_embed,
                    memory_bank=self.memory_bank.get_bank()
                )
                state_logits = F.softmax(next_state_logits, dim=-1)
                all_states.append(state_logits)
        
        return {
            "outputs": torch.stack(all_outputs, dim=1),
            "states": torch.stack(all_states, dim=1),
            "final_state": state_logits,
            "memory_bank": self.memory_bank.get_bank()
        } 