import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from ...core.base import NexusModule
from .state_transition import StateTransitionModule
from .memory_bank import MemoryBank

class NeuralStateMachine(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config following EnhancedGNN pattern
        self._validate_config(config)
        
        # Core dimensions
        self.hidden_dim = config["hidden_dim"]
        self.num_states = config["num_states"]
        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        
        # State embeddings with orthogonal initialization (following SFT pattern)
        self.state_embeddings = nn.Parameter(
            torch.randn(self.num_states, self.hidden_dim)
        )
        nn.init.orthogonal_(self.state_embeddings, gain=1.0)
        
        # Input encoder (following EnhancedRL pattern)
        self.input_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # State transition with uncertainty (following PlanningModule pattern)
        self.transition = StateTransitionModule(
            hidden_dim=self.hidden_dim,
            num_states=self.num_states,
            num_heads=config.get("num_heads", 8)
        )
        
        # Enhanced memory bank
        self.memory_bank = MemoryBank(
            hidden_dim=self.hidden_dim,
            bank_size=config.get("bank_size", 10000)
        )
        
        # Output network (following DecisionMakingModule pattern)
        self.output_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # Uncertainty estimation (following PlanningModule pattern)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Safety assessment (following DecisionMakingModule pattern)
        self.safety_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights (following SFT pattern)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights following SFT pattern"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
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
        initial_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = inputs.size(0)
        sequence_length = inputs.size(1)
        
        # Initialize or validate initial state
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
        all_uncertainties = []
        all_safety_scores = []
        
        for t in range(sequence_length):
            # Get current state embedding
            state_embed = torch.matmul(state_logits, self.state_embeddings)
            
            # Encode current input
            input_embed = self.input_encoder(inputs[:, t])
            
            # Update memory bank (following EnhancedReID pattern)
            self.memory_bank.update(state_embed)
            
            # Generate outputs and assessments
            output = self.output_net(state_embed)
            uncertainty = self.uncertainty_head(state_embed)
            safety_score = self.safety_head(state_embed)
            
            # Store results
            all_outputs.append(output)
            all_uncertainties.append(uncertainty)
            all_safety_scores.append(safety_score)
            
            # Predict next state if not last timestep
            if t < sequence_length - 1:
                transition_outputs = self.transition(
                    state_embed=state_embed,
                    input_embed=input_embed,
                    memory_bank=self.memory_bank.get_bank(),
                    attention_mask=attention_mask
                )
                state_logits = F.softmax(
                    transition_outputs["logits"], 
                    dim=-1
                )
                all_states.append(state_logits)
        
        return {
            "outputs": torch.stack(all_outputs, dim=1),
            "states": torch.stack(all_states, dim=1),
            "uncertainties": torch.stack(all_uncertainties, dim=1),
            "safety_scores": torch.stack(all_safety_scores, dim=1),
            "final_state": state_logits,
            "memory_bank": self.memory_bank.get_bank()
        } 