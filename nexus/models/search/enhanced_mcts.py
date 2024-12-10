from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from ...core.base import NexusModule
from .mcts_config import MCTSConfig
from .mcts_node import EnhancedMCTSNode

class EnhancedMCTS(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Convert dict config to MCTSConfig
        self.mcts_config = MCTSConfig(**config)
        
        # Neural network components
        self.state_encoder = nn.Sequential(
            nn.Linear(self.mcts_config.state_dim, self.mcts_config.hidden_dim),
            nn.LayerNorm(self.mcts_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mcts_config.hidden_dim, self.mcts_config.hidden_dim)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(self.mcts_config.hidden_dim, self.mcts_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mcts_config.hidden_dim, self.mcts_config.num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(self.mcts_config.hidden_dim, self.mcts_config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.mcts_config.hidden_dim // 2, 1),
            nn.Tanh()
        )
        
        self.uncertainty_head = nn.Linear(self.mcts_config.hidden_dim, self.mcts_config.num_actions)
        
        # Feature bank following EnhancedReID pattern
        self.register_buffer(
            "state_bank",
            torch.zeros(self.mcts_config.bank_size, self.mcts_config.hidden_dim)
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
    def update_state_bank(self, states: torch.Tensor):
        batch_size = states.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.state_bank.size(0):
            ptr = 0
            
        self.state_bank[ptr:ptr + batch_size] = states.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.state_bank.size(0) 