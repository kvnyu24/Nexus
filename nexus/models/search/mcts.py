from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import math
from ...core.base import NexusModule
from .mcts_node import MCTSNode
from .mcts_config import MCTSConfig

class MCTS(NexusModule):
    def __init__(self, config: MCTSConfig):
        super().__init__(config)
        
        # Core parameters
        self.hidden_dim = config.hidden_dim
        self.num_actions = config.num_actions
        self.num_simulations = config.num_simulations
        self.c_puct = config.c_puct
        
        # Neural network components
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Tanh()
        )
        
        # Feature bank for state storage
        self.register_buffer(
            "state_bank",
            torch.zeros(config.bank_size, self.hidden_dim)
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
    def update_state_bank(self, states: torch.Tensor):
        """Update state bank following EnhancedReID pattern"""
        batch_size = states.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.state_bank.size(0):
            ptr = 0
            
        self.state_bank[ptr:ptr + batch_size] = states.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.state_bank.size(0)
        
    def select_action(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """Select action using PUCT algorithm"""
        best_score = float('-inf')
        best_action = -1
        best_child = None
        
        for action, child in node.children.items():
            score = child.value() + self.c_puct * child.prior * \
                   math.sqrt(node.visit_count) / (1 + child.visit_count)
                   
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
        
    def forward(
        self,
        state: torch.Tensor,
        legal_actions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode state
        encoded_state = self.state_encoder(state)
        
        # Update state bank
        self.update_state_bank(encoded_state)
        
        # Get policy and value predictions
        policy_logits = self.policy_head(encoded_state)
        value = self.value_head(encoded_state)
        
        # Mask illegal actions
        if legal_actions is not None:
            policy_logits = policy_logits.masked_fill(~legal_actions, float('-inf'))
            
        policy_probs = torch.softmax(policy_logits, dim=-1)
        
        return {
            "policy": policy_probs,
            "value": value,
            "encoded_state": encoded_state
        }
        
    def simulate(
        self,
        root_state: torch.Tensor,
        num_simulations: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        sims = num_simulations or self.num_simulations
        root = MCTSNode(root_state)
        
        for _ in range(sims):
            node = root
            search_path = [node]
            
            # Selection
            while node.expanded():
                action, node = self.select_action(node)
                search_path.append(node)
                
            # Expansion and evaluation
            outputs = self.forward(node.state)
            policy, value = outputs["policy"], outputs["value"]
            
            # Backpropagation
            for node in reversed(search_path):
                node.value_sum += value.item()
                node.visit_count += 1
                
        # Calculate final action probabilities
        visit_counts = torch.zeros(self.num_actions)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
            
        action_probs = visit_counts / visit_counts.sum()
        
        return {
            "action_probs": action_probs,
            "root_value": root.value(),
            "visit_counts": visit_counts
        } 