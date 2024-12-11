from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule
from .mcts_node import MCTSNode
from .mcts import EnhancedMCTS
from .mcts_config import MCTSConfig

class MCTSTrainer(NexusModule):
    def __init__(self, config: MCTSConfig):
        super().__init__(config)
        
        # Core components
        self.mcts = EnhancedMCTS(config)
        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.KLDivLoss(reduction='batchmean')
        
        # Training parameters from config
        self.num_simulations = config.num_simulations
        self.temperature = config.init_temp
        self.value_weight = config.value_weight
        self.policy_weight = config.policy_weight
        self.entropy_weight = config.entropy_weight
        self.consistency_weight = config.consistency_weight
        
        # Prioritized experience replay
        self.register_buffer(
            "replay_states",
            torch.zeros(config.bank_size, config.state_dim)
        )
        self.register_buffer(
            "replay_values",
            torch.zeros(config.bank_size)
        )
        self.register_buffer(
            "replay_priorities",
            torch.ones(config.bank_size)
        )
        self.register_buffer("replay_ptr", torch.zeros(1, dtype=torch.long))
        
        self.alpha = config.prioritized_replay_alpha
        self.beta = config.prioritized_replay_beta
        self.min_replay_size = config.min_replay_size
        
    def update_replay_buffer(
        self, 
        states: torch.Tensor, 
        values: torch.Tensor,
        priorities: torch.Tensor
    ):
        batch_size = states.size(0)
        ptr = int(self.replay_ptr)
        
        if ptr + batch_size > self.replay_states.size(0):
            ptr = 0
            
        self.replay_states[ptr:ptr + batch_size] = states.detach()
        self.replay_values[ptr:ptr + batch_size] = values.detach()
        self.replay_priorities[ptr:ptr + batch_size] = priorities.detach()
        self.replay_ptr[0] = (ptr + batch_size) % self.replay_states.size(0)
        
    def sample_replay_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.replay_ptr[0] < self.min_replay_size:
            return None, None, None
            
        # Sample based on priorities
        probs = self.replay_priorities[:self.replay_ptr[0]] ** self.alpha
        probs = probs / probs.sum()
        
        indices = torch.multinomial(probs, batch_size, replacement=True)
        weights = (self.replay_ptr[0] * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        return (
            self.replay_states[indices],
            self.replay_values[indices],
            weights
        )
        
    def train_step(
        self,
        states: torch.Tensor,
        expert_policy: Optional[torch.Tensor] = None,
        expert_value: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Create root nodes for each state in batch
        root_nodes = [MCTSNode(state) for state in states]
        
        # Run MCTS simulations with noise for exploration
        mcts_outputs = self.mcts.simulate(
            root_nodes,
            self.num_simulations,
            add_exploration_noise=self.training
        )
        
        # Calculate losses
        losses = {}
        metrics = {}
        
        # Value loss with uncertainty
        if expert_value is not None:
            value_pred = torch.tensor([node.value() for node in root_nodes])
            value_uncertainty = mcts_outputs.get("value_uncertainty", 0.0)
            
            value_loss = self.value_loss(value_pred, expert_value)
            value_loss = value_loss * torch.exp(-value_uncertainty)  # Reduce loss for uncertain predictions
            
            losses["value_loss"] = value_loss * self.value_weight
            metrics["value_uncertainty"] = value_uncertainty.mean()
            
        # Policy loss with temperature annealing
        if expert_policy is not None:
            visit_counts = torch.tensor([[child.visit_count for child in node.children.values()] 
                                       for node in root_nodes])
            action_probs = visit_counts / visit_counts.sum(dim=-1, keepdim=True)
            action_logits = action_probs / self.temperature
            policy_pred = F.log_softmax(action_logits, dim=-1)
            
            policy_loss = self.policy_loss(policy_pred, expert_policy)
            losses["policy_loss"] = policy_loss * self.policy_weight
            
            # Add entropy bonus for exploration
            policy_entropy = -(F.softmax(action_logits, dim=-1) * policy_pred).sum(-1).mean()
            losses["entropy_loss"] = -policy_entropy * self.entropy_weight
            metrics["policy_entropy"] = policy_entropy
            
        # Value-policy consistency loss
        if expert_value is not None and expert_policy is not None:
            child_values = torch.tensor([[child.value() for child in node.children.values()]
                                       for node in root_nodes])
            consistency_loss = F.mse_loss(
                value_pred,
                (expert_policy * child_values).sum(-1)
            )
            losses["consistency_loss"] = consistency_loss * self.consistency_weight
            
        # Update replay buffer with priorities
        priorities = torch.abs(value_pred - expert_value) if expert_value is not None else torch.ones_like(value_pred)
        self.update_replay_buffer(states, value_pred, priorities)
        
        # Add replay batch if available
        replay_states, replay_values, importance_weights = self.sample_replay_batch(states.size(0))
        if replay_states is not None:
            replay_nodes = [MCTSNode(state) for state in replay_states]
            replay_outputs = self.mcts.simulate(replay_nodes, self.num_simulations // 2)
            replay_pred = torch.tensor([node.value() for node in replay_nodes])
            replay_loss = self.value_loss(replay_pred, replay_values)
            replay_loss = (replay_loss * importance_weights).mean()
            losses["replay_loss"] = replay_loss * self.value_weight
            
        metrics["value"] = value_pred.mean()
        
        return {
            "losses": losses,
            "metrics": metrics
        }