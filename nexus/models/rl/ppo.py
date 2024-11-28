import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from ...core.base import NexusModule
import numpy as np

class ActorCritic(NexusModule):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(state)
        action_logits = self.policy(features)
        value = self.value(features)
        return action_logits, value

class PPOAgent(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 128)
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 0.2)  # PPO clipping parameter
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        
        # Initialize actor-critic network
        self.network = ActorCritic(self.state_dim, self.action_dim, self.hidden_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_logits, value = self.network(state_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
            
            if training:
                action = torch.multinomial(action_probs, 1).item()
            else:
                action = action_probs.argmax().item()
                
            return action, {
                "value": value,
                "action_log_prob": F.log_softmax(action_logits, dim=-1)[0, action]
            }
            
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        states = batch["states"]
        actions = batch["actions"]
        old_values = batch["values"]
        old_log_probs = batch["log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        
        # Get current predictions
        action_logits, values = self.network(states)
        new_log_probs = F.log_softmax(action_logits, dim=-1)
        new_log_probs = new_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate ratios and surrogate objectives
        ratios = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
        
        # Calculate losses
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values.squeeze(), returns)
        entropy = -(F.softmax(action_logits, dim=-1) * F.log_softmax(action_logits, dim=-1)).sum(dim=1).mean()
        
        # Combined loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item()
        } 