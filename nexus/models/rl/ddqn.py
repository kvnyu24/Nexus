import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from ...core.base import NexusModule
import numpy as np

class DoubleDQNNetwork(NexusModule):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class DoubleDQNAgent(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 128)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.tau = config.get("tau", 0.005)  # Soft update parameter
        
        # Q-Networks
        self.online_network = DoubleDQNNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.target_network = DoubleDQNNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.target_network.load_state_dict(self.online_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.online_network(state_tensor)
            return q_values.argmax().item()
            
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        
        # Get next action using online network
        with torch.no_grad():
            next_actions = self.online_network(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))
        
        # Compute current Q values
        current_q = self.online_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {"loss": loss.item()}
        
    def _soft_update(self):
        """Soft update of target network parameters"""
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.online_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
