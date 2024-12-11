import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
from ...core.base import NexusModule
import numpy as np

class A2CNetwork(NexusModule):
    """Advanced A2C network with transformer-based architecture and latest RL enhancements"""
    def __init__(self, state_dim: Union[int, Dict[str, int]], action_dim: int, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_heads = config.get("num_heads", 4)
        self.dropout = config.get("dropout", 0.1)
        
        # Handle different input types
        if isinstance(state_dim, dict):
            self.vector_dim = state_dim.get("vector", 0)
            self.image_dim = state_dim.get("image", 0)
            input_dim = self.vector_dim + (self.image_dim if self.image_dim > 0 else 0)
        else:
            input_dim = state_dim
            
        # Advanced feature extractor with residual connections
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            ResidualBlock(self.hidden_dim),
            ResidualBlock(self.hidden_dim)
        )
        
        # Transformer-based processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Actor head with dueling architecture
        self.actor_advantage = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, action_dim)
        )
        self.actor_value = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Distributional critic head
        self.num_atoms = config.get("num_atoms", 51)
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_atoms)
        )
        
        # Auxiliary task heads
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, action_dim)
        )
        
        self.forward_dynamics = nn.Sequential(
            nn.Linear(self.hidden_dim + action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
    def forward(self, state: torch.Tensor, prev_state: Optional[torch.Tensor] = None,
                action: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.feature_net(state)
        features = self.transformer(features.unsqueeze(1)).squeeze(1)
        
        # Dueling actor architecture
        advantage = self.actor_advantage(features)
        value = self.actor_value(features)
        action_logits = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        # Distributional value prediction
        value_dist = self.critic(features)
        value = (F.softmax(value_dist, dim=-1) * torch.linspace(0, 1, self.num_atoms).to(value_dist.device)).sum(-1, keepdim=True)
        
        outputs = {
            "action_logits": action_logits,
            "value": value,
            "value_dist": value_dist,
            "features": features
        }
        
        # Auxiliary predictions if data available
        if prev_state is not None and action is not None:
            prev_features = self.feature_net(prev_state)
            prev_features = self.transformer(prev_features.unsqueeze(1)).squeeze(1)
            
            # Inverse dynamics prediction
            inv_features = torch.cat([prev_features, features], dim=-1)
            pred_action = self.inverse_dynamics(inv_features)
            
            # Forward dynamics prediction
            forward_input = torch.cat([prev_features, F.one_hot(action, num_classes=action_logits.size(-1)).float()], dim=-1)
            pred_next_features = self.forward_dynamics(forward_input)
            
            outputs.update({
                "pred_action": pred_action,
                "pred_next_features": pred_next_features
            })
            
        return outputs

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x))

class A2CAgent(NexusModule):
    """Advanced A2C agent with latest RL enhancements"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.aux_coef = config.get("aux_coef", 0.1)
        
        # Initialize network
        self.network = A2CNetwork(self.state_dim, self.action_dim, config)
        
        # Separate optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(
            list(self.network.feature_net.parameters()) +
            list(self.network.transformer.parameters()) +
            list(self.network.actor_advantage.parameters()) +
            list(self.network.actor_value.parameters()),
            lr=self.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.network.critic.parameters()) +
            list(self.network.inverse_dynamics.parameters()) +
            list(self.network.forward_dynamics.parameters()),
            lr=self.learning_rate
        )
        
    def select_action(self, state: np.ndarray, prev_state: Optional[np.ndarray] = None,
                     training: bool = True) -> Tuple[int, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            prev_state_tensor = torch.FloatTensor(prev_state).unsqueeze(0) if prev_state is not None else None
            
            outputs = self.network(state_tensor, prev_state_tensor)
            action_logits = outputs["action_logits"]
            action_probs = F.softmax(action_logits, dim=-1)
            
            if training:
                action = torch.multinomial(action_probs, 1).item()
            else:
                action = action_probs.argmax().item()
                
            outputs["action_log_prob"] = F.log_softmax(action_logits, dim=-1)[0, action]
            return action, outputs
            
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        states = batch["states"]
        prev_states = batch.get("prev_states")
        actions = batch["actions"]
        returns = batch["returns"]
        advantages = batch.get("advantages", None)
        
        # Get current predictions
        outputs = self.network(states, prev_states, actions)
        action_logits = outputs["action_logits"]
        values = outputs["value"]
        value_dist = outputs["value_dist"]
        
        # Calculate action probabilities
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Get log probs of taken actions
        action_log_probs = action_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate GAE if advantages not provided
        if advantages is None:
            advantages = returns - values.squeeze()
            if "next_values" in batch:
                next_values = batch["next_values"]
                deltas = batch["rewards"] + self.gamma * next_values - values.squeeze()
                advantages = self._compute_gae(deltas, batch["dones"])
        
        # Calculate losses
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values.squeeze(), returns)
        entropy = -(action_probs * action_log_probs).sum(dim=1).mean()
        
        # Auxiliary losses if data available
        aux_loss = 0
        if prev_states is not None:
            aux_loss = F.cross_entropy(outputs["pred_action"], actions)
            aux_loss += F.mse_loss(outputs["pred_next_features"], outputs["features"].detach())
        
        # Update actor
        actor_loss = policy_loss - self.entropy_coef * entropy
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        
        # Update critic
        critic_loss = self.value_coef * value_loss + self.aux_coef * aux_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "aux_loss": aux_loss.item(),
            "total_loss": (actor_loss + critic_loss).item()
        }
        
    def _compute_gae(self, deltas: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(deltas)
        gae = 0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages