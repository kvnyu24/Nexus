"""
AlphaZero-style Self-Play
Paper: "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
       (Silver et al., 2017)

AlphaZero combines:
- Monte Carlo Tree Search (MCTS) for planning
- Neural network for policy and value prediction
- Self-play for data generation
- No domain-specific knowledge beyond rules

Key components:
- Policy network π(a|s): predicts action probabilities
- Value network v(s): predicts game outcome
- MCTS with neural network guidance
- Self-play generates training data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional
from ...core.base import NexusModule
import numpy as np
import math


class AlphaZeroNetwork(nn.Module):
    """
    Combined policy and value network for AlphaZero.

    Uses a shared trunk with separate heads for policy and value.

    Args:
        state_dim: Dimension of state representation
        action_dim: Number of possible actions
        hidden_dim: Hidden layer size
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Value in [-1, 1]
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass predicting policy and value.

        Args:
            state: State representation [batch, state_dim]

        Returns:
            Tuple of (policy_logits, value)
        """
        features = self.trunk(state)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value


class MCTSNode:
    """
    Node in the MCTS tree.

    Stores visit counts, Q-values, and prior probabilities.
    """

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, MCTSNode] = {}

    def q_value(self) -> float:
        """Average Q-value."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visit_count: int, c_puct: float = 1.0) -> float:
        """UCB score for action selection."""
        u = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        return self.q_value() + u


class AlphaZeroAgent(NexusModule):
    """
    AlphaZero agent with MCTS and neural network.

    Combines neural network predictions with MCTS for action selection.
    Self-play generates training data for the network.

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state space
            - action_dim: Number of possible actions
            - hidden_dim: Hidden layer size (default: 256)
            - num_simulations: Number of MCTS simulations (default: 50)
            - c_puct: Exploration constant for UCB (default: 1.0)
            - temperature: Temperature for action selection (default: 1.0)
            - learning_rate: Learning rate (default: 1e-3)
            - value_loss_weight: Weight for value loss (default: 1.0)
            - l2_weight: L2 regularization weight (default: 1e-4)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_simulations = config.get("num_simulations", 50)
        self.c_puct = config.get("c_puct", 1.0)
        self.temperature = config.get("temperature", 1.0)
        self.value_loss_weight = config.get("value_loss_weight", 1.0)

        # Neural network
        self.network = AlphaZeroNetwork(
            self.state_dim, self.action_dim, self.hidden_dim
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("l2_weight", 1e-4),
        )

    def predict(self, state: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Network prediction for a single state.

        Args:
            state: State tensor

        Returns:
            Tuple of (policy_probs, value)
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)

            policy_logits, value = self.network(state)
            policy_probs = F.softmax(policy_logits, dim=-1).cpu().numpy()[0]
            value = value.item()

        return policy_probs, value

    def mcts_search(
        self, root_state: torch.Tensor, legal_actions: List[int]
    ) -> np.ndarray:
        """
        Run MCTS from root state.

        Args:
            root_state: Root state
            legal_actions: List of legal action indices

        Returns:
            Visit count distribution over actions
        """
        # Initialize root node
        policy_probs, _ = self.predict(root_state)
        root = MCTSNode(prior=0.0)

        # Initialize children for legal actions
        for action in legal_actions:
            root.children[action] = MCTSNode(prior=policy_probs[action])

        # Run simulations
        for _ in range(self.num_simulations):
            # Selection: traverse tree using UCB
            node = root
            search_path = [node]

            # In a real implementation, you would simulate the environment here
            # For this stub, we just do a single-step lookahead
            action = self._select_action(node, legal_actions)

            if action in node.children:
                node = node.children[action]
                search_path.append(node)

            # Expansion and evaluation
            _, value = self.predict(root_state)

            # Backpropagation
            for node in reversed(search_path):
                node.visit_count += 1
                node.value_sum += value
                value = -value  # Flip value for opponent

        # Return visit count distribution
        visit_counts = np.zeros(self.action_dim)
        for action in legal_actions:
            if action in root.children:
                visit_counts[action] = root.children[action].visit_count

        return visit_counts

    def _select_action(self, node: MCTSNode, legal_actions: List[int]) -> int:
        """Select action with highest UCB score."""
        best_score = -float('inf')
        best_action = legal_actions[0]

        for action in legal_actions:
            if action in node.children:
                score = node.children[action].ucb_score(node.visit_count, self.c_puct)
                if score > best_score:
                    best_score = score
                    best_action = action

        return best_action

    def select_action(
        self,
        state: torch.Tensor,
        legal_actions: List[int],
        temperature: Optional[float] = None,
    ) -> int:
        """
        Select action using MCTS with temperature-based sampling.

        Args:
            state: Current state
            legal_actions: List of legal actions
            temperature: Temperature for sampling (None = use default)

        Returns:
            Selected action index
        """
        if temperature is None:
            temperature = self.temperature

        # Run MCTS
        visit_counts = self.mcts_search(state, legal_actions)

        if temperature == 0:
            # Deterministic: select most visited
            return int(np.argmax(visit_counts))
        else:
            # Stochastic: sample proportional to visit_count^(1/temperature)
            visit_counts = visit_counts ** (1.0 / temperature)
            probs = visit_counts / visit_counts.sum()
            return np.random.choice(len(probs), p=probs)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update network using self-play data.

        Args:
            batch: Dictionary containing:
                - states: States [batch, state_dim]
                - policy_targets: Target policy distributions [batch, action_dim]
                - value_targets: Target values [batch]

        Returns:
            Dictionary with loss metrics
        """
        states = batch["states"]
        policy_targets = batch["policy_targets"]
        value_targets = batch["value_targets"]

        # Forward pass
        policy_logits, value_preds = self.network(states)

        # Policy loss (cross-entropy)
        policy_loss = -torch.sum(
            policy_targets * F.log_softmax(policy_logits, dim=-1), dim=-1
        ).mean()

        # Value loss (MSE)
        value_loss = F.mse_loss(value_preds.squeeze(-1), value_targets)

        # Total loss
        loss = policy_loss + self.value_loss_weight * value_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "mean_value_pred": value_preds.mean().item(),
        }

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: Input state

        Returns:
            Dictionary with policy_logits and value
        """
        policy_logits, value = self.network(state)

        return {
            "policy_logits": policy_logits,
            "policy_probs": F.softmax(policy_logits, dim=-1),
            "value": value,
        }
