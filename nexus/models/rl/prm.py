import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from ...core.base import NexusModule

class PRMNode:
    def __init__(self, state: np.ndarray):
        self.state = state
        self.neighbors: List[Tuple[int, float]] = []  # (node_idx, cost)

class PRMAgent(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # PRM specific parameters
        self.state_dim = config["state_dim"]
        self.num_samples = config.get("num_samples", 1000)
        self.max_neighbors = config.get("max_neighbors", 10)
        self.connection_radius = config.get("connection_radius", 0.5)
        
        # Neural components for state encoding
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Value estimation network
        self.value_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize roadmap
        self.nodes: List[PRMNode] = []
        self.optimizer = torch.optim.Adam(
            list(self.state_encoder.parameters()) + 
            list(self.value_net.parameters()),
            lr=config.get("learning_rate", 1e-3)
        )

    def sample_nodes(self, state_bounds: np.ndarray) -> None:
        """Sample nodes for the roadmap"""
        for _ in range(self.num_samples):
            # Sample random state within bounds
            state = np.random.uniform(
                state_bounds[:, 0],
                state_bounds[:, 1]
            )
            self.nodes.append(PRMNode(state))
        
        # Connect nodes
        self._connect_nodes()

    def _connect_nodes(self) -> None:
        """Connect nodes based on distance"""
        for i, node in enumerate(self.nodes):
            distances = [
                (j, np.linalg.norm(node.state - other.state))
                for j, other in enumerate(self.nodes)
                if i != j
            ]
            
            # Select nearest neighbors within radius
            valid_neighbors = [
                (j, dist) for j, dist in distances
                if dist <= self.connection_radius
            ]
            
            # Sort by distance and take top k
            node.neighbors = sorted(
                valid_neighbors,
                key=lambda x: x[1]
            )[:self.max_neighbors]

    def plan_path(
        self,
        start_state: np.ndarray,
        goal_state: np.ndarray
    ) -> Tuple[List[np.ndarray], float]:
        """Plan path using A* search on the roadmap"""
        # Add start and goal temporarily
        start_node = PRMNode(start_state)
        goal_node = PRMNode(goal_state)
        
        # Connect to roadmap
        self.nodes.append(start_node)
        self.nodes.append(goal_node)
        self._connect_nodes()
        
        # A* search
        path = self._astar_search(len(self.nodes)-2, len(self.nodes)-1)
        
        # Remove temporary nodes
        self.nodes = self.nodes[:-2]
        
        return path

    def _astar_search(
        self,
        start_idx: int,
        goal_idx: int
    ) -> Tuple[List[np.ndarray], float]:
        """A* search implementation"""
        from queue import PriorityQueue
        
        frontier = PriorityQueue()
        frontier.put((0, start_idx))
        came_from = {start_idx: None}
        cost_so_far = {start_idx: 0}
        
        while not frontier.empty():
            current_idx = frontier.get()[1]
            
            if current_idx == goal_idx:
                break
                
            for next_idx, step_cost in self.nodes[current_idx].neighbors:
                new_cost = cost_so_far[current_idx] + step_cost
                
                if next_idx not in cost_so_far or new_cost < cost_so_far[next_idx]:
                    cost_so_far[next_idx] = new_cost
                    priority = new_cost + self._heuristic(next_idx, goal_idx)
                    frontier.put((priority, next_idx))
                    came_from[next_idx] = current_idx
        
        # Reconstruct path
        path = []
        current_idx = goal_idx
        total_cost = cost_so_far.get(goal_idx, float('inf'))
        
        while current_idx is not None:
            path.append(self.nodes[current_idx].state)
            current_idx = came_from.get(current_idx)
            
        return list(reversed(path)), total_cost

    def _heuristic(self, node_idx: int, goal_idx: int) -> float:
        """Euclidean distance heuristic"""
        return np.linalg.norm(
            self.nodes[node_idx].state - self.nodes[goal_idx].state
        )

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update value estimates for states"""
        states = batch["states"]
        values = batch["values"]
        
        # Forward pass
        encoded_states = self.state_encoder(states)
        predicted_values = self.value_net(encoded_states)
        
        # Compute loss
        loss = nn.MSELoss()(predicted_values, values)
        
        # Update networks
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()} 