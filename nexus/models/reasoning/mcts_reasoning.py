from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from ...core.base import NexusModule
from ..search.mcts import EnhancedMCTS

class MCTSReasoning(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core components
        self.mcts = EnhancedMCTS(config)
        self.state_encoder = nn.Sequential(
            nn.Linear(config["state_dim"], config["hidden_dim"]),
            nn.LayerNorm(config["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"], config["hidden_dim"])
        )
        
        # Reasoning components (following ProactiveAgent pattern)
        self.reasoning_head = nn.Sequential(
            nn.Linear(config["hidden_dim"], config["hidden_dim"] // 2),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"] // 2, config["num_actions"])
        )
        
        # Uncertainty estimation (following PlanningModule pattern)
        self.uncertainty_head = nn.Linear(config["hidden_dim"], config["num_actions"])
        
    def forward(
        self,
        state: torch.Tensor,
        num_simulations: Optional[int] = None,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        # Encode state
        encoded_state = self.state_encoder(state)
        
        # Run MCTS
        mcts_outputs = self.mcts.simulate(state, num_simulations)
        
        # Generate reasoning outputs
        reasoning_logits = self.reasoning_head(encoded_state)
        uncertainty = self.uncertainty_head(encoded_state)
        
        # Combine MCTS and reasoning outputs
        combined_policy = (
            torch.softmax(reasoning_logits / temperature, dim=-1) +
            mcts_outputs["action_probs"]
        ) / 2
        
        return {
            "policy": combined_policy,
            "uncertainty": uncertainty,
            "mcts_value": mcts_outputs["root_value"],
            "visit_counts": mcts_outputs["visit_counts"]
        } 