import torch
import math
from typing import Dict, Optional, List

class EnhancedMCTSNode:
    def __init__(
        self,
        state: torch.Tensor,
        prior: float = 0.0,
        uncertainty: Optional[torch.Tensor] = None
    ):
        self.state = state
        self.prior = prior
        self.uncertainty = uncertainty or torch.zeros(1)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, 'EnhancedMCTSNode'] = {}
        self.reward_history: List[float] = []
        
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
        
    def uncertainty_adjusted_value(self) -> float:
        base_value = self.value()
        uncertainty_penalty = self.uncertainty.item() * math.sqrt(1 / (1 + self.visit_count))
        return base_value - uncertainty_penalty
        
    def expanded(self) -> bool:
        return len(self.children) > 0
        
    def add_exploration_noise(self, dirichlet_alpha: float, dirichlet_epsilon: float):
        noise = torch.distributions.Dirichlet(
            torch.ones(len(self.children)) * dirichlet_alpha
        ).sample()
        
        for idx, (action, child) in enumerate(self.children.items()):
            child.prior = (1 - dirichlet_epsilon) * child.prior + dirichlet_epsilon * noise[idx].item()