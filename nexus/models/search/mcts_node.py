import torch
import math
from typing import Dict, Optional, List, Tuple
import numpy as np

class MCTSNode:
    def __init__(
        self,
        state: torch.Tensor,
        prior: float = 0.0,
        uncertainty: Optional[torch.Tensor] = None,
        value_init: Optional[float] = None
    ):
        self.state = state
        self.prior = prior
        self.uncertainty = uncertainty or torch.zeros(1)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, 'MCTSNode'] = {}
        self.reward_history: List[float] = []
        self.value_init = value_init  # Initial value estimate for FPU
        
        # Track additional statistics
        self.max_value = float('-inf')
        self.min_value = float('inf')
        self.squared_value_sum = 0.0  # For variance calculation
        self.parent_visits = 0  # Track parent visit count for better exploration
        
    def value(self) -> float:
        if self.visit_count == 0:
            return self.value_init if self.value_init is not None else 0.0
        return self.value_sum / self.visit_count
        
    def value_stats(self) -> Tuple[float, float, float]:
        """Returns mean, std dev, and confidence bounds of value"""
        if self.visit_count < 2:
            return self.value(), 0.0, float('inf')
            
        mean = self.value()
        variance = (self.squared_value_sum / self.visit_count) - (mean * mean)
        std_dev = math.sqrt(max(variance, 0))
        confidence = 1.96 * std_dev / math.sqrt(self.visit_count)  # 95% confidence interval
        
        return mean, std_dev, confidence
        
    def uncertainty_adjusted_value(self) -> float:
        base_value, std_dev, confidence = self.value_stats()
        
        # Progressive widening with uncertainty
        visit_factor = math.sqrt(math.log(self.parent_visits + 1) / (self.visit_count + 1))
        uncertainty_penalty = (
            self.uncertainty.item() * visit_factor +  # Exploration term
            std_dev * 0.5 +  # Value uncertainty
            confidence * 0.5  # Statistical confidence
        )
        
        return base_value - uncertainty_penalty
        
    def expanded(self) -> bool:
        return len(self.children) > 0
        
    def add_exploration_noise(self, dirichlet_alpha: float, dirichlet_epsilon: float):
        """Add Dirichlet noise to priors for exploration"""
        # Scale alpha based on number of legal actions
        scaled_alpha = dirichlet_alpha / max(1, len(self.children))
        
        noise = torch.distributions.Dirichlet(
            torch.ones(len(self.children)) * scaled_alpha
        ).sample()
        
        # Temperature annealing based on visit count
        temp = max(0.1, 1.0 / math.sqrt(1 + self.visit_count))
        effective_epsilon = dirichlet_epsilon * temp
        
        for idx, (action, child) in enumerate(self.children.items()):
            # Mix noise with temperature-adjusted epsilon
            child.prior = ((1 - effective_epsilon) * child.prior + 
                         effective_epsilon * noise[idx].item())
            
            # Ensure priors remain normalized
            child.prior = max(0.01, min(0.99, child.prior))