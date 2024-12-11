from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import math

class ExplorationStrategy(Enum):
    """Different exploration strategies for MCTS"""
    UCB1 = "ucb1"  # Classic UCB1
    PUCT = "puct"  # PUCT from AlphaZero
    PROGRESSIVE = "progressive"  # Progressive widening

@dataclass
class MCTSConfig:
    # Core network dimensions
    hidden_dim: int
    state_dim: int 
    num_actions: int

    # Search parameters
    num_simulations: int = 800  # Increased for better exploration
    max_depth: int = 100  # Prevent infinite loops
    exploration_strategy: ExplorationStrategy = ExplorationStrategy.PUCT
    
    # PUCT hyperparameters
    c_puct: float = 2.5  # Increased exploration constant
    c_visit_factor: float = 0.5  # Visit count scaling
    fpu_reduction: float = 0.25  # First play urgency reduction
    
    # Memory parameters  
    bank_size: int = 50000  # Increased replay buffer
    min_replay_size: int = 1000  # Minimum before learning
    prioritized_replay_alpha: float = 0.6  # Prioritization exponent
    prioritized_replay_beta: float = 0.4  # Importance sampling
    
    # Temperature parameters
    init_temp: float = 1.0  # Initial temperature
    final_temp: float = 0.1  # Final temperature
    temp_decay: float = 0.98  # Temperature annealing
    
    # Noise parameters for exploration
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    noise_alpha: float = 0.03  # Root noise parameter
    
    # Loss weights
    value_weight: float = 1.0
    policy_weight: float = 1.0
    entropy_weight: float = 0.001  # Entropy bonus
    consistency_weight: float = 0.1  # Value/policy consistency
    
    # Validation
    def __post_init__(self):
        if self.hidden_dim <= 0 or self.state_dim <= 0 or self.num_actions <= 0:
            raise ValueError("Dimensions must be positive")
        if self.num_simulations < 1:
            raise ValueError("Must perform at least 1 simulation")
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError("Dirichlet epsilon must be between 0 and 1")
            
    def get_temperature(self, step: int) -> float:
        """Calculate annealed temperature"""
        return max(self.final_temp, 
                  self.init_temp * (self.temp_decay ** step))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, handling enums"""
        config_dict = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                if isinstance(v, Enum):
                    config_dict[k] = v.value
                else:
                    config_dict[k] = v
        return config_dict
