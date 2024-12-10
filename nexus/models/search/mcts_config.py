from dataclasses import dataclass
from typing import Optional

@dataclass
class MCTSConfig:
    hidden_dim: int
    state_dim: int
    num_actions: int
    num_simulations: int = 50
    c_puct: float = 1.0
    bank_size: int = 10000
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    value_weight: float = 1.0
    policy_weight: float = 1.0
    
    def to_dict(self):
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
