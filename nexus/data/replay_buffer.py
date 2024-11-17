import numpy as np
from collections import deque
from typing import Dict, List, Tuple
import torch
import random

class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        
        return {
            "states": torch.FloatTensor(np.array(states)),
            "actions": torch.LongTensor(actions),
            "rewards": torch.FloatTensor(rewards),
            "next_states": torch.FloatTensor(np.array(next_states)),
            "dones": torch.FloatTensor(dones)
        }
        
    def __len__(self) -> int:
        return len(self.buffer) 