import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Union, Optional
import torch
import random
import logging
from pathlib import Path

class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        """Initialize replay buffer with given capacity.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        if capacity <= 0:
            raise ValueError("Buffer capacity must be positive")
            
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.logger = logging.getLogger(__name__)
        
    def push(self, state: Union[np.ndarray, torch.Tensor], 
             action: Union[int, np.ndarray, torch.Tensor],
             reward: Union[float, np.ndarray, torch.Tensor],
             next_state: Union[np.ndarray, torch.Tensor], 
             done: Union[bool, np.ndarray, torch.Tensor]) -> None:
        """Add a transition to the buffer.
        
        Handles various input types and converts to numpy arrays for storage.
        """
        try:
            # Convert tensors to numpy
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            if isinstance(next_state, torch.Tensor):
                next_state = next_state.cpu().numpy()
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            if isinstance(reward, torch.Tensor):
                reward = reward.cpu().numpy()
            if isinstance(done, torch.Tensor):
                done = done.cpu().numpy()
                
            # Convert to correct types
            state = np.asarray(state, dtype=np.float32)
            next_state = np.asarray(next_state, dtype=np.float32)
            action = np.asarray(action)
            reward = float(reward)
            done = bool(done)
            
            self.buffer.append((state, action, reward, next_state, done))
            
        except Exception as e:
            self.logger.error(f"Failed to add transition to buffer: {str(e)}")
            raise
        
    def sample(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary of batched transitions as torch tensors,
            or None if batch_size is larger than buffer size
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
            
        if batch_size > len(self):
            self.logger.warning(f"Requested batch size {batch_size} > buffer size {len(self)}")
            return None
            
        try:
            transitions = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*transitions)
            
            # Convert to torch tensors with appropriate shapes
            return {
                "states": torch.FloatTensor(np.array(states)),
                "actions": torch.as_tensor(actions),
                "rewards": torch.FloatTensor(rewards).reshape(-1, 1),
                "next_states": torch.FloatTensor(np.array(next_states)), 
                "dones": torch.FloatTensor(dones).reshape(-1, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to sample from buffer: {str(e)}")
            raise
            
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)
        
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough transitions for sampling."""
        return len(self) >= batch_size