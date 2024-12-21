import torch
from ...core.base import NexusModule

class PositionalEncoding(NexusModule):
    def __init__(self, num_frequencies: int = 10, include_identity: bool = True):
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be >= 1")
        self.num_frequencies = num_frequencies
        self.include_identity = include_identity
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input validation
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if x.dim() < 1:
            raise ValueError("Input tensor must have at least 1 dimension")
            
        # Create frequencies for encoding
        frequencies = 2.0 ** torch.arange(self.num_frequencies, 
                                        dtype=x.dtype,
                                        device=x.device)
        
        # Handle numerical stability
        max_freq = torch.max(frequencies)
        if max_freq > 1e6:  # Prevent extreme frequencies
            frequencies = frequencies * (1e6 / max_freq)
            
        # Apply sin and cos to each frequency
        angles = x[..., None] * frequencies
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        
        # Optionally include original input
        if self.include_identity:
            encoding = torch.cat([x, encoding], dim=-1)
            
        return encoding.flatten(start_dim=-2)