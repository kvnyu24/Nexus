import torch
import torch.nn as nn
from ...core.base import NexusModule

class SSMProcessor(NexusModule):
    def __init__(self, hidden_dim: int, scan_factor: int, dropout: float = 0.1):
        super().__init__({"hidden_dim": hidden_dim, "dropout": dropout})
        
        self.hidden_dim = hidden_dim
        self.scan_factor = scan_factor
        
        # State processing layers
        self.state_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Sigmoid()
        )
        
        # Selective scan components
        self.scan_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) 