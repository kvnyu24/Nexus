import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from ...core.base import NexusModule
from .processor import SSMProcessor


class HierarchicalSSM(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config
        self._validate_config(config)
        
        # Core dimensions
        self.hidden_dim = config["hidden_dim"]
        self.num_levels = config["num_levels"]
        self.sequence_length = config["sequence_length"]
        
        # State space components for each level
        self.state_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(4, self.hidden_dim, self.hidden_dim))
            for _ in range(self.num_levels)
        ])
        
        # Level-specific processors with selective scanning
        self.level_processors = nn.ModuleList([
            SSMProcessor(
                hidden_dim=self.hidden_dim,
                scan_factor=2 ** i,  # Exponential scanning rate
                dropout=config.get("dropout", 0.1)
            ) for i in range(self.num_levels)
        ])
        
        # Cross-level attention (following RAG pattern)
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=config.get("num_heads", 8),
                dropout=config.get("dropout", 0.1)
            ) for _ in range(self.num_levels - 1)
        ])
        
        # Feature bank (following EnhancedReID pattern)
        self.register_buffer(
            "state_bank",
            torch.zeros(
                config.get("bank_size", 10000),
                self.hidden_dim
            )
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = ["hidden_dim", "num_levels", "sequence_length"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}") 
            
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Process through hierarchical levels
        level_states = []
        level_outputs = []
        
        # Initial state for each level
        h = x
        
        for level in range(self.num_levels):
            # Apply state space transformation
            state_matrices = self.state_matrices[level]
            
            # Selective scanning based on level
            scan_length = seq_len // self.level_processors[level].scan_factor
            reshaped_h = h.view(batch_size, -1, scan_length, self.hidden_dim)
            
            # Process states
            level_output = self.level_processors[level](reshaped_h)
            level_states.append(level_output)
            
            # Cross-level attention if not last level
            if level < self.num_levels - 1:
                attended_output, _ = self.cross_attention[level](
                    level_output,
                    level_output,
                    level_output,
                    key_padding_mask=mask
                )
                h = attended_output
            
            level_outputs.append(level_output)
        
        # Update state bank
        final_state = level_states[-1]
        self._update_state_bank(final_state)
        
        return {
            "level_outputs": level_outputs,
            "final_output": level_outputs[-1],
            "level_states": level_states
        }
        
    def _update_state_bank(self, states: torch.Tensor):
        """Update state bank following EnhancedReID pattern"""
        batch_size = states.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.state_bank.size(0):
            ptr = 0
            
        self.state_bank[ptr:ptr + batch_size] = states.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.state_bank.size(0)