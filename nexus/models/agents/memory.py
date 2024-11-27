from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from ...core.base import NexusModule

class AgentMemoryStream(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_dim = config["hidden_dim"]
        self.memory_size = config.get("memory_size", 1000)
        
        # Memory components (following EnhancedReID pattern)
        self.register_buffer(
            "episodic_memory",
            torch.zeros(self.memory_size, self.hidden_dim)
        )
        self.register_buffer(
            "semantic_memory",
            torch.zeros(self.memory_size, self.hidden_dim)
        )
        self.register_buffer("memory_ptr", torch.zeros(1, dtype=torch.long))
        
        # Memory processing
        self.memory_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Importance scoring (similar to AttentionModule)
        self.importance_scorer = nn.Linear(self.hidden_dim, 1)
        
    def forward(
        self,
        current_state: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Process current observation
        if context is not None:
            combined = torch.cat([current_state, context], dim=-1)
        else:
            combined = torch.cat([current_state, torch.zeros_like(current_state)], dim=-1)
            
        memory_encoding = self.memory_encoder(combined)
        
        # Score importance
        importance = self.importance_scorer(memory_encoding).sigmoid()
        
        # Update memory streams
        self._update_memory(memory_encoding, importance)
        
        return {
            "memory_encoding": memory_encoding,
            "importance": importance,
            "episodic_memory": self.episodic_memory,
            "semantic_memory": self.semantic_memory
        }
        
    def _update_memory(self, encoding: torch.Tensor, importance: torch.Tensor):
        """Update memory following EnhancedReID pattern"""
        batch_size = encoding.size(0)
        ptr = int(self.memory_ptr)
        
        if ptr + batch_size > self.memory_size:
            ptr = 0
            
        # Update memories based on importance
        self.episodic_memory[ptr:ptr + batch_size] = encoding.detach()
        self.semantic_memory[ptr:ptr + batch_size] = (
            encoding.detach() * importance
        )
        self.memory_ptr[0] = (ptr + batch_size) % self.memory_size