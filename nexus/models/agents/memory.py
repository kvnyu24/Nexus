from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule

class AgentMemoryStream(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_dim = config["hidden_dim"]
        self.memory_size = config.get("memory_size", 1000)
        self.num_heads = config.get("num_heads", 8)
        self.dropout = config.get("dropout", 0.1)
        
        # Enhanced memory components with separate short and long-term storage
        self.register_buffer(
            "episodic_memory",
            torch.zeros(self.memory_size, self.hidden_dim)
        )
        self.register_buffer(
            "semantic_memory", 
            torch.zeros(self.memory_size, self.hidden_dim)
        )
        self.register_buffer(
            "working_memory",
            torch.zeros(100, self.hidden_dim)  # Smaller, more active memory
        )
        self.register_buffer("memory_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("memory_mask", torch.ones(self.memory_size, dtype=torch.bool))
        
        # Enhanced memory processing with attention
        self.memory_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # Multi-head attention for memory retrieval
        self.memory_attention = nn.MultiheadAttention(
            self.hidden_dim,
            self.num_heads,
            dropout=self.dropout
        )
        
        # Enhanced importance scoring with context
        self.importance_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Memory consolidation network
        self.memory_consolidation = nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
    def forward(
        self,
        current_state: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = current_state.size(0)
        
        # Process current observation with enhanced context handling
        if context is not None:
            combined = torch.cat([current_state, context], dim=-1)
        else:
            context_padding = torch.zeros_like(current_state)
            combined = torch.cat([current_state, context_padding], dim=-1)
            
        memory_encoding = self.memory_encoder(combined)
        
        # Retrieve relevant memories using attention
        retrieved_memory, attention_weights = self.memory_attention(
            memory_encoding.unsqueeze(0),
            self.episodic_memory.unsqueeze(0),
            self.episodic_memory.unsqueeze(0),
            key_padding_mask=~self.memory_mask.unsqueeze(0)
        )
        retrieved_memory = retrieved_memory.squeeze(0)
        
        # Score importance using both encoding and retrieved memory
        importance = self.importance_scorer(
            torch.cat([memory_encoding, retrieved_memory], dim=-1)
        )
        
        # Update memory streams with consolidation
        self._update_memory(memory_encoding, importance, retrieved_memory)
        
        return {
            "memory_encoding": memory_encoding,
            "retrieved_memory": retrieved_memory,
            "importance": importance,
            "attention_weights": attention_weights,
            "episodic_memory": self.episodic_memory,
            "semantic_memory": self.semantic_memory,
            "working_memory": self.working_memory
        }
        
    def _update_memory(
        self,
        encoding: torch.Tensor,
        importance: torch.Tensor,
        retrieved_memory: torch.Tensor
    ):
        """Enhanced memory update with consolidation and pruning"""
        batch_size = encoding.size(0)
        ptr = int(self.memory_ptr)
        
        # Circular buffer implementation
        if ptr + batch_size > self.memory_size:
            ptr = 0
            
        # Update working memory (most recent memories)
        working_size = min(batch_size, self.working_memory.size(0))
        self.working_memory = torch.roll(self.working_memory, shifts=-working_size, dims=0)
        self.working_memory[-working_size:] = encoding[:working_size].detach()
        
        # Consolidate memories using GRU
        consolidated_memory, _ = self.memory_consolidation(
            torch.cat([encoding, retrieved_memory], dim=0).unsqueeze(0)
        )
        consolidated_memory = consolidated_memory.squeeze(0)
        
        # Update episodic and semantic memories
        self.episodic_memory[ptr:ptr + batch_size] = consolidated_memory[:batch_size].detach()
        self.semantic_memory[ptr:ptr + batch_size] = (
            consolidated_memory[:batch_size].detach() * importance
        )
        
        # Update memory mask and pointer
        self.memory_mask[ptr:ptr + batch_size] = True
        self.memory_ptr[0] = (ptr + batch_size) % self.memory_size
        
        # Prune low-importance memories periodically
        if ptr == 0:
            importance_threshold = importance.mean() * 0.5
            low_importance_mask = importance < importance_threshold
            self.memory_mask[ptr:ptr + batch_size][low_importance_mask] = False