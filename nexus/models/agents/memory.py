from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule
from ...core.mixins import FeatureBankMixin


class AgentMemoryStream(FeatureBankMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_dim = config["hidden_dim"]
        self.memory_size = config.get("memory_size", 1000)
        self.num_heads = config.get("num_heads", 8)
        self.dropout = config.get("dropout", 0.1)
        
        # Enhanced memory components with separate short and long-term storage
        # Using FeatureBankMixin for episodic and semantic memories
        self.register_feature_bank("episodic", self.memory_size, self.hidden_dim)
        self.register_feature_bank("semantic", self.memory_size, self.hidden_dim)
        self.register_feature_bank("working", 100, self.hidden_dim)  # Smaller, more active memory

        # Memory mask for attention (tracks which slots are valid)
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
        
        # Retrieve relevant memories using attention (using FeatureBankMixin buffers)
        episodic_bank = self.get_feature_bank("episodic")
        if episodic_bank.size(0) == 0:
            # Handle empty bank case
            episodic_bank = torch.zeros(1, self.hidden_dim, device=memory_encoding.device)
        retrieved_memory, attention_weights = self.memory_attention(
            memory_encoding.unsqueeze(0),
            episodic_bank.unsqueeze(0),
            episodic_bank.unsqueeze(0),
            key_padding_mask=~self.memory_mask[:episodic_bank.size(0)].unsqueeze(0) if self.is_bank_full("episodic") else None
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
            "episodic_memory": self.get_feature_bank("episodic"),
            "semantic_memory": self.get_feature_bank("semantic"),
            "working_memory": self.get_feature_bank("working")
        }
        
    def _update_memory(
        self,
        encoding: torch.Tensor,
        importance: torch.Tensor,
        retrieved_memory: torch.Tensor
    ):
        """Enhanced memory update with consolidation and pruning using FeatureBankMixin"""
        batch_size = encoding.size(0)

        # Update working memory using FeatureBankMixin
        self.update_feature_bank("working", encoding)

        # Consolidate memories using GRU
        consolidated_memory, _ = self.memory_consolidation(
            torch.cat([encoding, retrieved_memory], dim=0).unsqueeze(0)
        )
        consolidated_memory = consolidated_memory.squeeze(0)

        # Update episodic memory using FeatureBankMixin
        self.update_feature_bank("episodic", consolidated_memory[:batch_size])

        # Update semantic memory with importance weighting using FeatureBankMixin
        semantic_features = consolidated_memory[:batch_size] * importance
        self.update_feature_bank("semantic", semantic_features)

        # Update memory mask based on episodic pointer
        ptr = int(self.episodic_ptr)
        start_ptr = (ptr - batch_size) % self.memory_size
        if start_ptr < ptr:
            self.memory_mask[start_ptr:ptr] = True
        else:
            self.memory_mask[start_ptr:] = True
            self.memory_mask[:ptr] = True

        # Prune low-importance memories periodically (when pointer wraps)
        if ptr < batch_size:
            importance_threshold = importance.mean() * 0.5
            low_importance_mask = importance.squeeze(-1) < importance_threshold
            if start_ptr < ptr:
                self.memory_mask[start_ptr:ptr][low_importance_mask] = False
            else:
                # Handle wrap-around case
                first_part_size = self.memory_size - start_ptr
                self.memory_mask[start_ptr:][low_importance_mask[:first_part_size]] = False
                if ptr > 0:
                    self.memory_mask[:ptr][low_importance_mask[first_part_size:]] = False