from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...core.base import NexusModule
from .mcts_config import MCTSConfig 
from .mcts_node import MCTSNode
from ...components import CrossAttention

class TransformerMCTSWithMemory(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Convert dict config to MCTSConfig
        self.mcts_config = MCTSConfig(**config)
        
        # Hierarchical state encoder with multi-scale processing
        self.hierarchical_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.mcts_config.hidden_dim // (2 ** i),
                nhead=max(1, 16 // (2 ** i)),
                dim_feedforward=self.mcts_config.hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            ) for i in range(3)  # 3 scales of processing
        ])
        
        # Cross-attention for state-action history
        self.action_history_attention = CrossAttention(
            query_dim=self.mcts_config.hidden_dim,
            key_dim=self.mcts_config.hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Self-attention pooling
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=self.mcts_config.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.pool_norm = nn.LayerNorm(self.mcts_config.hidden_dim)
        
        # Contrastive learning head
        self.contrastive_proj = nn.Sequential(
            nn.Linear(self.mcts_config.hidden_dim, self.mcts_config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.mcts_config.hidden_dim, 128)
        )
        
        # Adaptive computation controller
        self.compute_controller = nn.Sequential(
            nn.Linear(self.mcts_config.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.mcts_config.hidden_dim, 2*self.mcts_config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2*self.mcts_config.hidden_dim, self.mcts_config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.mcts_config.hidden_dim, self.mcts_config.num_actions)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.mcts_config.hidden_dim, 2*self.mcts_config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2*self.mcts_config.hidden_dim, self.mcts_config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.mcts_config.hidden_dim, 1)
        )
        
        # Enhanced episodic memory with hierarchical storage
        self.register_buffer(
            "state_bank",
            torch.zeros(self.mcts_config.bank_size, self.mcts_config.hidden_dim)
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
        # Root node for MCTS
        self.root = None
        
    def update_state_bank(
        self,
        states: torch.Tensor,
        node: Optional[MCTSNode] = None
    ) -> None:
        batch_size = states.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.state_bank.size(0):
            ptr = 0
            
        self.state_bank[ptr:ptr + batch_size] = states.detach()
        
        if node is not None:
            # Update root node if provided
            self.root = node
            
        self.bank_ptr[0] = (ptr + batch_size) % self.state_bank.size(0)
        
    def forward(
        self,
        states: torch.Tensor,
        action_history: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Multi-scale hierarchical encoding
        features = states
        for encoder in self.hierarchical_encoder:
            features = encoder(features)
            
        # Process action history if provided
        if action_history is not None:
            features = self.action_history_attention(
                query=features,
                key=action_history,
                value=action_history
            )
            
        # Self-attention pooling
        pooled, _ = self.pool_attention(
            features,
            features,
            features
        )
        pooled = self.pool_norm(pooled)
        
        # Get computation allocation score
        compute_score = self.compute_controller(pooled.mean(dim=1))
        
        # Get policy and value predictions
        policy_logits = self.policy_head(pooled.mean(dim=1))
        value = self.value_head(pooled.mean(dim=1))
        
        # Project features for contrastive learning
        contrastive = self.contrastive_proj(pooled.mean(dim=1))
        
        outputs = {
            "policy_logits": policy_logits,
            "value": value,
            "compute_score": compute_score,
            "contrastive": contrastive
        }
        
        if return_features:
            outputs["features"] = features
            outputs["pooled"] = pooled
            
        return outputs