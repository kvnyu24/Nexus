import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from ...core.base import NexusModule
from .memory import AgentMemoryStream
from ..nlp import ChainOfThoughtModule, EnhancedRAGModule

class SocialAgent(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_dim = config["hidden_dim"]
        self.num_heads = config.get("num_heads", 8)
        self.dropout = config.get("dropout", 0.1)
        
        # Enhanced core components with multi-head attention
        self.state_encoder = nn.Sequential(
            nn.Linear(config["state_dim"], self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.memory_stream = AgentMemoryStream(config)
        
        # Advanced planning with chain-of-thought and attention
        self.chain_of_thought = ChainOfThoughtModule(config)
        self.plan_attention = nn.MultiheadAttention(
            self.hidden_dim,
            self.num_heads,
            dropout=self.dropout
        )
        
        self.plan_generator = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, config["num_actions"])
        )
        
        # Enhanced reflection with RAG and residual connections
        self.rag_module = EnhancedRAGModule(config)
        self.reflection_module = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ),
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
        ])
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(self.hidden_dim, 1)
        
    def forward(
        self,
        state: torch.Tensor,
        social_context: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Enhanced state encoding
        encoded_state = self.state_encoder(state)
        batch_size = encoded_state.size(0)
        
        # Process memory with error handling
        try:
            memory_outputs = self.memory_stream(
                encoded_state,
                context=social_context
            )
        except Exception as e:
            # Fallback to empty memory if error occurs
            memory_outputs = {
                "memory_encoding": torch.zeros(batch_size, self.hidden_dim).to(state.device),
                "episodic_memory": torch.zeros(batch_size, 10, self.hidden_dim).to(state.device),
                "semantic_memory": torch.zeros(batch_size, 5, self.hidden_dim).to(state.device)
            }
        
        # Generate plan with chain-of-thought reasoning
        thought_outputs = self.chain_of_thought(encoded_state, memory_outputs["memory_encoding"])
        
        # Apply attention over memories
        episodic_context = memory_outputs["episodic_memory"].mean(dim=1)
        semantic_context = memory_outputs["semantic_memory"].mean(dim=1)
        
        plan_query = encoded_state.unsqueeze(0)
        plan_key = torch.stack([episodic_context, semantic_context, thought_outputs], dim=0)
        plan_value = plan_key
        
        attended_plan, attention_weights = self.plan_attention(
            plan_query, plan_key, plan_value
        )
        
        # Generate actions with enhanced context
        plan_input = torch.cat([
            encoded_state,
            attended_plan.squeeze(0),
            memory_outputs["memory_encoding"],
            thought_outputs
        ], dim=-1)
        
        action_logits = self.plan_generator(plan_input)
        
        # Enhanced reflection with RAG and residual connections
        rag_output = self.rag_module(encoded_state)
        reflection_input = torch.cat([
            encoded_state,
            semantic_context,
            rag_output
        ], dim=-1)
        
        reflection_state = reflection_input
        for layer in self.reflection_module:
            reflection_state = layer(reflection_state) + reflection_state
            
        # Estimate uncertainty
        uncertainty = torch.sigmoid(self.uncertainty_head(reflection_state))
        
        outputs = {
            "action_logits": action_logits,
            "reflection_state": reflection_state,
            "memory_outputs": memory_outputs,
            "uncertainty": uncertainty,
            "thought_outputs": thought_outputs
        }
        
        if return_attention:
            outputs["attention_weights"] = attention_weights
            
        return outputs