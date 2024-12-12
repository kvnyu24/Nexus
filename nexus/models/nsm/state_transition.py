import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from ...core.base import NexusModule
from ...visualization.hierarchical import HierarchicalVisualizer
from ..gnn.attention import GraphAttention

class StateTransitionModule(NexusModule):
    """Enhanced state transition network with adaptive attention and uncertainty estimation"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_states: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        memory_layers: int = 2,
        use_graph_attention: bool = True
    ):
        super().__init__({
            "hidden_dim": hidden_dim,
            "num_states": num_states,
            "num_heads": num_heads
        })
        
        # Core dimensions
        self.hidden_dim = hidden_dim
        self.num_states = num_states
        self.num_heads = num_heads
        
        # Enhanced transition network with residual connections
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU()
        )
        
        self.transition_predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_states)
        )
        
        # Adaptive memory processing
        if use_graph_attention:
            self.memory_processor = GraphAttention({
                "hidden_dim": hidden_dim,
                "num_heads": num_heads,
                "dropout": dropout,
                "attention_type": "scaled_dot_product"
            })
        else:
            self.memory_processor = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=memory_layers
            )
        
        # Multi-head uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_states * num_heads)
        )
        
        # Confidence scoring for memory context
        self.memory_confidence = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Visualization support
        self.visualizer = HierarchicalVisualizer(self.config)
        
    def forward(
        self,
        state_embed: torch.Tensor,
        input_embed: torch.Tensor,
        memory_bank: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with attention visualization and confidence scoring
        
        Args:
            state_embed: Current state embeddings [batch_size, hidden_dim]
            input_embed: Input embeddings [batch_size, hidden_dim] 
            memory_bank: Memory features [num_memories, hidden_dim]
            attention_mask: Optional mask for memory attention
            return_attention: Whether to return attention weights
            
        Returns:
            Dict containing transition logits, uncertainty scores, memory context and optionally attention weights
        """
        batch_size = state_embed.shape[0]
        
        # Process memory with adaptive attention
        if isinstance(self.memory_processor, GraphAttention):
            memory_output = self.memory_processor(
                memory_bank.unsqueeze(0).expand(batch_size, -1, -1),
                attention_mask=attention_mask
            )
            memory_context = memory_output["output"]
            attention_weights = memory_output.get("attention_weights")
        else:
            memory_context = self.memory_processor(
                memory_bank.unsqueeze(0).expand(batch_size, -1, -1),
                src_key_padding_mask=attention_mask
            )
            attention_weights = None
            
        # Calculate memory confidence scores
        memory_confidence = self.memory_confidence(memory_context)
        
        # Combine inputs with confident memory context
        combined_features = torch.cat([
            state_embed,
            input_embed,
            memory_context * memory_confidence
        ], dim=-1)
        
        # Generate transition features
        transition_features = self.state_encoder(combined_features)
        
        # Predict transitions and uncertainty
        transition_logits = self.transition_predictor(transition_features)
        uncertainty_logits = self.uncertainty_estimator(transition_features)
        uncertainty = uncertainty_logits.view(batch_size, self.num_heads, self.num_states)
        
        outputs = {
            "logits": transition_logits,
            "uncertainty": uncertainty.mean(dim=1),
            "uncertainty_per_head": uncertainty,
            "memory_context": memory_context,
            "memory_confidence": memory_confidence
        }
        
        if return_attention and attention_weights is not None:
            outputs["attention_weights"] = attention_weights
            
        return outputs