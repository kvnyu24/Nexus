from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from ....core.base import NexusModule
from .document_encoder import DocumentEncoder
from .retriever import EfficientRetriever
from ....components.attention import MultiHeadSelfAttention

class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads
        )
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        query_states: torch.Tensor,
        doc_states: torch.Tensor
    ) -> torch.Tensor:
        # Cross-attention between query and documents
        fused = self.attention(
            query_states,
            key_value_states=doc_states
        )
        return self.norm(fused + query_states)

class EnhancedRAGModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize components
        self.document_encoder = DocumentEncoder(config)
        self.retriever = EfficientRetriever(config)
        self.fusion = CrossAttentionFusion(
            hidden_size=config["hidden_size"],
            num_heads=config.get("num_heads", 8)
        )
        
        # Output projection
        self.output_proj = nn.Linear(config["hidden_size"], config["vocab_size"])
        
    def forward(
        self,
        query_states: torch.Tensor,
        document_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode documents
        doc_embeddings = self.document_encoder(document_ids)
        
        # Retrieve relevant documents
        retrieval_outputs = self.retriever(
            query_states.mean(dim=1),  # Pool query embedding
            doc_embeddings,
            return_scores=True
        )
        
        # Fuse query with retrieved documents
        fused_states = self.fusion(
            query_states,
            retrieval_outputs["retrieved_docs"]
        )
        
        # Generate output logits
        logits = self.output_proj(fused_states)
        
        return {
            "logits": logits,
            "retrieved_docs": retrieval_outputs["retrieved_docs"],
            "retrieval_scores": retrieval_outputs["retrieval_scores"],
            "fused_states": fused_states
        } 