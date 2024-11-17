from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule
from ...components.attention import MultiHeadSelfAttention

class DocumentRetriever(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads
        )
        self.projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute relevance scores between query and documents
        scores = torch.matmul(
            self.projection(query_embeddings),
            document_embeddings.transpose(-2, -1)
        ) / torch.sqrt(torch.tensor(query_embeddings.size(-1), dtype=torch.float))
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, float("-inf"))
            
        # Get attention weights and retrieve relevant documents
        attention_weights = F.softmax(scores, dim=-1)
        retrieved_docs = torch.matmul(attention_weights, document_embeddings)
        
        return retrieved_docs, attention_weights

class RAGModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_size = config["hidden_size"]
        self.num_heads = config.get("num_heads", 8)
        self.num_retrieval_docs = config.get("num_retrieval_docs", 5)
        
        # Document retriever
        self.retriever = DocumentRetriever(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads
        )
        
        # Cross-attention for combining query and retrieved documents
        self.cross_attention = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
    def forward(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Retrieve relevant documents
        retrieved_docs, retrieval_weights = self.retriever(
            query_embeddings,
            document_embeddings,
            attention_mask
        )
        
        # Combine query with retrieved documents using cross-attention
        combined = self.cross_attention(
            query_embeddings,
            retrieved_docs
        )
        
        # Project to final representation
        output = self.output_proj(combined)
        
        return {
            "output": output,
            "retrieved_docs": retrieved_docs,
            "retrieval_weights": retrieval_weights
        } 