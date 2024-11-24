import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from ...core.base import NexusModule
from .rag import EnhancedRAGModule
from .hallucination_reducer import HallucinationReducer

class SupervisedFineTuningModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core components
        self.hidden_size = config["hidden_size"]
        self.num_heads = config.get("num_heads", 8)
        self.dropout = config.get("dropout", 0.1)
        
        # Initialize RAG for knowledge augmentation
        self.rag_module = EnhancedRAGModule(config)
        
        # Initialize hallucination reducer
        self.hallucination_reducer = HallucinationReducer(config)
        
        # SFT-specific components
        self.instruction_encoder = nn.Linear(self.hidden_size, self.hidden_size)
        self.response_encoder = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Cross-attention for instruction-response alignment
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Quality assessment head
        self.quality_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        instruction_ids: torch.Tensor,
        response_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        document_embeddings: Optional[torch.Tensor] = None,
        return_quality_scores: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Get knowledge context using RAG
        rag_outputs = self.rag_module(
            query_embeddings=instruction_ids,
            document_embeddings=document_embeddings,
            attention_mask=attention_mask
        )
        
        # Encode instructions and responses
        instruction_embeds = self.instruction_encoder(instruction_ids)
        response_embeds = self.response_encoder(response_ids)
        
        # Cross-attention between instructions and responses
        attended_response, attention_weights = self.cross_attention(
            instruction_embeds,
            response_embeds,
            response_embeds,
            key_padding_mask=attention_mask
        )
        
        # Verify factual consistency
        verification_outputs = self.hallucination_reducer(
            input_ids=response_ids,
            document_embeddings=rag_outputs["retrieved_docs"],
            attention_mask=attention_mask
        )
        
        outputs = {
            "attended_response": attended_response,
            "attention_weights": attention_weights,
            "verification_scores": verification_outputs["confidence_scores"],
            "retrieved_context": rag_outputs["retrieved_docs"]
        }
        
        if return_quality_scores:
            quality_scores = self.quality_head(attended_response)
            outputs["quality_scores"] = quality_scores
            
        return outputs
        