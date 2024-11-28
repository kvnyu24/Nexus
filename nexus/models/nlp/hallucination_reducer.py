import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from ...core.base import NexusModule
from ...components.attention import MultiHeadSelfAttention, CrossAttention, FlashAttention
from .chain_of_thoughts import ChainOfThoughtModule
from .rag import EnhancedRAGModule, DocumentEncoder

class FactVerifier(NexusModule):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Use FlashAttention for better performance when available
        self.cross_attention = CrossAttention(
            query_dim=hidden_size,
            key_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attn=True
        )
        
        # More robust confidence scoring with deeper network
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        query_states: torch.Tensor,
        evidence_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Cross-attend between query and evidence
        attended_states, attention_weights = self.cross_attention(
            query_states,
            evidence_states,
            mask=attention_mask
        )
        
        # Calculate confidence scores
        confidence = self.confidence_head(attended_states)
        
        # Add residual connection
        attended_states = attended_states + query_states
        
        return {
            "attended_states": attended_states,
            "confidence_scores": confidence,
            "attention_weights": attention_weights
        }

class HallucinationReducer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core components
        self.rag_module = EnhancedRAGModule(config) # Use enhanced RAG
        self.chain_of_thought = ChainOfThoughtModule(config)
        self.fact_verifier = FactVerifier(
            hidden_size=config["hidden_size"],
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1)
        )
        
        # Document encoder for handling raw text input
        self.document_encoder = DocumentEncoder(config)
        
        # Output projection with layer norm
        self.layer_norm = nn.LayerNorm(config["hidden_size"])
        self.output_proj = nn.Linear(config["hidden_size"], config["vocab_size"])
        
        # Temperature scaling for confidence calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        document_embeddings: Optional[Union[torch.Tensor, List[str]]] = None,
        return_reasoning_steps: bool = False,
        return_attention_patterns: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Handle raw text documents if provided
        if isinstance(document_embeddings, list):
            document_embeddings = self.document_encoder(document_embeddings)
            
        # Get retrieved context from enhanced RAG
        rag_outputs = self.rag_module(
            query_embeddings=input_ids,
            document_embeddings=document_embeddings,
            attention_mask=attention_mask
        )
        
        # Multi-step reasoning with chain-of-thought
        reasoning_outputs = self.chain_of_thought(
            hidden_states=rag_outputs["output"],
            attention_mask=attention_mask,
            context=rag_outputs["retrieved_docs"] # Pass retrieved docs as additional context
        )
        
        # Verify facts against retrieved evidence
        verification_outputs = self.fact_verifier(
            query_states=reasoning_outputs["hidden_states"],
            evidence_states=rag_outputs["retrieved_docs"],
            attention_mask=attention_mask
        )
        
        # Apply layer norm and temperature-scaled confidence to logits
        hidden_states = self.layer_norm(verification_outputs["attended_states"])
        logits = self.output_proj(hidden_states)
        confidence_scores = verification_outputs["confidence_scores"] / self.temperature
        logits = logits * confidence_scores
        
        outputs = {
            "logits": logits,
            "confidence_scores": confidence_scores,
            "retrieved_docs": rag_outputs["retrieved_docs"],
            "attention_weights": verification_outputs["attention_weights"]
        }
        
        if return_reasoning_steps:
            outputs["reasoning_steps"] = reasoning_outputs.get("reasoning_steps")
            
        if return_attention_patterns:
            outputs["rag_attention"] = rag_outputs.get("attention_patterns")
            outputs["reasoning_attention"] = reasoning_outputs.get("attention_weights")
            
        return outputs
