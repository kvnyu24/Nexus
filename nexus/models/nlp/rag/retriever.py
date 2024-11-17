from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ....core.base import NexusModule
import faiss

class EfficientRetriever(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_size = config["hidden_size"]
        self.num_retrieved = config.get("num_retrieved", 5)
        self.index = None
        self.document_store = []
        
    def build_index(self, document_embeddings: torch.Tensor):
        # Convert to numpy for FAISS
        embeddings_np = document_embeddings.cpu().numpy()
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.hidden_size)  # Inner product similarity
        self.index = faiss.IndexIDMap(self.index)
        
        # Add embeddings to index
        self.index.add_with_ids(
            embeddings_np,
            np.arange(len(embeddings_np))
        )
        
    def forward(
        self,
        query_embedding: torch.Tensor,
        document_embeddings: torch.Tensor,
        return_scores: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Compute similarity scores
        if self.index is None:
            self.build_index(document_embeddings)
            
        # Search for nearest neighbors
        scores, indices = self.index.search(
            query_embedding.cpu().numpy(),
            self.num_retrieved
        )
        
        # Get retrieved documents
        retrieved_docs = document_embeddings[indices]
        
        return {
            "retrieved_docs": retrieved_docs,
            "retrieval_scores": torch.from_numpy(scores) if return_scores else None,
            "doc_indices": torch.from_numpy(indices)
        } 