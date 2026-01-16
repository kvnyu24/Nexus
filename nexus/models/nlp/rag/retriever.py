from typing import Dict, Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ....core.base import NexusModule
from annoy import AnnoyIndex
import numpy as np
from ....utils.logging import Logger

class EfficientRetriever(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_size = config["hidden_size"]
        self.num_retrieved = config.get("num_retrieved", 5)
        self.num_trees = config.get("num_trees", 10)
        self.metric = config.get("metric", "dot")
        self.index = None
        self.document_store = []
        self.logger = Logger(__name__)
        
    def _validate_embeddings(self, embeddings: torch.Tensor) -> None:
        """Validate embedding dimensions and values."""
        if embeddings.dim() not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D tensor, got {embeddings.dim()}D")
        if embeddings.size(-1) != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {embeddings.size(-1)}")
        if torch.isnan(embeddings).any():
            raise ValueError("Embeddings contain NaN values")
            
    def build_index(self, document_embeddings: torch.Tensor) -> None:
        """Build Annoy index with error handling and validation."""
        try:
            self._validate_embeddings(document_embeddings)
            
            # Convert to numpy and ensure 2D
            embeddings_np = document_embeddings.detach().cpu().numpy()
            if embeddings_np.ndim == 3:
                embeddings_np = embeddings_np.mean(axis=1)  # Average sequence dimension
                
            # Build Annoy index
            self.index = AnnoyIndex(self.hidden_size, self.metric)
            self.index.set_seed(42)  # For reproducibility
            
            # Add embeddings with progress logging
            total_docs = len(embeddings_np)
            for i, embedding in enumerate(embeddings_np):
                if i % 1000 == 0:
                    self.logger.info(f"Indexing progress: {i}/{total_docs}")
                self.index.add_item(i, embedding)
                
            # Build the index
            self.index.build(self.num_trees)
            self.logger.info(f"Successfully built index with {total_docs} documents")
            
        except Exception as e:
            self.logger.error(f"Failed to build index: {str(e)}")
            raise
            
    def forward(
        self,
        query_embedding: torch.Tensor,
        document_embeddings: torch.Tensor,
        return_scores: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Retrieve relevant documents with error handling and validation."""
        try:
            self._validate_embeddings(query_embedding)
            self._validate_embeddings(document_embeddings)
            
            # Build index if not exists
            if self.index is None:
                self.build_index(document_embeddings)
                
            # Prepare query embeddings
            query_np = query_embedding.detach().cpu().numpy()
            if query_np.ndim == 3:
                query_np = query_np.mean(axis=1)
                
            # Search for nearest neighbors
            batch_size = len(query_np)
            indices_list = []
            scores_list = []
            
            for query in query_np:
                idx, distances = self.index.get_nns_by_vector(
                    query, 
                    self.num_retrieved, 
                    include_distances=True,
                    search_k=-1  # Use all trees for better accuracy
                )
                indices_list.append(idx)
                scores_list.append(distances)
                
            # Convert to numpy arrays with proper shape
            indices = np.array(indices_list)
            scores = np.array(scores_list)
            
            # Get retrieved documents
            retrieved_docs = document_embeddings[torch.from_numpy(indices).to(document_embeddings.device)]
            
            return {
                "retrieved_docs": retrieved_docs,
                "retrieval_scores": torch.from_numpy(scores).to(document_embeddings.device) if return_scores else None,
                "doc_indices": torch.from_numpy(indices).to(document_embeddings.device)
            }
            
        except Exception as e:
            self.logger.error(f"Error during retrieval: {str(e)}")
            raise

    def save_index(self, path: str) -> None:
        """Save the Annoy index to disk."""
        if self.index is not None:
            try:
                self.index.save(path)
                self.logger.info(f"Successfully saved index to {path}")
            except Exception as e:
                self.logger.error(f"Failed to save index: {str(e)}")
                raise

    def load_index(self, path: str) -> None:
        """Load the Annoy index from disk."""
        try:
            self.index = AnnoyIndex(self.hidden_size, self.metric)
            self.index.load(path)
            self.logger.info(f"Successfully loaded index from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load index: {str(e)}")
            raise