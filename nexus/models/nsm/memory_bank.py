import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union
from ...core.base import NexusModule
from ...visualization.hierarchical import HierarchicalVisualizer
import torch.nn.functional as F

class MemoryBank(NexusModule):
    """Enhanced memory bank for storing and retrieving important features with visualization capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core configuration
        self.hidden_dim = config.get("hidden_dim", 256)
        self.bank_size = config.get("bank_size", 1024)
        self.compression_ratio = config.get("compression_ratio", 2)
        self.min_importance_threshold = config.get("min_importance_threshold", 0.1)
        
        # Initialize memory banks
        self.register_buffer("feature_bank", torch.zeros(self.bank_size, self.hidden_dim))
        self.register_buffer("importance_scores", torch.zeros(self.bank_size))
        self.register_buffer("feature_timestamps", torch.zeros(self.bank_size))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
        # Enhanced feature compression with residual connection
        compressed_dim = self.hidden_dim // self.compression_ratio
        self.compressor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, compressed_dim),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(compressed_dim, self.hidden_dim)
        )
        
        # Importance estimation with attention
        self.importance_estimator = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, compressed_dim),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(compressed_dim, 1),
            nn.Sigmoid()
        )
        
        # Feature aggregation
        self.feature_aggregator = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU()
        )
        
        # Visualization
        self.visualizer = HierarchicalVisualizer(config)
        
    def update(
        self,
        features: torch.Tensor,
        importance_override: Optional[torch.Tensor] = None,
        update_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Update memory bank with new features
        
        Args:
            features: Input features to store [batch_size, hidden_dim]
            importance_override: Optional manual importance scores [batch_size]
            update_mask: Optional mask to selectively update entries [batch_size]
            
        Returns:
            Dict containing update statistics
        """
        batch_size = features.size(0)
        if features.size(-1) != self.hidden_dim:
            raise ValueError(f"Feature dimension mismatch. Expected {self.hidden_dim}, got {features.size(-1)}")
            
        # Compress and normalize features
        compressed = self.compressor(features)
        compressed = F.normalize(compressed, p=2, dim=-1)
        
        # Calculate importance scores
        if importance_override is None:
            importance = self.importance_estimator(features).squeeze(-1)
        else:
            importance = importance_override
            
        # Get current pointer
        ptr = int(self.bank_ptr)
        
        # Handle bank overflow
        if ptr + batch_size > self.bank_size:
            ptr = 0
            
        # Update entries
        update_range = slice(ptr, ptr + batch_size)
        if update_mask is not None:
            # Selective update
            mask = update_mask.bool()
            self.feature_bank[update_range][mask] = compressed[mask]
            self.importance_scores[update_range][mask] = importance[mask]
            self.feature_timestamps[update_range][mask] = self.get_current_timestamp()
        else:
            # Full update
            self.feature_bank[update_range] = compressed
            self.importance_scores[update_range] = importance
            self.feature_timestamps[update_range] = self.get_current_timestamp()
            
        # Update pointer
        self.bank_ptr[0] = (ptr + batch_size) % self.bank_size
        
        return {
            "num_updated": batch_size,
            "mean_importance": importance.mean().item(),
            "max_importance": importance.max().item()
        }
        
    def retrieve(
        self,
        query: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        min_importance: Optional[float] = None,
        max_age: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieve features from memory bank with flexible filtering
        
        Args:
            query: Optional query features for similarity-based retrieval
            top_k: Optional limit on number of features to return
            min_importance: Optional minimum importance threshold
            max_age: Optional maximum feature age in timestamps
            
        Returns:
            Dict containing retrieved features and metadata
        """
        # Build retrieval mask
        valid_mask = torch.ones_like(self.importance_scores, dtype=torch.bool)
        
        if min_importance is not None:
            valid_mask &= (self.importance_scores >= min_importance)
            
        if max_age is not None:
            current_time = self.get_current_timestamp()
            valid_mask &= (current_time - self.feature_timestamps <= max_age)
            
        # Get valid features
        valid_features = self.feature_bank[valid_mask]
        valid_importance = self.importance_scores[valid_mask]
        
        if len(valid_features) == 0:
            return {"features": torch.empty(0, self.hidden_dim)}
            
        # Similarity-based retrieval
        if query is not None:
            similarities = torch.matmul(query, valid_features.t())
            _, indices = similarities.topk(min(top_k or len(valid_features), len(valid_features)))
            retrieved_features = valid_features[indices]
            retrieved_importance = valid_importance[indices]
        else:
            # Importance-based retrieval
            if top_k is not None:
                _, indices = valid_importance.topk(min(top_k, len(valid_importance)))
                retrieved_features = valid_features[indices]
                retrieved_importance = valid_importance[indices]
            else:
                retrieved_features = valid_features
                retrieved_importance = valid_importance
                
        return {
            "features": retrieved_features,
            "importance_scores": retrieved_importance,
            "num_retrieved": len(retrieved_features)
        }
        
    def get_current_timestamp(self) -> int:
        """Get current timestamp for feature age tracking"""
        return torch.tensor(self.forward_count).long()
        
    def reset(self) -> None:
        """Reset memory bank state"""
        self.feature_bank.zero_()
        self.importance_scores.zero_()
        self.feature_timestamps.zero_()
        self.bank_ptr.zero_()