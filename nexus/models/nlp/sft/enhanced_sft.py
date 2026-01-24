import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Tuple
from ....core.base import NexusModule
from ....core.mixins import FeatureBankMixin
from .config_validator import SFTConfigValidator
from ..rag import EnhancedRAGModule
from ..hallucination_reducer import HallucinationReducer
from ....training.losses import EnhancedSFTLoss

class SFTModule(NexusModule, FeatureBankMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config with strict type checking and enhanced validation
        SFTConfigValidator.validate_config(config, strict=True)
        
        # Core architecture parameters with comprehensive validation
        self.hidden_size = config["hidden_size"]
        if not (64 <= self.hidden_size <= 8192 and self.hidden_size % 64 == 0):
            raise ValueError("hidden_size must be between 64 and 8192 and divisible by 64")
            
        self.num_heads = config["num_heads"]
        if not (1 <= self.num_heads <= 128 and self.hidden_size % self.num_heads == 0):
            raise ValueError("num_heads must be between 1 and 128 and divide hidden_size evenly")
            
        # Enhanced components with configurable parameters and validation
        self.use_rag = config.get("use_rag", True)
        if self.use_rag:
            self.rag_module = EnhancedRAGModule(config)
            
        self.use_hallucination_reduction = config.get("use_hallucination_reduction", True)
        if self.use_hallucination_reduction:
            self.hallucination_reducer = HallucinationReducer(config)
        
        # Dynamic feature bank with adaptive sizing and validation using mixin
        bank_size = config.get("bank_size", 10000)
        if not (100 <= bank_size <= 1000000):
            raise ValueError("bank_size must be between 100 and 1,000,000")

        self.register_feature_bank("instruction", bank_size, self.hidden_size)
        self.register_buffer("bank_mask", torch.zeros(bank_size, dtype=torch.bool))
        self.register_buffer("bank_quality_scores", torch.zeros(bank_size))
        self.register_buffer("bank_timestamps", torch.zeros(bank_size))
        
        # Enhanced quality assessment with uncertainty and confidence
        quality_hidden = config.get("quality_hidden_size", self.hidden_size // 2)
        if quality_hidden < 32:
            raise ValueError("quality_hidden_size must be at least 32")
            
        dropout = config.get("dropout", 0.1)
        if not (0.0 <= dropout <= 0.5):
            raise ValueError("dropout must be between 0.0 and 0.5")
        
        self.quality_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, quality_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(quality_hidden, quality_hidden // 2),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(quality_hidden // 2, 3),  # [quality, uncertainty, confidence]
            nn.Sigmoid()
        )
        
        # Improved weight initialization with validated gain
        self._init_gain = config.get("init_gain", 0.02)
        if not (0.001 <= self._init_gain <= 0.1):
            raise ValueError("init_gain must be between 0.001 and 0.1")
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Enhanced weight initialization with configurable gain and stability checks"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Use truncated normal with stability bounds
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=self._init_gain,
                a=-2*self._init_gain,
                b=2*self._init_gain
            )
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
            # Add weight norm clipping for stability
            with torch.no_grad():
                torch.clamp_(module.weight, min=-1.0, max=1.0)
                
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
            
    def update_instruction_bank(
        self, 
        features: torch.Tensor,
        quality_threshold: float = 0.8,
        uncertainty_threshold: float = 0.3,
        confidence_threshold: float = 0.7,
        max_age: int = 10000
    ):
        """Enhanced instruction bank update with quality filtering and temporal management"""
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
            
        batch_size = features.size(0)
        if features.size(1) != self.hidden_size:
            raise ValueError(f"Feature dimension {features.size(1)} does not match hidden_size {self.hidden_size}")
            
        # Get quality metrics
        quality_metrics = self.quality_head(features)
        quality_scores = quality_metrics[:, 0]
        uncertainty_scores = quality_metrics[:, 1]
        confidence_scores = quality_metrics[:, 2]
        
        # Enhanced filtering with multiple criteria
        quality_mask = (
            (quality_scores > quality_threshold) &
            (uncertainty_scores < uncertainty_threshold) &
            (confidence_scores > confidence_threshold)
        )
        features = features[quality_mask]
        quality_scores = quality_scores[quality_mask]
        batch_size = features.size(0)
        
        if batch_size == 0:
            return
            
        ptr = int(self.instruction_ptr)
        bank_size = self.instruction_bank.size(0)
        current_time = self.bank_timestamps.max() + 1
        
        # Remove old entries
        old_mask = (current_time - self.bank_timestamps) > max_age
        self.bank_mask[old_mask] = False
        
        # Handle wraparound with proper indexing and timestamp updates
        if ptr + batch_size > bank_size:
            first_part = bank_size - ptr
            second_part = batch_size - first_part
            
            # Update features and metadata
            self.instruction_bank[ptr:] = features[:first_part]
            self.instruction_bank[:second_part] = features[first_part:]
            self.bank_mask[ptr:] = True
            self.bank_mask[:second_part] = True
            self.bank_quality_scores[ptr:] = quality_scores[:first_part]
            self.bank_quality_scores[:second_part] = quality_scores[first_part:]
            self.bank_timestamps[ptr:] = current_time
            self.bank_timestamps[:second_part] = current_time
        else:
            # Update features and metadata
            self.instruction_bank[ptr:ptr + batch_size] = features
            self.bank_mask[ptr:ptr + batch_size] = True
            self.bank_quality_scores[ptr:ptr + batch_size] = quality_scores
            self.bank_timestamps[ptr:ptr + batch_size] = current_time
            
        self.instruction_ptr[0] = (ptr + batch_size) % bank_size