import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule

class FactorModel(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config
        self._validate_config(config)
        
        # Core dimensions
        self.hidden_dim = config["hidden_dim"]
        self.num_factors = config["num_factors"]
        self.sequence_length = config.get("sequence_length", 252)  # Default to 1 year
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(config["input_dim"], self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Factor discovery network
        self.factor_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim * 2, self.num_factors)
        )
        
        # Risk model
        self.risk_model = nn.Sequential(
            nn.Linear(self.hidden_dim + self.num_factors, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Feature bank for discovered factors
        self.register_buffer(
            "factor_bank",
            torch.zeros(config.get("bank_size", 10000), self.num_factors)
        )
        self.register_buffer("factor_importance", torch.zeros(config.get("bank_size", 10000)))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = ["hidden_dim", "input_dim", "num_factors"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
    def update_factor_bank(self, factors: torch.Tensor, importance: torch.Tensor):
        """Update factor bank following EnhancedReID pattern"""
        batch_size = factors.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.factor_bank.size(0):
            ptr = 0
            
        self.factor_bank[ptr:ptr + batch_size] = factors.detach()
        self.factor_importance[ptr:ptr + batch_size] = importance.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.factor_bank.size(0)
        
    def forward(
        self,
        market_data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.feature_extractor(market_data)
        
        # Discover factors
        factor_loadings = self.factor_network(features)
        
        # Calculate risk metrics
        combined = torch.cat([features, factor_loadings], dim=-1)
        risk_metrics = self.risk_model(combined)
        
        # Calculate factor importance
        factor_importance = torch.norm(factor_loadings, p=2, dim=-1, keepdim=True)
        
        # Update factor bank
        self.update_factor_bank(factor_loadings, factor_importance)
        
        return {
            "factor_loadings": factor_loadings,
            "risk_metrics": risk_metrics,
            "features": features,
            "factor_importance": factor_importance
        } 