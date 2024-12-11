import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
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
        self.bank_size = config.get("bank_size", 10000)
        
        # Feature extraction with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Linear(config["input_dim"], self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Factor discovery network with uncertainty
        self.factor_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim * 2, self.num_factors * 2)  # Mean and variance
        )
        
        # Risk model with attention
        self.risk_model = nn.Sequential(
            nn.Linear(self.hidden_dim + self.num_factors, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 2)  # Risk mean and variance
        )
        
        # Factor correlation network
        self.correlation_net = nn.Sequential(
            nn.Linear(self.num_factors, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_factors * self.num_factors)
        )
        
        # Temporal attention for time series
        self.temporal_attention = nn.MultiheadAttention(
            self.hidden_dim,
            num_heads=4,
            dropout=config.get("dropout", 0.1)
        )
        
        # Feature banks
        self.register_buffer(
            "factor_bank",
            torch.zeros(self.bank_size, self.num_factors)
        )
        self.register_buffer(
            "factor_importance",
            torch.zeros(self.bank_size)
        )
        self.register_buffer(
            "factor_uncertainty",
            torch.zeros(self.bank_size)
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("bank_is_full", torch.zeros(1, dtype=torch.bool))
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = ["hidden_dim", "input_dim", "num_factors"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
        if config["hidden_dim"] <= 0:
            raise ValueError("hidden_dim must be positive")
        if config["num_factors"] <= 0:
            raise ValueError("num_factors must be positive")
        if config["input_dim"] <= 0:
            raise ValueError("input_dim must be positive")
                
    def update_factor_bank(
        self,
        factors: torch.Tensor,
        importance: torch.Tensor,
        uncertainty: torch.Tensor
    ):
        """Update factor bank with importance weighting and uncertainty"""
        if not torch.isfinite(factors).all():
            return  # Skip update if factors contain NaN/inf
            
        batch_size = factors.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.factor_bank.size(0):
            ptr = 0
            self.bank_is_full[0] = True
            
        # Normalize factors
        factors = torch.nn.functional.normalize(factors, dim=-1)
        
        self.factor_bank[ptr:ptr + batch_size] = factors.detach()
        self.factor_importance[ptr:ptr + batch_size] = importance.detach().squeeze(-1)
        self.factor_uncertainty[ptr:ptr + batch_size] = uncertainty.detach().squeeze(-1)
        self.bank_ptr[0] = (ptr + batch_size) % self.bank_size
        
    def forward(
        self,
        market_data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if not torch.isfinite(market_data).all():
            raise ValueError("Input contains NaN/inf values")
            
        # Extract features with residual connection
        features = self.feature_extractor(market_data)
        features = features + torch.nn.functional.linear(
            market_data,
            torch.nn.Parameter(torch.eye(self.hidden_dim)[:, :market_data.size(-1)])
        )
        
        # Apply temporal attention if sequence
        if len(features.shape) == 3:
            features, _ = self.temporal_attention(
                features, features, features,
                key_padding_mask=attention_mask
            )
            
        # Discover factors with uncertainty
        factor_params = self.factor_network(features)
        factor_mean, factor_logvar = factor_params.chunk(2, dim=-1)
        factor_std = torch.exp(0.5 * factor_logvar)
        
        # Sample factors using reparameterization
        if self.training:
            epsilon = torch.randn_like(factor_std)
            factor_loadings = factor_mean + factor_std * epsilon
        else:
            factor_loadings = factor_mean
            
        # Calculate factor correlations
        factor_corr = self.correlation_net(factor_loadings)
        factor_corr = factor_corr.view(-1, self.num_factors, self.num_factors)
        factor_corr = 0.5 * (factor_corr + factor_corr.transpose(-2, -1))  # Symmetrize
        
        # Calculate risk metrics with uncertainty
        combined = torch.cat([features, factor_loadings], dim=-1)
        risk_params = self.risk_model(combined)
        risk_mean, risk_logvar = risk_params.chunk(2, dim=-1)
        risk_std = torch.exp(0.5 * risk_logvar)
        
        # Calculate factor importance and uncertainty
        factor_importance = torch.norm(factor_loadings, p=2, dim=-1, keepdim=True)
        factor_uncertainty = torch.mean(factor_std, dim=-1, keepdim=True)
        
        # Update factor bank with error handling
        try:
            self.update_factor_bank(factor_loadings, factor_importance, factor_uncertainty)
        except Exception:
            pass  # Fail silently on bank updates
        
        return {
            "factor_loadings": factor_loadings,
            "factor_uncertainty": factor_std,
            "risk_metrics": risk_mean,
            "risk_uncertainty": risk_std,
            "features": features,
            "factor_importance": factor_importance,
            "factor_correlations": factor_corr,
            "factor_bank_usage": self.bank_is_full
        }