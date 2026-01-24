import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from ...core.base import NexusModule
from ...core.mixins import ConfigValidatorMixin, FeatureBankMixin

class FactorModel(NexusModule, ConfigValidatorMixin, FeatureBankMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Validate config using mixin
        self.validate_config(config, required_keys=["hidden_dim", "input_dim", "num_factors"])
        self.validate_positive(config["hidden_dim"], "hidden_dim")
        self.validate_positive(config["input_dim"], "input_dim")
        self.validate_positive(config["num_factors"], "num_factors")

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

        # Feature banks using mixin
        self.register_feature_bank("factor", self.bank_size, self.num_factors)
        self.register_feature_bank("importance", self.bank_size, 1)
        self.register_feature_bank("uncertainty", self.bank_size, 1)
        
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
        
        # Update factor banks using mixin (normalizing factors first)
        normalized_factors = torch.nn.functional.normalize(factor_loadings, dim=-1)
        self.update_feature_bank("factor", normalized_factors)
        self.update_feature_bank("importance", factor_importance)
        self.update_feature_bank("uncertainty", factor_uncertainty)
        
        return {
            "factor_loadings": factor_loadings,
            "factor_uncertainty": factor_std,
            "risk_metrics": risk_mean,
            "risk_uncertainty": risk_std,
            "features": features,
            "factor_importance": factor_importance,
            "factor_correlations": factor_corr,
            "factor_bank_usage": self.is_bank_full("factor")
        }