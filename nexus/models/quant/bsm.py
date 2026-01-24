import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule
from ...core.mixins import ConfigValidatorMixin, FeatureBankMixin
from nexus.utils.logging import Logger

class BSM(NexusModule, ConfigValidatorMixin, FeatureBankMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = Logger(self.__class__.__name__)

        # Validate config using mixin
        self.validate_config(config, required_keys=["hidden_dim", "market_dim"])
        self.validate_positive(config["hidden_dim"], "hidden_dim")
        self.validate_positive(config["market_dim"], "market_dim")

        self.hidden_dim = config["hidden_dim"]
        self.bank_size = config.get("bank_size", 10000)
        self.min_vol = config.get("min_vol", 0.001)
        self.max_vol = config.get("max_vol", 2.0)

        # Market condition encoder with temporal attention
        self.market_encoder = nn.Sequential(
            nn.Linear(config["market_dim"], self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # Temporal attention for time series
        self.temporal_attention = nn.MultiheadAttention(
            self.hidden_dim,
            num_heads=4,
            dropout=config.get("dropout", 0.1)
        )

        # Volatility prediction network with uncertainty
        self.volatility_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim, 2),  # Mean and variance
        )

        # Risk-free rate prediction with uncertainty
        self.rate_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 2),  # Mean and variance
        )

        # Option price adjustment network with market impact
        self.price_adjuster = nn.Sequential(
            nn.Linear(self.hidden_dim + 6, self.hidden_dim),  # +6 for BSM parameters + Greeks
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 2)  # Price adjustment mean and variance
        )

        # Greeks calculation network
        self.greeks_net = nn.Sequential(
            nn.Linear(self.hidden_dim + 5, self.hidden_dim),  # +5 for BSM parameters
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 4)  # Delta, Gamma, Theta, Vega
        )

        # Feature bank for market conditions using mixin
        self.register_feature_bank("market", self.bank_size, self.hidden_dim)
        self.register_feature_bank("vol", self.bank_size, 1)
        
    def black_scholes(
        self,
        S: torch.Tensor,  # Spot price
        K: torch.Tensor,  # Strike price
        T: torch.Tensor,  # Time to maturity
        sigma: torch.Tensor,  # Volatility
        r: torch.Tensor,  # Risk-free rate
        option_type: str = "call"
    ) -> torch.Tensor:
        """Calculate BSM option price with bounds checking"""
        # Input validation
        if not (torch.isfinite(S).all() and torch.isfinite(K).all() and 
                torch.isfinite(T).all() and torch.isfinite(sigma).all() and
                torch.isfinite(r).all()):
            raise ValueError("Inputs contain NaN/inf values")
            
        # Ensure positive values
        S = torch.clamp(S, min=1e-6)
        K = torch.clamp(K, min=1e-6)
        T = torch.clamp(T, min=1e-6)
        sigma = torch.clamp(sigma, min=self.min_vol, max=self.max_vol)
        r = torch.clamp(r, min=0.0, max=1.0)
        
        sqrt_T = torch.sqrt(T)
        d1 = (torch.log(S/K) + (r + sigma**2/2)*T) / (sigma*sqrt_T)
        d2 = d1 - sigma*sqrt_T
        
        if option_type == "call":
            price = S * torch.normal_cdf(d1) - K * torch.exp(-r*T) * torch.normal_cdf(d2)
        else:
            price = K * torch.exp(-r*T) * torch.normal_cdf(-d2) - S * torch.normal_cdf(-d1)
            
        return torch.clamp(price, min=0.0)  # Ensure non-negative prices
            
    def forward(
        self,
        market_data: torch.Tensor,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        option_type: str = "call",
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Input validation
        if not torch.isfinite(market_data).all():
            raise ValueError("Market data contains NaN/inf values")
            
        # Encode market conditions with temporal attention
        market_features = self.market_encoder(market_data)
        if len(market_features.shape) == 3:
            market_features, _ = self.temporal_attention(
                market_features, market_features, market_features,
                key_padding_mask=attention_mask
            )
        
        # Predict volatility and rate with uncertainty
        vol_params = self.volatility_net(market_features)
        vol_mean, vol_logvar = vol_params.chunk(2, dim=-1)
        vol_std = torch.exp(0.5 * vol_logvar)
        sigma = torch.clamp(vol_mean, min=self.min_vol, max=self.max_vol)
        
        rate_params = self.rate_net(market_features)
        rate_mean, rate_logvar = rate_params.chunk(2, dim=-1)
        rate_std = torch.exp(0.5 * rate_logvar)
        r = torch.clamp(rate_mean, min=0.0, max=1.0)
        
        # Calculate base BSM price
        try:
            base_price = self.black_scholes(S, K, T, sigma, r, option_type)
        except Exception as e:
            # Fallback to minimum values on error
            base_price = torch.zeros_like(S)
            self.logger.error(f"BSM calculation failed: {str(e)}")
        
        # Calculate Greeks
        greeks_input = torch.cat([
            market_features,
            S.unsqueeze(-1),
            K.unsqueeze(-1),
            T.unsqueeze(-1),
            sigma,
            r
        ], dim=-1)
        greeks = self.greeks_net(greeks_input)
        delta, gamma, theta, vega = greeks.chunk(4, dim=-1)
        
        # Generate price adjustment with market impact
        adjustment_input = torch.cat([
            market_features,
            S.unsqueeze(-1),
            K.unsqueeze(-1),
            T.unsqueeze(-1),
            base_price,
            delta,
            vega
        ], dim=-1)
        
        adjustment_params = self.price_adjuster(adjustment_input)
        adj_mean, adj_logvar = adjustment_params.chunk(2, dim=-1)
        adj_std = torch.exp(0.5 * adj_logvar)
        
        # Final price with bounds
        final_price = torch.clamp(base_price + adj_mean, min=0.0)
        
        # Update market bank using mixin
        self.update_feature_bank("market", market_features)
        self.update_feature_bank("vol", sigma)
        
        return {
            "option_price": final_price,
            "price_uncertainty": adj_std,
            "implied_vol": sigma,
            "vol_uncertainty": vol_std,
            "risk_free_rate": r,
            "rate_uncertainty": rate_std,
            "base_price": base_price,
            "adjustment": adj_mean,
            "market_features": market_features,
            "greeks": {
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega
            },
            "bank_is_full": self.is_bank_full("market")
        }