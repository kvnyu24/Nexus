import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule

class BSM(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self._validate_config(config)
        self.hidden_dim = config["hidden_dim"]
        
        # Market condition encoder
        self.market_encoder = nn.Sequential(
            nn.Linear(config["market_dim"], self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Volatility prediction network
        self.volatility_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim, 1),
            nn.Softplus()  # Ensure positive volatility
        )
        
        # Risk-free rate prediction
        self.rate_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()  # Bound rates between 0 and 1
        )
        
        # Option price adjustment network
        self.price_adjuster = nn.Sequential(
            nn.Linear(self.hidden_dim + 4, self.hidden_dim),  # +4 for BSM parameters
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Feature bank for market conditions
        self.register_buffer(
            "market_bank",
            torch.zeros(config.get("bank_size", 10000), self.hidden_dim)
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = ["hidden_dim", "market_dim"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
    def update_market_bank(self, features: torch.Tensor):
        """Update market condition bank following EnhancedReID pattern"""
        batch_size = features.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.market_bank.size(0):
            ptr = 0
            
        self.market_bank[ptr:ptr + batch_size] = features.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.market_bank.size(0)
        
    def black_scholes(
        self,
        S: torch.Tensor,  # Spot price
        K: torch.Tensor,  # Strike price
        T: torch.Tensor,  # Time to maturity
        sigma: torch.Tensor,  # Volatility
        r: torch.Tensor,  # Risk-free rate
        option_type: str = "call"
    ) -> torch.Tensor:
        """Calculate BSM option price"""
        sqrt_T = torch.sqrt(T)
        d1 = (torch.log(S/K) + (r + sigma**2/2)*T) / (sigma*sqrt_T)
        d2 = d1 - sigma*sqrt_T
        
        if option_type == "call":
            return S * torch.normal_cdf(d1) - K * torch.exp(-r*T) * torch.normal_cdf(d2)
        else:
            return K * torch.exp(-r*T) * torch.normal_cdf(-d2) - S * torch.normal_cdf(-d1)
            
    def forward(
        self,
        market_data: torch.Tensor,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        option_type: str = "call"
    ) -> Dict[str, torch.Tensor]:
        # Encode market conditions
        market_features = self.market_encoder(market_data)
        
        # Predict volatility and rate
        sigma = self.volatility_net(market_features)
        r = self.rate_net(market_features)
        
        # Calculate base BSM price
        base_price = self.black_scholes(S, K, T, sigma, r, option_type)
        
        # Generate price adjustment
        adjustment_input = torch.cat([
            market_features,
            S.unsqueeze(-1),
            K.unsqueeze(-1),
            T.unsqueeze(-1),
            base_price
        ], dim=-1)
        
        price_adjustment = self.price_adjuster(adjustment_input)
        final_price = base_price + price_adjustment
        
        # Update market bank
        self.update_market_bank(market_features)
        
        return {
            "option_price": final_price,
            "implied_vol": sigma,
            "risk_free_rate": r,
            "base_price": base_price,
            "adjustment": price_adjustment,
            "market_features": market_features
        } 