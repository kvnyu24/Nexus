import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from ...core.base import NexusModule

class ARIMA(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self._validate_config(config)
        self.hidden_dim = config["hidden_dim"]
        self.seq_length = config["seq_length"]
        self.forecast_steps = config.get("forecast_steps", 1)
        self.bank_size = config.get("bank_size", 10000)
        
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.seq_length <= 0:
            raise ValueError("seq_length must be positive")
        if self.forecast_steps <= 0:
            raise ValueError("forecast_steps must be positive")
        if self.bank_size <= 0:
            raise ValueError("bank_size must be positive")
            
        # Time series encoder with skip connection
        self.series_encoder = nn.Sequential(
            nn.Linear(self.seq_length, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # ARIMA parameter prediction with constraints
        self.order_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 3),  # p, d, q
            nn.Softplus()  # Ensure positive orders
        )
        
        # Residual prediction network with uncertainty
        self.residual_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.forecast_steps * 2)  # Mean and variance
        )
        
        # Seasonal pattern detection with attention
        self.seasonal_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()  # Seasonal strength
        )
        
        # Feature bank with normalization
        self.register_buffer(
            "series_bank",
            torch.zeros(self.bank_size, self.hidden_dim)
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("bank_filled", torch.zeros(1, dtype=torch.bool))
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = ["hidden_dim", "seq_length"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        for key, value in config.items():
            if not isinstance(value, (int, float, str, bool, dict, list)):
                raise ValueError(f"Invalid config value type for {key}")
                
    def update_series_bank(self, features: torch.Tensor):
        """Update time series bank with normalization"""
        if not torch.isfinite(features).all():
            return  # Skip update if features contain NaN/inf
            
        batch_size = features.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.series_bank.size(0):
            ptr = 0
            self.bank_filled[0] = True
            
        # Normalize features
        features = torch.nn.functional.normalize(features, dim=-1)
        self.series_bank[ptr:ptr + batch_size] = features.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.series_bank.size(0)
        
    def differencing(self, x: torch.Tensor, d: int) -> torch.Tensor:
        """Apply d-th order differencing with bounds checking"""
        if not torch.isfinite(x).all():
            raise ValueError("Input contains NaN/inf values")
            
        d = max(0, min(int(d), x.size(1) - 1))  # Bound d to valid range
        diff = x
        
        for _ in range(d):
            if diff.size(1) <= 1:
                break
            diff = diff[:, 1:] - diff[:, :-1]
            
        return diff
        
    def forward(
        self,
        time_series: torch.Tensor,
        exog_variables: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if not torch.isfinite(time_series).all():
            raise ValueError("Input contains NaN/inf values")
            
        # Encode time series with residual connection
        series_features = self.series_encoder(time_series)
        series_features = series_features + torch.nn.functional.linear(
            time_series, 
            torch.nn.Parameter(torch.eye(self.hidden_dim)[:, :self.seq_length])
        )
        
        # Predict ARIMA orders with bounds
        orders = self.order_net(series_features)
        p, d, q = orders.chunk(3, dim=-1)
        
        # Apply differencing with error handling
        try:
            diff_series = self.differencing(time_series, d.mean().item())
        except Exception as e:
            diff_series = time_series
            
        # Predict residuals with uncertainty
        residual_params = self.residual_net(series_features)
        residual_mean, residual_var = residual_params.chunk(2, dim=-1)
        residual_var = torch.nn.functional.softplus(residual_var) + 1e-6
        
        # Detect seasonality with attention weights
        seasonal_strength = self.seasonal_net(series_features)
        
        # Update series bank with error handling
        try:
            self.update_series_bank(series_features)
        except Exception:
            pass  # Fail silently on bank updates
        
        return {
            "predicted_values": residual_mean,
            "prediction_variance": residual_var,
            "arima_orders": orders,
            "seasonal_strength": seasonal_strength,
            "series_features": series_features,
            "differenced_series": diff_series
        }