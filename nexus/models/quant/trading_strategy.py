import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from ...core.base import NexusModule

class TradingStrategy(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config
        self._validate_config(config)
        
        # Core dimensions
        self.hidden_dim = config["hidden_dim"]
        self.num_assets = config["num_assets"]
        self.sequence_length = config.get("sequence_length", 252)  # Default 1 year
        self.bank_size = config.get("bank_size", 10000)
        self.min_position = config.get("min_position", -1.0)
        self.max_position = config.get("max_position", 1.0)
        
        # Market state encoder with temporal attention
        self.state_encoder = nn.Sequential(
            nn.Linear(config["input_dim"], self.hidden_dim),
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
        
        # Portfolio optimizer with uncertainty
        self.portfolio_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim * 2, self.num_assets * 2)  # Mean and variance
        )
        
        # Risk assessment with multiple metrics
        self.risk_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim, 4),  # VaR, CVaR, Volatility, Drawdown
            nn.Sigmoid()
        )
        
        # Transaction cost model
        self.cost_model = nn.Sequential(
            nn.Linear(self.hidden_dim + self.num_assets, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.num_assets)
        )
        
        # Trading decision bank with importance weighting
        self.register_buffer(
            "decision_bank",
            torch.zeros(self.bank_size, self.hidden_dim)
        )
        self.register_buffer(
            "performance_metrics",
            torch.zeros(self.bank_size, 4)  # Sharpe, Sortino, Calmar, Info Ratio
        )
        self.register_buffer(
            "decision_importance",
            torch.zeros(self.bank_size)
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("bank_is_full", torch.zeros(1, dtype=torch.bool))
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = ["hidden_dim", "input_dim", "num_assets"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
        if config["hidden_dim"] <= 0:
            raise ValueError("hidden_dim must be positive")
        if config["num_assets"] <= 0:
            raise ValueError("num_assets must be positive")
        if config["input_dim"] <= 0:
            raise ValueError("input_dim must be positive")
                
    def update_decision_bank(
        self,
        decisions: torch.Tensor,
        metrics: torch.Tensor,
        importance: torch.Tensor
    ):
        """Update decision bank with importance weighting"""
        if not torch.isfinite(decisions).all():
            return  # Skip update if decisions contain NaN/inf
            
        batch_size = decisions.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.decision_bank.size(0):
            ptr = 0
            self.bank_is_full[0] = True
            
        # Normalize decisions
        decisions = torch.nn.functional.normalize(decisions, dim=-1)
        
        self.decision_bank[ptr:ptr + batch_size] = decisions.detach()
        self.performance_metrics[ptr:ptr + batch_size] = metrics.detach()
        self.decision_importance[ptr:ptr + batch_size] = importance.detach().squeeze(-1)
        self.bank_ptr[0] = (ptr + batch_size) % self.bank_size
        
    def forward(
        self,
        market_state: torch.Tensor,
        factor_data: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        current_positions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if not torch.isfinite(market_state).all():
            raise ValueError("Input contains NaN/inf values")
            
        # Encode market state with temporal attention
        encoded_state = self.state_encoder(market_state)
        
        if len(encoded_state.shape) == 3:
            encoded_state, _ = self.temporal_attention(
                encoded_state, encoded_state, encoded_state,
                key_padding_mask=attention_mask
            )
            
        # Generate portfolio weights with uncertainty
        portfolio_params = self.portfolio_head(encoded_state)
        weight_mean, weight_logvar = portfolio_params.chunk(2, dim=-1)
        weight_std = torch.exp(0.5 * weight_logvar)
        
        # Sample weights using reparameterization
        if self.training:
            epsilon = torch.randn_like(weight_std)
            portfolio_weights = weight_mean + weight_std * epsilon
        else:
            portfolio_weights = weight_mean
            
        # Apply position limits
        portfolio_weights = torch.clamp(
            portfolio_weights,
            min=self.min_position,
            max=self.max_position
        )
        
        # Normalize weights to sum to 1
        portfolio_weights = torch.softmax(portfolio_weights, dim=-1)
        
        # Calculate transaction costs
        if current_positions is not None:
            position_changes = portfolio_weights - current_positions
            cost_features = torch.cat([encoded_state, position_changes], dim=-1)
            transaction_costs = torch.sigmoid(self.cost_model(cost_features))
        else:
            transaction_costs = torch.zeros_like(portfolio_weights)
            
        # Assess multiple risk metrics
        risk_metrics = self.risk_head(encoded_state)
        var, cvar, vol, drawdown = risk_metrics.chunk(4, dim=-1)
        
        # Calculate performance metrics
        perf_metrics = self._calculate_performance_metrics(portfolio_weights, market_state)
        sharpe, sortino, calmar, info_ratio = perf_metrics.chunk(4, dim=-1)
        
        # Calculate decision importance
        importance = sharpe * (1 - vol)  # High Sharpe, low vol = important decision
        
        # Update decision bank
        self.update_decision_bank(encoded_state, perf_metrics, importance)
        
        return {
            "portfolio_weights": portfolio_weights,
            "weight_uncertainty": weight_std,
            "transaction_costs": transaction_costs,
            "risk_metrics": {
                "value_at_risk": var,
                "cond_value_at_risk": cvar,
                "volatility": vol,
                "max_drawdown": drawdown
            },
            "performance_metrics": {
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "calmar_ratio": calmar,
                "info_ratio": info_ratio
            },
            "encoded_state": encoded_state,
            "decision_importance": importance,
            "bank_usage": self.bank_is_full
        }
        
    def _calculate_performance_metrics(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """Calculate multiple portfolio performance metrics"""
        portfolio_returns = (weights * returns).sum(dim=-1)
        
        # Sharpe ratio
        sharpe = portfolio_returns.mean() / (portfolio_returns.std() + 1e-6)
        
        # Sortino ratio
        downside_returns = torch.minimum(portfolio_returns, torch.zeros_like(portfolio_returns))
        sortino = portfolio_returns.mean() / (downside_returns.std() + 1e-6)
        
        # Calmar ratio
        cumulative_returns = torch.cumsum(portfolio_returns, dim=0)
        drawdowns = cumulative_returns.max() - cumulative_returns
        calmar = portfolio_returns.mean() / (drawdowns.max() + 1e-6)
        
        # Information ratio vs equal weight
        equal_weight = torch.ones_like(weights) / weights.size(-1)
        active_returns = portfolio_returns - (equal_weight * returns).sum(dim=-1)
        info_ratio = active_returns.mean() / (active_returns.std() + 1e-6)
        
        return torch.stack([sharpe, sortino, calmar, info_ratio], dim=-1)