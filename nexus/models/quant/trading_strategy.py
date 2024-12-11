import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule

class TradingStrategy(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config
        self._validate_config(config)
        
        # Core dimensions
        self.hidden_dim = config["hidden_dim"]
        self.num_assets = config["num_assets"]
        
        # Market state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(config["input_dim"], self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Portfolio optimizer
        self.portfolio_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim * 2, self.num_assets)
        )
        
        # Risk assessment
        self.risk_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Trading decision bank
        self.register_buffer(
            "decision_bank",
            torch.zeros(config.get("bank_size", 10000), self.hidden_dim)
        )
        self.register_buffer("performance_metrics", torch.zeros(config.get("bank_size", 10000)))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = ["hidden_dim", "input_dim", "num_assets"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
    def update_decision_bank(self, decisions: torch.Tensor, metrics: torch.Tensor):
        """Update decision bank following EnhancedReID pattern"""
        batch_size = decisions.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.decision_bank.size(0):
            ptr = 0
            
        self.decision_bank[ptr:ptr + batch_size] = decisions.detach()
        self.performance_metrics[ptr:ptr + batch_size] = metrics.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.decision_bank.size(0)
        
    def forward(
        self,
        market_state: torch.Tensor,
        factor_data: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode market state
        encoded_state = self.state_encoder(market_state)
        
        # Generate portfolio weights
        portfolio_weights = torch.softmax(self.portfolio_head(encoded_state), dim=-1)
        
        # Assess risk
        risk_score = self.risk_head(encoded_state)
        
        # Calculate performance metric
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_weights, market_state)
        
        # Update decision bank
        self.update_decision_bank(encoded_state, sharpe_ratio)
        
        return {
            "portfolio_weights": portfolio_weights,
            "risk_score": risk_score,
            "encoded_state": encoded_state,
            "sharpe_ratio": sharpe_ratio
        }
        
    def _calculate_sharpe_ratio(self, weights: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        portfolio_returns = (weights * returns).sum(dim=-1)
        return (portfolio_returns.mean() / (portfolio_returns.std() + 1e-6)) 