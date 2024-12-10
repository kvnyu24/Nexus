from typing import Dict, Any, List, Optional
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ..visualization.base import BaseVisualizer

class RNNVisualizer(BaseVisualizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_cmap = config.get("hidden_cmap", "viridis")
        self.gate_cmap = config.get("gate_cmap", "RdYlBu")
        self.max_timesteps = config.get("max_timesteps", 50)
        self.max_units = config.get("max_units", 32)
        
    def visualize_hidden_states(
        self,
        hidden_states: torch.Tensor,
        timestamps: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Visualize RNN/LSTM hidden state evolution"""
        # Input validation
        if not isinstance(hidden_states, torch.Tensor):
            raise ValueError("hidden_states must be a torch.Tensor")
            
        # Handle batched input
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[0]
            
        # Truncate if needed
        if hidden_states.size(0) > self.max_timesteps:
            hidden_states = hidden_states[:self.max_timesteps]
            
        if hidden_states.size(1) > self.max_units:
            hidden_states = hidden_states[:, :self.max_units]
            
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            hidden_states.cpu(),
            cmap=self.hidden_cmap,
            xticklabels=range(hidden_states.size(1)),
            yticklabels=timestamps[:self.max_timesteps] if timestamps else range(hidden_states.size(0)),
            ax=ax
        )
        
        ax.set_xlabel('Hidden Units')
        ax.set_ylabel('Timesteps')
        ax.set_title('Hidden State Evolution')
        
        plt.tight_layout()
        self.save_figure(fig, "hidden_states.png")
        plt.close()
        
        return {"figure": fig}
        
    def visualize_gates(
        self,
        gate_values: Dict[str, torch.Tensor],
        timestep: int = -1
    ) -> Dict[str, Any]:
        """Visualize LSTM gate activations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for idx, (gate_name, values) in enumerate(gate_values.items()):
            # Take specified timestep
            gate_state = values[timestep].cpu() if values.dim() > 1 else values.cpu()
            
            sns.heatmap(
                gate_state.unsqueeze(0),
                cmap=self.gate_cmap,
                ax=axes[idx],
                xticklabels=False,
                yticklabels=False
            )
            axes[idx].set_title(f'{gate_name} Gate')
            
        plt.tight_layout()
        self.save_figure(fig, f"gates_t{timestep}.png")
        plt.close()
        
        return {"figure": fig} 