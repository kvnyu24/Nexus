from typing import Dict, Any, Optional
import torch
from ..visualization.base import BaseVisualizer

class GraphVisualizer(BaseVisualizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.node_size = config.get("node_size", 800)
        self.edge_width = config.get("edge_width", 2)
        self.node_color = config.get("node_color", "#2ecc71")
        self.edge_color = config.get("edge_color", "#3498db")
        self.alpha = config.get("alpha", 0.7)
        
    def visualize_graph(
        self,
        edge_index: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Visualize graph structure with optional attention weights"""
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # Create networkx graph
        G = nx.Graph()
        edge_list = edge_index.t().cpu().numpy()
        G.add_edges_from(edge_list)
        
        # Set up layout
        pos = nx.spring_layout(G)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=self.node_color,
            node_size=self.node_size,
            alpha=self.alpha,
            ax=ax
        )
        
        # Draw edges with optional attention weights
        if attention_weights is not None:
            weights = attention_weights.cpu().numpy()
            edges = nx.draw_networkx_edges(
                G, pos,
                width=[w * self.edge_width for w in weights],
                edge_color=self.edge_color,
                alpha=self.alpha,
                ax=ax
            )
        else:
            edges = nx.draw_networkx_edges(
                G, pos,
                width=self.edge_width,
                edge_color=self.edge_color,
                alpha=self.alpha,
                ax=ax
            )
            
        plt.axis('off')
        
        # Save visualization
        self.save_figure(fig, "graph_structure.png")
        plt.close()
        
        return {"figure": fig, "graph": G} 