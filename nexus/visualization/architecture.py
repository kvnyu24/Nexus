from typing import Dict, Any
import torch.nn as nn
import graphviz
from ..core import NexusModule
from ..visualization.base import BaseVisualizer

class ArchitectureVisualizer(BaseVisualizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.node_color = config.get("node_color", "#2ecc71")
        self.edge_color = config.get("edge_color", "#3498db")
        self.font_size = config.get("font_size", 10)
        
    def visualize_model(
        self,
        model: NexusModule,
        filename: str = "model_architecture"
    ) -> Dict[str, Any]:
        """Visualize model architecture as a graph"""
        try:
            import graphviz
        except ImportError:
            raise ImportError("graphviz package is required for architecture visualization")
            
        # Create directed graph
        dot = graphviz.Digraph(
            comment='Model Architecture',
            node_attr={'color': self.node_color, 'style': 'filled', 'fontsize': str(self.font_size)}
        )
        
        # Add nodes and edges
        added_nodes = set()
        
        def add_module(module: nn.Module, parent_name: str = ""):
            for name, child in module.named_children():
                full_name = f"{parent_name}/{name}" if parent_name else name
                
                # Add node if not already added
                if full_name not in added_nodes:
                    dot.node(full_name, name)
                    added_nodes.add(full_name)
                
                # Add edge from parent
                if parent_name:
                    dot.edge(parent_name, full_name, color=self.edge_color)
                    
                # Recursively add children
                add_module(child, full_name)
        
        # Build graph
        add_module(model)
        
        # Save visualization
        save_path = self.output_dir / f"{filename}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dot.render(str(save_path), view=False)
        
        return {"graph": dot} 