import graphviz
from typing import Optional, Dict, Any
from ..models.search.mcts_node import EnhancedMCTSNode
from ..models.search.enhanced_mcts import EnhancedMCTS
from ..visualization.base import BaseVisualizer

class MCTSVisualizer(BaseVisualizer):
    """Visualizer for Monte Carlo Tree Search trees using graphviz"""
    
    def __init__(self, **kwargs):
        """Initialize the MCTS visualizer
        
        Args:
            **kwargs: Additional arguments passed to graphviz.Digraph
        """
        super().__init__()
        self.dot = graphviz.Digraph(comment='MCTS Tree', **kwargs)
        self.dot.attr(rankdir='TB')
        
    def add_node(self, 
                 node: EnhancedMCTSNode, 
                 node_id: str, 
                 parent_id: Optional[str] = None,
                 node_attrs: Optional[Dict[str, Any]] = None):
        """Add a node to the visualization
        
        Args:
            node: The MCTS node to visualize
            node_id: Unique identifier for this node
            parent_id: ID of parent node if it exists
            node_attrs: Optional dict of node attributes to pass to graphviz
        """
        try:
            # Create node label with error handling
            value = node.value() if hasattr(node, 'value') else 0.0
            visits = node.visit_count if hasattr(node, 'visit_count') else 0
            uncertainty = node.uncertainty.item() if hasattr(node, 'uncertainty') else 0.0
            
            label = f"V: {value:.3f}\nN: {visits}\nU: {uncertainty:.3f}"
            
            # Add node with attributes
            attrs = {'label': label}
            if node_attrs:
                attrs.update(node_attrs)
            self.dot.node(node_id, **attrs)
            
            # Add edge from parent if exists
            if parent_id is not None:
                self.dot.edge(parent_id, node_id)
                
        except Exception as e:
            print(f"Error adding node {node_id}: {str(e)}")
            
    def visualize_tree(self, 
                      root: EnhancedMCTSNode, 
                      path: str,
                      view: bool = True,
                      cleanup: bool = True):
        """Generate visualization of MCTS tree
        
        Args:
            root: Root node of the MCTS tree
            path: Output path for rendered visualization
            view: Whether to open the rendered visualization
            cleanup: Whether to remove DOT source file after rendering
        """
        if root is None:
            raise ValueError("Root node cannot be None")
            
        def add_subtree(node: EnhancedMCTSNode, node_id: str):
            try:
                self.add_node(node, node_id)
                if hasattr(node, 'children'):
                    for action, child in node.children.items():
                        if child is not None:
                            child_id = f"{node_id}_{action}"
                            self.add_node(child, child_id, node_id)
                            add_subtree(child, child_id)
            except Exception as e:
                print(f"Error processing subtree at {node_id}: {str(e)}")
                
        try:
            add_subtree(root, "root")
            self.dot.render(path, view=view, cleanup=cleanup)
        except Exception as e:
            print(f"Error rendering visualization: {str(e)}")