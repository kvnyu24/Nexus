import graphviz
from typing import Optional, Dict, Any, Union
from ..models.search.mcts_node import MCTSNode
from ..models.search.mcts import MCTS
from ..models.search.transformer_mcts import TransformerMCTSWithMemory
from ..visualization.base import BaseVisualizer
from ..utils.logging import Logger

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
        self.node_counter = 0  # For generating unique node IDs
        self.logger = Logger(__name__)
        
    def _get_safe_value(self, node: MCTSNode, attr: str, default: Any = 0.0) -> Any:
        """Safely get attribute value from node with error handling
        
        Args:
            node: MCTS node to get value from
            attr: Name of attribute to get
            default: Default value if attribute doesn't exist
        """
        try:
            if not hasattr(node, attr):
                return default
            value = getattr(node, attr)
            if attr == 'uncertainty':
                return value.item() if hasattr(value, 'item') else float(value)
            if callable(value):
                return value()
            return value
        except Exception as e:
            self.logger.warning(f"Error getting {attr} from node: {str(e)}")
            return default
            
    def add_node(self, 
                 node: MCTSNode, 
                 node_id: str, 
                 parent_id: Optional[str] = None,
                 node_attrs: Optional[Dict[str, Any]] = None,
                 action: Optional[int] = None):
        """Add a node to the visualization
        
        Args:
            node: The MCTS node to visualize
            node_id: Unique identifier for this node
            parent_id: ID of parent node if it exists
            node_attrs: Optional dict of node attributes to pass to graphviz
            action: Action that led to this node (for edge labels)
        """
        try:
            if node is None:
                self.logger.warning(f"Attempted to add None node with ID {node_id}")
                return
                
            # Get node statistics safely
            value = self._get_safe_value(node, 'value', 0.0)
            visits = self._get_safe_value(node, 'visit_count', 0)
            uncertainty = self._get_safe_value(node, 'uncertainty', 0.0)
            prior = self._get_safe_value(node, 'prior', 0.0)
            
            # Create detailed node label
            label = f"Value: {value:.3f}\nVisits: {visits}\n"
            label += f"Prior: {prior:.3f}\nUncert: {uncertainty:.3f}"
            
            # Add node with attributes
            attrs = {
                'label': label,
                'shape': 'box',
                'style': 'rounded,filled',
                'fillcolor': f"0.0 {min(visits/100.0, 1.0)} 1.0"  # Color based on visits
            }
            if node_attrs:
                attrs.update(node_attrs)
            self.dot.node(node_id, **attrs)
            
            # Add edge from parent if exists
            if parent_id is not None:
                edge_attrs = {}
                if action is not None:
                    edge_attrs['label'] = str(action)
                self.dot.edge(parent_id, node_id, **edge_attrs)
                
        except Exception as e:
            self.logger.error(f"Error adding node {node_id}: {str(e)}")
            
    def visualize_tree(self, 
                      root: MCTSNode, 
                      path: str,
                      view: bool = True,
                      cleanup: bool = True,
                      max_depth: int = 10):
        """Generate visualization of MCTS tree
        
        Args:
            root: Root node of the MCTS tree
            path: Output path for rendered visualization
            view: Whether to open the rendered visualization
            cleanup: Whether to remove DOT source file after rendering
            max_depth: Maximum depth to visualize to prevent huge graphs
        """
        if root is None:
            raise ValueError("Root node cannot be None")
            
        def add_subtree(node: MCTSNode, node_id: str, depth: int = 0):
            if depth >= max_depth:
                return
                
            try:
                self.add_node(node, node_id)
                
                if hasattr(node, 'children'):
                    # Sort children by visit count for better visualization
                    sorted_children = sorted(
                        node.children.items(),
                        key=lambda x: getattr(x[1], 'visit_count', 0),
                        reverse=True
                    )
                    
                    for action, child in sorted_children:
                        if child is not None:
                            child_id = f"{node_id}_{action}"
                            self.add_node(
                                child, 
                                child_id, 
                                node_id,
                                action=action
                            )
                            add_subtree(child, child_id, depth + 1)
                            
            except Exception as e:
                self.logger.error(f"Error processing subtree at {node_id}: {str(e)}")
                
        try:
            self.dot.clear()  # Clear any previous graph
            add_subtree(root, "root")
            self.dot.render(path, view=view, cleanup=cleanup)
        except Exception as e:
            self.logger.error(f"Error rendering visualization: {str(e)}")
            raise