import graphviz
from typing import Optional
from ..models.search.mcts_node import EnhancedMCTSNode
from ..models.search.enhanced_mcts import EnhancedMCTS

class MCTSVisualizer:
    def __init__(self):
        self.dot = graphviz.Digraph(comment='MCTS Tree')
        self.dot.attr(rankdir='TB')
        
    def add_node(self, node: EnhancedMCTSNode, node_id: str, parent_id: Optional[str] = None):
        # Create node label
        label = f"V: {node.value():.3f}\nN: {node.visit_count}\nU: {node.uncertainty.item():.3f}"
        
        # Add node
        self.dot.node(node_id, label)
        
        # Add edge from parent if exists
        if parent_id is not None:
            self.dot.edge(parent_id, node_id)
            
    def visualize_tree(self, root: EnhancedMCTSNode, path: str):
        """Generate visualization of MCTS tree"""
        def add_subtree(node: EnhancedMCTSNode, node_id: str):
            self.add_node(node, node_id)
            for action, child in node.children.items():
                child_id = f"{node_id}_{action}"
                self.add_node(child, child_id, node_id)
                add_subtree(child, child_id)
                
        add_subtree(root, "root")
        self.dot.render(path, view=True) 