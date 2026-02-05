"""Graph of Thoughts (GoT): Solving Elaborate Problems with Large Language Models.

Reference: Besta et al., "Graph of Thoughts: Solving Elaborate Problems with
Large Language Models" (2024). https://arxiv.org/abs/2308.09687

Graph of Thoughts extends Tree of Thoughts by allowing arbitrary graph
structures over thoughts. Unlike ToT (which is a tree), GoT supports:
    - Generate: Create new thought nodes from existing ones.
    - Aggregate: Merge multiple thoughts into a single refined thought.
    - Refine: Iteratively improve a thought node in-place.
    - Score: Evaluate the quality of a thought.

This enables more expressive reasoning patterns such as merging parallel
reasoning chains, iterative refinement loops, and decomposition-aggregation
patterns that are not possible in a strict tree structure.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....core.base import NexusModule


class OperationType(Enum):
    """Types of operations that can be performed on the thought graph."""
    GENERATE = "generate"
    AGGREGATE = "aggregate"
    REFINE = "refine"
    SCORE = "score"


@dataclass
class GraphNode:
    """A node in the thought graph.

    Attributes:
        node_id: Unique identifier.
        embedding: Thought embedding, shape (hidden_size,).
        score: Quality score of this thought.
        predecessors: IDs of nodes that contributed to this one.
        successors: IDs of nodes generated from this one.
        operation: The operation that created this node.
        refinement_count: Number of times this node has been refined.
    """
    node_id: int
    embedding: Optional[torch.Tensor] = None
    score: float = 0.0
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    operation: Optional[OperationType] = None
    refinement_count: int = 0


class ThoughtGraph(NexusModule):
    """Directed graph data structure for thought nodes with neural operations.

    Maintains a set of thought nodes connected by directed edges. Supports
    adding nodes, connecting them, and performing message passing over the
    graph for contextual thought representations.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Dimension of thought embeddings.
            - max_nodes (int): Maximum number of nodes (default 20).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.max_nodes = config.get("max_nodes", 20)

        # Node embedding storage
        self.node_store = nn.Parameter(
            torch.zeros(self.max_nodes, self.hidden_size),
            requires_grad=False,
        )

        # Edge attention for message passing
        self.edge_attention = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )

        # Message passing update
        self.message_fn = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        self.update_fn = nn.GRUCell(self.hidden_size, self.hidden_size)

        self._num_nodes = 0
        self._adjacency: Dict[int, Set[int]] = {}

    def add_node(
        self,
        embedding: torch.Tensor,
        predecessors: Optional[List[int]] = None,
    ) -> int:
        """Add a thought node to the graph.

        Args:
            embedding: Node embedding, shape (hidden_size,).
            predecessors: Optional list of predecessor node IDs.

        Returns:
            ID of the newly added node.

        Raises:
            ValueError: If graph is at maximum capacity.
        """
        if self._num_nodes >= self.max_nodes:
            raise ValueError(
                f"Graph is full: {self._num_nodes}/{self.max_nodes} nodes."
            )

        node_id = self._num_nodes
        with torch.no_grad():
            self.node_store[node_id] = embedding.detach()

        self._adjacency[node_id] = set()
        if predecessors:
            for pred in predecessors:
                if pred in self._adjacency:
                    self._adjacency[pred].add(node_id)

        self._num_nodes += 1
        return node_id

    def get_node_embedding(self, node_id: int) -> torch.Tensor:
        """Get the embedding for a specific node.

        Args:
            node_id: ID of the node.

        Returns:
            Node embedding, shape (hidden_size,).
        """
        if node_id >= self._num_nodes:
            raise ValueError(f"Node {node_id} does not exist.")
        return self.node_store[node_id]

    def forward(
        self,
        node_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Perform message passing over the graph.

        Args:
            node_ids: Optional subset of node IDs to update. If None,
                updates all active nodes.

        Returns:
            Dictionary containing:
                - node_embeddings: Updated node embeddings,
                  shape (num_active_nodes, hidden_size).
                - attention_weights: Edge attention weights for the
                  message passing step.
        """
        if self._num_nodes == 0:
            return {
                "node_embeddings": torch.zeros(
                    0, self.hidden_size, device=self.node_store.device
                ),
                "attention_weights": torch.zeros(0),
            }

        active_nodes = self.node_store[:self._num_nodes]

        if node_ids is not None:
            target_ids = node_ids.tolist()
        else:
            target_ids = list(range(self._num_nodes))

        updated_embeddings = active_nodes.clone()
        all_attn_weights = []

        for target_id in target_ids:
            # Find predecessors of this node
            predecessors = []
            for src, dsts in self._adjacency.items():
                if target_id in dsts:
                    predecessors.append(src)

            if not predecessors:
                continue

            # Compute messages from predecessors
            target_emb = active_nodes[target_id].unsqueeze(0)
            pred_embs = active_nodes[predecessors]

            # Attention-weighted message aggregation
            target_expanded = target_emb.expand_as(pred_embs)
            combined = torch.cat([pred_embs, target_expanded], dim=-1)
            attn_scores = self.edge_attention(combined).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=0)
            all_attn_weights.append(attn_weights)

            # Compute weighted message
            messages = self.message_fn(combined)
            aggregated = (messages * attn_weights.unsqueeze(-1)).sum(dim=0)

            # Update node with GRU
            updated = self.update_fn(
                aggregated.unsqueeze(0), target_emb
            ).squeeze(0)
            updated_embeddings[target_id] = updated

        return {
            "node_embeddings": updated_embeddings,
            "attention_weights": all_attn_weights,
        }

    def reset(self):
        """Reset the graph, clearing all nodes and edges."""
        self._num_nodes = 0
        self._adjacency = {}
        with torch.no_grad():
            self.node_store.zero_()


class GenerateOperation(NexusModule):
    """Generate new thought nodes from existing ones.

    Takes one or more source thoughts and produces new candidate thoughts
    by transforming through a generation network.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Thought embedding dimension.
            - num_generated (int): Number of new thoughts to generate
              per source (default 3).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.num_generated = config.get("num_generated", 3)

        self.generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size * 2),
            nn.Linear(
                self.hidden_size * 2,
                self.hidden_size * self.num_generated,
            ),
        )

        # Diversity regularization
        self.diversity_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self, source_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generate new thoughts from a source thought.

        Args:
            source_embedding: Source thought embedding,
                shape (batch_size, hidden_size) or (hidden_size,).

        Returns:
            Dictionary containing:
                - generated: New thought embeddings,
                  shape (batch_size, num_generated, hidden_size) or
                  (num_generated, hidden_size).
                - diversity_loss: Scalar loss encouraging diversity among
                  generated thoughts.
        """
        was_1d = source_embedding.dim() == 1
        if was_1d:
            source_embedding = source_embedding.unsqueeze(0)

        generated_flat = self.generator(source_embedding)
        generated = generated_flat.view(
            -1, self.num_generated, self.hidden_size
        )

        # Compute diversity loss (encourage orthogonality)
        projected = self.diversity_proj(generated)
        projected_norm = F.normalize(projected, p=2, dim=-1)
        similarity_matrix = torch.bmm(
            projected_norm, projected_norm.transpose(1, 2)
        )
        # Penalize off-diagonal similarities
        identity = torch.eye(
            self.num_generated, device=generated.device
        ).unsqueeze(0)
        diversity_loss = (
            (similarity_matrix - identity).pow(2).mean()
        )

        if was_1d:
            generated = generated.squeeze(0)

        return {
            "generated": generated,
            "diversity_loss": diversity_loss,
        }


class AggregateOperation(NexusModule):
    """Aggregate multiple thought nodes into a single refined thought.

    Implements attention-based aggregation where a learned query attends
    to the set of input thoughts and produces a merged representation.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Thought embedding dimension.
            - num_heads (int): Attention heads for aggregation (default 4).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]

        # Attention-based aggregation
        self.aggregate_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.get("num_heads", 4),
            batch_first=True,
        )

        # Learnable aggregation query
        self.agg_query = nn.Parameter(
            torch.randn(1, 1, self.hidden_size) * 0.02
        )

        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

    def forward(
        self, thought_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Aggregate multiple thoughts into one.

        Args:
            thought_embeddings: Embeddings to aggregate,
                shape (num_thoughts, hidden_size) or
                (batch_size, num_thoughts, hidden_size).

        Returns:
            Dictionary containing:
                - aggregated: Merged thought embedding,
                  shape (hidden_size,) or (batch_size, hidden_size).
                - attention_weights: Aggregation attention weights.
        """
        was_2d = thought_embeddings.dim() == 2
        if was_2d:
            thought_embeddings = thought_embeddings.unsqueeze(0)

        batch_size = thought_embeddings.size(0)
        query = self.agg_query.expand(batch_size, -1, -1)

        aggregated, attn_weights = self.aggregate_attention(
            query, thought_embeddings, thought_embeddings,
            need_weights=True,
        )

        aggregated = self.output_proj(aggregated.squeeze(1))

        if was_2d:
            aggregated = aggregated.squeeze(0)
            attn_weights = attn_weights.squeeze(0)

        return {
            "aggregated": aggregated,
            "attention_weights": attn_weights,
        }


class RefineOperation(NexusModule):
    """Refine an existing thought to improve its quality.

    Applies an iterative refinement step that takes the current thought
    and the problem context, and produces an improved version of the thought.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Thought embedding dimension.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]

        self.refine_network = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size * 2),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )

        # Gating mechanism: decide how much to update
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid(),
        )

    def forward(
        self,
        thought_embedding: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Refine a thought given problem context.

        Args:
            thought_embedding: Current thought, shape (batch_size, hidden_size)
                or (hidden_size,).
            context_embedding: Problem context, shape (batch_size, hidden_size)
                or (hidden_size,).

        Returns:
            Dictionary containing:
                - refined: Refined thought embedding, same shape as input.
                - gate_values: How much the thought was updated, in [0, 1].
        """
        was_1d = thought_embedding.dim() == 1
        if was_1d:
            thought_embedding = thought_embedding.unsqueeze(0)
            context_embedding = context_embedding.unsqueeze(0)

        combined = torch.cat(
            [thought_embedding, context_embedding], dim=-1
        )
        refined_candidate = self.refine_network(combined)
        gate_values = self.gate(combined)

        refined = (
            gate_values * refined_candidate
            + (1 - gate_values) * thought_embedding
        )

        if was_1d:
            refined = refined.squeeze(0)
            gate_values = gate_values.squeeze(0)

        return {
            "refined": refined,
            "gate_values": gate_values,
        }


class ScoreOperation(NexusModule):
    """Score a thought for quality and progress toward the solution.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Thought embedding dimension.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]

        self.scorer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        thought_embedding: torch.Tensor,
        problem_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Score a thought in the context of the problem.

        Args:
            thought_embedding: Thought to score,
                shape (batch_size, hidden_size) or (hidden_size,).
            problem_embedding: Problem context,
                shape (batch_size, hidden_size) or (hidden_size,).

        Returns:
            Dictionary containing:
                - score: Quality score in [0, 1].
        """
        was_1d = thought_embedding.dim() == 1
        if was_1d:
            thought_embedding = thought_embedding.unsqueeze(0)
            problem_embedding = problem_embedding.unsqueeze(0)

        combined = torch.cat(
            [thought_embedding, problem_embedding], dim=-1
        )
        score = self.scorer(combined).squeeze(-1)

        if was_1d:
            score = score.squeeze(0)

        return {"score": score}


class GoTController(NexusModule):
    """Controller that orchestrates Graph of Thoughts operations.

    Manages the thought graph and decides which operations to apply,
    executing a sequence of generate/aggregate/refine/score steps
    to solve the problem.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Thought embedding dimension.
            - vocab_size (int): Output vocabulary size.
            - max_nodes (int): Maximum graph nodes (default 20).
            - num_generated (int): New thoughts per generate step (default 3).
            - max_refinements (int): Maximum refinements per thought
              (default 3).
            - operations (list): Sequence of operations to execute
              (default: ['generate', 'aggregate', 'refine', 'score']).
            - num_heads (int): Attention heads (default 8).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.max_refinements = config.get("max_refinements", 3)

        # Operation modules
        self.graph = ThoughtGraph(config)
        self.generate_op = GenerateOperation(config)
        self.aggregate_op = AggregateOperation(config)
        self.refine_op = RefineOperation(config)
        self.score_op = ScoreOperation(config)

        # Operation scheduler: predicts which operation to apply next
        self.operation_selector = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, len(OperationType)),
        )

        # Operations to execute (configurable pipeline)
        operation_names = config.get(
            "operations",
            ["generate", "aggregate", "refine", "score"],
        )
        self.operation_sequence = [
            OperationType(name) for name in operation_names
        ]

        # Solution extraction
        self.solution_extractor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        self.output_head = nn.Linear(
            self.hidden_size, config.get("vocab_size", self.hidden_size)
        )

    def forward(
        self,
        problem_embedding: torch.Tensor,
    ) -> Dict[str, Any]:
        """Execute the Graph of Thoughts reasoning pipeline.

        Args:
            problem_embedding: Problem representation,
                shape (batch_size, hidden_size).

        Returns:
            Dictionary containing:
                - logits: Output logits, shape (batch_size, vocab_size).
                - solution_embedding: Best solution representation.
                - best_score: Score of the best thought.
                - num_nodes: Number of nodes in the final graph.
                - operation_history: Sequence of operations performed.
                - diversity_loss: Diversity regularization loss.
        """
        batch_size = problem_embedding.size(0)
        device = problem_embedding.device

        # Reset graph for new problem
        self.graph.reset()

        # Initialize graph with problem as root node
        # Process each batch item independently through the graph
        all_thoughts = []
        all_scores = []
        total_diversity_loss = torch.tensor(0.0, device=device)
        operation_history = []

        for b in range(batch_size):
            prob_emb = problem_embedding[b]
            self.graph.reset()
            self.graph.add_node(prob_emb)

            current_thoughts = [prob_emb]
            best_thought = prob_emb
            best_score = torch.tensor(0.0, device=device)

            for op_type in self.operation_sequence:
                operation_history.append(op_type.value)

                if op_type == OperationType.GENERATE:
                    new_thoughts = []
                    for thought in current_thoughts:
                        gen_out = self.generate_op(thought)
                        generated = gen_out["generated"]
                        total_diversity_loss = (
                            total_diversity_loss + gen_out["diversity_loss"]
                        )

                        for i in range(generated.size(0)):
                            if self.graph._num_nodes < self.graph.max_nodes:
                                self.graph.add_node(generated[i])
                                new_thoughts.append(generated[i])

                    current_thoughts = new_thoughts if new_thoughts else current_thoughts

                elif op_type == OperationType.AGGREGATE:
                    if len(current_thoughts) > 1:
                        stacked = torch.stack(current_thoughts, dim=0)
                        agg_out = self.aggregate_op(stacked)
                        aggregated = agg_out["aggregated"]

                        if self.graph._num_nodes < self.graph.max_nodes:
                            self.graph.add_node(aggregated)
                        current_thoughts = [aggregated]

                elif op_type == OperationType.REFINE:
                    refined_thoughts = []
                    for thought in current_thoughts:
                        ref_out = self.refine_op(thought, prob_emb)
                        refined_thoughts.append(ref_out["refined"])
                    current_thoughts = refined_thoughts

                elif op_type == OperationType.SCORE:
                    scored_thoughts = []
                    for thought in current_thoughts:
                        sc_out = self.score_op(thought, prob_emb)
                        score = sc_out["score"]
                        if score > best_score:
                            best_score = score
                            best_thought = thought
                        scored_thoughts.append((thought, score))

                    # Keep only top thoughts for next operations
                    scored_thoughts.sort(key=lambda x: x[1], reverse=True)
                    keep_n = max(1, len(scored_thoughts) // 2)
                    current_thoughts = [
                        t for t, _ in scored_thoughts[:keep_n]
                    ]

            all_thoughts.append(best_thought)
            all_scores.append(best_score)

        # Stack batch results
        best_thoughts_batch = torch.stack(all_thoughts, dim=0)
        best_scores_batch = torch.stack(all_scores, dim=0)

        # Extract solution
        solution = self.solution_extractor(
            torch.cat([best_thoughts_batch, problem_embedding], dim=-1)
        )
        logits = self.output_head(solution)

        return {
            "logits": logits,
            "solution_embedding": solution,
            "best_score": best_scores_batch,
            "num_nodes": self.graph._num_nodes,
            "operation_history": operation_history,
            "diversity_loss": total_diversity_loss / max(batch_size, 1),
        }
