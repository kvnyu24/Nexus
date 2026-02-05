"""Tree of Thoughts (ToT): Deliberate Problem Solving with Large Language Models.

Reference:
    Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T.L., Cao, Y., &
    Narasimhan, K. (2023). "Tree of Thoughts: Deliberate Problem Solving with
    Large Language Models." Advances in Neural Information Processing Systems
    (NeurIPS 2023). https://arxiv.org/abs/2305.10601

Tree of Thoughts (ToT) generalizes over Chain-of-Thought (CoT) prompting by
maintaining a tree where each node represents a partial solution (a "thought").
The framework enables:

    1. **Thought decomposition**: Breaking a problem into coherent intermediate
       reasoning steps.
    2. **Thought generation**: Proposing multiple candidate next-thoughts from
       the current state (branching).
    3. **State evaluation**: Scoring each thought for progress toward a solution,
       enabling pruning of unpromising branches.
    4. **Search algorithm**: Systematic exploration via BFS (breadth-first) or
       DFS (depth-first with backtracking) to find the best reasoning path.

This implementation provides both a data structure for the thought tree
(ThoughtNode) and a NexusModule (TreeOfThoughts) that orchestrates the
generation, evaluation, and search over the thought space.

Example usage::

    config = {
        "model": language_model,
        "max_depth": 3,
        "branching_factor": 3,
        "search_method": "bfs",
        "evaluation_method": "vote",
    }
    tot = TreeOfThoughts(config)
    result = tot.solve("What is the best strategy to solve 24-game with 4 5 6 10?")
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque

import torch
import torch.nn as nn

from nexus.core.base import NexusModule


@dataclass
class ThoughtNode:
    """A single node in the Tree of Thoughts.

    Each node stores a textual thought (an intermediate reasoning step),
    an evaluation score indicating how promising the thought is, and
    structural references to its parent and children for tree traversal.

    Attributes:
        thought: The textual content of this intermediate reasoning step.
        value: Evaluation score in [0, 1] indicating quality/promise of this
            thought toward the final solution.
        children: List of child ThoughtNode instances branching from this node.
        parent: Reference to the parent node (None for the root).
        depth: Depth in the tree (0 for root).
        is_terminal: Whether this node represents a final answer/solution.
    """

    thought: str
    value: float = 0.0
    children: List["ThoughtNode"] = field(default_factory=list)
    parent: Optional["ThoughtNode"] = None
    depth: int = 0
    is_terminal: bool = False

    def add_child(self, child: "ThoughtNode") -> "ThoughtNode":
        """Add a child node and set its parent reference.

        Args:
            child: The child ThoughtNode to add.

        Returns:
            The child node (for chaining).
        """
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
        return child

    def get_path(self) -> List[str]:
        """Retrieve the full reasoning path from root to this node.

        Returns:
            List of thought strings from the root down to this node.
        """
        path = []
        node: Optional[ThoughtNode] = self
        while node is not None:
            path.append(node.thought)
            node = node.parent
        return list(reversed(path))

    def best_child(self) -> Optional["ThoughtNode"]:
        """Return the child with the highest evaluation value.

        Returns:
            The highest-valued child, or None if there are no children.
        """
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.value)

    def __repr__(self) -> str:
        return (
            f"ThoughtNode(depth={self.depth}, value={self.value:.3f}, "
            f"children={len(self.children)}, "
            f"thought={self.thought[:60]!r}{'...' if len(self.thought) > 60 else ''})"
        )


class TreeOfThoughts(NexusModule):
    """Tree of Thoughts reasoning framework.

    Implements the ToT framework from Yao et al. (NeurIPS 2023), which
    explores a tree of intermediate reasoning states to solve problems
    requiring deliberate planning and search.

    The module orchestrates:
        - **Thought generation**: Using a language model to propose multiple
          candidate next-thoughts from the current reasoning state.
        - **Thought evaluation**: Scoring each candidate via voting among
          multiple model calls or direct scalar scoring.
        - **Tree search**: BFS or DFS over the thought tree, pruning low-value
          branches and expanding promising ones.

    Args:
        config: Configuration dictionary with the following keys:

            - **model**: A callable language model (or object with a
              ``generate(prompt) -> str`` method) used to generate and
              evaluate thoughts.
            - **max_depth** (int): Maximum depth of the thought tree.
              Default: 3.
            - **branching_factor** (int): Number of candidate thoughts to
              generate at each node. Default: 3.
            - **search_method** (str): Tree search strategy, one of
              ``"bfs"`` (breadth-first) or ``"dfs"`` (depth-first with
              backtracking). Default: ``"bfs"``.
            - **evaluation_method** (str): How to evaluate thoughts, one
              of ``"vote"`` (majority vote among model calls) or
              ``"score"`` (direct scalar scoring). Default: ``"vote"``.
            - **num_votes** (int): Number of voting calls when using
              ``evaluation_method="vote"``. Default: 3.
            - **value_threshold** (float): Minimum thought value to continue
              expanding a branch. Default: 0.3.
            - **beam_width** (int): Number of top candidates to keep at each
              BFS level. Default: 5.

    Example::

        model = MyLanguageModel()
        tot = TreeOfThoughts({
            "model": model,
            "max_depth": 3,
            "branching_factor": 3,
            "search_method": "bfs",
            "evaluation_method": "vote",
        })
        result = tot.solve("Solve the 24 game with numbers 4, 5, 6, 10.")
        print(result["answer"])
        print(result["reasoning_path"])
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.model = self.config.get("model")
        self.max_depth = self.config.get("max_depth", 3)
        self.branching_factor = self.config.get("branching_factor", 3)
        self.search_method = self.config.get("search_method", "bfs")
        self.evaluation_method = self.config.get("evaluation_method", "vote")
        self.num_votes = self.config.get("num_votes", 3)
        self.value_threshold = self.config.get("value_threshold", 0.3)
        self.beam_width = self.config.get("beam_width", 5)

        if self.search_method not in ("bfs", "dfs"):
            raise ValueError(
                f"search_method must be 'bfs' or 'dfs', got {self.search_method!r}"
            )
        if self.evaluation_method not in ("vote", "score"):
            raise ValueError(
                f"evaluation_method must be 'vote' or 'score', "
                f"got {self.evaluation_method!r}"
            )

        # Placeholder parameter so the module registers as having parameters
        # (required by NexusModule / nn.Module conventions for device tracking)
        self._placeholder = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _call_model(self, prompt: str) -> str:
        """Call the underlying language model with a prompt.

        Supports callables and objects with a ``generate`` method.

        Args:
            prompt: The text prompt to send to the model.

        Returns:
            The model's text response.

        Raises:
            RuntimeError: If no model is configured.
        """
        if self.model is None:
            raise RuntimeError(
                "No language model configured. Provide 'model' in config."
            )
        if callable(self.model) and not hasattr(self.model, "generate"):
            return self.model(prompt)
        return self.model.generate(prompt)

    def generate_thoughts(self, state: str, k: int) -> List[str]:
        """Generate k candidate thoughts from the current reasoning state.

        Uses the language model to propose multiple distinct next reasoning
        steps given the current problem state.

        Args:
            state: The current reasoning state (concatenation of the problem
                statement and the chain of thoughts so far).
            k: Number of candidate thoughts to generate.

        Returns:
            A list of k candidate thought strings.
        """
        prompt = (
            f"Given the following problem and reasoning so far:\n\n"
            f"{state}\n\n"
            f"Generate {k} distinct possible next steps for reasoning about "
            f"this problem. Number each step and provide clear, specific "
            f"reasoning.\n\n"
            f"Possible next steps:"
        )
        response = self._call_model(prompt)

        # Parse numbered responses
        thoughts: List[str] = []
        current_thought = ""
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Detect numbered items (e.g., "1.", "1)", "Step 1:")
            is_new = False
            for i in range(1, k + 2):
                for prefix in (f"{i}.", f"{i})", f"Step {i}"):
                    if line.startswith(prefix):
                        if current_thought:
                            thoughts.append(current_thought.strip())
                        current_thought = line[len(prefix):].strip()
                        is_new = True
                        break
                if is_new:
                    break
            if not is_new:
                current_thought += " " + line

        if current_thought.strip():
            thoughts.append(current_thought.strip())

        # Ensure exactly k thoughts (pad with re-generation or truncate)
        while len(thoughts) < k:
            fallback_prompt = (
                f"Given the problem state:\n{state}\n\n"
                f"Suggest one more distinct reasoning step:"
            )
            thoughts.append(self._call_model(fallback_prompt).strip())
        return thoughts[:k]

    def evaluate_thought(self, state: str, thought: str) -> float:
        """Evaluate a single thought for its promise toward solving the problem.

        Supports two evaluation methods:

        - **"vote"**: Ask the model multiple times whether the thought is
          promising and aggregate via majority vote.
        - **"score"**: Ask the model to directly assign a numerical score.

        Args:
            state: The current reasoning state.
            thought: The candidate thought to evaluate.

        Returns:
            A float score in [0, 1] indicating how promising the thought is.
        """
        if self.evaluation_method == "vote":
            return self._evaluate_by_vote(state, thought)
        else:
            return self._evaluate_by_score(state, thought)

    def _evaluate_by_vote(self, state: str, thought: str) -> float:
        """Evaluate a thought via majority voting.

        Asks the model ``num_votes`` times to classify the thought as
        "sure" (good), "maybe" (uncertain), or "impossible" (bad), then
        converts the votes into a score.

        Args:
            state: Current problem state.
            thought: Candidate thought.

        Returns:
            Score in [0, 1] based on vote distribution.
        """
        prompt = (
            f"Given the problem and reasoning so far:\n{state}\n\n"
            f"Evaluate the following next step:\n{thought}\n\n"
            f"Is this step promising for reaching the solution? "
            f"Answer with exactly one word: 'sure', 'maybe', or 'impossible'."
        )

        vote_scores = {"sure": 1.0, "maybe": 0.5, "impossible": 0.0}
        total = 0.0
        for _ in range(self.num_votes):
            response = self._call_model(prompt).strip().lower()
            # Extract the classification
            if "sure" in response:
                total += vote_scores["sure"]
            elif "impossible" in response:
                total += vote_scores["impossible"]
            else:
                total += vote_scores["maybe"]

        return total / self.num_votes

    def _evaluate_by_score(self, state: str, thought: str) -> float:
        """Evaluate a thought by asking for a direct score.

        Args:
            state: Current problem state.
            thought: Candidate thought.

        Returns:
            Score in [0, 1].
        """
        prompt = (
            f"Given the problem and reasoning so far:\n{state}\n\n"
            f"Rate the following reasoning step on a scale from 0.0 to 1.0, "
            f"where 0.0 means completely wrong and 1.0 means certainly correct "
            f"and helpful.\n\n"
            f"Step: {thought}\n\n"
            f"Score (0.0 to 1.0):"
        )
        response = self._call_model(prompt).strip()

        # Parse numeric score from response
        try:
            for token in response.replace(",", " ").split():
                token = token.strip("()[]{}:")
                val = float(token)
                if 0.0 <= val <= 1.0:
                    return val
        except (ValueError, IndexError):
            pass

        # Fallback: heuristic based on positive/negative language
        positive_words = {"good", "correct", "promising", "right", "yes", "sure"}
        negative_words = {"bad", "wrong", "impossible", "no", "incorrect"}
        words = set(response.lower().split())
        pos = len(words & positive_words)
        neg = len(words & negative_words)
        if pos + neg == 0:
            return 0.5
        return pos / (pos + neg)

    def bfs_search(self, initial_state: str) -> ThoughtNode:
        """Breadth-first search over the thought tree.

        Explores all candidate thoughts at each depth level, evaluates them,
        keeps the top ``beam_width`` nodes, then expands the next level.
        This is effective for finding the best reasoning path when the
        solution depth is unknown.

        Args:
            initial_state: The problem statement / initial reasoning state.

        Returns:
            The best ThoughtNode found (highest value at maximum depth or
            a terminal node).
        """
        root = ThoughtNode(thought=initial_state, depth=0)
        current_level: List[ThoughtNode] = [root]
        best_node = root

        for depth in range(self.max_depth):
            next_level: List[ThoughtNode] = []

            for node in current_level:
                state = "\n".join(node.get_path())
                thoughts = self.generate_thoughts(state, self.branching_factor)

                for thought_text in thoughts:
                    child = ThoughtNode(thought=thought_text)
                    node.add_child(child)

                    # Evaluate
                    child.value = self.evaluate_thought(state, thought_text)
                    next_level.append(child)

                    if child.value > best_node.value:
                        best_node = child

            # Prune: keep only top beam_width candidates
            next_level.sort(key=lambda n: n.value, reverse=True)
            current_level = [
                n for n in next_level[:self.beam_width]
                if n.value >= self.value_threshold
            ]

            if not current_level:
                break

        return best_node

    def dfs_search(self, initial_state: str) -> ThoughtNode:
        """Depth-first search with backtracking over the thought tree.

        Explores the most promising branch first, backtracking when a
        thought's value falls below ``value_threshold``. This is
        memory-efficient and effective when good solutions are deep.

        Args:
            initial_state: The problem statement / initial reasoning state.

        Returns:
            The best ThoughtNode found during the search.
        """
        root = ThoughtNode(thought=initial_state, depth=0)
        best_node = root

        stack: List[ThoughtNode] = [root]

        while stack:
            node = stack.pop()

            if node.depth >= self.max_depth:
                continue

            state = "\n".join(node.get_path())
            thoughts = self.generate_thoughts(state, self.branching_factor)

            children: List[ThoughtNode] = []
            for thought_text in thoughts:
                child = ThoughtNode(thought=thought_text)
                node.add_child(child)
                child.value = self.evaluate_thought(state, thought_text)
                children.append(child)

                if child.value > best_node.value:
                    best_node = child

            # Push promising children onto stack (sorted so best is popped first)
            children.sort(key=lambda c: c.value)
            for child in children:
                if child.value >= self.value_threshold:
                    stack.append(child)

        return best_node

    def solve(self, problem: str) -> Dict[str, Any]:
        """Solve a problem using Tree of Thoughts reasoning.

        This is the main entry point. It builds and searches a thought tree
        using the configured search method, then returns the best reasoning
        path and answer.

        Args:
            problem: The problem statement to solve.

        Returns:
            Dictionary containing:
                - **answer** (str): The final thought/answer from the best
                  reasoning path.
                - **reasoning_path** (List[str]): The full chain of thoughts
                  from root to the best node.
                - **best_value** (float): The evaluation score of the best
                  thought node.
                - **best_node** (ThoughtNode): The best node object for
                  further inspection.
                - **search_method** (str): The search method used.
        """
        if self.search_method == "bfs":
            best_node = self.bfs_search(problem)
        else:
            best_node = self.dfs_search(problem)

        reasoning_path = best_node.get_path()

        return {
            "answer": best_node.thought,
            "reasoning_path": reasoning_path,
            "best_value": best_node.value,
            "best_node": best_node,
            "search_method": self.search_method,
        }

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        """Forward pass delegates to solve().

        Accepts a problem string as the first positional argument or
        as the ``problem`` keyword argument.

        Returns:
            Result dictionary from :meth:`solve`.
        """
        problem = kwargs.get("problem") or (args[0] if args else "")
        return self.solve(str(problem))
