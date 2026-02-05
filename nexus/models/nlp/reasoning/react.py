"""ReAct: Synergizing Reasoning and Acting in Language Models.

Reference: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language
Models" (2023). https://arxiv.org/abs/2210.03629

ReAct interleaves reasoning (Thought) and acting (Action) in an alternating
loop, grounding the model's reasoning in real observations from tool use:

    Loop:
        1. Thought: The model reasons about the current state and what
           action to take next.
        2. Action: The model selects and parameterizes an action from a
           tool registry.
        3. Observation: The action is executed and the observation is fed
           back to the model.

This loop continues until the model produces a "Finish" action or a
maximum number of steps is reached. The interleaved design allows the
model to adapt its reasoning based on real-world feedback, reducing
hallucination and improving factual grounding.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....core.base import NexusModule


@dataclass
class Tool:
    """Represents a tool/action available to the ReAct agent.

    Attributes:
        name: Unique name of the tool.
        description: Human-readable description of what the tool does.
        input_size: Expected input dimension for the tool.
        output_size: Dimension of the tool's output observation.
    """
    name: str
    description: str = ""
    input_size: int = 0
    output_size: int = 0


@dataclass
class ActionResult:
    """Result of executing an action.

    Attributes:
        tool_name: Name of the tool that was invoked.
        action_embedding: The action representation used,
            shape (hidden_size,).
        observation_embedding: Embedding of the observation returned,
            shape (hidden_size,).
        is_terminal: Whether this action indicates completion (e.g., "Finish").
    """
    tool_name: str
    action_embedding: Optional[torch.Tensor] = None
    observation_embedding: Optional[torch.Tensor] = None
    is_terminal: bool = False


class ToolRegistry(NexusModule):
    """Registry of tools available to the ReAct agent.

    Maintains tool descriptions and learned tool embeddings that the agent
    uses to select and parameterize actions. Each tool has an associated
    embedding that encodes its capabilities, and an execution network that
    simulates tool behavior during training.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Dimension of tool and action embeddings.
            - tools (list): List of tool name strings to register.
            - stop_token (str): Name of the terminal action (default "Finish").
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        tool_names = config.get("tools", ["Search", "Lookup", "Finish"])
        self.stop_token = config.get("stop_token", "Finish")

        self.tool_names = tool_names
        self.num_tools = len(tool_names)
        self.tool_name_to_idx = {
            name: idx for idx, name in enumerate(tool_names)
        }

        # Learned tool embeddings
        self.tool_embeddings = nn.Embedding(self.num_tools, self.hidden_size)

        # Tool description encoder
        self.tool_description_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        # Tool execution simulator (for differentiable training)
        self.tool_executors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            for _ in range(self.num_tools)
        ])

    def get_tool_embeddings(self) -> torch.Tensor:
        """Get all tool embeddings.

        Returns:
            Tool embeddings, shape (num_tools, hidden_size).
        """
        indices = torch.arange(
            self.num_tools, device=self.tool_embeddings.weight.device
        )
        return self.tool_embeddings(indices)

    def execute_tool(
        self,
        tool_idx: torch.Tensor,
        action_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute a tool and return the observation.

        Args:
            tool_idx: Index of the tool to execute,
                shape (batch_size,).
            action_input: Input to the tool,
                shape (batch_size, hidden_size).

        Returns:
            Tuple of:
                - observation: Tool output embedding,
                  shape (batch_size, hidden_size).
                - is_terminal: Whether the stop token was selected,
                  shape (batch_size,).
        """
        batch_size = action_input.size(0)
        device = action_input.device

        # Execute each tool for the corresponding batch items
        observations = torch.zeros(
            batch_size, self.hidden_size, device=device
        )

        for i in range(self.num_tools):
            mask = tool_idx == i
            if mask.any():
                tool_input = action_input[mask]
                tool_output = self.tool_executors[i](tool_input)
                observations[mask] = tool_output

        # Check for terminal action
        stop_idx = self.tool_name_to_idx.get(self.stop_token, -1)
        is_terminal = tool_idx == stop_idx

        return observations, is_terminal

    def forward(
        self, query_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute tool selection scores for a query.

        Args:
            query_embedding: Current state embedding,
                shape (batch_size, hidden_size).

        Returns:
            Dictionary containing:
                - tool_scores: Similarity scores for each tool,
                  shape (batch_size, num_tools).
                - tool_probabilities: Softmax probabilities,
                  shape (batch_size, num_tools).
                - selected_tool: Index of the most likely tool,
                  shape (batch_size,).
        """
        tool_embs = self.get_tool_embeddings()
        tool_embs = self.tool_description_proj(tool_embs)

        scores = torch.matmul(
            query_embedding, tool_embs.t()
        ) / (self.hidden_size ** 0.5)

        probabilities = F.softmax(scores, dim=-1)
        selected = torch.argmax(probabilities, dim=-1)

        return {
            "tool_scores": scores,
            "tool_probabilities": probabilities,
            "selected_tool": selected,
        }


class ActionParser(NexusModule):
    """Parses the agent's hidden state into a structured action.

    Converts the agent's current hidden representation into:
    1. A tool selection (which tool to use).
    2. An action input (parameterization of the tool call).

    Args:
        config: Dictionary containing:
            - hidden_size (int): Hidden state dimension.
            - num_tools (int): Number of available tools.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        num_tools = config.get(
            "num_tools", len(config.get("tools", ["Search", "Lookup", "Finish"]))
        )

        # Tool selection head
        self.tool_selector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, num_tools),
        )

        # Action input generator
        self.action_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def forward(
        self, hidden_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Parse hidden state into action components.

        Args:
            hidden_state: Agent's current hidden state,
                shape (batch_size, hidden_size).

        Returns:
            Dictionary containing:
                - tool_logits: Logits for tool selection,
                  shape (batch_size, num_tools).
                - tool_probabilities: Softmax probabilities,
                  shape (batch_size, num_tools).
                - selected_tool: Argmax tool index,
                  shape (batch_size,).
                - action_input: Parameterized action input,
                  shape (batch_size, hidden_size).
        """
        tool_logits = self.tool_selector(hidden_state)
        tool_probs = F.softmax(tool_logits, dim=-1)
        selected_tool = torch.argmax(tool_probs, dim=-1)

        action_input = self.action_generator(hidden_state)

        return {
            "tool_logits": tool_logits,
            "tool_probabilities": tool_probs,
            "selected_tool": selected_tool,
            "action_input": action_input,
        }


class ReActAgent(NexusModule):
    """ReAct agent: alternates Thought-Action-Observation loops.

    The agent maintains an internal state that evolves through:
    1. Thought: Process current state to reason about next steps.
    2. Action: Select a tool and generate input parameters.
    3. Observation: Execute the tool and incorporate the result.

    The loop continues until a terminal action is selected or the
    maximum number of steps is reached.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Agent hidden state dimension.
            - vocab_size (int): Output vocabulary size.
            - max_steps (int): Maximum Thought-Action-Observation iterations
              (default 10).
            - tools (list): List of tool name strings (default:
              ["Search", "Lookup", "Finish"]).
            - stop_token (str): Name of the terminal action (default "Finish").
            - num_heads (int): Attention heads for state processing (default 8).
            - max_seq_length (int): Maximum sequence length (default 512).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.max_steps = config.get("max_steps", 10)

        # Tool registry
        self.tool_registry = ToolRegistry(config)

        # Action parser
        config_with_num_tools = {
            **config,
            "num_tools": self.tool_registry.num_tools,
        }
        self.action_parser = ActionParser(config_with_num_tools)

        # Thought module: processes current state
        self.thought_module = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size * 4),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )

        # State update: GRU that integrates observation into state
        self.state_updater = nn.GRUCell(self.hidden_size, self.hidden_size)

        # Observation processor
        self.observation_processor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        # Step embedding (to differentiate reasoning steps)
        self.step_embeddings = nn.Embedding(self.max_steps, self.hidden_size)

        # Output head
        self.output_head = nn.Linear(
            self.hidden_size, config.get("vocab_size", self.hidden_size)
        )

        # Problem encoder (optional, for encoding input tokens)
        if "vocab_size" in config:
            self.token_embedding = nn.Embedding(
                config["vocab_size"], self.hidden_size
            )
            self.position_embedding = nn.Parameter(
                torch.zeros(
                    1,
                    config.get("max_seq_length", 512),
                    self.hidden_size,
                )
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=config.get("num_heads", 8),
                dim_feedforward=self.hidden_size * 4,
                dropout=config.get("dropout", 0.1),
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=config.get("num_layers", 4)
            )
        else:
            self.token_embedding = None
            self.encoder = None

    def _encode_input(
        self,
        input_ids: Optional[torch.Tensor] = None,
        problem_embedding: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode the input problem into an initial state.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_length).
            problem_embedding: Pre-computed embedding,
                shape (batch_size, hidden_size).
            attention_mask: Optional mask for input_ids.

        Returns:
            Initial agent state, shape (batch_size, hidden_size).
        """
        if problem_embedding is not None:
            return problem_embedding

        if input_ids is None:
            raise ValueError(
                "Must provide either input_ids or problem_embedding."
            )

        if self.token_embedding is None or self.encoder is None:
            raise ValueError(
                "Token embedding and encoder required for input_ids. "
                "Provide vocab_size in config."
            )

        seq_len = input_ids.size(1)
        emb = self.token_embedding(input_ids)
        emb = emb + self.position_embedding[:, :seq_len, :]

        mask = (attention_mask == 0) if attention_mask is not None else None
        hidden = self.encoder(emb, src_key_padding_mask=mask)

        return hidden.mean(dim=1)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        problem_embedding: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Execute the ReAct agent loop.

        Args:
            input_ids: Problem token IDs, shape (batch_size, seq_length).
            problem_embedding: Pre-computed problem embedding,
                shape (batch_size, hidden_size).
            attention_mask: Optional attention mask.

        Returns:
            Dictionary containing:
                - logits: Final output logits,
                  shape (batch_size, vocab_size).
                - final_state: Agent's final hidden state,
                  shape (batch_size, hidden_size).
                - thoughts: List of thought embeddings per step.
                - actions: List of (tool_index, action_input) per step.
                - observations: List of observation embeddings per step.
                - num_steps: Number of steps taken before termination.
                - tool_probabilities: Tool selection probs per step.
                - terminated: Boolean tensor indicating which batch items
                  terminated naturally (via stop token).
        """
        # Encode initial state
        state = self._encode_input(
            input_ids, problem_embedding, attention_mask
        )
        batch_size = state.size(0)
        device = state.device

        # Tracking
        thoughts = []
        actions = []
        observations = []
        tool_probs_list = []
        terminated = torch.zeros(batch_size, dtype=torch.bool, device=device)
        num_steps = 0

        for step in range(self.max_steps):
            if terminated.all():
                break

            num_steps = step + 1

            # Add step embedding
            step_idx = torch.tensor(
                [step], device=device
            ).expand(batch_size)
            step_emb = self.step_embeddings(step_idx)
            state_with_step = state + step_emb

            # 1. Thought: reason about current state
            thought = self.thought_module(state_with_step)
            thought = thought + state  # Residual
            thoughts.append(thought)

            # 2. Action: select tool and generate input
            action_out = self.action_parser(thought)
            selected_tool = action_out["selected_tool"]
            action_input = action_out["action_input"]
            actions.append((selected_tool, action_input))
            tool_probs_list.append(action_out["tool_probabilities"])

            # 3. Observation: execute tool
            observation, is_terminal = self.tool_registry.execute_tool(
                selected_tool, action_input
            )
            observation = self.observation_processor(observation)
            observations.append(observation)

            # Update termination status
            terminated = terminated | is_terminal

            # 4. Update state with observation (only for non-terminated items)
            active = ~terminated
            if active.any():
                new_state = self.state_updater(
                    observation[active], state[active]
                )
                state = state.clone()
                state[active] = new_state

        # Final output
        logits = self.output_head(state)

        return {
            "logits": logits,
            "final_state": state,
            "thoughts": thoughts,
            "actions": actions,
            "observations": observations,
            "num_steps": num_steps,
            "tool_probabilities": tool_probs_list,
            "terminated": terminated,
        }
