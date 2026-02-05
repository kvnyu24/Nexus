"""
Mixture-of-Agents (MoA): Multi-LLM Layered Collaboration.

MoA is a framework that leverages multiple LLMs in a layered architecture
where each layer consists of multiple LLM agents that process outputs from
the previous layer. This iterative refinement approach achieves state-of-the-art
results by allowing models to learn from and build upon each other's responses.

Key differences from Mixture-of-Experts (MoE):
- MoA: Multiple complete LLM models collaborate across layers
- MoE: Single model with sparse expert routing within layers

Reference:
    Mixture-of-Agents Enhances Large Language Model Capabilities
    https://arxiv.org/abs/2406.04692
    Together AI, June 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable, Union
from nexus.core.base import NexusModule


class MoALayer(NexusModule):
    """Single layer in Mixture-of-Agents architecture.

    A MoA layer consists of multiple LLM agents (proposers) that each
    generate responses based on the previous layer's outputs. Their
    responses are then aggregated for the next layer.

    Args:
        agents: List of agent modules (LLMs or callable generators)
        aggregation: How to combine agent outputs ('concat', 'mean', 'max', 'attention')
        use_reference: Whether to include reference/previous responses in prompts
    """

    def __init__(
        self,
        agents: List[Union[nn.Module, Callable]],
        aggregation: str = 'concat',
        use_reference: bool = True,
    ):
        super().__init__()
        self.num_agents = len(agents)
        self.aggregation = aggregation
        self.use_reference = use_reference

        # Store agents as a ModuleList if they're nn.Modules
        if all(isinstance(a, nn.Module) for a in agents):
            self.agents = nn.ModuleList(agents)
        else:
            # Store as regular list if they're callables
            self.agents = agents

        # Attention-based aggregation
        if aggregation == 'attention':
            # Assume all agents have same hidden dim
            # This is a placeholder - actual implementation would need
            # the hidden dimension
            self.attn_weights = nn.Parameter(torch.ones(self.num_agents))

    def forward(
        self,
        inputs: torch.Tensor,
        references: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Process inputs through all agents in parallel.

        Args:
            inputs: Input tensor (batch, seq_len, hidden_dim)
            references: Optional list of reference outputs from previous agents
            attention_mask: Optional attention mask

        Returns:
            List of outputs from each agent
        """
        agent_outputs = []

        for i, agent in enumerate(self.agents):
            # Prepare input (optionally including references)
            agent_input = inputs

            if self.use_reference and references is not None:
                # Concatenate reference responses
                # In practice, this would involve prompt engineering
                # Here we simply concatenate embeddings
                ref_concat = torch.cat(references, dim=1)  # (batch, ref_len, hidden)
                agent_input = torch.cat([agent_input, ref_concat], dim=1)

            # Generate response
            if isinstance(agent, nn.Module):
                output = agent(agent_input, attention_mask=attention_mask)
                # Handle different output formats
                if isinstance(output, tuple):
                    output = output[0]
            else:
                # Callable agent
                output = agent(agent_input)

            agent_outputs.append(output)

        return agent_outputs

    def aggregate(self, agent_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate multiple agent outputs into a single representation.

        Args:
            agent_outputs: List of tensors from each agent

        Returns:
            Aggregated output tensor
        """
        if self.aggregation == 'concat':
            # Concatenate along sequence dimension
            return torch.cat(agent_outputs, dim=1)

        elif self.aggregation == 'mean':
            # Average pooling
            stacked = torch.stack(agent_outputs, dim=0)
            return stacked.mean(dim=0)

        elif self.aggregation == 'max':
            # Max pooling
            stacked = torch.stack(agent_outputs, dim=0)
            return stacked.max(dim=0)[0]

        elif self.aggregation == 'attention':
            # Weighted combination using learned attention weights
            weights = F.softmax(self.attn_weights, dim=0)
            stacked = torch.stack(agent_outputs, dim=0)  # (num_agents, batch, seq, hidden)
            weighted = stacked * weights.view(-1, 1, 1, 1)
            return weighted.sum(dim=0)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


class MixtureOfAgents(NexusModule):
    """Mixture-of-Agents: Layered multi-LLM collaboration architecture.

    Implements a multi-layer architecture where each layer contains multiple
    LLM agents that process and refine outputs from the previous layer.

    The final layer typically uses a stronger "aggregator" model to synthesize
    the final response from all proposer outputs.

    Architecture:
        Layer 1: [Agent_1, Agent_2, ..., Agent_N] -> Aggregate
        Layer 2: [Agent_1, Agent_2, ..., Agent_N] -> Aggregate
        ...
        Layer L: [Aggregator] -> Final Output

    Args:
        layers: List of MoALayer modules
        final_aggregator: Optional final aggregation layer/model
        enable_caching: Whether to cache intermediate outputs

    Example:
        >>> # Create a 3-layer MoA with 4 agents per layer
        >>> agents_per_layer = 4
        >>> layers = [
        ...     MoALayer([create_agent() for _ in range(agents_per_layer)])
        ...     for _ in range(3)
        ... ]
        >>> moa = MixtureOfAgents(layers)
        >>> output = moa(input_ids)
    """

    def __init__(
        self,
        layers: List[MoALayer],
        final_aggregator: Optional[nn.Module] = None,
        enable_caching: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.final_aggregator = final_aggregator
        self.enable_caching = enable_caching
        self.num_layers = len(layers)

        # Cache for intermediate results
        self._cache = None

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Process inputs through all MoA layers.

        Args:
            inputs: Input tensor (batch, seq_len, hidden_dim)
            attention_mask: Optional attention mask
            return_all_layers: Whether to return outputs from all layers

        Returns:
            Final output tensor, or tuple of (final_output, all_layer_outputs)
        """
        all_layer_outputs = []
        current_input = inputs
        references = None

        # Process through each layer
        for layer_idx, layer in enumerate(self.layers):
            # Get agent outputs
            agent_outputs = layer(
                current_input,
                references=references,
                attention_mask=attention_mask
            )

            # Aggregate for next layer input
            aggregated = layer.aggregate(agent_outputs)

            if return_all_layers or self.enable_caching:
                all_layer_outputs.append({
                    'agent_outputs': agent_outputs,
                    'aggregated': aggregated
                })

            # Update for next layer
            references = agent_outputs
            current_input = aggregated

        # Final aggregation
        if self.final_aggregator is not None:
            final_output = self.final_aggregator(current_input)
        else:
            final_output = current_input

        if return_all_layers:
            return final_output, all_layer_outputs
        return final_output

    def generate(
        self,
        inputs: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using MoA in an autoregressive manner.

        Args:
            inputs: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            **kwargs: Additional generation parameters

        Returns:
            Generated token IDs
        """
        # This is a simplified version - actual implementation would
        # need proper generation logic for each agent
        batch_size = inputs.shape[0]
        current_length = inputs.shape[1]

        generated = inputs

        for _ in range(max_length):
            # Forward pass through MoA
            outputs = self.forward(generated)

            # Get next token logits (assuming output has a language modeling head)
            if hasattr(outputs, 'logits'):
                next_token_logits = outputs.logits[:, -1, :]
            else:
                # Assume outputs are logits directly
                next_token_logits = outputs[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Nucleus sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits = next_token_logits.masked_fill(
                    indices_to_remove, float('-inf')
                )

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS token (assuming 0 is EOS)
            if (next_token == 0).all():
                break

        return generated

    def clear_cache(self):
        """Clear cached intermediate outputs."""
        self._cache = None


class SimpleMoA(NexusModule):
    """Simplified Mixture-of-Agents for easier experimentation.

    A streamlined version that uses the same agent architecture across
    all layers with simple mean aggregation.

    Args:
        base_model: Base LLM model to use for all agents
        num_layers: Number of MoA layers
        agents_per_layer: Number of agents in each layer
        share_weights: Whether agents share weights (memory efficient)

    Example:
        >>> from transformers import AutoModel
        >>> base = AutoModel.from_pretrained("gpt2")
        >>> moa = SimpleMoA(base, num_layers=3, agents_per_layer=4)
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_layers: int = 3,
        agents_per_layer: int = 4,
        share_weights: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.agents_per_layer = agents_per_layer
        self.share_weights = share_weights

        # Create layers
        layers = []
        for _ in range(num_layers):
            if share_weights:
                # All agents share the same model
                agents = [base_model for _ in range(agents_per_layer)]
            else:
                # Each agent is a separate copy
                import copy
                agents = [
                    copy.deepcopy(base_model)
                    for _ in range(agents_per_layer)
                ]

            layer = MoALayer(agents, aggregation='mean')
            layers.append(layer)

        self.moa = MixtureOfAgents(layers)

    def forward(self, *args, **kwargs):
        """Forward pass through MoA."""
        return self.moa(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generate using MoA."""
        return self.moa.generate(*args, **kwargs)
