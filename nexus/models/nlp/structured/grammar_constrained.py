"""Grammar-Constrained Decoding for structured output generation.

Reference:
    Willard, B. T., & Louf, R. "Efficient Guided Generation for LLMs."
    2023. https://arxiv.org/abs/2307.09702

Grammar-constrained decoding ensures that generated tokens conform to a formal
grammar (e.g., JSON, Python, SQL). This is achieved by masking invalid tokens
at each decoding step based on the current parse state.

Applications:
- Generating valid JSON for structured APIs
- Synthesizing syntactically correct code
- Producing domain-specific languages

Key innovation: Use formal grammar parsing to constrain the token space at
each generation step, guaranteeing valid outputs.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Set, Tuple, Union
import re
from enum import Enum

from nexus.core.base import NexusModule


class GrammarState(Enum):
    """States in grammar parsing."""
    START = "start"
    IN_OBJECT = "in_object"
    IN_ARRAY = "in_array"
    IN_STRING = "in_string"
    IN_NUMBER = "in_number"
    IN_BOOLEAN = "in_boolean"
    IN_NULL = "in_null"
    EXPECT_KEY = "expect_key"
    EXPECT_VALUE = "expect_value"
    EXPECT_COMMA_OR_END = "expect_comma_or_end"


@dataclass
class GrammarConfig:
    """Configuration for grammar-constrained decoding.

    Attributes:
        grammar_type: Type of grammar to enforce:
            - "json": JSON grammar
            - "python": Python grammar (subset)
            - "custom": Custom grammar rules
        allow_partial: Whether to allow partial/incomplete outputs.
        max_depth: Maximum nesting depth for structured formats.
    """
    grammar_type: str = "json"
    allow_partial: bool = False
    max_depth: int = 10


class GrammarParser:
    """Parser for tracking grammar state during generation.

    Maintains the current parse state and determines which tokens are valid
    at each step based on the grammar rules.
    """

    def __init__(self, config: GrammarConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.state_stack: List[GrammarState] = [GrammarState.START]
        self.depth = 0

        # Token sets for different contexts
        self._initialize_token_sets()

    def _initialize_token_sets(self):
        """Initialize sets of valid tokens for different contexts."""
        vocab = self.tokenizer.get_vocab()

        # JSON structural tokens
        self.structural_tokens = {
            vocab.get('{', -1),
            vocab.get('}', -1),
            vocab.get('[', -1),
            vocab.get(']', -1),
            vocab.get(',', -1),
            vocab.get(':', -1),
            vocab.get('"', -1),
        }

        # Value start tokens
        self.value_start_tokens = {
            vocab.get('{', -1),  # object
            vocab.get('[', -1),  # array
            vocab.get('"', -1),  # string
            vocab.get('t', -1),  # true
            vocab.get('f', -1),  # false
            vocab.get('n', -1),  # null
        }

        # Digit tokens
        self.digit_tokens = {vocab.get(str(i), -1) for i in range(10)}
        self.digit_tokens.update({vocab.get('-', -1)})  # negative numbers

        # Boolean tokens
        self.true_tokens = {
            vocab.get('true', -1),
            vocab.get('t', -1),
            vocab.get('tr', -1),
            vocab.get('tru', -1),
        }
        self.false_tokens = {
            vocab.get('false', -1),
            vocab.get('f', -1),
            vocab.get('fa', -1),
            vocab.get('fal', -1),
            vocab.get('fals', -1),
        }

        # Null tokens
        self.null_tokens = {
            vocab.get('null', -1),
            vocab.get('n', -1),
            vocab.get('nu', -1),
            vocab.get('nul', -1),
        }

        # Remove -1 (unknown tokens)
        self.structural_tokens.discard(-1)
        self.value_start_tokens.discard(-1)
        self.digit_tokens.discard(-1)

    def get_valid_tokens(self) -> Set[int]:
        """Get the set of valid token IDs for the current parse state.

        Returns:
            Set of valid token IDs.
        """
        current_state = self.state_stack[-1]

        if current_state == GrammarState.START:
            # At start, can begin any value
            return self.value_start_tokens

        elif current_state == GrammarState.IN_OBJECT:
            # In object, expect key (string) or closing brace
            vocab = self.tokenizer.get_vocab()
            return {vocab.get('"', -1), vocab.get('}', -1)} - {-1}

        elif current_state == GrammarState.EXPECT_KEY:
            # Expect a string key
            vocab = self.tokenizer.get_vocab()
            return {vocab.get('"', -1)} - {-1}

        elif current_state == GrammarState.EXPECT_VALUE:
            # Expect any value
            return self.value_start_tokens

        elif current_state == GrammarState.IN_ARRAY:
            # In array, expect value or closing bracket
            vocab = self.tokenizer.get_vocab()
            return self.value_start_tokens.union({vocab.get(']', -1)}) - {-1}

        elif current_state == GrammarState.EXPECT_COMMA_OR_END:
            # After value, expect comma or end of container
            vocab = self.tokenizer.get_vocab()
            if len(self.state_stack) > 1:
                parent_state = self.state_stack[-2]
                if parent_state == GrammarState.IN_OBJECT:
                    return {vocab.get(',', -1), vocab.get('}', -1)} - {-1}
                elif parent_state == GrammarState.IN_ARRAY:
                    return {vocab.get(',', -1), vocab.get(']', -1)} - {-1}
            return {vocab.get('}', -1), vocab.get(']', -1)} - {-1}

        elif current_state == GrammarState.IN_STRING:
            # In string, any character except unescaped quote
            # For simplicity, allow all tokens except structural ones
            vocab = self.tokenizer.get_vocab()
            all_tokens = set(range(len(vocab)))
            return all_tokens - self.structural_tokens

        elif current_state == GrammarState.IN_NUMBER:
            # In number, allow digits and decimal point
            vocab = self.tokenizer.get_vocab()
            return self.digit_tokens.union({vocab.get('.', -1), vocab.get('e', -1), vocab.get('E', -1)}) - {-1}

        elif current_state == GrammarState.IN_BOOLEAN:
            return self.true_tokens.union(self.false_tokens)

        elif current_state == GrammarState.IN_NULL:
            return self.null_tokens

        # Default: allow any token
        vocab = self.tokenizer.get_vocab()
        return set(range(len(vocab)))

    def update_state(self, token_id: int):
        """Update parse state after generating a token.

        Args:
            token_id: The generated token ID.
        """
        token_str = self.tokenizer.decode([token_id])
        vocab = self.tokenizer.get_vocab()

        # Update state based on token
        if token_id == vocab.get('{', -1):
            self.state_stack.append(GrammarState.IN_OBJECT)
            self.depth += 1
        elif token_id == vocab.get('}', -1):
            if self.state_stack and self.state_stack[-1] == GrammarState.IN_OBJECT:
                self.state_stack.pop()
                self.depth -= 1
                self.state_stack.append(GrammarState.EXPECT_COMMA_OR_END)
        elif token_id == vocab.get('[', -1):
            self.state_stack.append(GrammarState.IN_ARRAY)
            self.depth += 1
        elif token_id == vocab.get(']', -1):
            if self.state_stack and self.state_stack[-1] == GrammarState.IN_ARRAY:
                self.state_stack.pop()
                self.depth -= 1
                self.state_stack.append(GrammarState.EXPECT_COMMA_OR_END)
        elif token_id == vocab.get('"', -1):
            # Toggle string state
            if self.state_stack and self.state_stack[-1] == GrammarState.IN_STRING:
                self.state_stack.pop()
                self.state_stack.append(GrammarState.EXPECT_COMMA_OR_END)
            else:
                self.state_stack.append(GrammarState.IN_STRING)
        elif token_id == vocab.get(':', -1):
            # After colon, expect value
            self.state_stack.append(GrammarState.EXPECT_VALUE)
        elif token_id == vocab.get(',', -1):
            # After comma, expect next element
            if self.state_stack[-1] == GrammarState.EXPECT_COMMA_OR_END:
                self.state_stack.pop()

    def reset(self):
        """Reset parser state."""
        self.state_stack = [GrammarState.START]
        self.depth = 0


class GrammarConstrainedDecoder(NexusModule):
    """Decoder with grammar constraints for structured generation.

    Args:
        model: The language model.
        tokenizer: Tokenizer for the model.
        config: Grammar configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: GrammarConfig,
    ):
        super().__init__(config.__dict__)

        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.parser = GrammarParser(config, tokenizer)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Generate tokens with grammar constraints.

        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len).
            max_length: Maximum generation length.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated token IDs, shape (batch_size, seq_len + generated_len).
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Reset parser
        self.parser.reset()

        # Generate tokens
        generated = input_ids.clone()

        for _ in range(max_length):
            # Get model logits
            with torch.no_grad():
                outputs = self.model(generated)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                next_token_logits = logits[:, -1, :] / temperature

            # Apply grammar constraints
            valid_tokens = self.parser.get_valid_tokens()

            # Create mask for invalid tokens
            mask = torch.full_like(next_token_logits, float('-inf'))
            for token_id in valid_tokens:
                mask[:, token_id] = 0.0

            # Apply mask
            next_token_logits = next_token_logits + mask

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)

            # Nucleus sampling if top_p < 1.0
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(probs, num_samples=1)

            # Update parser state
            self.parser.update_state(next_token[0, 0].item())

            # Append token
            generated = torch.cat([generated, next_token], dim=1)

            # Check for end conditions
            if self.parser.depth == 0 and len(self.parser.state_stack) == 1:
                # Complete valid structure
                break

        return generated


def create_grammar_decoder(
    model: nn.Module,
    tokenizer: Any,
    config: Optional[GrammarConfig] = None
) -> GrammarConstrainedDecoder:
    """Create a grammar-constrained decoder.

    Args:
        model: Language model.
        tokenizer: Tokenizer.
        config: Grammar configuration (uses defaults if None).

    Returns:
        GrammarConstrainedDecoder instance.
    """
    config = config or GrammarConfig()
    return GrammarConstrainedDecoder(model, tokenizer, config)
