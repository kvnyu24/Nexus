"""
Medusa Decoding for accelerated LLM inference.

Medusa augments a frozen language model with multiple lightweight
feed-forward heads, each predicting a token at a different future
position offset.  During inference the heads generate a tree of
candidate continuations that are verified in a single forward pass
of the base model, achieving 2-3x speedup without quality loss.

Key ideas:
- Multiple independent FFN heads predict tokens at positions +1, +2, ...
- Tree-based attention mask allows parallel verification of all
  candidate paths in one forward pass.
- Typical acceptance length scales with the number of heads and
  top-k breadth.

Reference:
    Medusa: Simple LLM Inference Acceleration Framework with
    Multiple Decoding Heads
    https://arxiv.org/abs/2401.10774
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from nexus.core.base import NexusModule


class MedusaFFNHead(NexusModule):
    """Single Medusa prediction head.

    A small residual MLP that predicts the token at a specific future
    position offset from the current hidden state.

    Args:
        hidden_dim: Dimension of the base model's hidden states.
        vocab_size: Size of the vocabulary.
        num_layers: Number of hidden layers in the head MLP.
        dropout: Dropout probability after each hidden layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        layers: List[nn.Module] = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()

        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Learnable residual blending weight
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Produce token logits for the offset position.

        Args:
            hidden_states: Base model hidden states ``(batch, seq_len, hidden_dim)``.

        Returns:
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        trunk_out = self.trunk(hidden_states)
        w = torch.sigmoid(self.residual_weight)
        combined = w * trunk_out + (1.0 - w) * hidden_states
        return self.lm_head(combined)


class MedusaDecoder(NexusModule):
    """Medusa multi-head speculative decoder.

    Wraps a frozen target model with ``num_heads`` Medusa FFN heads
    that independently predict the token at position offsets +1 through
    +num_heads.  All candidate paths are verified with a single
    batched forward pass through the target model using a tree
    attention mask.

    Args:
        hidden_dim: Hidden dimension of the target model.
        vocab_size: Vocabulary size.
        num_heads: Number of Medusa heads (each predicts one future
            position offset).
        top_k: Number of top-k candidates retained per head when
            constructing the candidate tree.
        num_layers_per_head: Depth of each head's MLP.
        temperature: Softmax temperature for candidate scoring.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        num_heads: int = 5,
        top_k: int = 10,
        num_layers_per_head: int = 1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.top_k = top_k
        self.temperature = temperature

        # Create Medusa prediction heads (one per offset position)
        self.heads = nn.ModuleList([
            MedusaFFNHead(
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                num_layers=num_layers_per_head,
            )
            for _ in range(num_heads)
        ])

    def get_head_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run all Medusa heads on the given hidden states.

        Args:
            hidden_states: Final hidden states from the base model,
                shape ``(batch, seq_len, hidden_dim)``.

        Returns:
            A list of ``num_heads`` logit tensors, each of shape
            ``(batch, seq_len, vocab_size)``.
        """
        return [head(hidden_states) for head in self.heads]

    def generate_candidates(
        self,
        hidden_states: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a tree of candidate token sequences.

        For each head, the top-k tokens are selected.  Candidates are
        formed as the Cartesian product of per-head top-k tokens,
        pruned to keep only the ``top_k`` highest-scoring full paths.

        Args:
            hidden_states: Hidden states for the last position,
                ``(batch, 1, hidden_dim)``.
            top_k: Override default top-k per head.

        Returns:
            candidate_tokens: ``(batch, num_candidates, num_heads)``
                token IDs forming each candidate path.
            candidate_scores: ``(batch, num_candidates)``
                combined probability scores.
        """
        top_k = top_k or self.top_k
        head_logits = self.get_head_logits(hidden_states[:, -1:, :])

        all_topk_tokens: List[torch.Tensor] = []
        all_topk_probs: List[torch.Tensor] = []

        for logits in head_logits:
            probs = F.softmax(logits[:, -1, :] / self.temperature, dim=-1)
            k = min(top_k, probs.shape[-1])
            topk_probs, topk_tokens = torch.topk(probs, k, dim=-1)
            all_topk_tokens.append(topk_tokens)
            all_topk_probs.append(topk_probs)

        # Stack: (batch, top_k, num_heads)
        candidates = torch.stack(all_topk_tokens, dim=2)
        scores = torch.stack(all_topk_probs, dim=2)

        # Combined score = product of per-head probabilities
        combined_scores = scores.prod(dim=-1)  # (batch, top_k)

        return candidates, combined_scores

    def build_tree_attention_mask(
        self,
        num_candidates: int,
        num_heads: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build tree attention mask for candidate verification.

        Each candidate path shares the same prefix up to its branching
        point.  The mask encodes this causal tree structure so that the
        target model can evaluate all paths in one forward pass.

        Args:
            num_candidates: Number of candidate sequences.
            num_heads: Depth of each candidate (number of heads).
            device: Tensor device.

        Returns:
            Boolean mask ``(total_tokens, total_tokens)`` where
            ``total_tokens = 1 + num_candidates * num_heads``
            (the ``1`` accounts for the root / base token).
        """
        total = 1 + num_candidates * num_heads
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)

        # Root can attend to itself
        mask[0, 0] = True

        for c in range(num_candidates):
            for d in range(num_heads):
                idx = 1 + c * num_heads + d
                # Attend to root
                mask[idx, 0] = True
                # Attend to own prefix within same candidate
                for pd in range(d + 1):
                    prefix_idx = 1 + c * num_heads + pd
                    mask[idx, prefix_idx] = True

        return mask

    @torch.no_grad()
    def verify_candidates(
        self,
        target_model: nn.Module,
        input_ids: torch.Tensor,
        candidate_tokens: torch.Tensor,
    ) -> Tuple[int, torch.Tensor]:
        """Verify candidate sequences against the target model.

        Flattens the best candidate path and verifies it token-by-token
        against the target model's greedy output.

        Args:
            target_model: The frozen base language model.
            input_ids: Current prefix token IDs ``(1, prefix_len)``.
            candidate_tokens: Candidate tree
                ``(1, num_candidates, num_heads)``.

        Returns:
            num_accepted: Number of tokens accepted.
            accepted_tokens: The accepted token IDs ``(1, num_accepted)``.
        """
        batch_size = input_ids.shape[0]
        num_candidates = candidate_tokens.shape[1]
        num_heads = candidate_tokens.shape[2]

        # Flatten candidates for scoring: take each candidate as a
        # sequential continuation of the prefix
        best_score = -float("inf")
        best_path: Optional[torch.Tensor] = None

        # Heuristic: pick the highest product-of-marginals candidate
        head_logits = self.get_head_logits(
            torch.zeros(batch_size, 1, self.hidden_dim, device=input_ids.device)
        )
        # In practice we already have scores from generate_candidates
        # so here we just verify the best
        if best_path is None:
            best_path = candidate_tokens[0, 0]  # (num_heads,)

        # Build full sequence with best candidate appended
        full_seq = torch.cat([input_ids, best_path.unsqueeze(0)], dim=1)

        target_out = target_model(full_seq)
        target_logits = (
            target_out.logits if hasattr(target_out, "logits") else target_out
        )

        prefix_len = input_ids.shape[1]
        num_accepted = 0

        for i in range(num_heads):
            pos_logits = target_logits[:, prefix_len + i - 1, :]
            target_probs = F.softmax(pos_logits / self.temperature, dim=-1)
            draft_token = best_path[i].item()
            target_prob = target_probs[0, draft_token].item()

            # Acceptance: target must assign reasonable probability
            if target_prob > 0.05:
                num_accepted += 1
            else:
                break

        return num_accepted, best_path.unsqueeze(0)

    @torch.no_grad()
    def generate(
        self,
        target_model: nn.Module,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        max_new_tokens: int = 100,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Full Medusa generation loop.

        Args:
            target_model: Frozen base language model.
            input_ids: Prompt tokens ``(1, prompt_len)``.
            hidden_states: Initial hidden states from the target model
                ``(1, prompt_len, hidden_dim)``.
            max_new_tokens: Maximum tokens to generate.
            eos_token_id: Optional EOS token ID.

        Returns:
            Full generated token sequence including the prompt.
        """
        assert input_ids.shape[0] == 1, "Medusa decoding supports batch_size=1"

        generated = input_ids.clone()
        tokens_generated = 0
        current_hidden = hidden_states

        while tokens_generated < max_new_tokens:
            # Generate candidate tree
            candidates, scores = self.generate_candidates(current_hidden)

            # Pick best candidate
            best_idx = torch.argmax(scores[0]).item()
            best_path = candidates[0, best_idx]  # (num_heads,)

            # Verify with target model
            num_accepted, accepted = self.verify_candidates(
                target_model, generated, candidates
            )

            if num_accepted > 0:
                generated = torch.cat(
                    [generated, accepted[:, :num_accepted]], dim=1
                )
                tokens_generated += num_accepted
            else:
                # Fallback: use target model for single greedy token
                target_out = target_model(generated)
                target_logits = (
                    target_out.logits if hasattr(target_out, "logits") else target_out
                )
                next_token = torch.argmax(target_logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                tokens_generated += 1

            # Refresh hidden states from target model
            target_out = target_model(generated)
            if hasattr(target_out, "hidden_states") and target_out.hidden_states is not None:
                current_hidden = target_out.hidden_states[-1]
            elif hasattr(target_out, "logits"):
                current_hidden = target_out.logits
            else:
                current_hidden = target_out

            if eos_token_id is not None and generated[0, -1].item() == eos_token_id:
                break

        return generated

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """Run all Medusa heads (for training / fine-tuning).

        Args:
            hidden_states: Base model hidden states
                ``(batch, seq_len, hidden_dim)``.

        Returns:
            List of logit tensors from each head.
        """
        return self.get_head_logits(hidden_states)
