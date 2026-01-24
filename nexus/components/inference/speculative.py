"""
Speculative Decoding for faster autoregressive generation.

Uses a smaller draft model to propose multiple tokens, then verifies
with the target model in a single pass.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Callable
from nexus.core.base import NexusModule


class SpeculativeDecoder(NexusModule):
    """Speculative Decoding for accelerated inference.

    The draft model proposes K tokens, the target model verifies them
    in parallel. Accepted tokens are kept, rejected ones are resampled.

    Reference: https://arxiv.org/abs/2211.17192

    Args:
        target_model: The main (large) model
        draft_model: The smaller draft model for speculation
        num_speculative_tokens: Number of tokens to speculate (K)
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        num_speculative_tokens: int = 5,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ):
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.num_speculative_tokens = num_speculative_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens using speculative decoding.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID

        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # For simplicity, assume batch_size = 1
        assert batch_size == 1, "Speculative decoding currently supports batch_size=1"

        generated = input_ids.clone()
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            # Step 1: Draft model proposes K tokens
            draft_tokens, draft_probs = self._draft_tokens(generated)

            # Step 2: Target model verifies all proposals in parallel
            num_accepted, next_token = self._verify_tokens(
                generated, draft_tokens, draft_probs
            )

            # Step 3: Accept tokens and append
            if num_accepted > 0:
                accepted = draft_tokens[:, :num_accepted]
                generated = torch.cat([generated, accepted], dim=1)
                tokens_generated += num_accepted

            # Append the corrected/new token from target
            if next_token is not None:
                generated = torch.cat([generated, next_token], dim=1)
                tokens_generated += 1

            # Check for EOS
            if eos_token_id is not None and generated[0, -1].item() == eos_token_id:
                break

        return generated

    def _draft_tokens(
        self,
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use draft model to propose speculative tokens.

        Returns:
            draft_tokens: Proposed tokens (1, K)
            draft_probs: Probabilities for each token (1, K, vocab_size)
        """
        draft_tokens = []
        draft_probs = []
        current_ids = input_ids.clone()

        for _ in range(self.num_speculative_tokens):
            # Get draft model prediction
            outputs = self.draft_model(current_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # Get last token logits
            next_logits = logits[:, -1, :] / self.temperature

            # Apply top-k/top-p filtering
            next_logits = self._filter_logits(next_logits)

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            draft_tokens.append(next_token)
            draft_probs.append(probs)

            # Append for next iteration
            current_ids = torch.cat([current_ids, next_token], dim=1)

        return (
            torch.cat(draft_tokens, dim=1),  # (1, K)
            torch.stack(draft_probs, dim=1)  # (1, K, vocab_size)
        )

    def _verify_tokens(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor
    ) -> Tuple[int, Optional[torch.Tensor]]:
        """
        Verify draft tokens with target model.

        Uses rejection sampling: accept if target_prob >= draft_prob,
        otherwise accept with probability target_prob / draft_prob.

        Returns:
            num_accepted: Number of accepted tokens
            next_token: Next token from target (either correction or continuation)
        """
        K = draft_tokens.shape[1]

        # Concatenate input with draft tokens for parallel verification
        full_sequence = torch.cat([input_ids, draft_tokens], dim=1)

        # Get target model probabilities for all positions
        outputs = self.target_model(full_sequence)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        target_logits = logits[:, -K-1:, :] / self.temperature

        # Get probabilities
        target_probs = F.softmax(target_logits, dim=-1)

        # Verify each draft token
        num_accepted = 0
        for i in range(K):
            draft_token = draft_tokens[0, i].item()
            p_draft = draft_probs[0, i, draft_token].item()
            p_target = target_probs[0, i, draft_token].item()

            # Rejection sampling
            if p_target >= p_draft:
                # Always accept
                num_accepted += 1
            else:
                # Accept with probability p_target / p_draft
                r = torch.rand(1).item()
                if r < p_target / p_draft:
                    num_accepted += 1
                else:
                    # Reject - need to resample from adjusted distribution
                    break

        # Get next token from target model
        # If all accepted, sample from position after last draft token
        # If some rejected, sample from the rejection position
        next_pos = num_accepted
        next_logits = target_logits[0, next_pos]

        # Adjust distribution if rejection occurred
        if num_accepted < K:
            # Sample from (target - draft) distribution, clipped to non-negative
            draft_token = draft_tokens[0, num_accepted].item()
            adjusted_probs = torch.clamp(
                target_probs[0, next_pos] - draft_probs[0, num_accepted],
                min=0.0
            )
            adjusted_probs = adjusted_probs / (adjusted_probs.sum() + 1e-10)
            next_token = torch.multinomial(adjusted_probs.unsqueeze(0), num_samples=1)
        else:
            # Sample normally from next position
            next_probs = F.softmax(next_logits.unsqueeze(0), dim=-1)
            next_token = torch.multinomial(next_probs, num_samples=1)

        return num_accepted, next_token

    def _filter_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply top-k and top-p filtering."""
        if self.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits


class NGramSpeculator(NexusModule):
    """N-gram based speculation from prompt.

    Looks for n-gram matches in the prompt to propose continuations,
    without needing a draft model.

    Args:
        n: N-gram size for matching
        max_speculation: Maximum tokens to speculate
    """

    def __init__(
        self,
        n: int = 3,
        max_speculation: int = 5
    ):
        super().__init__()
        self.n = n
        self.max_speculation = max_speculation

    def find_matches(
        self,
        input_ids: torch.Tensor,
        context_ids: torch.Tensor
    ) -> List[int]:
        """
        Find n-gram matches and return proposed continuation.

        Args:
            input_ids: Current generation prefix
            context_ids: Full context to search for matches

        Returns:
            List of proposed token IDs
        """
        if input_ids.shape[1] < self.n:
            return []

        # Get the last n tokens as query
        query = input_ids[0, -self.n:].tolist()
        context = context_ids[0].tolist()

        # Search for matches in context
        proposals = []
        for i in range(len(context) - self.n):
            if context[i:i+self.n] == query:
                # Found match, propose following tokens
                end_idx = min(i + self.n + self.max_speculation, len(context))
                proposals = context[i+self.n:end_idx]
                break

        return proposals
