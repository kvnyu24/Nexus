"""
Lookahead Decoding for parallel autoregressive inference.

Lookahead decoding uses Jacobi iteration to convert the inherently
sequential autoregressive generation process into a parallelisable
fixed-point computation.  At each step, the model simultaneously
generates *and* verifies multiple n-gram candidates in a single
forward pass -- no draft model and no extra training required.

Key ideas:
- Maintain a lookahead window of tentative future tokens.
- Update all positions in parallel via Jacobi iteration until the
  window converges (fixed-point).
- Collect confirmed n-gram matches from the converged prefix to
  skip past already-verified tokens.
- Falls back to standard autoregressive decoding when no n-gram
  candidates are found.

Reference:
    Break the Sequential Dependency of LLM Inference Using
    Lookahead Decoding
    https://arxiv.org/abs/2402.02057
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Set
from nexus.core.base import NexusModule


class NGramPool:
    """Pool of verified n-grams collected during generation.

    As Jacobi iterations converge, confirmed token sequences are
    stored here for future use as candidate continuations.

    Args:
        n_gram_size: Size of n-grams to track.
        max_pool_size: Maximum number of n-grams to retain.
    """

    def __init__(self, n_gram_size: int = 5, max_pool_size: int = 10000):
        self.n_gram_size = n_gram_size
        self.max_pool_size = max_pool_size
        # Maps (n-1)-gram prefix -> set of possible next tokens
        self._pool: Dict[Tuple[int, ...], Set[int]] = {}

    def add(self, tokens: List[int]) -> None:
        """Add all n-grams from a token sequence.

        Args:
            tokens: A list of token IDs whose n-grams will be indexed.
        """
        for i in range(len(tokens) - self.n_gram_size + 1):
            prefix = tuple(tokens[i : i + self.n_gram_size - 1])
            next_tok = tokens[i + self.n_gram_size - 1]
            if prefix not in self._pool:
                if len(self._pool) >= self.max_pool_size:
                    # Evict oldest entry (FIFO approximation)
                    oldest = next(iter(self._pool))
                    del self._pool[oldest]
                self._pool[prefix] = set()
            self._pool[prefix].add(next_tok)

    def lookup(self, prefix: List[int]) -> List[int]:
        """Look up candidate next tokens for a given prefix.

        Args:
            prefix: The (n-1)-token context.

        Returns:
            List of candidate next token IDs (may be empty).
        """
        key = tuple(prefix[-(self.n_gram_size - 1) :])
        return list(self._pool.get(key, set()))

    def size(self) -> int:
        """Number of distinct prefix entries in the pool."""
        return len(self._pool)

    def clear(self) -> None:
        """Remove all stored n-grams."""
        self._pool.clear()


class LookaheadBranch(NexusModule):
    """Generates n-gram candidate continuations via Jacobi iteration.

    Given the current prefix, this branch maintains a *lookahead window*
    of tentative future tokens.  At each step every position in the
    window is updated in parallel by the target model.  Positions that
    stabilise (same token across two consecutive iterations) contribute
    confirmed n-grams to the ``NGramPool``.

    Args:
        n_gram_size: N-gram length used for candidate matching.
        lookahead_window: Number of future positions in the Jacobi window.
        max_jacobi_iterations: Maximum Jacobi iterations per decode step
            before falling back.
    """

    def __init__(
        self,
        n_gram_size: int = 5,
        lookahead_window: int = 7,
        max_jacobi_iterations: int = 16,
    ):
        super().__init__()
        self.n_gram_size = n_gram_size
        self.lookahead_window = lookahead_window
        self.max_jacobi_iterations = max_jacobi_iterations

    @torch.no_grad()
    def step(
        self,
        model: nn.Module,
        prefix_ids: torch.Tensor,
        window: torch.Tensor,
        ngram_pool: NGramPool,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Run one round of Jacobi iteration on the lookahead window.

        All positions in ``window`` are fed to the model simultaneously
        and their predictions replace the current guesses.

        Args:
            model: Target language model.
            prefix_ids: Confirmed prefix ``(1, prefix_len)``.
            window: Current lookahead window ``(1, window_len)``.
            ngram_pool: Pool to collect confirmed n-grams.

        Returns:
            updated_window: New window after one parallel update.
            confirmed_tokens: Tokens confirmed (converged) during this step.
        """
        # Build full input: prefix + window
        full_input = torch.cat([prefix_ids, window], dim=1)

        output = model(full_input)
        logits = output.logits if hasattr(output, "logits") else output

        # Extract logits for the window positions
        prefix_len = prefix_ids.shape[1]
        window_logits = logits[:, prefix_len - 1 : prefix_len - 1 + window.shape[1], :]

        # Greedy update: replace each window slot with argmax prediction
        new_window = torch.argmax(window_logits, dim=-1)  # (1, window_len)

        # Check convergence -- positions where old == new
        converged = (new_window == window).squeeze(0)  # (window_len,)

        # Collect confirmed n-grams from converged prefix
        confirmed: List[int] = []
        all_tokens = prefix_ids[0].tolist() + new_window[0].tolist()

        if converged.any():
            # Find longest converged prefix of the window
            converged_len = 0
            for i in range(converged.shape[0]):
                if converged[i].item():
                    converged_len += 1
                else:
                    break

            if converged_len > 0:
                confirmed = new_window[0, :converged_len].tolist()
                ngram_pool.add(all_tokens)

        return new_window, confirmed

    def forward(self, *args, **kwargs):
        """Not used directly. Use step() instead."""
        raise NotImplementedError("Use step() for lookahead branch iteration.")


class VerificationBranch(NexusModule):
    """Validates candidate n-gram continuations in parallel.

    Given a set of candidate continuations sourced from the
    ``NGramPool``, this branch evaluates all of them simultaneously
    in a single forward pass of the target model and accepts the
    longest match.

    Args:
        max_candidates: Maximum number of candidate continuations
            to evaluate simultaneously.
    """

    def __init__(self, max_candidates: int = 10):
        super().__init__()
        self.max_candidates = max_candidates

    @torch.no_grad()
    def verify(
        self,
        model: nn.Module,
        prefix_ids: torch.Tensor,
        candidates: List[List[int]],
        temperature: float = 1.0,
    ) -> Tuple[int, List[int]]:
        """Verify candidate continuations against the target model.

        All candidates are evaluated in a single batched forward pass.
        The candidate whose tokens all match the model's greedy
        predictions (longest match wins) is accepted.

        Args:
            model: Target language model.
            prefix_ids: Current prefix ``(1, prefix_len)``.
            candidates: List of candidate token sequences.
            temperature: Softmax temperature.

        Returns:
            num_accepted: Length of the accepted continuation.
            accepted_tokens: Accepted token list.
        """
        if not candidates:
            return 0, []

        device = prefix_ids.device

        # Limit candidates to max_candidates
        candidates = candidates[: self.max_candidates]

        # Pad candidates to equal length
        max_len = max(len(c) for c in candidates)
        num_cands = len(candidates)

        # Build batched input: repeat prefix for each candidate
        prefix_len = prefix_ids.shape[1]

        batched_input = torch.zeros(
            num_cands, prefix_len + max_len, dtype=torch.long, device=device
        )
        for i, cand in enumerate(candidates):
            batched_input[i, :prefix_len] = prefix_ids[0]
            cand_t = torch.tensor(cand, dtype=torch.long, device=device)
            batched_input[i, prefix_len : prefix_len + len(cand)] = cand_t

        # Single batched forward pass
        output = model(batched_input)
        logits = output.logits if hasattr(output, "logits") else output

        # Verify each candidate
        best_accepted = 0
        best_tokens: List[int] = []

        for i, cand in enumerate(candidates):
            accepted = 0
            for j, token in enumerate(cand):
                pos = prefix_len + j - 1
                if pos < 0 or pos >= logits.shape[1]:
                    break

                pos_logits = logits[i, pos, :] / temperature
                pred_token = torch.argmax(pos_logits).item()

                if pred_token == token:
                    accepted += 1
                else:
                    break

            if accepted > best_accepted:
                best_accepted = accepted
                best_tokens = cand[:accepted]

        return best_accepted, best_tokens

    def forward(self, *args, **kwargs):
        """Not used directly. Use verify() instead."""
        raise NotImplementedError("Use verify() for candidate verification.")


class LookaheadDecoder(NexusModule):
    """Complete Lookahead Decoding pipeline.

    Orchestrates the ``LookaheadBranch`` (Jacobi iteration) and
    ``VerificationBranch`` (n-gram matching) to perform accelerated
    autoregressive generation *without* a separate draft model and
    *without* additional training.

    Workflow per decode step:

    1. Query the ``NGramPool`` for candidate continuations matching
       the current suffix.
    2. If candidates exist, verify them in parallel with the target
       model.  Accept the longest match.
    3. Regardless of acceptance, run one Jacobi iteration on the
       lookahead window to collect fresh n-grams.
    4. Advance the sequence by the number of accepted tokens (at
       least one from standard greedy decoding).

    Args:
        n_gram_size: Size of n-grams for candidate matching.
        max_candidates: Maximum candidates per verification step.
        lookahead_window: Jacobi iteration window size.
        max_jacobi_iterations: Max Jacobi iterations before fallback.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        n_gram_size: int = 5,
        max_candidates: int = 10,
        lookahead_window: int = 7,
        max_jacobi_iterations: int = 16,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_gram_size = n_gram_size
        self.temperature = temperature

        self.ngram_pool = NGramPool(
            n_gram_size=n_gram_size,
        )

        self.lookahead_branch = LookaheadBranch(
            n_gram_size=n_gram_size,
            lookahead_window=lookahead_window,
            max_jacobi_iterations=max_jacobi_iterations,
        )

        self.verification_branch = VerificationBranch(
            max_candidates=max_candidates,
        )

    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens using lookahead decoding.

        Args:
            model: Target language model.
            input_ids: Prompt token IDs ``(1, prompt_len)``.
            max_new_tokens: Maximum tokens to generate.
            eos_token_id: Optional end-of-sequence token.

        Returns:
            Full token sequence including the prompt.
        """
        assert input_ids.shape[0] == 1, "Lookahead decoding supports batch_size=1"
        device = input_ids.device

        generated = input_ids.clone()
        tokens_generated = 0

        # Seed the n-gram pool from the prompt
        self.ngram_pool.add(input_ids[0].tolist())

        # Initialize lookahead window (random guesses; will converge)
        window = torch.randint(
            0, 1000, (1, self.lookahead_branch.lookahead_window),
            dtype=torch.long, device=device,
        )

        while tokens_generated < max_new_tokens:
            prefix_tokens = generated[0].tolist()

            # --- Phase 1: n-gram candidate lookup and verification ----
            suffix = prefix_tokens[-(self.n_gram_size - 1):]
            pool_next = self.ngram_pool.lookup(suffix)

            candidates: List[List[int]] = []
            if pool_next:
                # Build candidate continuations from pool
                for tok in pool_next:
                    cand = [tok]
                    # Extend candidate greedily via pool lookups
                    cur_suffix = suffix[1:] + [tok]
                    for _ in range(self.n_gram_size - 1):
                        nxt = self.ngram_pool.lookup(cur_suffix)
                        if nxt:
                            cand.append(nxt[0])
                            cur_suffix = cur_suffix[1:] + [nxt[0]]
                        else:
                            break
                    candidates.append(cand)

            num_accepted = 0
            accepted_tokens: List[int] = []

            if candidates:
                num_accepted, accepted_tokens = self.verification_branch.verify(
                    model, generated, candidates, self.temperature,
                )

            if num_accepted > 0:
                accepted_t = torch.tensor(
                    [accepted_tokens], dtype=torch.long, device=device,
                )
                generated = torch.cat([generated, accepted_t], dim=1)
                tokens_generated += num_accepted
            else:
                # --- Fallback: standard greedy decode -----------------
                output = model(generated)
                logits = output.logits if hasattr(output, "logits") else output
                next_logits = logits[:, -1, :] / self.temperature
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                tokens_generated += 1

            # --- Phase 2: Jacobi iteration to harvest n-grams --------
            if tokens_generated < max_new_tokens:
                window, confirmed = self.lookahead_branch.step(
                    model, generated, window, self.ngram_pool,
                )

                # Also feed the newly generated tokens into the pool
                self.ngram_pool.add(generated[0].tolist())

            # --- Check EOS -------------------------------------------
            if eos_token_id is not None and generated[0, -1].item() == eos_token_id:
                break

        return generated

    def reset(self) -> None:
        """Clear internal state for a new generation session."""
        self.ngram_pool.clear()

    def forward(self, *args, **kwargs):
        """Use generate() for inference."""
        raise NotImplementedError(
            "LookaheadDecoder is an inference-only module. Use generate() instead."
        )
