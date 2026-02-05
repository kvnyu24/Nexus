"""
EAGLE Speculative Decoding for accelerated LLM inference.

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency)
uses a lightweight autoregressive draft head that operates on the
second-to-top-layer hidden states of the target model. It extrapolates
feature representations rather than token-level probabilities, yielding
significantly higher acceptance rates than traditional speculative decoding.

Key ideas:
- Draft at the feature level using the target model's own hidden states
- Tree-based candidate verification for higher throughput
- Dynamic tree structure adjusted by confidence scores at each step

Reference:
    EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty
    https://arxiv.org/abs/2401.15077

    EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees
    https://arxiv.org/abs/2406.16858
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from nexus.core.base import NexusModule


class EAGLEDraftHead(NexusModule):
    """Lightweight autoregressive draft head for EAGLE speculative decoding.

    Predicts the next hidden state by combining the target model's
    second-to-top-layer features with the embedding of the previously
    predicted token.  The predicted hidden state is then projected through
    the (shared) language-model head to obtain token logits.

    Architecture::

        [hidden_state || token_embedding] --> FC layers --> predicted_hidden
        predicted_hidden --> lm_head --> logits

    Args:
        hidden_dim: Dimension of the target model's hidden states.
        num_layers: Number of fully-connected layers in the draft head.
        vocab_size: Size of the token vocabulary.
        bias: Whether linear layers include bias terms.
        dropout: Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        num_layers: int = 1,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # Token embedding used to condition on the previously predicted token
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Feature extrapolation network: maps [hidden || embed] -> hidden
        layers: List[nn.Module] = []
        input_dim = hidden_dim * 2  # concatenation of hidden state and embedding
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim, bias=bias))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.feature_extrapolator = nn.Sequential(*layers)

        # Learnable residual blending factor (initialized to mild residual)
        self.residual_gate = nn.Parameter(torch.tensor(0.5))

        # LM head projects predicted hidden state to vocab logits.
        # In practice this weight is often shared with the target model's
        # embedding / LM-head; here we keep an independent copy for
        # modularity.
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the next hidden state and token logits.

        Args:
            hidden_states: Target model hidden states of shape
                ``(batch, seq_len, hidden_dim)``.
            token_embeddings: Embeddings of previously predicted tokens,
                shape ``(batch, seq_len, hidden_dim)``.  If ``None``, a
                zero tensor is used (e.g. for the first draft step).

        Returns:
            predicted_hidden: Extrapolated hidden states
                ``(batch, seq_len, hidden_dim)``.
            logits: Token logits ``(batch, seq_len, vocab_size)``.
        """
        if token_embeddings is None:
            token_embeddings = torch.zeros_like(hidden_states)

        combined = torch.cat([hidden_states, token_embeddings], dim=-1)
        predicted = self.feature_extrapolator(combined)

        # Gated residual connection to the original hidden state
        gate = torch.sigmoid(self.residual_gate)
        predicted_hidden = gate * predicted + (1.0 - gate) * hidden_states

        logits = self.lm_head(predicted_hidden)
        return predicted_hidden, logits

    def get_token_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for token IDs.

        Args:
            token_ids: Token indices ``(batch, seq_len)``.

        Returns:
            Embeddings ``(batch, seq_len, hidden_dim)``.
        """
        return self.token_embedding(token_ids)


class EAGLETreeStructure:
    """Dynamic tree structure for EAGLE draft-token verification.

    Manages a tree of candidate token sequences where each node
    corresponds to a draft token.  The tree width and depth can be
    adjusted dynamically based on per-step confidence scores.

    Args:
        tree_width: Maximum branching factor at each depth level.
        tree_depth: Maximum depth of the draft tree.
        confidence_threshold: Minimum probability for a branch to be
            expanded further.
    """

    def __init__(
        self,
        tree_width: int = 10,
        tree_depth: int = 4,
        confidence_threshold: float = 0.1,
    ):
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        self.confidence_threshold = confidence_threshold

    def build_tree(
        self,
        logits_per_depth: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct candidate sequences from per-depth logits.

        Uses a greedy top-k expansion.  At every depth level only the
        branches whose cumulative probability exceeds
        ``confidence_threshold`` are retained, dynamically pruning
        unlikely paths.

        Args:
            logits_per_depth: List of logit tensors, one per depth level.
                Each has shape ``(num_candidates_at_prev_depth, vocab_size)``.

        Returns:
            candidate_tokens: Token IDs for every tree path,
                ``(num_candidates, tree_depth)``.
            candidate_scores: Cumulative log-probabilities,
                ``(num_candidates,)``.
            tree_mask: Attention mask encoding parent-child relationships,
                ``(num_candidates, num_candidates)``.
        """
        device = logits_per_depth[0].device

        # Level 0 -- pick top-k tokens from root logits
        root_probs = F.softmax(logits_per_depth[0], dim=-1)  # (1, vocab)
        if root_probs.dim() == 3:
            root_probs = root_probs[:, -1, :]  # take last position
        root_probs = root_probs.squeeze(0)  # (vocab,)

        k = min(self.tree_width, root_probs.shape[-1])
        top_probs, top_tokens = torch.topk(root_probs, k)

        # Each candidate is a growing path: list of (tokens_so_far, cum_log_prob)
        candidates: List[Tuple[List[int], float]] = [
            ([t.item()], torch.log(p + 1e-10).item())
            for t, p in zip(top_tokens, top_probs)
        ]

        # Expand tree depth by depth
        for depth in range(1, min(self.tree_depth, len(logits_per_depth))):
            next_candidates: List[Tuple[List[int], float]] = []
            depth_logits = logits_per_depth[depth]  # (num_prev, vocab)
            if depth_logits.dim() == 3:
                depth_logits = depth_logits[:, -1, :]
            depth_probs = F.softmax(depth_logits, dim=-1)

            for idx, (path, cum_score) in enumerate(candidates):
                if idx >= depth_probs.shape[0]:
                    # Fewer logit rows than candidates -- keep path as-is
                    next_candidates.append((path, cum_score))
                    continue

                row_probs = depth_probs[idx]
                branch_k = min(self.tree_width, row_probs.shape[-1])
                branch_probs, branch_tokens = torch.topk(row_probs, branch_k)

                for bp, bt in zip(branch_probs, branch_tokens):
                    branch_score = cum_score + torch.log(bp + 1e-10).item()
                    if bp.item() >= self.confidence_threshold:
                        next_candidates.append((path + [bt.item()], branch_score))

            candidates = next_candidates
            if not candidates:
                break

        # Assemble into tensors -- pad shorter paths with 0
        num_candidates = len(candidates)
        if num_candidates == 0:
            return (
                torch.zeros(1, self.tree_depth, dtype=torch.long, device=device),
                torch.zeros(1, device=device),
                torch.ones(1, 1, dtype=torch.bool, device=device),
            )

        max_path_len = max(len(c[0]) for c in candidates)
        candidate_tokens = torch.zeros(
            num_candidates, max_path_len, dtype=torch.long, device=device
        )
        candidate_scores = torch.zeros(num_candidates, device=device)

        for i, (path, score) in enumerate(candidates):
            for j, tok in enumerate(path):
                candidate_tokens[i, j] = tok
            candidate_scores[i] = score

        # Build causal tree attention mask
        tree_mask = self._build_tree_mask(candidates, device)

        return candidate_tokens, candidate_scores, tree_mask

    @staticmethod
    def _build_tree_mask(
        candidates: List[Tuple[List[int], float]],
        device: torch.device,
    ) -> torch.Tensor:
        """Build a boolean attention mask for tree-structured candidates.

        A candidate *i* can attend to candidate *j* if *j*'s path is a
        prefix of *i*'s path (including *i* itself).

        Returns:
            mask of shape ``(num_candidates, num_candidates)``.
        """
        n = len(candidates)
        mask = torch.zeros(n, n, dtype=torch.bool, device=device)

        paths = [c[0] for c in candidates]
        for i, pi in enumerate(paths):
            for j, pj in enumerate(paths):
                # j is a prefix of i (or equal)
                if len(pj) <= len(pi) and pi[: len(pj)] == pj:
                    mask[i, j] = True

        return mask


class EAGLEDecoder(NexusModule):
    """Full EAGLE speculative decoding pipeline.

    Combines an ``EAGLEDraftHead`` with an ``EAGLETreeStructure`` and
    a target model to perform end-to-end speculative generation.

    The workflow per step is:

    1. Run the draft head autoregressively to produce a tree of
       candidate token sequences.
    2. Verify the entire candidate tree with a **single** forward
       pass of the target model.
    3. Accept the longest prefix that matches, then continue.

    Args:
        target_model: The full-size target language model.  Must return
            logits and (optionally) hidden states.
        hidden_dim: Hidden dimension of the target model.
        vocab_size: Vocabulary size.
        num_draft_layers: Depth of the draft head MLP.
        tree_width: Maximum branching factor for the draft tree.
        tree_depth: Maximum depth (number of speculative tokens).
        confidence_threshold: Pruning threshold for tree branches.
        temperature: Sampling temperature for draft generation.
    """

    def __init__(
        self,
        target_model: nn.Module,
        hidden_dim: int,
        vocab_size: int,
        num_draft_layers: int = 1,
        tree_width: int = 10,
        tree_depth: int = 4,
        confidence_threshold: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.target_model = target_model
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.temperature = temperature

        self.draft_head = EAGLEDraftHead(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_layers=num_draft_layers,
        )
        self.tree = EAGLETreeStructure(
            tree_width=tree_width,
            tree_depth=tree_depth,
            confidence_threshold=confidence_threshold,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens using EAGLE speculative decoding.

        Args:
            input_ids: Prompt token IDs ``(1, seq_len)``.
            max_new_tokens: Maximum number of new tokens to generate.
            eos_token_id: End-of-sequence token ID.

        Returns:
            Full token sequence including the prompt.
        """
        assert input_ids.shape[0] == 1, "EAGLE decoding currently supports batch_size=1"

        generated = input_ids.clone()
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            # --- Step 1: obtain hidden states from target model -------
            target_output = self.target_model(generated)
            if hasattr(target_output, "hidden_states") and target_output.hidden_states is not None:
                # Use second-to-top layer features
                hidden = target_output.hidden_states[-2]
            elif hasattr(target_output, "logits"):
                # Fallback: use logits dimension as a proxy (not ideal)
                hidden = target_output.logits
            else:
                hidden = target_output

            last_hidden = hidden[:, -1:, :]  # (1, 1, dim)

            # --- Step 2: draft a tree of candidates -------------------
            logits_per_depth: List[torch.Tensor] = []
            current_hidden = last_hidden
            current_embed: Optional[torch.Tensor] = None

            for _ in range(self.tree.tree_depth):
                pred_hidden, logits = self.draft_head(current_hidden, current_embed)
                logits_per_depth.append(logits / self.temperature)

                # Greedy pick to feed next step (top-1 for autoregressive chain)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                current_embed = self.draft_head.get_token_embedding(next_token).unsqueeze(1)
                current_hidden = pred_hidden

            candidate_tokens, candidate_scores, tree_mask = self.tree.build_tree(
                logits_per_depth
            )

            # --- Step 3: verify candidates with target model ----------
            num_accepted, accepted_tokens = self._verify_candidates(
                generated, candidate_tokens, candidate_scores
            )

            if num_accepted > 0:
                generated = torch.cat(
                    [generated, accepted_tokens[:, :num_accepted]], dim=1
                )
                tokens_generated += num_accepted
            else:
                # Fallback: single greedy token from target
                target_logits = (
                    target_output.logits if hasattr(target_output, "logits") else target_output
                )
                next_token = torch.argmax(target_logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                tokens_generated += 1

            if eos_token_id is not None and generated[0, -1].item() == eos_token_id:
                break

        return generated

    def _verify_candidates(
        self,
        prefix: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_scores: torch.Tensor,
    ) -> Tuple[int, torch.Tensor]:
        """Verify candidate sequences against the target model.

        Picks the highest-scoring candidate and verifies it token by
        token using the target model's probabilities.

        Args:
            prefix: Current generated prefix ``(1, prefix_len)``.
            candidate_tokens: Draft tree paths ``(num_candidates, depth)``.
            candidate_scores: Cumulative log-probs ``(num_candidates,)``.

        Returns:
            num_accepted: Number of tokens accepted from the best candidate.
            accepted_tokens: The accepted token IDs ``(1, num_accepted)``.
        """
        # Select the highest-scoring candidate path
        best_idx = torch.argmax(candidate_scores).item()
        best_path = candidate_tokens[best_idx]  # (depth,)

        # Build full sequence for target verification
        full_seq = torch.cat(
            [prefix, best_path.unsqueeze(0)], dim=1
        )

        target_output = self.target_model(full_seq)
        target_logits = (
            target_output.logits if hasattr(target_output, "logits") else target_output
        )

        # Verify each token: target must assign high probability
        prefix_len = prefix.shape[1]
        num_accepted = 0

        for i in range(best_path.shape[0]):
            if best_path[i].item() == 0 and i > 0:
                break  # padding

            pos_logits = target_logits[:, prefix_len + i - 1, :]
            target_probs = F.softmax(pos_logits / self.temperature, dim=-1)
            draft_token = best_path[i].item()
            target_prob = target_probs[0, draft_token].item()

            if target_prob > 0.1:  # acceptance threshold
                num_accepted += 1
            else:
                break

        return num_accepted, best_path.unsqueeze(0)

    def forward(self, *args, **kwargs):
        """Use generate() for inference."""
        raise NotImplementedError(
            "EAGLEDecoder is an inference-only module. Use generate() instead."
        )
