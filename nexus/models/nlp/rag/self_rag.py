"""Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.

Reference: Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique
through Self-Reflection" (2023). https://arxiv.org/abs/2310.11511

Self-RAG trains a single LM that adaptively retrieves passages on-demand,
generates text informed by retrieved passages, and reflects on its own
generation using special reflection tokens:
    - [Retrieve]: Decides whether to retrieve passages
    - [IsRelevant]: Assesses relevance of retrieved passages to the query
    - [IsSupported]: Checks if the generation is supported by the passage
    - [IsUseful]: Evaluates overall utility of the response

This module implements the core Self-RAG inference pipeline where the model
dynamically decides when retrieval is needed and self-critiques the quality
of retrieved documents and generated text.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....core.base import NexusModule


class ReflectionTokenType(Enum):
    """Special reflection tokens used in Self-RAG for self-assessment.

    Each token type corresponds to a different aspect of the retrieval-augmented
    generation process that the model evaluates.
    """
    RETRIEVE = "retrieve"
    IS_RELEVANT = "is_relevant"
    IS_SUPPORTED = "is_supported"
    IS_USEFUL = "is_useful"


class ReflectionTokens(NexusModule):
    """Learned embeddings and classifiers for Self-RAG reflection tokens.

    Each reflection token type has an associated classifier head that predicts
    a categorical assessment (e.g., whether to retrieve, whether a document is
    relevant, whether output is supported, whether it is useful).

    Args:
        config: Dictionary containing:
            - hidden_size (int): Dimension of the hidden representations.
            - num_retrieve_classes (int): Number of classes for the retrieve
              decision (default 2: yes/no).
            - num_relevance_classes (int): Number of relevance levels
              (default 2: relevant/irrelevant).
            - num_support_classes (int): Number of support levels
              (default 3: fully_supported/partially_supported/not_supported).
            - num_utility_classes (int): Number of utility levels
              (default 5: rating 1-5).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.num_retrieve_classes = config.get("num_retrieve_classes", 2)
        self.num_relevance_classes = config.get("num_relevance_classes", 2)
        self.num_support_classes = config.get("num_support_classes", 3)
        self.num_utility_classes = config.get("num_utility_classes", 5)

        # Classifier heads for each reflection token type
        self.retrieve_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size // 2),
            nn.Linear(self.hidden_size // 2, self.num_retrieve_classes),
        )

        self.relevance_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size // 2),
            nn.Linear(self.hidden_size // 2, self.num_relevance_classes),
        )

        self.support_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size // 2),
            nn.Linear(self.hidden_size // 2, self.num_support_classes),
        )

        self.utility_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size // 2),
            nn.Linear(self.hidden_size // 2, self.num_utility_classes),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type: ReflectionTokenType,
    ) -> Dict[str, torch.Tensor]:
        """Predict reflection token values from hidden states.

        Args:
            hidden_states: Tensor of shape (batch_size, hidden_size) representing
                the contextualized state at which to evaluate.
            token_type: Which reflection token classifier to apply.

        Returns:
            Dictionary containing:
                - logits: Raw classifier logits.
                - probabilities: Softmax probabilities over classes.
                - prediction: Argmax predicted class index.
        """
        if token_type == ReflectionTokenType.RETRIEVE:
            logits = self.retrieve_head(hidden_states)
        elif token_type == ReflectionTokenType.IS_RELEVANT:
            logits = self.relevance_head(hidden_states)
        elif token_type == ReflectionTokenType.IS_SUPPORTED:
            logits = self.support_head(hidden_states)
        elif token_type == ReflectionTokenType.IS_USEFUL:
            logits = self.utility_head(hidden_states)
        else:
            raise ValueError(f"Unknown reflection token type: {token_type}")

        probabilities = F.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "prediction": prediction,
        }


class SelfRAGModel(NexusModule):
    """Self-RAG model that decides when to retrieve and self-critiques outputs.

    This module wraps a language model with Self-RAG capabilities: it generates
    text segment by segment, deciding at each segment boundary whether retrieval
    is needed. When retrieval is triggered, it scores retrieved documents for
    relevance, generates candidate continuations conditioned on each document,
    and scores those continuations for factual support and overall utility.

    The final output is selected from the candidate that best balances relevance,
    support, and utility scores.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Model hidden dimension.
            - vocab_size (int): Vocabulary size.
            - max_seq_length (int): Maximum sequence length.
            - num_heads (int): Number of attention heads (default 8).
            - num_layers (int): Number of transformer layers (default 6).
            - retrieve_threshold (float): Probability threshold above which
              retrieval is triggered (default 0.5).
            - num_retrieved (int): Number of documents to retrieve (default 5).
            - segment_length (int): Number of tokens per generation segment
              before evaluating retrieval need (default 64).
            - relevance_weight (float): Weight for relevance score in candidate
              ranking (default 1.0).
            - support_weight (float): Weight for support score in candidate
              ranking (default 1.0).
            - utility_weight (float): Weight for utility score in candidate
              ranking (default 0.5).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.max_seq_length = config.get("max_seq_length", 1024)
        self.num_heads = config.get("num_heads", 8)
        self.num_layers = config.get("num_layers", 6)
        self.retrieve_threshold = config.get("retrieve_threshold", 0.5)
        self.num_retrieved = config.get("num_retrieved", 5)
        self.segment_length = config.get("segment_length", 64)

        # Scoring weights for candidate ranking
        self.relevance_weight = config.get("relevance_weight", 1.0)
        self.support_weight = config.get("support_weight", 1.0)
        self.utility_weight = config.get("utility_weight", 0.5)

        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_seq_length, self.hidden_size)
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=config.get("dropout", 0.1),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Reflection token classifiers
        self.reflection_tokens = ReflectionTokens(config)

        # Query projection for retrieval scoring
        self.query_projection = nn.Linear(self.hidden_size, self.hidden_size)

        # Document-conditioned fusion layer
        self.doc_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        # Output language model head
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # Tie weights with token embeddings
        self.lm_head.weight = self.token_embedding.weight

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input tokens to hidden representations.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_length).
            attention_mask: Optional mask of shape (batch_size, seq_length).

        Returns:
            Hidden states of shape (batch_size, seq_length, hidden_size).
        """
        seq_length = input_ids.size(1)
        embeddings = self.token_embedding(input_ids)
        embeddings = embeddings + self.position_embedding[:, :seq_length, :]

        if attention_mask is not None:
            # Convert to additive mask for TransformerEncoder
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        hidden_states = self.encoder(
            embeddings, src_key_padding_mask=src_key_padding_mask
        )
        return hidden_states

    def _should_retrieve(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decide whether retrieval is needed based on current hidden state.

        Uses the [Retrieve] reflection token classifier on the pooled hidden
        state to predict whether external knowledge retrieval would be beneficial.

        Args:
            hidden_states: Encoded hidden states of shape
                (batch_size, seq_length, hidden_size).

        Returns:
            Tuple of:
                - retrieve_decision: Boolean tensor of shape (batch_size,)
                  indicating whether to retrieve for each example.
                - retrieve_prob: Probability of retrieval for each example,
                  shape (batch_size,).
        """
        # Pool hidden states (use last token representation)
        pooled = hidden_states[:, -1, :]

        reflection = self.reflection_tokens(
            pooled, ReflectionTokenType.RETRIEVE
        )
        retrieve_prob = reflection["probabilities"][:, 1]  # P(retrieve=yes)
        retrieve_decision = retrieve_prob > self.retrieve_threshold

        return retrieve_decision, retrieve_prob

    def _score_relevance(
        self,
        query_hidden: torch.Tensor,
        document_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Score retrieved documents for relevance using [IsRelevant] token.

        Args:
            query_hidden: Pooled query representation of shape
                (batch_size, hidden_size).
            document_embeddings: Retrieved document embeddings of shape
                (batch_size, num_docs, hidden_size).

        Returns:
            Dictionary containing:
                - relevance_logits: Shape (batch_size, num_docs, num_relevance_classes).
                - relevance_scores: Shape (batch_size, num_docs), probability of
                  relevance for each document.
        """
        batch_size, num_docs, _ = document_embeddings.shape

        # Combine query with each document for relevance scoring
        query_expanded = query_hidden.unsqueeze(1).expand(-1, num_docs, -1)
        combined = self.doc_fusion(
            torch.cat([query_expanded, document_embeddings], dim=-1)
        )

        # Flatten for reflection head
        combined_flat = combined.view(batch_size * num_docs, -1)
        relevance_output = self.reflection_tokens(
            combined_flat, ReflectionTokenType.IS_RELEVANT
        )

        relevance_logits = relevance_output["logits"].view(
            batch_size, num_docs, -1
        )
        relevance_scores = relevance_output["probabilities"].view(
            batch_size, num_docs, -1
        )[:, :, 1]  # P(relevant)

        return {
            "relevance_logits": relevance_logits,
            "relevance_scores": relevance_scores,
        }

    def _score_support(
        self, generation_hidden: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Score whether generation is supported by the evidence using [IsSupported].

        Args:
            generation_hidden: Hidden states of generated output, shape
                (batch_size, hidden_size).

        Returns:
            Dictionary with support logits, probabilities, and prediction.
        """
        return self.reflection_tokens(
            generation_hidden, ReflectionTokenType.IS_SUPPORTED
        )

    def _score_utility(
        self, generation_hidden: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Score overall utility of the generation using [IsUseful].

        Args:
            generation_hidden: Hidden states of generated output, shape
                (batch_size, hidden_size).

        Returns:
            Dictionary with utility logits, probabilities, and prediction.
        """
        return self.reflection_tokens(
            generation_hidden, ReflectionTokenType.IS_USEFUL
        )

    def _rank_candidates(
        self,
        relevance_scores: torch.Tensor,
        support_scores: torch.Tensor,
        utility_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Rank candidate generations using weighted combination of reflection scores.

        Args:
            relevance_scores: Shape (batch_size, num_candidates).
            support_scores: Shape (batch_size, num_candidates).
            utility_scores: Shape (batch_size, num_candidates).

        Returns:
            Indices of best candidates per batch, shape (batch_size,).
        """
        combined_score = (
            self.relevance_weight * relevance_scores
            + self.support_weight * support_scores
            + self.utility_weight * utility_scores
        )
        best_indices = torch.argmax(combined_score, dim=-1)
        return best_indices

    def forward(
        self,
        input_ids: torch.Tensor,
        document_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive retrieval and self-reflection.

        The model encodes the input, decides whether retrieval is needed, and
        if so, scores retrieved documents, generates document-conditioned
        outputs, evaluates support and utility, and selects the best candidate.

        Args:
            input_ids: Input token IDs, shape (batch_size, seq_length).
            document_embeddings: Pre-computed document embeddings for retrieval,
                shape (batch_size, num_docs, hidden_size). If None, retrieval
                is skipped regardless of the retrieve decision.
            attention_mask: Optional attention mask, shape (batch_size, seq_length).

        Returns:
            Dictionary containing:
                - logits: Output logits, shape (batch_size, seq_length, vocab_size).
                - hidden_states: Final hidden states.
                - retrieve_decision: Whether retrieval was triggered per example.
                - retrieve_probability: Probability of retrieval per example.
                - relevance_scores: Relevance scores for documents (if retrieved).
                - support_scores: Support assessment of final output.
                - utility_scores: Utility assessment of final output.
                - selected_candidates: Indices of selected document candidates.
        """
        # Encode input
        hidden_states = self._encode(input_ids, attention_mask)

        # Decide whether to retrieve
        retrieve_decision, retrieve_prob = self._should_retrieve(hidden_states)

        outputs = {
            "retrieve_decision": retrieve_decision,
            "retrieve_probability": retrieve_prob,
        }

        if document_embeddings is not None and retrieve_decision.any():
            batch_size = hidden_states.size(0)
            num_docs = document_embeddings.size(1)

            # Pool query for retrieval scoring
            query_hidden = self.query_projection(hidden_states[:, -1, :])

            # Score relevance of each retrieved document
            relevance_out = self._score_relevance(
                query_hidden, document_embeddings
            )
            outputs["relevance_scores"] = relevance_out["relevance_scores"]

            # Generate conditioned on each document
            query_expanded = hidden_states.unsqueeze(1).expand(
                -1, num_docs, -1, -1
            )
            docs_expanded = document_embeddings.unsqueeze(2).expand(
                -1, -1, hidden_states.size(1), -1
            )

            # Fuse query with documents
            fused = self.doc_fusion(
                torch.cat([query_expanded, docs_expanded], dim=-1)
            )  # (batch, num_docs, seq_len, hidden)

            # Compute LM logits for each candidate
            candidate_logits = self.lm_head(fused)

            # Score support for each candidate (pool over sequence)
            candidate_pooled = fused.mean(dim=2)  # (batch, num_docs, hidden)
            candidate_flat = candidate_pooled.view(batch_size * num_docs, -1)

            support_out = self._score_support(candidate_flat)
            support_probs = support_out["probabilities"].view(
                batch_size, num_docs, -1
            )
            # Use probability of "fully supported" as the support score
            support_scores = support_probs[:, :, 0]
            outputs["support_scores"] = support_scores

            utility_out = self._score_utility(candidate_flat)
            utility_probs = utility_out["probabilities"].view(
                batch_size, num_docs, -1
            )
            # Use expected utility (weighted sum)
            utility_levels = torch.arange(
                utility_probs.size(-1),
                device=utility_probs.device,
                dtype=utility_probs.dtype,
            )
            utility_scores = (utility_probs * utility_levels).sum(dim=-1)
            # Normalize to [0, 1]
            utility_scores = utility_scores / max(
                utility_probs.size(-1) - 1, 1
            )
            outputs["utility_scores"] = utility_scores

            # Rank candidates and select best
            best_indices = self._rank_candidates(
                relevance_out["relevance_scores"],
                support_scores,
                utility_scores,
            )
            outputs["selected_candidates"] = best_indices

            # Gather best candidate logits
            best_indices_expanded = best_indices.view(
                batch_size, 1, 1, 1
            ).expand(-1, -1, candidate_logits.size(2), candidate_logits.size(3))
            logits = candidate_logits.gather(1, best_indices_expanded).squeeze(1)
            outputs["logits"] = logits

            # Gather best candidate hidden states
            best_hidden_idx = best_indices.view(batch_size, 1, 1).expand(
                -1, -1, fused.size(2), -1
            )[:, :, :, :1].expand(-1, -1, fused.size(2), fused.size(3))
            best_hidden = fused.gather(1, best_hidden_idx).squeeze(1)
            outputs["hidden_states"] = best_hidden
        else:
            # No retrieval: generate directly
            logits = self.lm_head(hidden_states)
            outputs["logits"] = logits
            outputs["hidden_states"] = hidden_states

            # Still produce support/utility for the direct generation
            pooled = hidden_states[:, -1, :]
            support_out = self._score_support(pooled)
            outputs["support_scores"] = support_out["probabilities"]
            utility_out = self._score_utility(pooled)
            outputs["utility_scores"] = utility_out["probabilities"]

        return outputs
