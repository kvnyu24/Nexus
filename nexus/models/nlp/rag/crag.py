"""Corrective RAG (CRAG): Corrective Retrieval Augmented Generation.

Reference: Yan et al., "Corrective Retrieval Augmented Generation" (2024).
https://arxiv.org/abs/2401.15884

CRAG addresses the problem of unreliable retrieval in RAG systems by adding
a lightweight retrieval evaluator that assesses the quality of retrieved
documents before they are used for generation. Based on the evaluator's
assessment, the pipeline takes one of three actions:

    - Correct: Documents are relevant; use them directly (with filtering).
    - Incorrect: Documents are irrelevant; fall back to web search.
    - Ambiguous: Uncertain relevance; combine filtered documents with web
      search results.

The pipeline also includes a document decomposition-then-recomposition step
that strips irrelevant content from retrieved documents before feeding them
to the generator, improving factual grounding.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....core.base import NexusModule


class RetrievalConfidence(Enum):
    """Confidence levels for the retrieval evaluator's assessment."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    AMBIGUOUS = "ambiguous"


class RetrievalEvaluator(NexusModule):
    """Lightweight evaluator that scores retrieved documents for relevance.

    Takes a query and a set of retrieved documents and classifies each
    document as Correct (relevant), Incorrect (irrelevant), or Ambiguous
    (uncertain). This three-way classification drives the corrective
    retrieval pipeline's behavior.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Dimension of input embeddings.
            - confidence_threshold (float): Threshold above which a document
              is considered "Correct" (default 0.7).
            - ambiguity_threshold (float): Threshold below which a document
              is considered "Incorrect" (default 0.3). Between the two
              thresholds, it is "Ambiguous".
            - num_heads (int): Number of attention heads for query-document
              interaction (default 8).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.ambiguity_threshold = config.get("ambiguity_threshold", 0.3)

        # Query-document interaction via cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.get("num_heads", 8),
            batch_first=True,
        )
        self.interaction_norm = nn.LayerNorm(self.hidden_size)

        # Relevance classifier
        self.relevance_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_size // 2, 3),  # Correct/Ambiguous/Incorrect
        )

        # Fine-grained relevance scoring (continuous score)
        self.relevance_scorer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        query_embedding: torch.Tensor,
        document_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate relevance of retrieved documents to the query.

        Args:
            query_embedding: Query representation,
                shape (batch_size, hidden_size) or
                (batch_size, query_len, hidden_size).
            document_embeddings: Retrieved document embeddings,
                shape (batch_size, num_docs, hidden_size) or
                (batch_size, num_docs, doc_len, hidden_size).

        Returns:
            Dictionary containing:
                - confidence_labels: Categorical labels per document
                  (0=Correct, 1=Ambiguous, 2=Incorrect),
                  shape (batch_size, num_docs).
                - confidence_logits: Raw logits for three-way classification,
                  shape (batch_size, num_docs, 3).
                - relevance_scores: Continuous relevance scores in [0, 1],
                  shape (batch_size, num_docs).
                - action: Recommended pipeline action per batch item
                  ("correct", "incorrect", "ambiguous").
        """
        # Ensure 3D tensors for cross-attention
        if query_embedding.dim() == 2:
            query_embedding = query_embedding.unsqueeze(1)
        if document_embeddings.dim() == 4:
            # Pool over document token dimension
            document_embeddings = document_embeddings.mean(dim=2)

        batch_size, num_docs, _ = document_embeddings.shape

        # Cross-attention: query attends to each document
        query_expanded = query_embedding.expand(-1, num_docs, -1)
        query_flat = query_expanded.reshape(batch_size * num_docs, 1, -1)
        docs_flat = document_embeddings.reshape(
            batch_size * num_docs, 1, -1
        )

        interaction, _ = self.cross_attention(
            query_flat, docs_flat, docs_flat
        )
        interaction = self.interaction_norm(
            interaction.squeeze(1) + query_flat.squeeze(1)
        )

        # Classification
        confidence_logits = self.relevance_classifier(interaction)
        confidence_logits = confidence_logits.view(batch_size, num_docs, 3)

        # Continuous relevance scores
        relevance_scores = self.relevance_scorer(interaction)
        relevance_scores = relevance_scores.view(batch_size, num_docs)

        # Determine categorical labels based on thresholds
        confidence_labels = torch.where(
            relevance_scores >= self.confidence_threshold,
            torch.zeros_like(relevance_scores, dtype=torch.long),  # Correct
            torch.where(
                relevance_scores >= self.ambiguity_threshold,
                torch.ones_like(relevance_scores, dtype=torch.long),  # Ambiguous
                torch.full_like(
                    relevance_scores, 2, dtype=torch.long
                ),  # Incorrect
            ),
        )

        # Determine overall action per batch item (based on best document)
        max_scores, _ = relevance_scores.max(dim=-1)
        actions = []
        for score in max_scores:
            if score >= self.confidence_threshold:
                actions.append(RetrievalConfidence.CORRECT.value)
            elif score >= self.ambiguity_threshold:
                actions.append(RetrievalConfidence.AMBIGUOUS.value)
            else:
                actions.append(RetrievalConfidence.INCORRECT.value)

        return {
            "confidence_labels": confidence_labels,
            "confidence_logits": confidence_logits,
            "relevance_scores": relevance_scores,
            "actions": actions,
        }


class WebSearchFallback(NexusModule):
    """Simulates web search fallback when retrieved documents are unreliable.

    In practice, this module would interface with a web search API. In the
    neural implementation, it transforms the query into a search-optimized
    representation and produces synthetic search result embeddings that can
    be used as alternative context for generation.

    This serves as a differentiable proxy for the web search step in the
    CRAG pipeline, enabling end-to-end training.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Embedding dimension.
            - num_search_results (int): Number of synthetic search results
              to generate (default 5).
            - search_hidden_size (int): Internal dimension for the search
              module (default: same as hidden_size).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.num_search_results = config.get("num_search_results", 5)
        search_hidden = config.get("search_hidden_size", self.hidden_size)

        # Query reformulation for search
        self.query_reformulator = nn.Sequential(
            nn.Linear(self.hidden_size, search_hidden),
            nn.GELU(),
            nn.LayerNorm(search_hidden),
            nn.Linear(search_hidden, self.hidden_size),
        )

        # Generate synthetic search result embeddings
        self.search_result_generator = nn.Sequential(
            nn.Linear(self.hidden_size, search_hidden),
            nn.GELU(),
            nn.Linear(
                search_hidden, self.hidden_size * self.num_search_results
            ),
        )

        # Relevance scoring for generated results
        self.result_scorer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        query_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Generate fallback search results for the query.

        Args:
            query_embedding: Query representation,
                shape (batch_size, hidden_size).

        Returns:
            Dictionary containing:
                - search_results: Synthetic search result embeddings,
                  shape (batch_size, num_search_results, hidden_size).
                - reformulated_query: Search-optimized query representation,
                  shape (batch_size, hidden_size).
                - result_scores: Relevance scores for each result,
                  shape (batch_size, num_search_results).
        """
        batch_size = query_embedding.size(0)

        # Reformulate query for search
        reformulated = self.query_reformulator(query_embedding)

        # Generate search result embeddings
        result_flat = self.search_result_generator(reformulated)
        search_results = result_flat.view(
            batch_size, self.num_search_results, self.hidden_size
        )

        # Score each result against the query
        query_expanded = query_embedding.unsqueeze(1).expand_as(search_results)
        combined = torch.cat([query_expanded, search_results], dim=-1)
        result_scores = self.result_scorer(combined).squeeze(-1)

        return {
            "search_results": search_results,
            "reformulated_query": reformulated,
            "result_scores": result_scores,
        }


class DocumentFilter(NexusModule):
    """Decompose-then-recompose filter for retrieved documents.

    Implements the knowledge refinement step of CRAG: decomposes retrieved
    documents into fine-grained knowledge strips (segments), evaluates the
    relevance of each strip independently, filters out irrelevant strips,
    and recomposes the remaining strips into a refined document representation.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Embedding dimension.
            - num_strips (int): Number of strips to decompose each document
              into (default 8).
            - strip_threshold (float): Minimum relevance score for a strip
              to be kept (default 0.5).
            - num_heads (int): Attention heads for strip scoring (default 8).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.num_strips = config.get("num_strips", 8)
        self.strip_threshold = config.get("strip_threshold", 0.5)

        # Decomposition: project document embedding into strips
        self.decomposer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * self.num_strips),
            nn.GELU(),
        )

        # Per-strip normalization
        self.strip_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size) for _ in range(self.num_strips)
        ])

        # Strip relevance scoring against the query
        self.strip_scorer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )

        # Recomposition: aggregate relevant strips back into a document
        self.recomposer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

    def forward(
        self,
        query_embedding: torch.Tensor,
        document_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Filter documents by decomposing, scoring, and recomposing.

        Args:
            query_embedding: Query representation,
                shape (batch_size, hidden_size).
            document_embeddings: Document embeddings to filter,
                shape (batch_size, num_docs, hidden_size).

        Returns:
            Dictionary containing:
                - filtered_embeddings: Refined document embeddings after
                  filtering, shape (batch_size, num_docs, hidden_size).
                - strip_scores: Relevance scores for each strip,
                  shape (batch_size, num_docs, num_strips).
                - strip_mask: Binary mask of retained strips,
                  shape (batch_size, num_docs, num_strips).
                - retention_ratio: Fraction of strips retained per document,
                  shape (batch_size, num_docs).
        """
        batch_size, num_docs, _ = document_embeddings.shape

        # Decompose documents into knowledge strips
        doc_flat = document_embeddings.view(batch_size * num_docs, -1)
        strips_flat = self.decomposer(doc_flat)
        strips = strips_flat.view(
            batch_size * num_docs, self.num_strips, self.hidden_size
        )

        # Apply per-strip normalization
        normalized_strips = []
        for i, norm in enumerate(self.strip_norms):
            normalized_strips.append(norm(strips[:, i, :]))
        strips = torch.stack(normalized_strips, dim=1)

        # Score each strip against the query
        query_expanded = query_embedding.unsqueeze(1).unsqueeze(1).expand(
            -1, num_docs, self.num_strips, -1
        ).reshape(batch_size * num_docs, self.num_strips, -1)

        combined = torch.cat([query_expanded, strips], dim=-1)
        strip_scores = self.strip_scorer(combined).squeeze(-1)
        strip_scores = strip_scores.view(
            batch_size, num_docs, self.num_strips
        )

        # Filter strips below threshold
        strip_mask = (strip_scores >= self.strip_threshold).float()

        # Recompose: weighted sum of retained strips
        strips_reshaped = strips.view(
            batch_size, num_docs, self.num_strips, self.hidden_size
        )
        weighted_strips = strips_reshaped * (
            strip_scores * strip_mask
        ).unsqueeze(-1)
        aggregated = weighted_strips.sum(dim=2)

        # Normalize by number of retained strips
        num_retained = strip_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        aggregated = aggregated / num_retained

        # Final refinement
        filtered = self.recomposer(aggregated)

        retention_ratio = strip_mask.mean(dim=-1)

        return {
            "filtered_embeddings": filtered,
            "strip_scores": strip_scores,
            "strip_mask": strip_mask,
            "retention_ratio": retention_ratio,
        }


class CRAGPipeline(NexusModule):
    """Full Corrective Retrieval-Augmented Generation pipeline.

    Orchestrates the CRAG workflow:
        1. Evaluate retrieved documents for relevance.
        2. Based on the evaluation:
           - Correct: Filter and use retrieved documents.
           - Incorrect: Fall back to web search results.
           - Ambiguous: Combine filtered documents with web search results.
        3. Generate output conditioned on the selected context.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Common embedding dimension.
            - vocab_size (int): Output vocabulary size (optional).
            - confidence_threshold (float): Threshold for "Correct" (default 0.7).
            - ambiguity_threshold (float): Threshold for "Incorrect" (default 0.3).
            - num_search_results (int): Number of web search fallback results
              (default 5).
            - num_strips (int): Number of knowledge strips for filtering
              (default 8).
            - strip_threshold (float): Minimum strip relevance (default 0.5).
            - num_heads (int): Number of attention heads (default 8).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]

        # Pipeline components
        self.evaluator = RetrievalEvaluator(config)
        self.web_search = WebSearchFallback(config)
        self.document_filter = DocumentFilter(config)

        # Context fusion: merge query with (possibly combined) context
        self.context_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        # Optional output projection
        if "vocab_size" in config:
            self.output_proj = nn.Linear(
                self.hidden_size, config["vocab_size"]
            )
        else:
            self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        query_embedding: torch.Tensor,
        document_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run the CRAG pipeline.

        Args:
            query_embedding: Query representation,
                shape (batch_size, hidden_size).
            document_embeddings: Retrieved document embeddings,
                shape (batch_size, num_docs, hidden_size).
            attention_mask: Optional mask for documents,
                shape (batch_size, num_docs).

        Returns:
            Dictionary containing:
                - output: Final fused representation,
                  shape (batch_size, hidden_size).
                - logits: Output logits,
                  shape (batch_size, vocab_size or hidden_size).
                - evaluation: Retrieval evaluator results.
                - filtered_docs: Filtered document embeddings (if used).
                - search_results: Web search results (if triggered).
                - action_taken: Which action was taken per batch item.
                - context_source: Describes the context source per batch item.
        """
        batch_size = query_embedding.size(0)

        # Step 1: Evaluate retrieved documents
        evaluation = self.evaluator(query_embedding, document_embeddings)
        actions = evaluation["actions"]

        # Step 2: Filter documents (always compute for differentiability)
        filter_out = self.document_filter(query_embedding, document_embeddings)
        filtered_docs = filter_out["filtered_embeddings"]

        # Step 3: Web search fallback (always compute for differentiability)
        search_out = self.web_search(query_embedding)
        search_results = search_out["search_results"]

        # Step 4: Select context based on evaluation
        # Use soft gating for differentiability during training
        relevance_scores = evaluation["relevance_scores"]
        max_relevance = relevance_scores.max(dim=-1, keepdim=True)[0]

        # Soft confidence gate: high confidence -> use filtered docs,
        # low confidence -> use search results
        confidence_gate = torch.sigmoid(
            (max_relevance - self.evaluator.ambiguity_threshold) * 10.0
        ).unsqueeze(-1)

        # Pool contexts
        filtered_context = filtered_docs.mean(dim=1)  # (batch, hidden)
        search_context = search_results.mean(dim=1)  # (batch, hidden)

        # Blended context
        blended_context = (
            confidence_gate * filtered_context
            + (1 - confidence_gate) * search_context
        )

        # Step 5: Fuse query with selected context
        fused = self.context_fusion(
            torch.cat([query_embedding, blended_context], dim=-1)
        )

        logits = self.output_proj(fused)

        # Track context sources for interpretability
        context_sources = []
        for action in actions:
            if action == RetrievalConfidence.CORRECT.value:
                context_sources.append("filtered_documents")
            elif action == RetrievalConfidence.INCORRECT.value:
                context_sources.append("web_search")
            else:
                context_sources.append("blended")

        return {
            "output": fused,
            "logits": logits,
            "evaluation": evaluation,
            "filtered_docs": filtered_docs,
            "search_results": search_results,
            "action_taken": actions,
            "context_source": context_sources,
            "confidence_gate": confidence_gate.squeeze(-1),
            "filter_retention": filter_out["retention_ratio"],
        }
