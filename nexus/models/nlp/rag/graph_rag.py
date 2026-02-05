"""GraphRAG: Graph-based Retrieval-Augmented Generation.

Reference: Edge et al., "From Local to Global: A Graph RAG Approach to
Query-Focused Summarization" (2024). https://arxiv.org/abs/2404.16130

GraphRAG constructs a knowledge graph from source documents by extracting
entities and relationships, then uses hierarchical community detection
(e.g., Leiden algorithm) to partition the graph into communities at multiple
levels of granularity. Each community is pre-summarized. At query time,
relevant community summaries are retrieved and used as context for generation,
enabling the model to answer global sensemaking queries that require
synthesizing information across many documents.

Pipeline:
    1. EntityExtractor: Extract entities and relationships from text.
    2. KnowledgeGraph: Build and maintain the entity-relationship graph.
    3. CommunityDetector: Hierarchical community detection on the graph.
    4. CommunitySummarizer: Generate summaries for each detected community.
    5. GraphRAGRetriever: Retrieve relevant community summaries for queries.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....core.base import NexusModule


@dataclass
class Entity:
    """Represents a named entity extracted from text.

    Attributes:
        name: Canonical name of the entity.
        entity_type: Category of entity (e.g., PERSON, ORG, LOCATION).
        description: Brief description or context of the entity.
        source_doc_ids: IDs of documents where this entity appears.
    """
    name: str
    entity_type: str
    description: str = ""
    source_doc_ids: List[int] = field(default_factory=list)


@dataclass
class Relationship:
    """Represents a relationship between two entities.

    Attributes:
        source: Name of the source entity.
        target: Name of the target entity.
        relation_type: Type/label of the relationship.
        weight: Strength or confidence of the relationship.
        description: Textual description of the relationship.
    """
    source: str
    target: str
    relation_type: str
    weight: float = 1.0
    description: str = ""


@dataclass
class Community:
    """Represents a community of related entities in the knowledge graph.

    Attributes:
        community_id: Unique identifier for the community.
        level: Hierarchical level (0 = finest granularity).
        entity_names: Names of entities belonging to this community.
        summary_embedding: Pre-computed embedding of the community summary.
        summary_text: Human-readable summary of the community.
    """
    community_id: int
    level: int
    entity_names: List[str] = field(default_factory=list)
    summary_embedding: Optional[torch.Tensor] = None
    summary_text: str = ""


class EntityExtractor(NexusModule):
    """Neural entity and relationship extractor from document text.

    Uses a transformer encoder to identify entity spans and classify their
    types, then extracts pairwise relationships between identified entities
    using a bilinear relation classifier.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Model hidden dimension.
            - vocab_size (int): Vocabulary size for token embeddings.
            - num_entity_types (int): Number of entity type categories
              (default 10).
            - num_relation_types (int): Number of relationship types
              (default 20).
            - max_seq_length (int): Maximum input sequence length
              (default 512).
            - num_heads (int): Number of attention heads (default 8).
            - num_layers (int): Number of encoder layers (default 4).
            - entity_threshold (float): Confidence threshold for entity
              extraction (default 0.5).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.num_entity_types = config.get("num_entity_types", 10)
        self.num_relation_types = config.get("num_relation_types", 20)
        self.max_seq_length = config.get("max_seq_length", 512)
        self.entity_threshold = config.get("entity_threshold", 0.5)

        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_seq_length, self.hidden_size)
        )

        # Encoder
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

        # Entity span detection (BIO tagging)
        self.entity_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_size // 2, self.num_entity_types * 3),
        )  # *3 for B, I, O per type

        # Relation classifier: bilinear scoring between entity pair embeddings
        self.relation_head_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.relation_tail_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.relation_bilinear = nn.Bilinear(
            self.hidden_size, self.hidden_size, self.num_relation_types
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract entities and relationships from input text.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_length).
            attention_mask: Optional mask of shape (batch_size, seq_length).

        Returns:
            Dictionary containing:
                - entity_logits: BIO tag logits per token,
                  shape (batch_size, seq_length, num_entity_types * 3).
                - entity_probabilities: Softmax over BIO tags for each entity
                  type, shape (batch_size, seq_length, num_entity_types, 3).
                - hidden_states: Encoded representations,
                  shape (batch_size, seq_length, hidden_size).
                - relation_logits: Pairwise relation scores between all token
                  positions, shape (batch_size, seq_length, seq_length,
                  num_relation_types).
        """
        seq_length = input_ids.size(1)
        embeddings = self.token_embedding(input_ids)
        embeddings = embeddings + self.position_embedding[:, :seq_length, :]

        src_key_padding_mask = (
            (attention_mask == 0) if attention_mask is not None else None
        )
        hidden_states = self.encoder(
            embeddings, src_key_padding_mask=src_key_padding_mask
        )

        # Entity classification
        entity_logits = self.entity_classifier(hidden_states)
        batch_size = entity_logits.size(0)
        entity_logits_reshaped = entity_logits.view(
            batch_size, seq_length, self.num_entity_types, 3
        )
        entity_probabilities = F.softmax(entity_logits_reshaped, dim=-1)

        # Relation classification via bilinear scoring between all positions
        head_repr = self.relation_head_proj(hidden_states)
        tail_repr = self.relation_tail_proj(hidden_states)

        # Expand for pairwise scoring: (batch, seq_i, seq_j, hidden)
        head_expanded = head_repr.unsqueeze(2).expand(
            -1, -1, seq_length, -1
        )
        tail_expanded = tail_repr.unsqueeze(1).expand(
            -1, seq_length, -1, -1
        )

        # Flatten for bilinear, then reshape
        head_flat = head_expanded.reshape(-1, self.hidden_size)
        tail_flat = tail_expanded.reshape(-1, self.hidden_size)
        relation_logits = self.relation_bilinear(head_flat, tail_flat)
        relation_logits = relation_logits.view(
            batch_size, seq_length, seq_length, self.num_relation_types
        )

        return {
            "entity_logits": entity_logits,
            "entity_probabilities": entity_probabilities,
            "hidden_states": hidden_states,
            "relation_logits": relation_logits,
        }


class KnowledgeGraph(NexusModule):
    """Neural knowledge graph that stores entities and relationships as embeddings.

    Maintains a set of entity embeddings and a relation adjacency structure,
    enabling message passing over the graph for contextual entity representations.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Embedding dimension for entities and relations.
            - max_entities (int): Maximum number of entities the graph can hold
              (default 10000).
            - num_relation_types (int): Number of distinct relation types
              (default 20).
            - num_gnn_layers (int): Number of graph neural network message
              passing layers (default 2).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.max_entities = config.get("max_entities", 10000)
        self.num_relation_types = config.get("num_relation_types", 20)
        self.num_gnn_layers = config.get("num_gnn_layers", 2)

        # Entity embedding store
        self.entity_embeddings = nn.Embedding(
            self.max_entities, self.hidden_size
        )

        # Relation type embeddings
        self.relation_embeddings = nn.Embedding(
            self.num_relation_types, self.hidden_size
        )

        # Graph attention layers for message passing
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            self.gnn_layers.append(
                nn.ModuleDict({
                    "message_proj": nn.Linear(
                        self.hidden_size * 2, self.hidden_size
                    ),
                    "attention": nn.Linear(self.hidden_size, 1),
                    "update": nn.GRUCell(self.hidden_size, self.hidden_size),
                    "norm": nn.LayerNorm(self.hidden_size),
                })
            )

        # Track number of stored entities
        self._num_entities = 0

    def add_entities(
        self, entity_features: torch.Tensor
    ) -> torch.Tensor:
        """Add new entity embeddings to the graph.

        Args:
            entity_features: Feature vectors for new entities,
                shape (num_new_entities, hidden_size).

        Returns:
            Tensor of assigned entity indices, shape (num_new_entities,).
        """
        num_new = entity_features.size(0)
        if self._num_entities + num_new > self.max_entities:
            raise ValueError(
                f"Cannot add {num_new} entities: would exceed max_entities "
                f"({self.max_entities}). Current count: {self._num_entities}."
            )

        start_idx = self._num_entities
        indices = torch.arange(
            start_idx, start_idx + num_new, device=entity_features.device
        )

        # Initialize embeddings for new entities
        with torch.no_grad():
            self.entity_embeddings.weight[start_idx:start_idx + num_new] = (
                entity_features
            )

        self._num_entities += num_new
        return indices

    def forward(
        self,
        entity_indices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Perform message passing over the knowledge graph.

        Args:
            entity_indices: Indices of entities to compute representations for,
                shape (num_nodes,).
            edge_index: Edge list of shape (2, num_edges), where edge_index[0]
                contains source node indices and edge_index[1] contains target.
            edge_type: Relation type for each edge, shape (num_edges,).

        Returns:
            Dictionary containing:
                - entity_embeddings: Updated entity representations after
                  message passing, shape (num_nodes, hidden_size).
                - attention_weights: Attention weights from each GNN layer,
                  list of tensors.
        """
        node_features = self.entity_embeddings(entity_indices)
        all_attention_weights = []

        for layer in self.gnn_layers:
            source_idx = edge_index[0]
            target_idx = edge_index[1]

            # Get source features and relation embeddings
            source_features = node_features[source_idx]
            rel_features = self.relation_embeddings(edge_type)

            # Compose messages: concat source and relation
            messages = layer["message_proj"](
                torch.cat([source_features, rel_features], dim=-1)
            )

            # Compute attention scores
            attn_scores = layer["attention"](messages).squeeze(-1)

            # Scatter-based softmax per target node
            num_nodes = node_features.size(0)
            attn_max = torch.zeros(
                num_nodes, device=node_features.device
            ).scatter_reduce(
                0, target_idx, attn_scores, reduce="amax",
                include_self=False,
            )
            attn_scores = attn_scores - attn_max[target_idx]
            attn_exp = attn_scores.exp()
            attn_sum = torch.zeros(
                num_nodes, device=node_features.device
            ).scatter_add(0, target_idx, attn_exp)
            attn_weights = attn_exp / (attn_sum[target_idx] + 1e-8)
            all_attention_weights.append(attn_weights)

            # Aggregate weighted messages
            weighted_messages = messages * attn_weights.unsqueeze(-1)
            aggregated = torch.zeros_like(node_features).scatter_add(
                0,
                target_idx.unsqueeze(-1).expand_as(weighted_messages),
                weighted_messages,
            )

            # Update node features with GRU
            node_features = layer["update"](aggregated, node_features)
            node_features = layer["norm"](node_features)

        return {
            "entity_embeddings": node_features,
            "attention_weights": all_attention_weights,
        }


class CommunityDetector(NexusModule):
    """Hierarchical community detection on the knowledge graph.

    Implements a differentiable approximation of hierarchical graph clustering.
    At each level, nodes are softly assigned to communities using a learned
    assignment matrix, and the graph is coarsened for the next level.

    This follows the approach of DiffPool (Ying et al., 2018) adapted for
    hierarchical community detection as used in GraphRAG.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Dimension of node embeddings.
            - num_communities (int): Number of communities at the finest
              level (default 10).
            - community_levels (int): Number of hierarchical levels
              (default 3).
            - coarsen_ratio (float): Ratio of nodes to communities at each
              level (default 0.5).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.num_communities = config.get("num_communities", 10)
        self.community_levels = config.get("community_levels", 3)
        self.coarsen_ratio = config.get("coarsen_ratio", 0.5)

        # Assignment networks for each hierarchical level
        self.assignment_layers = nn.ModuleList()
        current_nodes = self.num_communities
        for level in range(self.community_levels):
            num_clusters = max(int(current_nodes * self.coarsen_ratio), 2)
            self.assignment_layers.append(
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.GELU(),
                    nn.Linear(self.hidden_size, num_clusters),
                )
            )
            current_nodes = num_clusters

        # Embedding transform per level
        self.embedding_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.LayerNorm(self.hidden_size),
            )
            for _ in range(self.community_levels)
        ])

    def forward(
        self,
        node_embeddings: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Perform hierarchical community detection.

        Args:
            node_embeddings: Entity embeddings, shape (num_nodes, hidden_size).
            adjacency: Optional adjacency matrix, shape (num_nodes, num_nodes).
                If None, a fully connected graph is assumed.

        Returns:
            Dictionary containing:
                - assignments: List of soft assignment matrices, one per level.
                  Each has shape (num_nodes_at_level, num_clusters_at_level).
                - community_embeddings: List of community embeddings per level.
                  Each has shape (num_clusters_at_level, hidden_size).
                - coarsened_adjacencies: List of coarsened adjacency matrices.
        """
        if adjacency is None:
            num_nodes = node_embeddings.size(0)
            adjacency = torch.ones(
                num_nodes, num_nodes, device=node_embeddings.device
            )

        current_embeddings = node_embeddings
        current_adjacency = adjacency

        all_assignments = []
        all_community_embeddings = []
        all_adjacencies = []

        for level in range(self.community_levels):
            # Compute soft assignment
            assignment_logits = self.assignment_layers[level](
                current_embeddings
            )
            assignment = F.softmax(assignment_logits, dim=-1)
            all_assignments.append(assignment)

            # Coarsen embeddings: S^T * X
            community_embeddings = torch.matmul(
                assignment.t(), current_embeddings
            )
            community_embeddings = self.embedding_transforms[level](
                community_embeddings
            )
            all_community_embeddings.append(community_embeddings)

            # Coarsen adjacency: S^T * A * S
            coarsened_adjacency = torch.matmul(
                torch.matmul(assignment.t(), current_adjacency), assignment
            )
            all_adjacencies.append(coarsened_adjacency)

            # Prepare for next level
            current_embeddings = community_embeddings
            current_adjacency = coarsened_adjacency

        return {
            "assignments": all_assignments,
            "community_embeddings": all_community_embeddings,
            "coarsened_adjacencies": all_adjacencies,
        }


class CommunitySummarizer(NexusModule):
    """Generates summaries for detected communities in the knowledge graph.

    Takes community embeddings (aggregated entity representations) and produces
    dense summary representations that capture the key information within each
    community. These summaries are later used for retrieval.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Dimension of embeddings.
            - summary_hidden_size (int): Hidden size of the summarization
              network (default: same as hidden_size).
            - num_summary_heads (int): Number of attention heads for
              cross-attention summarization (default 8).
            - summary_length (int): Number of summary tokens to produce
              per community (default 16).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.summary_hidden_size = config.get(
            "summary_hidden_size", self.hidden_size
        )
        self.num_summary_heads = config.get("num_summary_heads", 8)
        self.summary_length = config.get("summary_length", 16)

        # Learnable summary query tokens
        self.summary_queries = nn.Parameter(
            torch.randn(1, self.summary_length, self.hidden_size) * 0.02
        )

        # Cross-attention: summary queries attend to community entity embeddings
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_summary_heads,
            batch_first=True,
        )

        # Feedforward refinement
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.summary_hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.summary_hidden_size),
            nn.Linear(self.summary_hidden_size, self.hidden_size),
        )

        self.output_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        community_embeddings: torch.Tensor,
        entity_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate summary embeddings for communities.

        Args:
            community_embeddings: Embeddings of community nodes,
                shape (num_communities, hidden_size).
            entity_embeddings: Optional fine-grained entity embeddings to
                attend to, shape (num_communities, max_entities_per_community,
                hidden_size). If None, uses community_embeddings directly as
                key/value.

        Returns:
            Dictionary containing:
                - summary_embeddings: Dense summary for each community,
                  shape (num_communities, summary_length, hidden_size).
                - pooled_summaries: Mean-pooled summary per community,
                  shape (num_communities, hidden_size).
                - attention_weights: Cross-attention weights,
                  shape (num_communities, summary_length, num_kv_tokens).
        """
        num_communities = community_embeddings.size(0)

        # Expand summary queries for each community
        queries = self.summary_queries.expand(num_communities, -1, -1)

        # Key/value context
        if entity_embeddings is not None:
            kv = entity_embeddings
        else:
            kv = community_embeddings.unsqueeze(1)

        # Cross-attention
        summary, attn_weights = self.cross_attention(
            queries, kv, kv, need_weights=True
        )

        # Feedforward refinement with residual
        summary = summary + self.ffn(summary)
        summary = self.output_norm(summary)

        # Pool summaries
        pooled = summary.mean(dim=1)

        return {
            "summary_embeddings": summary,
            "pooled_summaries": pooled,
            "attention_weights": attn_weights,
        }


class GraphRAGRetriever(NexusModule):
    """Retrieves relevant community summaries for a given query.

    Maps the query into the same embedding space as community summaries,
    computes similarity scores, and returns the top-k most relevant
    community summaries to use as generation context.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Embedding dimension.
            - num_retrieved (int): Number of community summaries to retrieve
              (default 5).
            - temperature (float): Temperature for similarity softmax
              (default 1.0).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.num_retrieved = config.get("num_retrieved", 5)
        self.temperature = config.get("temperature", 1.0)

        # Query projection to community summary space
        self.query_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.summary_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        query_embedding: torch.Tensor,
        community_summaries: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Retrieve relevant community summaries for the query.

        Args:
            query_embedding: Query representation,
                shape (batch_size, hidden_size).
            community_summaries: Pre-computed community summary embeddings,
                shape (num_communities, hidden_size).

        Returns:
            Dictionary containing:
                - retrieved_summaries: Top-k community summaries,
                  shape (batch_size, num_retrieved, hidden_size).
                - retrieval_scores: Similarity scores for retrieved summaries,
                  shape (batch_size, num_retrieved).
                - retrieval_indices: Indices of retrieved communities,
                  shape (batch_size, num_retrieved).
                - all_scores: Scores for all communities,
                  shape (batch_size, num_communities).
        """
        # Project query and summaries
        query_proj = self.query_proj(query_embedding)
        summary_proj = self.summary_proj(community_summaries)

        # Compute similarity scores
        scores = torch.matmul(
            query_proj, summary_proj.t()
        ) / (self.hidden_size ** 0.5 * self.temperature)

        # Softmax over communities
        score_probs = F.softmax(scores, dim=-1)

        # Select top-k
        num_to_retrieve = min(
            self.num_retrieved, community_summaries.size(0)
        )
        top_scores, top_indices = torch.topk(
            score_probs, num_to_retrieve, dim=-1
        )

        # Gather retrieved summaries
        batch_size = query_embedding.size(0)
        expanded_indices = top_indices.unsqueeze(-1).expand(
            -1, -1, self.hidden_size
        )
        summaries_expanded = community_summaries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        retrieved = torch.gather(summaries_expanded, 1, expanded_indices)

        return {
            "retrieved_summaries": retrieved,
            "retrieval_scores": top_scores,
            "retrieval_indices": top_indices,
            "all_scores": score_probs,
        }


class GraphRAGPipeline(NexusModule):
    """Full GraphRAG pipeline: extract, build graph, detect communities,
    summarize, and retrieve for generation.

    Orchestrates the full GraphRAG process from raw document token IDs through
    entity extraction, knowledge graph construction, community detection,
    community summarization, and query-time retrieval.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Common hidden dimension across components.
            - vocab_size (int): Vocabulary size for the entity extractor.
            - num_entity_types (int): Number of entity types (default 10).
            - num_relation_types (int): Number of relation types (default 20).
            - max_entities (int): Max entities in the knowledge graph
              (default 10000).
            - num_communities (int): Target community count (default 10).
            - community_levels (int): Hierarchical levels (default 3).
            - num_retrieved (int): Communities to retrieve at query time
              (default 5).
            - max_seq_length (int): Max sequence length (default 512).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]

        self.entity_extractor = EntityExtractor(config)
        self.knowledge_graph = KnowledgeGraph(config)
        self.community_detector = CommunityDetector(config)
        self.community_summarizer = CommunitySummarizer(config)
        self.retriever = GraphRAGRetriever(config)

        # Generation: fuse query with retrieved community summaries
        self.context_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        self.output_proj = nn.Linear(
            self.hidden_size, config.get("vocab_size", self.hidden_size)
        )

    def forward(
        self,
        query_embedding: torch.Tensor,
        community_summaries: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the GraphRAG pipeline for query-time retrieval and generation.

        In the typical workflow:
          1. Documents are pre-processed offline to build entity_extractor ->
             knowledge_graph -> community_detector -> community_summarizer.
          2. At query time, pre-computed community_summaries are provided and
             the retriever selects the most relevant ones.

        If community_summaries are not provided but document_ids are given,
        the pipeline runs the full extraction and summarization on-the-fly
        (primarily for training or small-scale use).

        Args:
            query_embedding: Query representation,
                shape (batch_size, hidden_size).
            community_summaries: Pre-computed community summaries,
                shape (num_communities, hidden_size). If None, computed from
                document_ids.
            document_ids: Document token IDs for on-the-fly processing,
                shape (num_docs, seq_length). Only used if community_summaries
                is None.
            attention_mask: Attention mask for document_ids,
                shape (num_docs, seq_length).

        Returns:
            Dictionary containing:
                - output: Fused output representations,
                  shape (batch_size, hidden_size).
                - logits: Output logits, shape (batch_size, vocab_size).
                - retrieved_summaries: Retrieved community summaries.
                - retrieval_scores: Scores of retrieved communities.
                - community_embeddings: All community embeddings (if computed).
        """
        outputs = {}

        # Build community summaries on-the-fly if not provided
        if community_summaries is None:
            if document_ids is None:
                raise ValueError(
                    "Either community_summaries or document_ids must be provided."
                )

            # Extract entities from documents
            extraction = self.entity_extractor(document_ids, attention_mask)
            entity_hidden = extraction["hidden_states"]

            # Use mean-pooled document representations as entity proxies
            entity_features = entity_hidden.mean(dim=1)
            outputs["entity_features"] = entity_features

            # Detect communities
            community_out = self.community_detector(entity_features)
            outputs["community_assignments"] = community_out["assignments"]

            # Use the finest level community embeddings
            finest_community_embs = community_out["community_embeddings"][0]

            # Summarize communities
            summary_out = self.community_summarizer(finest_community_embs)
            community_summaries = summary_out["pooled_summaries"]
            outputs["community_embeddings"] = community_summaries

        # Retrieve relevant community summaries for the query
        retrieval_out = self.retriever(query_embedding, community_summaries)
        outputs["retrieved_summaries"] = retrieval_out["retrieved_summaries"]
        outputs["retrieval_scores"] = retrieval_out["retrieval_scores"]

        # Fuse query with retrieved summaries (mean-pool retrieved summaries)
        retrieved_context = retrieval_out["retrieved_summaries"].mean(dim=1)
        fused = self.context_fusion(
            torch.cat([query_embedding, retrieved_context], dim=-1)
        )
        outputs["output"] = fused
        outputs["logits"] = self.output_proj(fused)

        return outputs
