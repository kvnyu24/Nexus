"""RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

Reference: Sarthi et al., "RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval" (2024). https://arxiv.org/abs/2401.18059

RAPTOR builds a hierarchical tree of document summaries through recursive
clustering and summarization:
    1. Start with leaf-level text chunks.
    2. Cluster chunks by semantic similarity using their embeddings.
    3. Summarize each cluster to produce higher-level nodes.
    4. Repeat clustering and summarization on the summaries to build a tree.

At query time, retrieval can occur at any level of the tree, allowing the
model to access both fine-grained details (leaf level) and high-level
abstractions (root level), depending on the query's scope.
"""

from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....core.base import NexusModule


class TextClusterer(NexusModule):
    """Clusters text chunk embeddings into semantically coherent groups.

    Uses a soft clustering approach where embeddings are projected into a
    clustering space and assigned to clusters based on cosine similarity.
    A learnable set of cluster centroids is maintained and refined through
    training.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Dimension of input embeddings.
            - max_clusters (int): Maximum number of clusters (default 50).
            - cluster_threshold (float): Minimum similarity for cluster
              assignment (default 0.5).
            - temperature (float): Temperature for soft assignment
              (default 0.1).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.max_clusters = config.get("max_clusters", 50)
        self.cluster_threshold = config.get("cluster_threshold", 0.5)
        self.temperature = config.get("temperature", 0.1)

        # Projection into clustering space
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        # Learnable cluster centroids
        self.centroids = nn.Parameter(
            torch.randn(self.max_clusters, self.hidden_size) * 0.02
        )

        # Cluster number predictor (predicts how many clusters to use)
        self.num_cluster_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.max_clusters),
        )

    def _compute_similarities(
        self,
        embeddings: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarities between embeddings and centroids.

        Args:
            embeddings: Shape (num_chunks, hidden_size).
            centroids: Shape (num_centroids, hidden_size).

        Returns:
            Similarity matrix of shape (num_chunks, num_centroids).
        """
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        centroids_norm = F.normalize(centroids, p=2, dim=-1)
        return torch.matmul(embeddings_norm, centroids_norm.t())

    def forward(
        self,
        chunk_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Cluster text chunk embeddings.

        Args:
            chunk_embeddings: Embeddings of text chunks,
                shape (num_chunks, hidden_size).

        Returns:
            Dictionary containing:
                - assignments: Soft cluster assignments,
                  shape (num_chunks, num_active_clusters).
                - hard_assignments: Hard cluster labels,
                  shape (num_chunks,).
                - cluster_embeddings: Weighted average embeddings per cluster,
                  shape (num_active_clusters, hidden_size).
                - num_clusters: Predicted number of active clusters.
                - similarities: Raw similarity scores,
                  shape (num_chunks, max_clusters).
        """
        projected = self.projection(chunk_embeddings)

        # Compute similarities to all centroids
        similarities = self._compute_similarities(projected, self.centroids)

        # Predict number of clusters based on global representation
        global_repr = chunk_embeddings.mean(dim=0)
        cluster_logits = self.num_cluster_predictor(global_repr)
        num_clusters = torch.argmax(cluster_logits).item() + 1
        num_clusters = min(
            max(num_clusters, 2), chunk_embeddings.size(0)
        )

        # Restrict to active centroids
        active_similarities = similarities[:, :num_clusters]

        # Soft assignments via temperature-scaled softmax
        assignments = F.softmax(
            active_similarities / self.temperature, dim=-1
        )

        # Hard assignments
        hard_assignments = torch.argmax(assignments, dim=-1)

        # Compute cluster embeddings as weighted averages
        cluster_embeddings = torch.matmul(
            assignments.t(), chunk_embeddings
        )
        # Normalize by cluster size
        cluster_sizes = assignments.sum(dim=0).unsqueeze(-1).clamp(min=1e-8)
        cluster_embeddings = cluster_embeddings / cluster_sizes

        return {
            "assignments": assignments,
            "hard_assignments": hard_assignments,
            "cluster_embeddings": cluster_embeddings,
            "num_clusters": num_clusters,
            "similarities": similarities,
        }


class RecursiveSummarizer(NexusModule):
    """Summarizes clusters of text chunks into higher-level representations.

    Uses a transformer-based architecture to read a set of chunk embeddings
    belonging to a cluster and produce a condensed summary embedding.
    Learnable summary tokens attend to the cluster members via cross-attention.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Dimension of embeddings.
            - num_summary_tokens (int): Number of summary tokens per cluster
              (default 4).
            - num_heads (int): Attention heads for cross-attention (default 8).
            - num_layers (int): Number of summarization layers (default 2).
            - dropout (float): Dropout rate (default 0.1).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.num_summary_tokens = config.get("num_summary_tokens", 4)
        self.num_heads = config.get("num_heads", 8)
        self.num_layers = config.get("num_layers", 2)

        # Learnable summary tokens
        self.summary_tokens = nn.Parameter(
            torch.randn(1, self.num_summary_tokens, self.hidden_size) * 0.02
        )

        # Cross-attention layers: summary tokens attend to cluster members
        self.cross_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()

        for _ in range(self.num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=self.hidden_size,
                    num_heads=self.num_heads,
                    dropout=config.get("dropout", 0.1),
                    batch_first=True,
                )
            )
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size * 4),
                    nn.GELU(),
                    nn.Dropout(config.get("dropout", 0.1)),
                    nn.Linear(self.hidden_size * 4, self.hidden_size),
                )
            )
            self.norms1.append(nn.LayerNorm(self.hidden_size))
            self.norms2.append(nn.LayerNorm(self.hidden_size))

        self.output_projection = nn.Linear(
            self.hidden_size * self.num_summary_tokens, self.hidden_size
        )

    def forward(
        self,
        cluster_embeddings: torch.Tensor,
        cluster_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Summarize a batch of clusters into higher-level representations.

        Args:
            cluster_embeddings: Embeddings of chunks within clusters,
                shape (num_clusters, max_chunks_per_cluster, hidden_size).
            cluster_mask: Optional mask indicating valid chunks per cluster,
                shape (num_clusters, max_chunks_per_cluster). False values
                indicate padding positions.

        Returns:
            Dictionary containing:
                - summary_embeddings: Summary embedding per cluster,
                  shape (num_clusters, hidden_size).
                - summary_tokens: Full summary token embeddings,
                  shape (num_clusters, num_summary_tokens, hidden_size).
                - attention_weights: Cross-attention weights from last layer.
        """
        num_clusters = cluster_embeddings.size(0)
        summary = self.summary_tokens.expand(num_clusters, -1, -1)

        # Convert mask to key_padding_mask format
        key_padding_mask = None
        if cluster_mask is not None:
            key_padding_mask = ~cluster_mask

        attn_weights = None
        for cross_attn, ffn, norm1, norm2 in zip(
            self.cross_attn_layers,
            self.ffn_layers,
            self.norms1,
            self.norms2,
        ):
            # Cross-attention
            attended, attn_weights = cross_attn(
                summary,
                cluster_embeddings,
                cluster_embeddings,
                key_padding_mask=key_padding_mask,
                need_weights=True,
            )
            summary = norm1(summary + attended)

            # Feedforward
            summary = norm2(summary + ffn(summary))

        # Pool summary tokens into single embedding per cluster
        summary_flat = summary.reshape(num_clusters, -1)
        pooled = self.output_projection(summary_flat)

        return {
            "summary_embeddings": pooled,
            "summary_tokens": summary,
            "attention_weights": attn_weights,
        }


class TreeRetriever(NexusModule):
    """Retrieves from the RAPTOR tree at multiple abstraction levels.

    Given a query, computes similarity scores against nodes at every level
    of the tree (leaf chunks, cluster summaries, higher-level summaries)
    and returns the top-k most relevant nodes across all levels.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Embedding dimension.
            - num_retrieved (int): Total number of nodes to retrieve across
              all levels (default 10).
            - level_weights (list): Optional weights for each tree level
              (higher weight = higher priority). Defaults to uniform.
            - max_depth (int): Maximum tree depth (default 3).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.num_retrieved = config.get("num_retrieved", 10)
        self.max_depth = config.get("max_depth", 3)

        level_weights = config.get("level_weights", None)
        if level_weights is not None:
            self.register_buffer(
                "level_weights",
                torch.tensor(level_weights, dtype=torch.float32),
            )
        else:
            self.register_buffer(
                "level_weights",
                torch.ones(self.max_depth + 1, dtype=torch.float32),
            )

        # Query projection
        self.query_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # Level-specific key projections
        self.level_key_projs = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size)
            for _ in range(self.max_depth + 1)
        ])

    def forward(
        self,
        query_embedding: torch.Tensor,
        tree_nodes: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """Retrieve relevant nodes from the RAPTOR tree.

        Args:
            query_embedding: Query representation,
                shape (batch_size, hidden_size).
            tree_nodes: List of node embeddings per tree level.
                tree_nodes[0] = leaf chunks, shape (num_leaves, hidden_size).
                tree_nodes[i] = level-i summaries, shape (num_nodes_i, hidden_size).

        Returns:
            Dictionary containing:
                - retrieved_embeddings: Top-k retrieved node embeddings,
                  shape (batch_size, num_retrieved, hidden_size).
                - retrieval_scores: Similarity scores for retrieved nodes,
                  shape (batch_size, num_retrieved).
                - level_indices: Which tree level each retrieved node came from,
                  shape (batch_size, num_retrieved).
                - node_indices: Index within the level for each retrieved node,
                  shape (batch_size, num_retrieved).
                - all_level_scores: Scores per level, list of tensors.
        """
        batch_size = query_embedding.size(0)
        query_proj = self.query_proj(query_embedding)

        all_scores = []
        all_embeddings = []
        all_level_ids = []
        all_node_ids = []

        for level_idx, level_nodes in enumerate(tree_nodes):
            if level_idx >= len(self.level_key_projs):
                break

            key_proj = self.level_key_projs[level_idx]
            keys = key_proj(level_nodes)

            # Compute similarity
            scores = torch.matmul(
                query_proj, keys.t()
            ) / (self.hidden_size ** 0.5)

            # Apply level weight
            level_weight = self.level_weights[level_idx]
            weighted_scores = scores * level_weight

            all_scores.append(weighted_scores)
            all_embeddings.append(level_nodes)
            all_level_ids.extend([level_idx] * level_nodes.size(0))
            all_node_ids.extend(range(level_nodes.size(0)))

        # Concatenate scores across all levels
        concat_scores = torch.cat(all_scores, dim=-1)
        concat_embeddings = torch.cat(all_embeddings, dim=0)
        level_ids = torch.tensor(
            all_level_ids, device=query_embedding.device
        )
        node_ids = torch.tensor(
            all_node_ids, device=query_embedding.device
        )

        # Select top-k
        num_to_retrieve = min(
            self.num_retrieved, concat_scores.size(-1)
        )
        top_scores, top_flat_indices = torch.topk(
            concat_scores, num_to_retrieve, dim=-1
        )

        # Gather embeddings
        expanded_indices = top_flat_indices.unsqueeze(-1).expand(
            -1, -1, self.hidden_size
        )
        all_emb_expanded = concat_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        retrieved_embeddings = torch.gather(
            all_emb_expanded, 1, expanded_indices
        )

        # Map flat indices back to level and node indices
        retrieved_level_ids = level_ids[top_flat_indices]
        retrieved_node_ids = node_ids[top_flat_indices]

        return {
            "retrieved_embeddings": retrieved_embeddings,
            "retrieval_scores": top_scores,
            "level_indices": retrieved_level_ids,
            "node_indices": retrieved_node_ids,
            "all_level_scores": all_scores,
        }


class RAPTOR(NexusModule):
    """Full RAPTOR pipeline: Recursive Abstractive Processing for
    Tree-Organized Retrieval.

    Builds a hierarchical tree from text chunks by iteratively clustering
    and summarizing, then retrieves from the tree at multiple abstraction
    levels to answer queries.

    Pipeline:
        1. Embed text chunks (leaf level).
        2. Cluster leaves by semantic similarity.
        3. Summarize each cluster to produce level-1 nodes.
        4. Repeat clustering and summarization up to max_depth.
        5. At query time, retrieve from any tree level.

    Args:
        config: Dictionary containing:
            - hidden_size (int): Embedding dimension.
            - vocab_size (int): Vocabulary size for optional text encoding.
            - max_depth (int): Maximum tree depth / recursion levels
              (default 3).
            - cluster_threshold (float): Minimum similarity for clustering
              (default 0.5).
            - max_clusters (int): Maximum clusters per level (default 50).
            - num_retrieved (int): Number of nodes to retrieve (default 10).
            - num_summary_tokens (int): Summary tokens per cluster (default 4).
            - num_heads (int): Attention heads (default 8).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config["hidden_size"]
        self.max_depth = config.get("max_depth", 3)

        # Text encoding (optional: embeddings may be pre-computed)
        if "vocab_size" in config:
            self.text_encoder = nn.Sequential(
                nn.Embedding(config["vocab_size"], self.hidden_size),
                nn.LayerNorm(self.hidden_size),
            )
        else:
            self.text_encoder = None

        # Clustering and summarization modules
        self.clusterer = TextClusterer(config)
        self.summarizer = RecursiveSummarizer(config)

        # Tree retriever
        self.retriever = TreeRetriever(config)

        # Context fusion for query + retrieved context
        self.context_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        self.output_proj = nn.Linear(
            self.hidden_size, config.get("vocab_size", self.hidden_size)
        )

    def build_tree(
        self,
        chunk_embeddings: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Build the RAPTOR tree by recursive clustering and summarization.

        Args:
            chunk_embeddings: Leaf-level text chunk embeddings,
                shape (num_chunks, hidden_size).

        Returns:
            List of node embeddings per level:
                tree[0] = original chunks (leaves).
                tree[i] = level-i summaries.
        """
        tree_levels = [chunk_embeddings]
        current_embeddings = chunk_embeddings

        for depth in range(self.max_depth):
            if current_embeddings.size(0) <= 1:
                break

            # Cluster current level
            cluster_out = self.clusterer(current_embeddings)
            num_clusters = cluster_out["num_clusters"]
            assignments = cluster_out["hard_assignments"]

            # Group embeddings by cluster for summarization
            max_cluster_size = 0
            cluster_groups = []
            for c in range(num_clusters):
                mask = assignments == c
                if mask.any():
                    cluster_members = current_embeddings[mask]
                    cluster_groups.append(cluster_members)
                    max_cluster_size = max(
                        max_cluster_size, cluster_members.size(0)
                    )

            if len(cluster_groups) == 0:
                break

            # Pad clusters to uniform size for batched summarization
            padded_clusters = torch.zeros(
                len(cluster_groups),
                max_cluster_size,
                self.hidden_size,
                device=current_embeddings.device,
            )
            cluster_mask = torch.zeros(
                len(cluster_groups),
                max_cluster_size,
                dtype=torch.bool,
                device=current_embeddings.device,
            )

            for i, group in enumerate(cluster_groups):
                padded_clusters[i, :group.size(0)] = group
                cluster_mask[i, :group.size(0)] = True

            # Summarize clusters
            summary_out = self.summarizer(padded_clusters, cluster_mask)
            level_summaries = summary_out["summary_embeddings"]

            tree_levels.append(level_summaries)
            current_embeddings = level_summaries

        return tree_levels

    def forward(
        self,
        query_embedding: torch.Tensor,
        chunk_embeddings: Optional[torch.Tensor] = None,
        tree_nodes: Optional[List[torch.Tensor]] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run the full RAPTOR pipeline.

        Either provide pre-built tree_nodes, or provide chunk_embeddings
        (or input_ids) to build the tree on-the-fly.

        Args:
            query_embedding: Query representation,
                shape (batch_size, hidden_size).
            chunk_embeddings: Pre-computed chunk embeddings,
                shape (num_chunks, hidden_size). Used if tree_nodes is None.
            tree_nodes: Pre-built tree node embeddings (list per level).
                If provided, skips tree construction.
            input_ids: Raw token IDs for text chunks,
                shape (num_chunks, seq_length). Only used if text_encoder
                is available and chunk_embeddings is None.

        Returns:
            Dictionary containing:
                - output: Fused representation, shape (batch_size, hidden_size).
                - logits: Output logits, shape (batch_size, vocab_size).
                - retrieved_embeddings: Retrieved tree node embeddings.
                - retrieval_scores: Retrieval similarity scores.
                - level_indices: Tree levels of retrieved nodes.
                - tree_depth: Number of levels in the tree.
        """
        # Build tree if not provided
        if tree_nodes is None:
            if chunk_embeddings is None:
                if input_ids is not None and self.text_encoder is not None:
                    chunk_embeddings = self.text_encoder(input_ids).mean(dim=1)
                else:
                    raise ValueError(
                        "Must provide one of: tree_nodes, chunk_embeddings, "
                        "or input_ids (with text_encoder configured)."
                    )
            tree_nodes = self.build_tree(chunk_embeddings)

        # Retrieve from tree
        retrieval_out = self.retriever(query_embedding, tree_nodes)

        # Fuse query with retrieved context
        retrieved_context = retrieval_out["retrieved_embeddings"].mean(dim=1)
        fused = self.context_fusion(
            torch.cat([query_embedding, retrieved_context], dim=-1)
        )

        logits = self.output_proj(fused)

        return {
            "output": fused,
            "logits": logits,
            "retrieved_embeddings": retrieval_out["retrieved_embeddings"],
            "retrieval_scores": retrieval_out["retrieval_scores"],
            "level_indices": retrieval_out["level_indices"],
            "tree_depth": len(tree_nodes),
        }
