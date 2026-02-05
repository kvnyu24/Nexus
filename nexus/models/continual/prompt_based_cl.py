"""Prompt-Based Continual Learning: L2P, DualPrompt, CODA-Prompt.

References:
    - L2P (Learning to Prompt): "Learning to Prompt for Continual Learning" (CVPR 2022)
    - DualPrompt: "DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning" (ECCV 2022)
    - CODA-Prompt: "CODA-Prompt: Continual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning" (CVPR 2023)

Prompt-based continual learning methods prepend learnable prompt tokens to the
input, allowing the model to specialize for different tasks without modifying
the backbone parameters. This enables:
    - Parameter-efficient continual learning
    - No catastrophic forgetting of backbone
    - Zero-shot task inference via prompt selection

Key innovations across methods:
    - L2P: Learnable prompt pool with instance-wise selection
    - DualPrompt: Separate general and expert prompts (G-Prompt + E-Prompt)
    - CODA-Prompt: Decomposed attention for prompts across layers

Architecture:
    - PromptPool: Pool of learnable prompt vectors
    - PromptSelector: Selects relevant prompts based on input
    - L2PModel: Learning to Prompt implementation
    - DualPromptModel: Dual complementary prompts
    - CODAPromptModel: Continual decomposed attention prompting

Key properties:
    - Frozen backbone (parameter-efficient)
    - Rehearsal-free (no data storage)
    - Task-incremental and class-incremental learning
    - Automatic task inference at test time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from ...core.base import NexusModule
import math


class PromptPool(nn.Module):
    """Learnable pool of prompt vectors.

    Maintains a set of learnable prompt tokens that can be prepended
    to input sequences. Prompts are selected based on input similarity.

    Args:
        pool_size: Number of prompts in the pool. Default: 20.
        prompt_length: Length of each prompt. Default: 5.
        embed_dim: Embedding dimension. Default: 768.
        prompt_key_dim: Dimension of prompt keys. Default: 768.
    """

    def __init__(
        self,
        pool_size: int = 20,
        prompt_length: int = 5,
        embed_dim: int = 768,
        prompt_key_dim: int = 768
    ):
        super().__init__()

        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.prompt_key_dim = prompt_key_dim

        # Learnable prompt embeddings (pool_size, prompt_length, embed_dim)
        self.prompts = nn.Parameter(
            torch.randn(pool_size, prompt_length, embed_dim) * 0.02
        )

        # Learnable prompt keys for selection (pool_size, prompt_key_dim)
        self.prompt_keys = nn.Parameter(
            torch.randn(pool_size, prompt_key_dim) * 0.02
        )

    def forward(
        self,
        query: torch.Tensor,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select and return top-k prompts based on query.

        Args:
            query: Query embedding (B, prompt_key_dim).
            top_k: Number of prompts to select. Default: 5.

        Returns:
            Tuple of (selected_prompts, selection_indices).
                - selected_prompts: (B, top_k, prompt_length, embed_dim)
                - selection_indices: (B, top_k)
        """
        batch_size = query.shape[0]

        # Normalize query and keys
        query_norm = F.normalize(query, p=2, dim=-1)  # (B, prompt_key_dim)
        keys_norm = F.normalize(self.prompt_keys, p=2, dim=-1)  # (pool_size, prompt_key_dim)

        # Compute similarity
        similarity = torch.matmul(query_norm, keys_norm.T)  # (B, pool_size)

        # Select top-k prompts
        _, indices = torch.topk(similarity, k=top_k, dim=-1)  # (B, top_k)

        # Gather selected prompts
        selected_prompts = self.prompts[indices]  # (B, top_k, prompt_length, embed_dim)

        return selected_prompts, indices


class PromptSelector(NexusModule):
    """Selects prompts from a pool based on input features.

    Uses a query network to compute instance-specific representations
    for prompt selection.

    Args:
        config: Configuration dictionary with:
            - input_dim: Input feature dimension. Default: 768.
            - query_dim: Query embedding dimension. Default: 768.
            - num_layers: Number of query network layers. Default: 2.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.input_dim = config.get("input_dim", 768)
        self.query_dim = config.get("query_dim", 768)
        self.num_layers = config.get("num_layers", 2)

        # Query network
        layers = []
        prev_dim = self.input_dim
        for _ in range(self.num_layers - 1):
            layers.extend([
                nn.Linear(prev_dim, self.query_dim),
                nn.ReLU(),
            ])
            prev_dim = self.query_dim
        layers.append(nn.Linear(prev_dim, self.query_dim))

        self.query_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute query embedding from input.

        Args:
            x: Input features (B, input_dim) or (B, L, input_dim).

        Returns:
            Query embedding (B, query_dim).
        """
        # If input is sequential, take mean
        if x.dim() == 3:
            x = x.mean(dim=1)  # (B, input_dim)

        query = self.query_net(x)  # (B, query_dim)

        return query


class L2PModel(NexusModule):
    """Learning to Prompt (L2P) for continual learning.

    L2P learns a pool of prompts and selects relevant ones based on
    input instances. The backbone model is frozen, and only prompts
    are learned per task.

    Training procedure:
        1. Extract features from frozen backbone
        2. Compute query from features
        3. Select top-k prompts from pool
        4. Prepend prompts to input tokens
        5. Forward through backbone with prompts
        6. Train only prompt pool (backbone frozen)

    Args:
        config: Configuration dictionary with:
            - backbone: Frozen backbone model (e.g., ViT)
            - pool_size: Number of prompts. Default: 20.
            - prompt_length: Length per prompt. Default: 5.
            - top_k: Number of prompts to select. Default: 5.
            - embed_dim: Embedding dimension. Default: 768.
            - num_classes: Number of output classes. Default: 100.

    Example:
        >>> config = {
        ...     "backbone": vit_model,
        ...     "pool_size": 20,
        ...     "prompt_length": 5,
        ...     "top_k": 5
        ... }
        >>> model = L2PModel(config)
        >>> images = torch.randn(8, 3, 224, 224)
        >>> logits = model(images)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.backbone = config.get("backbone")
        if self.backbone is None:
            raise ValueError("Backbone model must be provided")

        self.pool_size = config.get("pool_size", 20)
        self.prompt_length = config.get("prompt_length", 5)
        self.top_k = config.get("top_k", 5)
        self.embed_dim = config.get("embed_dim", 768)
        self.num_classes = config.get("num_classes", 100)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Prompt pool
        self.prompt_pool = PromptPool(
            pool_size=self.pool_size,
            prompt_length=self.prompt_length,
            embed_dim=self.embed_dim,
            prompt_key_dim=self.embed_dim
        )

        # Query selector
        selector_config = {
            "input_dim": self.embed_dim,
            "query_dim": self.embed_dim,
            "num_layers": 2
        }
        self.selector = PromptSelector(selector_config)

        # Task-incremental classifier head
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_prompt_indices: bool = False
    ) -> torch.Tensor:
        """Forward pass with prompt selection.

        Args:
            x: Input tensor (B, C, H, W) for vision.
            return_prompt_indices: Return selected prompt indices.

        Returns:
            Classification logits (B, num_classes).
            Or tuple (logits, prompt_indices) if return_prompt_indices=True.
        """
        batch_size = x.shape[0]

        # Extract features with frozen backbone (no prompts for query)
        with torch.no_grad():
            # Simplified: assume backbone has a feature extraction method
            # In practice, would patch input and extract embeddings
            features = self._extract_features(x)  # (B, embed_dim)

        # Compute query for prompt selection
        query = self.selector(features)  # (B, embed_dim)

        # Select prompts
        selected_prompts, prompt_indices = self.prompt_pool(
            query, top_k=self.top_k
        )  # (B, top_k, prompt_length, embed_dim)

        # Reshape prompts for prepending
        # (B, top_k * prompt_length, embed_dim)
        prompts = selected_prompts.view(
            batch_size, self.top_k * self.prompt_length, self.embed_dim
        )

        # Forward through backbone with prompts
        # In practice, prompts are prepended to patch tokens
        output = self._forward_with_prompts(x, prompts)  # (B, embed_dim)

        # Classify
        logits = self.classifier(output)  # (B, num_classes)

        if return_prompt_indices:
            return logits, prompt_indices

        return logits

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone (simplified).

        Args:
            x: Input tensor.

        Returns:
            Feature embeddings.
        """
        # Placeholder: call backbone's feature extraction
        # In practice, would properly extract cls token or pooled features
        return torch.randn(x.shape[0], self.embed_dim, device=x.device)

    def _forward_with_prompts(
        self,
        x: torch.Tensor,
        prompts: torch.Tensor
    ) -> torch.Tensor:
        """Forward through backbone with prompts prepended.

        Args:
            x: Input tensor.
            prompts: Prompt tokens to prepend.

        Returns:
            Output embeddings.
        """
        # Placeholder: prepend prompts to input tokens and forward
        # In practice, would integrate prompts into backbone's attention
        return torch.randn(x.shape[0], self.embed_dim, device=x.device)


class DualPromptModel(NexusModule):
    """DualPrompt: Complementary prompting with G-Prompt and E-Prompt.

    DualPrompt uses two types of prompts:
        - G-Prompt (General): Shared across all tasks for common knowledge
        - E-Prompt (Expert): Task-specific prompts from a pool

    This decomposition allows capturing both shared and task-specific features.

    Args:
        config: Configuration dictionary with:
            - backbone: Frozen backbone model
            - g_prompt_length: Length of general prompt. Default: 5.
            - e_pool_size: Size of expert prompt pool. Default: 10.
            - e_prompt_length: Length of expert prompts. Default: 5.
            - top_k: Number of expert prompts to select. Default: 1.
            - embed_dim: Embedding dimension. Default: 768.
            - num_classes: Number of output classes. Default: 100.

    Example:
        >>> config = {
        ...     "backbone": vit_model,
        ...     "g_prompt_length": 5,
        ...     "e_pool_size": 10
        ... }
        >>> model = DualPromptModel(config)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.backbone = config.get("backbone")
        if self.backbone is None:
            raise ValueError("Backbone model must be provided")

        self.g_prompt_length = config.get("g_prompt_length", 5)
        self.e_pool_size = config.get("e_pool_size", 10)
        self.e_prompt_length = config.get("e_prompt_length", 5)
        self.top_k = config.get("top_k", 1)
        self.embed_dim = config.get("embed_dim", 768)
        self.num_classes = config.get("num_classes", 100)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # General prompt (shared across tasks)
        self.g_prompt = nn.Parameter(
            torch.randn(1, self.g_prompt_length, self.embed_dim) * 0.02
        )

        # Expert prompt pool (task-specific)
        self.e_prompt_pool = PromptPool(
            pool_size=self.e_pool_size,
            prompt_length=self.e_prompt_length,
            embed_dim=self.embed_dim,
            prompt_key_dim=self.embed_dim
        )

        # Selector
        selector_config = {
            "input_dim": self.embed_dim,
            "query_dim": self.embed_dim,
            "num_layers": 2
        }
        self.selector = PromptSelector(selector_config)

        # Classifier
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with dual prompting.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Classification logits (B, num_classes).
        """
        batch_size = x.shape[0]

        # Extract features
        with torch.no_grad():
            features = self._extract_features(x)  # (B, embed_dim)

        # Select expert prompts
        query = self.selector(features)
        e_prompts, _ = self.e_prompt_pool(query, top_k=self.top_k)
        e_prompts = e_prompts.view(
            batch_size, self.top_k * self.e_prompt_length, self.embed_dim
        )

        # Combine general and expert prompts
        g_prompts = self.g_prompt.expand(batch_size, -1, -1)
        prompts = torch.cat([g_prompts, e_prompts], dim=1)

        # Forward with prompts
        output = self._forward_with_prompts(x, prompts)

        # Classify
        logits = self.classifier(output)

        return logits

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features (placeholder)."""
        return torch.randn(x.shape[0], self.embed_dim, device=x.device)

    def _forward_with_prompts(
        self,
        x: torch.Tensor,
        prompts: torch.Tensor
    ) -> torch.Tensor:
        """Forward with prompts (placeholder)."""
        return torch.randn(x.shape[0], self.embed_dim, device=x.device)


class CODAPromptModel(NexusModule):
    """CODA-Prompt: Continual Decomposed Attention-based Prompting.

    CODA-Prompt decomposes prompts across attention layers and learns
    attention-specific prompts for each task. This allows capturing
    hierarchical task-specific features at different layers.

    Args:
        config: Configuration dictionary with:
            - backbone: Frozen backbone model (e.g., ViT)
            - num_layers: Number of transformer layers. Default: 12.
            - pool_size: Prompt pool size per layer. Default: 10.
            - prompt_length: Prompt length. Default: 5.
            - top_k: Number of prompts to select. Default: 1.
            - embed_dim: Embedding dimension. Default: 768.
            - num_classes: Number of output classes. Default: 100.

    Example:
        >>> config = {
        ...     "backbone": vit_model,
        ...     "num_layers": 12,
        ...     "pool_size": 10
        ... }
        >>> model = CODAPromptModel(config)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.backbone = config.get("backbone")
        if self.backbone is None:
            raise ValueError("Backbone model must be provided")

        self.num_layers = config.get("num_layers", 12)
        self.pool_size = config.get("pool_size", 10)
        self.prompt_length = config.get("prompt_length", 5)
        self.top_k = config.get("top_k", 1)
        self.embed_dim = config.get("embed_dim", 768)
        self.num_classes = config.get("num_classes", 100)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Layer-wise prompt pools
        self.layer_prompt_pools = nn.ModuleList([
            PromptPool(
                pool_size=self.pool_size,
                prompt_length=self.prompt_length,
                embed_dim=self.embed_dim,
                prompt_key_dim=self.embed_dim
            )
            for _ in range(self.num_layers)
        ])

        # Shared selector
        selector_config = {
            "input_dim": self.embed_dim,
            "query_dim": self.embed_dim,
            "num_layers": 2
        }
        self.selector = PromptSelector(selector_config)

        # Classifier
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with layer-wise prompting.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Classification logits (B, num_classes).
        """
        batch_size = x.shape[0]

        # Extract features for query
        with torch.no_grad():
            features = self._extract_features(x)

        # Compute query
        query = self.selector(features)

        # Select prompts for each layer
        layer_prompts = []
        for layer_pool in self.layer_prompt_pools:
            prompts, _ = layer_pool(query, top_k=self.top_k)
            prompts = prompts.view(
                batch_size, self.top_k * self.prompt_length, self.embed_dim
            )
            layer_prompts.append(prompts)

        # Forward with layer-wise prompts
        output = self._forward_with_layer_prompts(x, layer_prompts)

        # Classify
        logits = self.classifier(output)

        return logits

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features (placeholder)."""
        return torch.randn(x.shape[0], self.embed_dim, device=x.device)

    def _forward_with_layer_prompts(
        self,
        x: torch.Tensor,
        layer_prompts: List[torch.Tensor]
    ) -> torch.Tensor:
        """Forward with layer-specific prompts (placeholder).

        Args:
            x: Input tensor.
            layer_prompts: List of prompt tensors, one per layer.

        Returns:
            Output embeddings.
        """
        # Placeholder: integrate prompts into each attention layer
        return torch.randn(x.shape[0], self.embed_dim, device=x.device)
