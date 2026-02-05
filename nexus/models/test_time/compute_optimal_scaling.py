"""
Compute-Optimal Scaling for Test-Time Compute.

Dynamically allocates test-time compute resources based on query difficulty.
Instead of using fixed compute budgets (e.g., always generating N samples or
running K refinement steps), this approach adapts compute allocation per prompt
to maximize performance under a compute budget constraint.

Key innovations:
- Difficulty-aware compute allocation: hard queries get more compute
- Adaptive sampling: generate fewer high-quality samples for easy queries,
  more samples for hard queries
- Verifier-guided early stopping: stop refinement when confidence is high
- Compute budget optimization: maximize expected performance under constraints
- Meta-learned difficulty predictor: learns which queries need more compute

The system includes:
1. Difficulty predictor: estimates query hardness
2. Compute allocator: decides how much compute to use
3. Adaptive sampler: generates variable number of samples
4. Early stopping criterion: terminates when sufficient quality reached

Paper: "Compute-Optimal Test-Time Scaling for Language Models"
       Inspired by:
       - "Scaling Laws for Test-Time Compute" (2024)
       - "Adaptive Compute Time" principles
       - Meta-learning for compute allocation

Applications:
- Language model inference with variable sample counts
- Code generation with adaptive verification
- Mathematical reasoning with dynamic search
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Callable
from nexus.core.base import NexusModule
import math


class DifficultyPredictor(NexusModule):
    """Predicts query difficulty to guide compute allocation.

    Args:
        config: Configuration dictionary with keys:
            - embed_dim (int): Embedding dimension
            - hidden_dim (int): Hidden layer dimension. Default 256
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 512)
        self.hidden_dim = config.get('hidden_dim', 256)

        # Difficulty prediction network
        self.predictor = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()  # Difficulty score in [0, 1]
        )

    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Predict query difficulty.

        Args:
            query_embedding: Query representation (B, embed_dim)

        Returns:
            Difficulty scores (B, 1) in [0, 1]
        """
        return self.predictor(query_embedding)


class ComputeAllocator(NexusModule):
    """Determines optimal compute allocation based on difficulty and budget.

    Args:
        config: Configuration dictionary with keys:
            - min_samples (int): Minimum samples per query. Default 1
            - max_samples (int): Maximum samples per query. Default 16
            - min_steps (int): Minimum refinement steps. Default 1
            - max_steps (int): Maximum refinement steps. Default 10
            - allocation_strategy (str): 'linear', 'exponential', 'learned'
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_samples = config.get('min_samples', 1)
        self.max_samples = config.get('max_samples', 16)
        self.min_steps = config.get('min_steps', 1)
        self.max_steps = config.get('max_steps', 10)
        self.strategy = config.get('allocation_strategy', 'linear')

        if self.strategy == 'learned':
            # Learnable allocation function
            hidden_dim = config.get('hidden_dim', 128)
            self.allocation_net = nn.Sequential(
                nn.Linear(1, hidden_dim),  # Input: difficulty score
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 2),  # Output: (num_samples, num_steps)
            )

    def allocate(self, difficulty: torch.Tensor,
                 compute_budget: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Allocate compute resources based on difficulty.

        Args:
            difficulty: Difficulty scores (B, 1)
            compute_budget: Available compute budget (multiplier)

        Returns:
            Tuple of:
                - num_samples: Number of samples to generate (B,)
                - num_steps: Number of refinement steps (B,)
        """
        B = difficulty.shape[0]

        if self.strategy == 'linear':
            # Linear scaling: easy queries get min, hard queries get max
            samples_range = self.max_samples - self.min_samples
            steps_range = self.max_steps - self.min_steps

            num_samples = (self.min_samples +
                          difficulty.squeeze(-1) * samples_range * compute_budget)
            num_steps = (self.min_steps +
                        difficulty.squeeze(-1) * steps_range * compute_budget)

        elif self.strategy == 'exponential':
            # Exponential scaling: allocate exponentially more to hard queries
            # This can lead to better performance on hard queries
            difficulty_exp = torch.pow(difficulty.squeeze(-1), 2)

            num_samples = (self.min_samples +
                          difficulty_exp * (self.max_samples - self.min_samples) * compute_budget)
            num_steps = (self.min_steps +
                        difficulty_exp * (self.max_steps - self.min_steps) * compute_budget)

        elif self.strategy == 'learned':
            # Learned allocation function
            allocation = self.allocation_net(difficulty)
            num_samples = torch.sigmoid(allocation[:, 0]) * self.max_samples * compute_budget
            num_steps = torch.sigmoid(allocation[:, 1]) * self.max_steps * compute_budget

            num_samples = torch.clamp(num_samples, self.min_samples, self.max_samples)
            num_steps = torch.clamp(num_steps, self.min_steps, self.max_steps)
        else:
            # Default: uniform allocation
            num_samples = torch.full((B,), self.min_samples, device=difficulty.device, dtype=torch.float)
            num_steps = torch.full((B,), self.min_steps, device=difficulty.device, dtype=torch.float)

        # Round to integers
        num_samples = num_samples.round().long()
        num_steps = num_steps.round().long()

        return num_samples, num_steps


class ConfidenceEstimator(NexusModule):
    """Estimates confidence in generated outputs for early stopping.

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 512)
        self.hidden_dim = config.get('hidden_dim', 256)

        # Confidence estimation network
        self.confidence_net = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, output_embedding: torch.Tensor) -> torch.Tensor:
        """Estimate confidence in output.

        Args:
            output_embedding: Output representation (B, embed_dim)

        Returns:
            Confidence scores (B, 1) in [0, 1]
        """
        return self.confidence_net(output_embedding)


class ComputeOptimalScaling(NexusModule):
    """Compute-optimal test-time scaling system.

    Adaptively allocates test-time compute based on query difficulty and
    available budget to maximize overall performance.

    Args:
        config: Configuration dictionary with keys:
            - embed_dim (int): Embedding dimension. Default 512
            - min_samples (int): Min samples per query. Default 1
            - max_samples (int): Max samples per query. Default 16
            - min_steps (int): Min refinement steps. Default 1
            - max_steps (int): Max refinement steps. Default 10
            - allocation_strategy (str): Compute allocation strategy
            - confidence_threshold (float): Early stopping threshold. Default 0.9
            - use_early_stopping (bool): Enable early stopping. Default True
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Components
        self.difficulty_predictor = DifficultyPredictor(config)
        self.compute_allocator = ComputeAllocator(config)
        self.confidence_estimator = ConfidenceEstimator(config)

        # Hyperparameters
        self.confidence_threshold = config.get('confidence_threshold', 0.9)
        self.use_early_stopping = config.get('use_early_stopping', True)

    def predict_allocation(self,
                          query_embedding: torch.Tensor,
                          compute_budget: float = 1.0) -> Dict[str, torch.Tensor]:
        """Predict compute allocation for queries.

        Args:
            query_embedding: Query representations (B, embed_dim)
            compute_budget: Available compute budget

        Returns:
            Dictionary containing:
                - difficulty: Predicted difficulty scores
                - num_samples: Allocated number of samples
                - num_steps: Allocated refinement steps
        """
        # Predict difficulty
        difficulty = self.difficulty_predictor(query_embedding)

        # Allocate compute
        num_samples, num_steps = self.compute_allocator.allocate(difficulty, compute_budget)

        return {
            'difficulty': difficulty,
            'num_samples': num_samples,
            'num_steps': num_steps
        }

    def should_stop_early(self,
                         output_embedding: torch.Tensor,
                         current_step: int,
                         max_steps: int) -> bool:
        """Determine if we should stop refinement early.

        Args:
            output_embedding: Current output representation (B, embed_dim)
            current_step: Current refinement step
            max_steps: Maximum allowed steps

        Returns:
            True if should stop early
        """
        if not self.use_early_stopping:
            return False

        # Estimate confidence
        confidence = self.confidence_estimator(output_embedding)

        # Stop if confidence is high enough
        return confidence.mean().item() >= self.confidence_threshold

    def adaptive_generate(self,
                         model: nn.Module,
                         query_embedding: torch.Tensor,
                         generate_fn: Callable,
                         refine_fn: Optional[Callable] = None,
                         compute_budget: float = 1.0) -> Dict[str, Any]:
        """Generate outputs with adaptive compute allocation.

        Args:
            model: The base model to use for generation
            query_embedding: Query representations (B, embed_dim)
            generate_fn: Function(query, num_samples) -> List[outputs]
                        Generates multiple candidate outputs
            refine_fn: Optional function(output, num_steps) -> refined_output
                      Refines output over multiple steps
            compute_budget: Available compute budget

        Returns:
            Dictionary containing:
                - outputs: Generated outputs
                - num_samples_used: Actual samples generated per query
                - num_steps_used: Actual refinement steps used
                - confidences: Confidence scores
                - difficulties: Difficulty scores
        """
        # Get compute allocation
        allocation = self.predict_allocation(query_embedding, compute_budget)

        B = query_embedding.shape[0]
        all_outputs = []
        samples_used = []
        steps_used = []
        confidences_list = []

        for i in range(B):
            num_samples = allocation['num_samples'][i].item()
            max_steps = allocation['num_steps'][i].item()

            query = query_embedding[i:i+1]

            # Generate multiple samples
            candidates = generate_fn(query, num_samples)

            # Optionally refine with early stopping
            if refine_fn is not None:
                refined_candidates = []
                for candidate in candidates:
                    # Iterative refinement with early stopping
                    current = candidate
                    for step in range(max_steps):
                        current = refine_fn(current, step)

                        # Check if we should stop early
                        # (Assuming current has embedding attribute)
                        if hasattr(current, 'embedding'):
                            if self.should_stop_early(current.embedding, step, max_steps):
                                steps_used.append(step + 1)
                                break
                    else:
                        steps_used.append(max_steps)

                    refined_candidates.append(current)

                candidates = refined_candidates
            else:
                steps_used.append(0)

            # Estimate confidence for all candidates
            # (Assuming candidates have embeddings)
            if hasattr(candidates[0], 'embedding'):
                cand_embeds = torch.stack([c.embedding for c in candidates])
                confidences = self.confidence_estimator(cand_embeds)

                # Select best candidate by confidence
                best_idx = confidences.argmax().item()
                best_output = candidates[best_idx]
                best_confidence = confidences[best_idx]
            else:
                # No confidence available, just take first
                best_output = candidates[0]
                best_confidence = torch.tensor([0.5])

            all_outputs.append(best_output)
            samples_used.append(num_samples)
            confidences_list.append(best_confidence)

        return {
            'outputs': all_outputs,
            'num_samples_used': torch.tensor(samples_used),
            'num_steps_used': torch.tensor(steps_used),
            'confidences': torch.stack(confidences_list) if confidences_list else None,
            'difficulties': allocation['difficulty']
        }

    def compute_efficiency_loss(self,
                                 predicted_difficulty: torch.Tensor,
                                 actual_performance: torch.Tensor,
                                 compute_used: torch.Tensor) -> torch.Tensor:
        """Compute loss for training difficulty predictor.

        Args:
            predicted_difficulty: Predicted difficulty (B, 1)
            actual_performance: Actual performance scores (B, 1)
            compute_used: Actual compute used (B, 1)

        Returns:
            Loss encouraging accurate difficulty prediction
        """
        # Difficulty should correlate with compute needs
        # Higher difficulty -> more compute -> lower performance without compute
        performance_loss = F.mse_loss(
            predicted_difficulty,
            1.0 - actual_performance  # Low performance = high difficulty
        )

        # Encourage compute efficiency
        efficiency_loss = (compute_used * (1.0 - actual_performance)).mean()

        return performance_loss + 0.1 * efficiency_loss


__all__ = [
    'DifficultyPredictor',
    'ComputeAllocator',
    'ConfidenceEstimator',
    'ComputeOptimalScaling'
]
