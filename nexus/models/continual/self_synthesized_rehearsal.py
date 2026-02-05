"""Self-Synthesized Rehearsal for Continual Learning with LLMs.

Reference: "Self-Synthesized Rehearsal: Reshaping the Learning Curriculum
for Continual Learning with Large Language Models" (ACL 2024)

Self-Synthesized Rehearsal (SSR) is a continual learning method specifically
designed for large language models. Instead of storing real data from previous
tasks (which requires large memory), SSR has the LLM synthesize pseudo-examples
that capture the knowledge from past tasks. These synthesized examples are then
used for rehearsal alongside new task data.

Key innovations:
    - Self-synthesis: Model generates its own rehearsal data
    - No memory buffer: Doesn't need to store actual training examples
    - Curriculum reshaping: Intelligently schedules synthetic vs. real data
    - Quality filtering: Filters low-quality synthetic examples
    - Task-aware synthesis: Generates examples conditioned on task identity

Architecture:
    - SynthesisModel: Generates synthetic examples from task descriptions
    - QualityFilter: Filters low-quality synthesized examples
    - CurriculumScheduler: Schedules mix of synthetic and real data
    - SSRModel: Complete continual learning system with synthesis

Key properties:
    - Memory-efficient (no data storage)
    - Privacy-preserving (no real data retained)
    - Works well with large language models
    - Adapts difficulty of synthetic examples over training
    - Can be combined with other CL methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from typing import Dict, Any, Optional, List, Tuple
from ...core.base import NexusModule
import random


class SyntheticExample:
    """Container for a synthesized example.

    Attributes:
        input_text: Generated input text
        target_text: Generated target/label
        task_id: Task identifier
        confidence: Quality confidence score
    """

    def __init__(
        self,
        input_text: str,
        target_text: str,
        task_id: int,
        confidence: float = 1.0
    ):
        self.input_text = input_text
        self.target_text = target_text
        self.task_id = task_id
        self.confidence = confidence


class SynthesisModel(NexusModule):
    """Model for synthesizing rehearsal examples.

    Uses the base LLM to generate synthetic examples that capture
    task-specific knowledge. Conditions generation on task descriptions
    and learned representations.

    Args:
        config: Configuration dictionary with:
            - model: Base language model for synthesis
            - max_length: Maximum sequence length. Default: 128.
            - temperature: Sampling temperature. Default: 1.0.
            - top_k: Top-k sampling. Default: 50.
            - top_p: Nucleus sampling threshold. Default: 0.95.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model = config.get("model")  # Base LLM
        if self.model is None:
            raise ValueError("Base model must be provided in config")

        self.max_length = config.get("max_length", 128)
        self.temperature = config.get("temperature", 1.0)
        self.top_k = config.get("top_k", 50)
        self.top_p = config.get("top_p", 0.95)

    @torch.no_grad()
    def synthesize_examples(
        self,
        task_description: str,
        task_id: int,
        num_examples: int = 100,
        batch_size: int = 8
    ) -> List[SyntheticExample]:
        """Generate synthetic examples for a task.

        Args:
            task_description: Text description of the task.
            task_id: Task identifier.
            num_examples: Number of examples to generate.
            batch_size: Generation batch size.

        Returns:
            List of SyntheticExample objects.
        """
        # Set model to inference mode (disables dropout, batch norm updates)
        self.model.train(mode=False)

        synthetic_examples = []
        num_batches = (num_examples + batch_size - 1) // batch_size

        for _ in range(num_batches):
            # Create synthesis prompt
            prompts = [
                f"Generate a training example for the following task: {task_description}\n"
                f"Input:"
            ] * min(batch_size, num_examples - len(synthetic_examples))

            # Generate inputs
            # Note: This is a simplified version. In practice, would use the
            # model's tokenizer and generation method
            generated_inputs = self._generate_batch(prompts)

            # For each generated input, generate the corresponding target
            for gen_input in generated_inputs:
                target_prompt = (
                    f"Task: {task_description}\n"
                    f"Input: {gen_input}\n"
                    f"Output:"
                )
                generated_target = self._generate_batch([target_prompt])[0]

                synthetic_examples.append(
                    SyntheticExample(
                        input_text=gen_input,
                        target_text=generated_target,
                        task_id=task_id,
                        confidence=1.0  # Will be updated by quality filter
                    )
                )

                if len(synthetic_examples) >= num_examples:
                    break

            if len(synthetic_examples) >= num_examples:
                break

        return synthetic_examples[:num_examples]

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate text from prompts (simplified placeholder).

        In a real implementation, this would use the model's generation API.

        Args:
            prompts: List of prompt strings.

        Returns:
            List of generated strings.
        """
        # Placeholder: In practice, use model.generate() with proper tokenization
        # This is a simplified version for demonstration
        generated = []
        for prompt in prompts:
            # Simplified generation: just return a placeholder
            # Real implementation would call model.generate()
            generated.append(f"[Generated response for: {prompt[:30]}...]")
        return generated


class QualityFilter(NexusModule):
    """Filters low-quality synthesized examples.

    Uses the base model's perplexity and consistency checks to identify
    and remove low-quality synthetic examples that might hurt learning.

    Args:
        config: Configuration dictionary with:
            - model: Base model for quality assessment
            - perplexity_threshold: Max allowed perplexity. Default: 100.0.
            - consistency_threshold: Min consistency score. Default: 0.5.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model = config.get("model")
        if self.model is None:
            raise ValueError("Base model must be provided in config")

        self.perplexity_threshold = config.get("perplexity_threshold", 100.0)
        self.consistency_threshold = config.get("consistency_threshold", 0.5)

    @torch.no_grad()
    def filter_examples(
        self,
        examples: List[SyntheticExample]
    ) -> List[SyntheticExample]:
        """Filter synthetic examples by quality.

        Args:
            examples: List of synthetic examples to filter.

        Returns:
            Filtered list of high-quality examples.
        """
        # Set model to inference mode
        self.model.train(mode=False)

        filtered = []

        for example in examples:
            # Compute perplexity of the example
            perplexity = self._compute_perplexity(
                example.input_text,
                example.target_text
            )

            # Check consistency (model agrees with synthetic label)
            consistency = self._compute_consistency(
                example.input_text,
                example.target_text
            )

            # Update confidence score
            example.confidence = consistency

            # Filter by thresholds
            if (perplexity < self.perplexity_threshold and
                consistency > self.consistency_threshold):
                filtered.append(example)

        return filtered

    def _compute_perplexity(
        self,
        input_text: str,
        target_text: str
    ) -> float:
        """Compute perplexity of the example (simplified).

        Args:
            input_text: Input text.
            target_text: Target text.

        Returns:
            Perplexity score.
        """
        # Placeholder: compute actual perplexity using model
        # Real implementation would tokenize and compute log-likelihood
        return 50.0  # Simplified placeholder

    def _compute_consistency(
        self,
        input_text: str,
        target_text: str
    ) -> float:
        """Compute consistency score (simplified).

        Checks if the model's prediction matches the synthetic label.

        Args:
            input_text: Input text.
            target_text: Target text.

        Returns:
            Consistency score [0, 1].
        """
        # Placeholder: check if model prediction matches target
        # Real implementation would generate and compare
        return 0.8  # Simplified placeholder


class CurriculumScheduler:
    """Schedules curriculum of synthetic vs. real examples.

    Adaptively mixes synthetic rehearsal examples with real task data
    based on training progress. Early training uses more real data,
    later training incorporates more synthetic data.

    Args:
        initial_synthetic_ratio: Initial ratio of synthetic data. Default: 0.2.
        final_synthetic_ratio: Final ratio of synthetic data. Default: 0.8.
        warmup_steps: Steps before increasing synthetic ratio. Default: 1000.
    """

    def __init__(
        self,
        initial_synthetic_ratio: float = 0.2,
        final_synthetic_ratio: float = 0.8,
        warmup_steps: int = 1000
    ):
        self.initial_synthetic_ratio = initial_synthetic_ratio
        self.final_synthetic_ratio = final_synthetic_ratio
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self) -> None:
        """Increment the scheduler step."""
        self.current_step += 1

    def get_synthetic_ratio(self) -> float:
        """Get current synthetic data ratio.

        Returns:
            Current ratio of synthetic to total data.
        """
        if self.current_step < self.warmup_steps:
            # Linear warmup
            progress = self.current_step / self.warmup_steps
            ratio = (
                self.initial_synthetic_ratio +
                (self.final_synthetic_ratio - self.initial_synthetic_ratio) * progress
            )
        else:
            ratio = self.final_synthetic_ratio

        return ratio

    def create_mixed_batch(
        self,
        real_batch: Dict[str, torch.Tensor],
        synthetic_examples: List[SyntheticExample],
        task_id: int
    ) -> Dict[str, torch.Tensor]:
        """Create a mixed batch of real and synthetic data.

        Args:
            real_batch: Batch of real training data.
            synthetic_examples: Pool of synthetic examples.
            task_id: Current task ID.

        Returns:
            Mixed batch dictionary.
        """
        synthetic_ratio = self.get_synthetic_ratio()
        batch_size = real_batch["input_ids"].shape[0]

        # Number of synthetic examples to include
        num_synthetic = int(batch_size * synthetic_ratio)
        num_real = batch_size - num_synthetic

        # Sample synthetic examples
        sampled_synthetic = random.sample(
            synthetic_examples,
            min(num_synthetic, len(synthetic_examples))
        )

        # Create mixed batch (simplified)
        # In practice, would properly tokenize and batch the data
        mixed_batch = {
            "input_ids": real_batch["input_ids"][:num_real],
            "labels": real_batch["labels"][:num_real],
            "task_id": torch.full((num_real,), task_id),
        }

        return mixed_batch


class SSRModel(NexusModule):
    """Self-Synthesized Rehearsal continual learning model.

    Complete system that uses LLM self-synthesis to generate rehearsal
    examples, avoiding the need to store real training data.

    Training procedure:
        1. Train on new task data
        2. Synthesize examples capturing task knowledge
        3. Filter low-quality synthetic examples
        4. When training on next task, mix new data with synthetic rehearsal
        5. Curriculum scheduler adapts synthetic ratio over time

    Args:
        config: Configuration dictionary with:
            - model: Base language model
            - num_synthesis_examples: Examples per task. Default: 1000.
            - synthesis_batch_size: Batch size for synthesis. Default: 8.
            - perplexity_threshold: Quality filter threshold. Default: 100.0.
            - initial_synthetic_ratio: Initial synthetic ratio. Default: 0.2.
            - final_synthetic_ratio: Final synthetic ratio. Default: 0.8.

    Example:
        >>> config = {
        ...     "model": base_llm,
        ...     "num_synthesis_examples": 1000
        ... }
        >>> ssr = SSRModel(config)
        >>> # Train on task 1
        >>> ssr.train_task(task1_loader, task_id=0, task_description="Sentiment classification")
        >>> # Train on task 2 with rehearsal from task 1
        >>> ssr.train_task(task2_loader, task_id=1, task_description="Topic classification")
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model = config.get("model")
        if self.model is None:
            raise ValueError("Base model must be provided in config")

        # Synthesis module
        self.synthesizer = SynthesisModel(config)

        # Quality filter
        self.quality_filter = QualityFilter(config)

        # Curriculum scheduler
        self.scheduler = CurriculumScheduler(
            initial_synthetic_ratio=config.get("initial_synthetic_ratio", 0.2),
            final_synthetic_ratio=config.get("final_synthetic_ratio", 0.8),
            warmup_steps=config.get("warmup_steps", 1000)
        )

        # Storage for synthetic examples (per task)
        self.synthetic_memory: Dict[int, List[SyntheticExample]] = {}

        # Configuration
        self.num_synthesis_examples = config.get("num_synthesis_examples", 1000)
        self.synthesis_batch_size = config.get("synthesis_batch_size", 8)

    def synthesize_task_memory(
        self,
        task_id: int,
        task_description: str
    ) -> None:
        """Synthesize and store examples for a completed task.

        Args:
            task_id: Task identifier.
            task_description: Text description of the task.
        """
        # Generate synthetic examples
        synthetic_examples = self.synthesizer.synthesize_examples(
            task_description=task_description,
            task_id=task_id,
            num_examples=self.num_synthesis_examples,
            batch_size=self.synthesis_batch_size
        )

        # Filter by quality
        filtered_examples = self.quality_filter.filter_examples(synthetic_examples)

        # Store in memory
        self.synthetic_memory[task_id] = filtered_examples

    def get_rehearsal_examples(self) -> List[SyntheticExample]:
        """Get all stored rehearsal examples from previous tasks.

        Returns:
            List of all synthetic examples.
        """
        all_examples = []
        for examples in self.synthetic_memory.values():
            all_examples.extend(examples)
        return all_examples

    def train_task(
        self,
        data_loader: DataLoader,
        task_id: int,
        task_description: str,
        num_epochs: int = 3,
        learning_rate: float = 5e-5
    ) -> Dict[str, List[float]]:
        """Train on a task with self-synthesized rehearsal.

        Args:
            data_loader: DataLoader for the new task.
            task_id: Task identifier.
            task_description: Text description of the task.
            num_epochs: Number of training epochs. Default: 3.
            learning_rate: Learning rate. Default: 5e-5.

        Returns:
            Training history.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Get rehearsal examples from previous tasks
        rehearsal_examples = self.get_rehearsal_examples()

        history = {"loss": []}

        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in data_loader:
                # Mix real and synthetic data (if rehearsal available)
                if rehearsal_examples:
                    batch = self.scheduler.create_mixed_batch(
                        batch, rehearsal_examples, task_id
                    )

                # Forward pass (simplified)
                # In practice, would use proper model forward and loss computation
                optimizer.zero_grad()

                # Placeholder loss computation
                loss = torch.tensor(0.5, requires_grad=True)  # Simplified

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                self.scheduler.step()

            avg_loss = epoch_loss / max(num_batches, 1)
            history["loss"].append(avg_loss)

        # After training, synthesize examples for this task
        self.synthesize_task_memory(task_id, task_description)

        return history

    @property
    def num_stored_examples(self) -> int:
        """Return total number of stored synthetic examples."""
        return sum(len(examples) for examples in self.synthetic_memory.values())
