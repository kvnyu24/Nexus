"""
Self-Consistency: Improves Chain of Thought Reasoning in Language Models.

Paper: "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
       Wang et al., ICLR 2023
       https://arxiv.org/abs/2203.11171

Self-Consistency samples multiple diverse reasoning paths using temperature sampling,
then aggregates the final answers via majority voting. This simple technique
significantly improves performance on arithmetic, commonsense, and symbolic reasoning
tasks by marginalizing over the space of reasoning paths.

Key innovations:
- Sample diverse CoT paths with temperature > 0
- Aggregate answers via majority vote or weighted consensus
- No training required - inference-time technique
- Consistent improvements across models and tasks
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Callable
from collections import Counter
from nexus.core.base import NexusModule


class SelfConsistency(NexusModule):
    """Self-Consistency for robust reasoning with language models.

    Args:
        config: Configuration dictionary with keys:
            - model: Language model with generate() method
            - num_samples (int): Number of reasoning paths to sample. Default 40
            - temperature (float): Sampling temperature for diversity. Default 0.7
            - aggregation (str): 'majority' or 'weighted'. Default 'majority'
            - answer_extractor (callable): Function to extract answer from reasoning. Default None
            - max_new_tokens (int): Max tokens to generate. Default 256
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config['model']
        self.num_samples = config.get('num_samples', 40)
        self.temperature = config.get('temperature', 0.7)
        self.aggregation = config.get('aggregation', 'majority')
        self.answer_extractor = config.get('answer_extractor', None)
        self.max_new_tokens = config.get('max_new_tokens', 256)

        # Default answer extractor (looks for "answer is X" pattern)
        if self.answer_extractor is None:
            self.answer_extractor = self._default_answer_extractor

    def _default_answer_extractor(self, reasoning: str) -> Optional[str]:
        """Default answer extraction from reasoning chain.

        Args:
            reasoning: Full reasoning text

        Returns:
            Extracted answer or None
        """
        # Try multiple patterns
        patterns = [
            "the answer is ",
            "answer: ",
            "therefore, ",
            "#### ",  # Common in math datasets
        ]

        reasoning_lower = reasoning.lower()

        for pattern in patterns:
            if pattern in reasoning_lower:
                # Find answer after pattern
                idx = reasoning_lower.rfind(pattern)
                answer_text = reasoning[idx + len(pattern):].strip()

                # Extract first word/number as answer
                answer = answer_text.split()[0] if answer_text else None

                # Clean up punctuation
                if answer:
                    answer = answer.rstrip('.,;:')

                return answer

        # Fallback: last line of reasoning
        lines = reasoning.strip().split('\n')
        if lines:
            return lines[-1].strip()

        return None

    def sample_reasoning_paths(
        self,
        question: str,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[str]:
        """Sample multiple diverse reasoning paths for a question.

        Args:
            question: Input question/problem
            num_samples: Number of paths to sample (overrides config)
            temperature: Sampling temperature (overrides config)

        Returns:
            List of reasoning paths as strings
        """
        if num_samples is None:
            num_samples = self.num_samples
        if temperature is None:
            temperature = self.temperature

        reasoning_paths = []

        # Create prompt with CoT instruction
        prompt = f"{question}\n\nLet's think step by step:"

        for i in range(num_samples):
            # Generate reasoning path with temperature sampling
            if hasattr(self.model, 'generate'):
                output = self.model.generate(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9
                )
            else:
                # Fallback for models without generate method
                output = f"Reasoning path {i+1}"

            reasoning_paths.append(output)

        return reasoning_paths

    def extract_answers(self, reasoning_paths: List[str]) -> List[Optional[str]]:
        """Extract final answers from reasoning paths.

        Args:
            reasoning_paths: List of reasoning strings

        Returns:
            List of extracted answers
        """
        answers = []
        for reasoning in reasoning_paths:
            answer = self.answer_extractor(reasoning)
            answers.append(answer)

        return answers

    def _majority_vote(self, answers: List[Optional[str]]) -> Tuple[Optional[str], float]:
        """Aggregate answers via majority voting.

        Args:
            answers: List of candidate answers

        Returns:
            Tuple of (most_common_answer, confidence)
        """
        # Filter out None answers
        valid_answers = [a for a in answers if a is not None]

        if not valid_answers:
            return None, 0.0

        # Count votes
        counter = Counter(valid_answers)
        most_common = counter.most_common(1)[0]
        answer, count = most_common

        # Confidence = fraction of votes for winner
        confidence = count / len(valid_answers)

        return answer, confidence

    def _weighted_consensus(
        self,
        answers: List[Optional[str]],
        reasoning_paths: List[str]
    ) -> Tuple[Optional[str], float]:
        """Aggregate answers with confidence-weighted voting.

        Weights each answer by the log probability of its reasoning path
        (if available from the model).

        Args:
            answers: List of candidate answers
            reasoning_paths: Corresponding reasoning paths

        Returns:
            Tuple of (best_answer, confidence)
        """
        # Simplified version: weight by reasoning length as proxy for confidence
        # (longer, more detailed reasoning may be more confident)

        answer_weights = {}

        for answer, reasoning in zip(answers, reasoning_paths):
            if answer is None:
                continue

            # Weight by log(reasoning_length) as proxy
            weight = torch.log(torch.tensor(len(reasoning.split()) + 1)).item()

            if answer in answer_weights:
                answer_weights[answer] += weight
            else:
                answer_weights[answer] = weight

        if not answer_weights:
            return None, 0.0

        # Get answer with highest weight
        best_answer = max(answer_weights, key=answer_weights.get)
        total_weight = sum(answer_weights.values())
        confidence = answer_weights[best_answer] / total_weight

        return best_answer, confidence

    def aggregate_answers(
        self,
        answers: List[Optional[str]],
        reasoning_paths: Optional[List[str]] = None
    ) -> Tuple[Optional[str], float]:
        """Aggregate multiple answers into final answer.

        Args:
            answers: List of candidate answers
            reasoning_paths: Optional reasoning paths for weighted aggregation

        Returns:
            Tuple of (final_answer, confidence)
        """
        if self.aggregation == 'majority':
            return self._majority_vote(answers)
        elif self.aggregation == 'weighted' and reasoning_paths is not None:
            return self._weighted_consensus(answers, reasoning_paths)
        else:
            # Fallback to majority vote
            return self._majority_vote(answers)

    def solve(
        self,
        question: str,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """Solve a question using self-consistency.

        Args:
            question: Input question/problem
            return_details: Whether to return all reasoning paths and answers

        Returns:
            Dictionary with final answer and optional details
        """
        # Sample reasoning paths
        reasoning_paths = self.sample_reasoning_paths(question)

        # Extract answers
        answers = self.extract_answers(reasoning_paths)

        # Aggregate
        final_answer, confidence = self.aggregate_answers(answers, reasoning_paths)

        result = {
            'answer': final_answer,
            'confidence': confidence
        }

        if return_details:
            result['reasoning_paths'] = reasoning_paths
            result['all_answers'] = answers
            result['answer_distribution'] = Counter(
                [a for a in answers if a is not None]
            )

        return result

    def forward(self, question: str) -> str:
        """Forward pass returns the final answer.

        Args:
            question: Input question

        Returns:
            Final answer string
        """
        result = self.solve(question, return_details=False)
        return result['answer'] if result['answer'] is not None else ""


# Convenience function
def solve_with_self_consistency(
    model,
    question: str,
    num_samples: int = 40,
    temperature: float = 0.7
) -> str:
    """Convenience function to solve a question with self-consistency.

    Args:
        model: Language model
        question: Question to solve
        num_samples: Number of reasoning paths
        temperature: Sampling temperature

    Returns:
        Final answer
    """
    config = {
        'model': model,
        'num_samples': num_samples,
        'temperature': temperature
    }

    sc = SelfConsistency(config)
    result = sc.solve(question)

    return result['answer']
