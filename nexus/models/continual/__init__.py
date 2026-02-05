"""Nexus continual learning models.

Provides implementations of modern continual learning methods for mitigating
catastrophic forgetting in neural networks. Includes regularization-based,
rehearsal-based, and parameter isolation approaches.
"""

from .ewc import FisherInformation, EWCRegularizer, EWCTrainer
from .evcl import (
    VariationalLayer,
    TaskPosterior,
    EVCLRegularizer,
    EVCLModel,
)
from .self_synthesized_rehearsal import (
    SyntheticExample,
    SynthesisModel,
    QualityFilter,
    CurriculumScheduler,
    SSRModel,
)
from .prompt_based_cl import (
    PromptPool,
    PromptSelector,
    L2PModel,
    DualPromptModel,
    CODAPromptModel,
)

__all__ = [
    # EWC (Elastic Weight Consolidation)
    "FisherInformation",
    "EWCRegularizer",
    "EWCTrainer",
    # EVCL (Elastic Variational Continual Learning)
    "VariationalLayer",
    "TaskPosterior",
    "EVCLRegularizer",
    "EVCLModel",
    # Self-Synthesized Rehearsal
    "SyntheticExample",
    "SynthesisModel",
    "QualityFilter",
    "CurriculumScheduler",
    "SSRModel",
    # Prompt-Based CL (L2P, DualPrompt, CODA-Prompt)
    "PromptPool",
    "PromptSelector",
    "L2PModel",
    "DualPromptModel",
    "CODAPromptModel",
]
