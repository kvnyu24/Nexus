"""
Test-Time Compute optimization and adaptation modules.

Includes:
- Test-Time Training (TTT) Layers: Hidden state as learnable model
- Compute-Optimal Scaling: Adaptive compute allocation
- Best-of-N with PRM: Sample multiple, verify with process reward model
"""

from .ttt_layers import TTTLinearModel, TTTLayer, TTTBlock, TTTTransformer
from .compute_optimal_scaling import (
    DifficultyPredictor,
    ComputeAllocator,
    ConfidenceEstimator,
    ComputeOptimalScaling
)
from .best_of_n_prm import (
    ProcessRewardModel,
    BestOfNSelector,
    BestOfNWithPRM,
    BeamSearchWithPRM
)

__all__ = [
    # TTT Layers
    'TTTLinearModel',
    'TTTLayer',
    'TTTBlock',
    'TTTTransformer',
    # Compute-Optimal Scaling
    'DifficultyPredictor',
    'ComputeAllocator',
    'ConfidenceEstimator',
    'ComputeOptimalScaling',
    # Best-of-N with PRM
    'ProcessRewardModel',
    'BestOfNSelector',
    'BestOfNWithPRM',
    'BeamSearchWithPRM',
]
