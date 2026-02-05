from .base_trainer import BaseTrainer
from .trainer import Trainer

from .losses import (
    ContrastiveLoss,
    FocalLoss,
    CircleLoss,
    TripletLoss,
    NTXentLoss,
    WingLoss,
    DiceLoss,
    InfoNCELoss,
    AdaCosLoss,
    PolyLoss,
    WeightedFocalLoss
)

from .scheduler import (
    CosineWarmupScheduler,
)

from .mixed_precision import (
    MixedPrecisionConfig,
    FP8Format,
    FP8Linear,
    FP8ScalingManager,
    FP8LayerNorm,
    GradScaler,
    AdaptiveGradScaler,
    convert_to_fp8,
    get_fp8_memory_savings,
    is_fp8_available,
    get_recommended_config,
)

from .gradient_methods import (
    SelectiveCheckpoint,
    SelectiveCheckpointConfig,
    CheckpointPolicy,
    ActivationOffloader,
    apply_selective_checkpointing,
    estimate_checkpointing_memory_savings,
    estimate_offloading_memory_savings,
)

__all__ = [
    # Core training
    'BaseTrainer',
    'Trainer',

    # Losses
    'ContrastiveLoss',
    'FocalLoss',
    'CircleLoss',
    'TripletLoss',
    'NTXentLoss',
    'WingLoss',
    'DiceLoss',
    'InfoNCELoss',
    'AdaCosLoss',
    'PolyLoss',
    'WeightedFocalLoss',

    # Learning rate schedulers
    'CosineWarmupScheduler',

    # Mixed precision training
    'MixedPrecisionConfig',
    'FP8Format',
    'FP8Linear',
    'FP8ScalingManager',
    'FP8LayerNorm',
    'GradScaler',
    'AdaptiveGradScaler',
    'convert_to_fp8',
    'get_fp8_memory_savings',
    'is_fp8_available',
    'get_recommended_config',

    # Gradient methods
    'SelectiveCheckpoint',
    'SelectiveCheckpointConfig',
    'CheckpointPolicy',
    'ActivationOffloader',
    'apply_selective_checkpointing',
    'estimate_checkpointing_memory_savings',
    'estimate_offloading_memory_savings',
]
