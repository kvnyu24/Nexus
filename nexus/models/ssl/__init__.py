"""Nexus self-supervised learning models.

Provides implementations of modern self-supervised learning methods
including joint-embedding, masked autoencoding, and redundancy reduction.
"""

from .jepa import ContextEncoder, TargetEncoder, Predictor, JEPAModel
from .mae import MAEEncoder, MAEDecoder, MAE
from .barlow_twins import ProjectionHead, BarlowTwinsModel
from .v_jepa import (
    VideoContextEncoder,
    VideoTargetEncoder,
    VideoPredictor,
    VJEPAModel,
)
from .data2vec import (
    StudentEncoder,
    TeacherEncoder,
    ContextualizedDecoder,
    Data2VecModel,
)
from .vicreg import (
    VICRegProjector,
    VICRegLoss,
    VICRegEncoder,
    VICRegModel,
)

__all__ = [
    # I-JEPA (Image)
    "ContextEncoder",
    "TargetEncoder",
    "Predictor",
    "JEPAModel",
    # V-JEPA 2 (Video)
    "VideoContextEncoder",
    "VideoTargetEncoder",
    "VideoPredictor",
    "VJEPAModel",
    # MAE
    "MAEEncoder",
    "MAEDecoder",
    "MAE",
    # Barlow Twins
    "ProjectionHead",
    "BarlowTwinsModel",
    # data2vec 2.0
    "StudentEncoder",
    "TeacherEncoder",
    "ContextualizedDecoder",
    "Data2VecModel",
    # VICReg
    "VICRegProjector",
    "VICRegLoss",
    "VICRegEncoder",
    "VICRegModel",
]
