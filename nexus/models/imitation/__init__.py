"""Imitation Learning algorithms."""

from .gail import GAILAgent, GAILDiscriminator, train_gail
from .dagger import DAggerAgent, DAggerPolicy, train_with_dagger
from .mega_dagger import MEGADAggerPolicy, MEGADAggerAgent, ExpertWeightingModule, UncertaintyEstimator, train_mega_dagger
from .airl import RewardNetwork, ValueNetwork, AIRLDiscriminator, AIRLAgent, train_airl

__all__ = [
    'GAILAgent',
    'GAILDiscriminator',
    'train_gail',
    'DAggerAgent',
    'DAggerPolicy',
    'train_with_dagger',
    # MEGA-DAgger
    'MEGADAggerPolicy',
    'MEGADAggerAgent',
    'ExpertWeightingModule',
    'UncertaintyEstimator',
    'train_mega_dagger',
    # AIRL
    'RewardNetwork',
    'ValueNetwork',
    'AIRLDiscriminator',
    'AIRLAgent',
    'train_airl',
]
