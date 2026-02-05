"""Imitation Learning algorithms."""

from .gail import GAILAgent, GAILDiscriminator, train_gail
from .dagger import DAggerAgent, DAggerPolicy, train_with_dagger

__all__ = [
    'GAILAgent',
    'GAILDiscriminator',
    'train_gail',
    'DAggerAgent',
    'DAggerPolicy',
    'train_with_dagger',
]
