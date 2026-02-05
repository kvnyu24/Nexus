"""Nexus optimizers module.

Provides modern optimizers for deep learning training, including
memory-efficient, second-order, and schedule-free variants.
"""

from .lion import Lion
from .sophia import Sophia
from .prodigy import Prodigy
from .schedule_free import ScheduleFreeAdamW
from .muon import Muon
from .soap import SOAP

__all__ = [
    "Lion",
    "Sophia",
    "Prodigy",
    "ScheduleFreeAdamW",
    "Muon",
    "SOAP",
]
