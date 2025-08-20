"""
Dataset loaders for different physical reasoning datasets.
"""

from .base_loader import BaseDatasetLoader
from .phybench import PHYBenchLoader
from .seephys import SeePhysLoader
from .ugphysics import UGPhysicsLoader
from .jeebench import JEEBenchLoader
from .scibench import SciBenchLoader
from .tpbench import TPBenchLoader
from .physreason import PhysReasonLoader

__all__ = [
    "BaseDatasetLoader",
    "PHYBenchLoader",
    "SeePhysLoader",
    "UGPhysicsLoader",
    "JEEBenchLoader",
    "SciBenchLoader",
    "TPBenchLoader",
    "PhysReasonLoader",
]
