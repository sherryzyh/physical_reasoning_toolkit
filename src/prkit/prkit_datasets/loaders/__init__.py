"""
Dataset loaders for different physical reasoning datasets.
"""

from .base_loader import BaseDatasetLoader
from .jeebench_loader import JEEBenchLoader
from .phybench_loader import PHYBenchLoader
from .phyx_loader import PhyXLoader
from .physreason_loader import PhysReasonLoader
from .scibench_loader import SciBenchLoader
from .seephys_loader import SeePhysLoader
from .tpbench_loader import TPBenchLoader
from .ugphysics_loader import UGPhysicsLoader

__all__ = [
    "BaseDatasetLoader",
    "PHYBenchLoader",
    "PhyXLoader",
    "SeePhysLoader",
    "UGPhysicsLoader",
    "JEEBenchLoader",
    "SciBenchLoader",
    "TPBenchLoader",
    "PhysReasonLoader",
]
