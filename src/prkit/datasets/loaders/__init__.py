"""
Dataset loaders for different physical reasoning datasets.
"""

from .base_loader import BaseDatasetLoader
from .jeebench_loader import JEEBenchLoader
from .physbench_loader import PhysBenchLoader
from .phybench_loader import PHYBenchLoader
from .physics_loader import PhysicsLoader
from .phyx_loader import PhyXLoader
from .physreason_loader import PhysReasonLoader
from .seephys_loader import SeePhysLoader
from .tpbench_loader import TPBenchLoader
from .ugphysics_loader import UGPhysicsLoader

__all__ = [
    "BaseDatasetLoader",
    "PhysBenchLoader",
    "PHYBenchLoader",
    "PhysicsLoader",
    "PhyXLoader",
    "SeePhysLoader",
    "UGPhysicsLoader",
    "JEEBenchLoader",
    "TPBenchLoader",
    "PhysReasonLoader",
]
