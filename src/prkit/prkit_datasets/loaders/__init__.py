"""
Dataset loaders for different physical reasoning datasets.
"""

__version__ = "0.1.0"
__author__ = "Yinghuan Zhang"
__author_email__ = "yinghuan.flash@gmail.com"

from .base_loader import BaseDatasetLoader
from .jeebench import JEEBenchLoader
from .phybench import PHYBenchLoader
from .physreason import PhysReasonLoader
from .scibench import SciBenchLoader
from .seephys import SeePhysLoader
from .tpbench import TPBenchLoader
from .ugphysics import UGPhysicsLoader

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
