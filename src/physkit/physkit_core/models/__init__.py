"""
PhysKit Models

This module provides the core data models for PhysKit.
"""

__version__ = "0.1.0"
__author__ = "Yinghuan Zhang"
__author_email__ = "yinghuan.flash@gmail.com"

from ..definitions.physics_domain import PhysicsDomain
from .physics_problem import PhysicsProblem
from .physics_dataset import PhysicalDataset
from .answer import Answer

__all__ = [
    "PhysicsDomain",
    "PhysicsProblem",
    "PhysicalDataset",
    "Answer",
]
