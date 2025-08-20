"""
PhysKit Models

This module provides the core data models for PhysKit.
"""

from ..definitions.physics_domain import PhysicsDomain
from .physics_problem import PhysicsProblem
from .physics_dataset import PhysicalDataset

__all__ = [
    "PhysicsDomain",
    "PhysicsProblem",
    "PhysicalDataset",
]
