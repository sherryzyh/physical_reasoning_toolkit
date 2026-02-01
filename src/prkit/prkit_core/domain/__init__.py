"""
PRKit Domain Package

This package provides domain-specific data models, definitions, and enumerations
for PRKit (physical-reasoning-toolkit).

It consolidates:
- Domain models: Answer, PhysicsProblem, PhysicalDataset, PhysicsSolution
- Domain definitions: AnswerType, PhysicsDomain
"""

# Domain definitions (enums/constants)
from .answer_type import AnswerType
from .physics_domain import PhysicsDomain

# Domain models (data classes)
from .answer import Answer
from .physics_problem import PhysicsProblem
from .physics_dataset import PhysicalDataset
from .physics_solution import PhysicsSolution

__all__ = [
    # Definitions
    "PhysicsDomain",
    "AnswerType",
    # Models
    "Answer",
    "PhysicsProblem",
    "PhysicalDataset",
    "PhysicsSolution",
]
