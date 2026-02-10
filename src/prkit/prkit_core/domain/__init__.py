"""
PRKit Domain Package

This package provides domain-specific data models, definitions, and enumerations
for PRKit (physical-reasoning-toolkit).

It consolidates:
- Domain models: Answer, PhysicsProblem, PhysicalDataset, PhysicsSolution
- Domain definitions: AnswerCategory, PhysicsDomain
"""

# Domain definitions (enums/constants)
from .answer_category import AnswerCategory
from .physics_domain import PhysicsDomain

# Domain models (data classes)
from .answer import Answer
from .physics_problem import PhysicsProblem
from .physics_dataset import PhysicalDataset
from .physics_solution import PhysicsSolution

__all__ = [
    # Definitions
    "PhysicsDomain",
    "AnswerCategory",
    # Models
    "Answer",
    "PhysicsProblem",
    "PhysicalDataset",
    "PhysicsSolution",
]
