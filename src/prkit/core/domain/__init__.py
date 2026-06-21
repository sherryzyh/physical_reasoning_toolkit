"""
PRKit Domain Package

This package provides domain-specific data models, definitions, and enumerations
for PRKit (physical-reasoning-toolkit).

It consolidates:
- Domain models: Answer, PhysicsProblem, PhysicalDataset, PhysicsSolution
- Domain definitions: AnswerObjectKind, AnswerStructure, PhysicsDomain
"""

# Domain definitions (enums/constants)
# Domain models (data classes)
from .answer import Answer
from .answer_taxonomy import AnswerObjectKind, AnswerStructure
from .license_spec import LicenseSpec
from .physics_dataset import PhysicalDataset
from .physics_domain import PhysicsDomain
from .physics_problem import PhysicsProblem
from .physics_solution import PhysicsSolution

__all__ = [
    # Canonical answer ontology (single contract for the whole toolkit)
    "AnswerObjectKind",
    "AnswerStructure",
    # Definitions
    "PhysicsDomain",
    # Models
    "Answer",
    "PhysicsProblem",
    "PhysicalDataset",
    "PhysicsSolution",
    "LicenseSpec",
]
