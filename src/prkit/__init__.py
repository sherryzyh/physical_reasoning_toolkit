"""
PRKit - Physical Reasoning Toolkit

A comprehensive toolkit for physical reasoning, annotation, and dataset management.

Package name: physical-reasoning-toolkit
Import name: prkit
"""

# Make subpackages importable at top level (e.g., `from prkit.prkit_datasets import DatasetHub` or `from prkit_datasets import DatasetHub`)
# Note: PyPI package name is "physical-reasoning-toolkit", import name is "prkit"
import sys

# Import and expose subpackages at top level
try:
    from . import prkit_annotation, prkit_core, prkit_datasets, prkit_evaluation

    # Make them available as top-level modules
    sys.modules["prkit_core"] = prkit_core
    sys.modules["prkit_datasets"] = prkit_datasets
    sys.modules["prkit_annotation"] = prkit_annotation
    sys.modules["prkit_evaluation"] = prkit_evaluation

    # Import main components for easy access
    from .prkit_core import PRKitLogger
    from .prkit_core.domain import AnswerType, PhysicsDomain, PhysicalDataset, PhysicsProblem
except ImportError:
    # Allow package to be imported even if subpackages aren't installed
    pass

__all__ = [
    "PRKitLogger",
    "PhysicsProblem",
    "PhysicalDataset",
    "PhysicsDomain",
    "AnswerType",
]
