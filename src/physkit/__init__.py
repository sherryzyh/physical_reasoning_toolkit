"""
PhysKit - Physical Reasoning Toolkit

A comprehensive toolkit for physical reasoning, annotation, and dataset management.
"""

__version__ = "0.1.0"
__author__ = "Yinghuan Zhang"
__author_email__ = "yinghuan.flash@gmail.com"

# Make subpackages importable at top level (e.g., `from physkit_datasets import DatasetHub`)
import sys

# Import and expose subpackages at top level
try:
    from . import physkit_core
    from . import physkit_datasets
    from . import physkit_annotation
    from . import physkit_evaluation
    
    # Make them available as top-level modules
    sys.modules['physkit_core'] = physkit_core
    sys.modules['physkit_datasets'] = physkit_datasets
    sys.modules['physkit_annotation'] = physkit_annotation
    sys.modules['physkit_evaluation'] = physkit_evaluation
    
    # Import main components for easy access
    from .physkit_core import PhysKitLogger
    from .physkit_core.models import PhysicsProblem, PhysicalDataset
    from .physkit_core.definitions import PhysicsDomain, AnswerType
except ImportError:
    # Allow package to be imported even if subpackages aren't installed
    pass

__all__ = [
    "PhysKitLogger",
    "PhysicsProblem", 
    "PhysicalDataset",
    "PhysicsDomain",
    "AnswerType",
]
