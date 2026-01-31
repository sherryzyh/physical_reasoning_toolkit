"""
Annotation data models and types.

This module provides data models for annotations including domain labels
and theorem annotations.
"""

from .domain import DomainAnnotation
from .theorem import TheoremAnnotation

__all__ = [
    "DomainAnnotation",
    "TheoremAnnotation",
]
