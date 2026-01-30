"""
Annotation data models and types.

This module provides data models for annotations including domain labels
and theorem annotations.
"""

__version__ = "0.1.0"
__author__ = "Yinghuan Zhang"
__author_email__ = "yinghuan.flash@gmail.com"

from .domain import DomainAnnotation
from .theorem import TheoremAnnotation

__all__ = [
    "DomainAnnotation",
    "TheoremAnnotation",
]
