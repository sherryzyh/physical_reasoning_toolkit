"""
Annotators module for physical problem annotation.

This module contains individual annotator classes for each step of the
physical problem annotation process.
"""

from .domain_labeler import DomainLabeler
from .theorem_detector import TheoremDetector

__all__ = [
    "DomainLabeler", 
    "TheoremDetector",
]
