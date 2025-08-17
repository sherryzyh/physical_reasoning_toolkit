"""
Annotators module for physical problem annotation.

This module contains individual annotator classes for each step of the
physical problem annotation process.
"""

from .base import BaseAnnotator
from .domain import DomainAnnotator
from .theorem import TheoremAnnotator
from .variable import VariableAnnotator
from .final_answer import FinalAnswerAnnotator

__all__ = [
    "BaseAnnotator",
    "DomainAnnotator", 
    "TheoremAnnotator",
    "VariableAnnotator",
    "FinalAnswerAnnotator",
]
