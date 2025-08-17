"""
Assessment module for evaluating annotation quality.

This module provides tools to assess the correctness of LLM-generated annotations
by comparing them against golden truth annotations.
"""

from .domain_assessor import DomainAssessor
from .base_assessor import BaseAssessor

__all__ = [
    "BaseAssessor",
    "DomainAssessor"
]
