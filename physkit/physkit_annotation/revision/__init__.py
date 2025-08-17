"""
Revision module for correcting annotation errors.

This module provides tools to revise incorrect annotations or replace them
with golden truth annotations when LLM annotations fail quality assessment.
"""

from .domain_revisor import DomainRevisor
from .base_revisor import BaseRevisor

__all__ = [
    "BaseRevisor",
    "DomainRevisor"
]
