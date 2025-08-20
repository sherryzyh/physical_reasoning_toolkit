"""
Physical annotation package for physics problems using LLMs.
"""
from physkit_core.models import PhysicsDomain

from .annotators.domain import DomainAnnotation
from .annotators.theorem import TheoremAnnotation
from .annotators.variable import VariableAnnotation
from .annotators.final_answer import FinalAnswer

from .annotators import (
    BaseAnnotator, DomainAnnotator, TheoremAnnotator, 
    VariableAnnotator, FinalAnswerAnnotator
)

from .workflows import SupervisedAnnotationWorkflow, AnnotationWorkflow
from .assessment import DomainAssessor
from .revision import DomainRevisor

__version__ = "2.0.0"

__all__ = [
    # Core models
    "PhysicsDomain",
    "DomainAnnotation",
    "TheoremAnnotation",
    "VariableAnnotation",
    "FinalAnswer",
    
    # Individual annotators
    "BaseAnnotator",
    "DomainAnnotator",
    "TheoremAnnotator", 
    "VariableAnnotator",
    "FinalAnswerAnnotator",
    
    # Orchestration classes
    "AnnotationWorkflow",
    "SupervisedAnnotationWorkflow",
    
    # Quality Control classes
    "DomainAssessor",
    "DomainRevisor",
]
