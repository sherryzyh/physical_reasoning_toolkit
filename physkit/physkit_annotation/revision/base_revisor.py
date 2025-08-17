"""
Base class for annotation revision.

All revisors should inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class RevisionResult:
    """Result of an annotation revision."""
    original_annotation: Any
    revised_annotation: Any
    revision_type: str  # "correction", "golden_truth", "no_change"
    explanation: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseRevisor(ABC):
    """
    Abstract base class for annotation revisors.
    
    Revisors correct incorrect annotations or replace them with golden truth
    annotations when LLM annotations fail quality assessment.
    """
    
    def __init__(self):
        """Initialize the revisor."""
        pass
    
    @abstractmethod
    def revise(
        self,
        assessment_result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> RevisionResult:
        """
        Revise an annotation based on assessment results.
        
        Args:
            llm_annotation: The original LLM annotation
            golden_annotation: The golden truth annotation
            assessment_result: Result from the assessment step
            context: Additional context (e.g., question, problem description)
            
        Returns:
            RevisionResult with the revised annotation
        """
        pass
    
    @abstractmethod
    def get_revision_strategy(self) -> Dict[str, Any]:
        """
        Get the revision strategy and rules.
        
        Returns:
            Dictionary describing revision strategy and rules
        """
        pass

