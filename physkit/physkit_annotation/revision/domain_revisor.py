"""
Domain annotation revisor.

This revisor corrects incorrect domain annotations or replaces them with
golden annotations when LLM annotations fail quality assessment.
"""

from typing import Dict, Any, Optional
from .base_revisor import BaseRevisor, RevisionResult
from ..annotators.domain import DomainAnnotation
from ..assessment.base_assessor import AssessmentResult
from physkit.models import PhysicsDomain


class DomainRevisor(BaseRevisor):
    """
    Revisor for domain annotations.
    
    Corrects incorrect domain classifications or replaces them with
    golden annotation domain labels when LLM annotations fail assessment.
    """
    
    def __init__(self):
        """Initialize the domain revisor."""
        super().__init__()
        self.revision_strategy = {
            "no_change": "No revision needed",
            "golden_annotation": "Replace incorrect LLM annotation with golden annotation",
            "human_answer": "Use human answer to correct the annotation",
        }
    
    def revise(
        self,
        assessment_result: AssessmentResult,
        context: Optional[Dict[str, Any]] = None
    ) -> RevisionResult:
        """
        Revise domain annotation based on assessment results.
        
        Args:
            assessment_result: AssessmentResult from domain assessment
            context: Additional context (e.g., question text)
            
        Returns:
            RevisionResult with the revised annotation
        """
        
        # Check if revision is needed
        if assessment_result.suggested_correction == "no_change":
            return RevisionResult(
                original_annotation=assessment_result.metadata.get("llm_prediction", ""),
                revised_annotation=assessment_result.metadata.get("llm_prediction", ""),
                revision_type="no_change",
                explanation="No revision needed",
                metadata={
                    "assessment_result": assessment_result.assessment_result,
                    "suggested_correction": assessment_result.suggested_correction,
                    "revision_strategy": self.revision_strategy["no_change"]
                }
            )
        elif assessment_result.suggested_correction == "golden_annotation":
            return RevisionResult(
                original_annotation=assessment_result.metadata.get("llm_prediction", ""),
                revised_annotation=assessment_result.metadata.get("golden_annotation", ""),
                revision_type="golden_annotation",
                explanation="Using golden annotation",
                metadata={
                    "assessment_result": assessment_result.assessment_result,
                    "suggested_correction": assessment_result.suggested_correction,
                    "revision_strategy": self.revision_strategy["golden_annotation"]
                }
            )
        elif assessment_result.suggested_correction == "human_answer":
            if not assessment_result.metadata.get("human_answer", ""):
                raise ValueError("Human answer is not provided")
            
            return RevisionResult(
                original_annotation=assessment_result.metadata.get("llm_prediction", ""),
                revised_annotation=assessment_result.metadata.get("human_answer", ""),
                revision_type="human_answer",
                explanation="Asking human to provide the correct answer",
                metadata={
                    "assessment_result": assessment_result.assessment_result,
                    "suggested_correction": assessment_result.suggested_correction,
                    "revision_strategy": self.revision_strategy["human_answer"]
                }
            )
        else:
            raise ValueError("No revision strategy provided")

    def get_revision_strategy(self) -> Dict[str, Any]:
        """Get the revision strategy and rules."""
        return self.revision_strategy.copy()
