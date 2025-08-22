"""
Domain assessment workflow module for physics problem classification.

This module provides domain annotation and assessment functionality that can be composed
into larger annotation workflows.
"""

from typing import Any, Dict, Optional, List
from physkit_core.definitions.physics_domain import PhysicsDomain
from physkit_annotation.annotators import DomainAnnotator
from physkit_annotation.annotations.domain import DomainAnnotation
from .base_module import BaseWorkflowModule


class DomainAssessmentModule(BaseWorkflowModule):
    """
    Workflow module for domain assessment of physics problems.
    
    This module classifies physics problems into their respective domains
    (e.g., mechanics, electromagnetism, quantum physics, etc.) and assesses
    the quality of the annotations.
    """
    
    def __init__(
        self,
        name: str = "domain_assessment",
        model: str = "o3-mini",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, model, config)
        
        # Initialize the domain annotator
        self.domain_annotator = DomainAnnotator(model=model)
        
        # Update stats with domain-specific information
        self.stats.update({
            "annotation_type": "domain",
            "domains_identified": 0,
            "confidence_scores": [],
            "problems_with_multiple_domains": 0,
            "assessment_successes": 0,
            "assessment_failures": 0
        })
    
    def _annotate_domain(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform domain annotation for a problem.
        
        Args:
            problem_data: Problem data dictionary
            
        Returns:
            Dictionary containing annotation results and status
        """
        problem_id = problem_data.get("problem_id", "unknown")
        question = problem_data.get("question", "")
        
        try:
            # Perform domain annotation - returns List[DomainAnnotation]
            domain_annotation = self.domain_annotator.annotate(question)
            
            if not domain_annotation:
                return {
                    "status": "FAILED",
                    "error": "No domain annotations returned",
                }
            
            # Extract confidence scores and update statistics
            if hasattr(domain_annotation, 'confidence') and domain_annotation.confidence is not None:
                self.stats["confidence_scores"].append(domain_annotation.confidence)
                
            if hasattr(domain_annotation, 'domains') and domain_annotation.domains:
                self.stats["domains_identified"] += len(domain_annotation.domains)
            
            return {
                "status": "SUCCESS",
                "llm_annotation": domain_annotation,
                "metadata": {
                    "module_name": self.name,
                    "model_used": self.model,
                    "annotation_type": "domain",
                    "annotation_count": len(domain_annotation.domains),
                    "timestamp": self.stats.get("start_time", "").isoformat() if self.stats.get("start_time") else None
                }
            }
            
        except Exception as e:
            self.logger.error("Domain annotation failed for problem %s: %s", problem_id, str(e))
            return {
                "status": "FAILED",
                "error": str(e)
            }
    
    def _assess_annotated_domain(
        self,
        problem_data: Dict[str, Any],
        llm_annotation: DomainAnnotation,
    ) -> Dict[str, Any]:
        """
        Assess the quality of domain annotations.
        
        Args:
            problem_data: Original problem data
            llm_annotations: List of domain annotations to assess
            
        Returns:
            Dictionary containing assessment results
        """
        
        try:
            annotated_domains = llm_annotation.domains
            
            # Check if any domains are annotated
            if not annotated_domains:
                return {
                    "status": "FAILED",
                    "error": "No domains annotated by LLM"
                }
            
            valid_annotated_domains = []
            
            # Check if the annotated domains are valid
            for domain in annotated_domains:
                if domain in PhysicsDomain:
                    valid_annotated_domains.append(domain)
                else:
                    self.logger.warning("Invalid domain: %s", domain)
            
            if not valid_annotated_domains:
                return {
                    "status": "FAILED",
                    "error": "No valid domains annotated by LLM"
                }
            
            # Check the annotation's correctness
            gold_domain = PhysicsDomain.from_string(problem_data.get("domain"))
            
            if gold_domain:
                if gold_domain in valid_annotated_domains:
                    return {
                        "status": "SUCCESS",
                        "domain_to_proceed": gold_domain,
                        "metadata": {
                            "reasoning": "The gold domain is in the annotated domains",
                            "valid_annotated_domains": valid_annotated_domains,
                            "gold_domain": gold_domain
                        }
                    }
                else:
                    return {
                        "status": "SUCCESS",
                        "metadata": {
                            "reasoning": "DUMMY SUCCESS - GOLD DOMAIN not in annotated domains",
                            "valid_annotated_domains": valid_annotated_domains,
                            "gold_domain": gold_domain
                        }
                    }
            else:
                # TODO: decide if domain is correct when gold domain is not provided
                return {
                    "status": "SUCCESS",
                    "metadata": {
                        "reasoning": "DUMMY SUCCESS - NO GOLD DOMAIN PROVIDED",
                    }
                }


            
        except Exception as e:
            self.logger.error("Domain assessment failed: %s", str(e))
            self.stats["assessment_failures"] += 1
            return {
                "status": "FAILED",
                "error": str(e)
            }
    
    def process(
        self,
        problem_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single problem for domain annotation and assessment.
        
        Args:
            problem_data: Problem data dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing the problem data and domain assessment results
        """
        problem_id = problem_data.get("problem_id", "unknown")
        question = problem_data.get("question", problem_data.get("content", ""))
        
        if not question:
            self.logger.warning("No question found for problem: %s", problem_id)
            return {
                "problem_id": problem_id,
                "problem_data": {**problem_data},
                "status": "FAILED",
                "error": "No question found"
            }
        
        # Step 1: Domain annotation
        domain_annotation_result = self._annotate_domain(problem_data)
        if domain_annotation_result["status"] == "FAILED":
            return {
                "problem_id": problem_id,
                "problem_data": {**problem_data},
                "status": "FAILED",
                "error": domain_annotation_result.get("error", "Unknown error in domain annotation")
            }
        
        # Step 2: Assess the domain annotation
        domain_assessment_result = self._assess_annotated_domain(
            problem_data,
            domain_annotation_result.get("llm_annotation", [])
        )
        
        if domain_assessment_result["status"] == "FAILED":
            return {
                "problem_id": problem_id,
                "problem_data": {**problem_data},
                "status": "FAILED",
                "error": domain_assessment_result.get("error", "Unknown error in domain assessment")
            }
        
        # Return successful result with all information
        return {
            "problem_id": problem_id,
            "problem_data": {**problem_data},
            "status": "SUCCESS",
            "llm_annotation": domain_annotation_result.get("llm_annotation", []),
            "domain_to_proceed": domain_assessment_result.get("domain_to_proceed", ""),
            "annotation_metadata": domain_annotation_result.get("metadata", {}),
            "assessment_metadata": domain_assessment_result.get("metadata", {})
        }
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get domain-specific statistics."""
        stats = self.stats.copy()
        
        # Calculate average confidence if available
        if stats["confidence_scores"]:
            stats["average_confidence"] = sum(stats["confidence_scores"]) / len(stats["confidence_scores"])
            stats["min_confidence"] = min(stats["confidence_scores"])
            stats["max_confidence"] = max(stats["confidence_scores"])
        
        # Calculate success rates
        total_assessments = stats["assessment_successes"] + stats["assessment_failures"]
        if total_assessments > 0:
            stats["assessment_success_rate"] = stats["assessment_successes"] / total_assessments
            stats["assessment_failure_rate"] = stats["assessment_failures"] / total_assessments
        
        return stats
    
    def _validate_config(self) -> None:
        """Validate module configuration."""
        # Add any domain-specific validation here
        pass
